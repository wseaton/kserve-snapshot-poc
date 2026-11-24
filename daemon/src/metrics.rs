//! What the daemon has done and what its stores hold, for Prometheus.
//!
//! ```text
//! snapshotd_pods_total{namespace, workload, how}       pods brought to serving
//! snapshotd_pod_failures_total{namespace, workload}    attempts that ended in an error
//! snapshotd_step_seconds{step}                         cold_boot, checkpoint, restore, pull
//! snapshotd_store_bytes{class, namespace, workload}    as of the last look at the stores
//! snapshotd_store_revisions{class, namespace, workload}
//! snapshotd_store_free_bytes{class}
//! snapshotd_swept_revisions_total{class}               removed for being unused
//! snapshotd_swept_bytes_total{class}
//! snapshotd_timeslice_operations_total{action, outcome}
//! snapshotd_timeslice_operation_seconds{action}
//! ```
use std::convert::Infallible;
use std::fmt::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::header::CONTENT_TYPE;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;
use tokio::net::TcpListener;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::store::Usage;

const OPENMETRICS: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

macro_rules! label_values {
    ($($name:ident { $($(#[$meta:meta])* $variant:ident => $text:literal),+ $(,)? })+) => {$(
        #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
        pub enum $name {
            $($(#[$meta])* $variant),+
        }

        impl EncodeLabelValue for $name {
            fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
                encoder.write_str(match self {
                    $(Self::$variant => $text),+
                })
            }
        }
    )+};
}

label_values! {
    How {
        Restore => "restore",
        ColdBoot => "cold_boot",
        Capture => "capture",
        StartedEarlier => "started_earlier",
    }
    Step {
        ColdBoot => "cold_boot",
        Checkpoint => "checkpoint",
        Restore => "restore",
        #[cfg_attr(not(feature = "nixl"), allow(dead_code))]
        Pull => "pull",
    }
    Action {
        Snapshot => "snapshot",
        Restore => "restore",
    }
    Outcome {
        Complete => "complete",
        Failed => "failed",
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Workload {
    namespace: String,
    workload: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Served {
    namespace: String,
    workload: String,
    how: How,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Stepped {
    step: Step,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Stored {
    class: String,
    namespace: String,
    workload: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Class {
    class: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Operated {
    action: Action,
    outcome: Outcome,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Acted {
    action: Action,
}

type Seconds<L> = Family<L, Histogram, fn() -> Histogram>;

/// 0.25 s to about 17 minutes.
fn seconds() -> Histogram {
    Histogram::new(exponential_buckets(0.25, 2.0, 13))
}

pub struct Metrics {
    registry: Registry,
    pods: Family<Served, Counter>,
    pod_failures: Family<Workload, Counter>,
    step_seconds: Seconds<Stepped>,
    store_bytes: Family<Stored, Gauge>,
    store_revisions: Family<Stored, Gauge>,
    store_free_bytes: Family<Class, Gauge>,
    swept_revisions: Family<Class, Counter>,
    swept_bytes: Family<Class, Counter>,
    timeslice_operations: Family<Operated, Counter>,
    timeslice_operation_seconds: Seconds<Acted>,
}

fn clamped(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

impl Default for Metrics {
    fn default() -> Self {
        let mut metrics = Self {
            registry: Registry::with_prefix("snapshotd"),
            pods: Family::default(),
            pod_failures: Family::default(),
            step_seconds: Family::new_with_constructor(seconds),
            store_bytes: Family::default(),
            store_revisions: Family::default(),
            store_free_bytes: Family::default(),
            swept_revisions: Family::default(),
            swept_bytes: Family::default(),
            timeslice_operations: Family::default(),
            timeslice_operation_seconds: Family::new_with_constructor(seconds),
        };
        let registry = &mut metrics.registry;
        registry.register(
            "pods",
            "Pods brought to serving, by how",
            metrics.pods.clone(),
        );
        registry.register(
            "pod_failures",
            "Attempts to bring a pod to serving that ended in an error",
            metrics.pod_failures.clone(),
        );
        registry.register(
            "step_seconds",
            "How long a cold boot, checkpoint, restore or pull took",
            metrics.step_seconds.clone(),
        );
        registry.register(
            "store_bytes",
            "Bytes a workload's revisions hold in a store class",
            metrics.store_bytes.clone(),
        );
        registry.register(
            "store_revisions",
            "Revisions of a workload in a store class",
            metrics.store_revisions.clone(),
        );
        registry.register(
            "store_free_bytes",
            "Free bytes on a store class's filesystem",
            metrics.store_free_bytes.clone(),
        );
        registry.register(
            "swept_revisions",
            "Revisions removed for being unused past the class's retention",
            metrics.swept_revisions.clone(),
        );
        registry.register(
            "swept_bytes",
            "Bytes those revisions held",
            metrics.swept_bytes.clone(),
        );
        registry.register(
            "timeslice_operations",
            "Time-slicing snapshots and restores, by outcome",
            metrics.timeslice_operations.clone(),
        );
        registry.register(
            "timeslice_operation_seconds",
            "How long a time-slicing snapshot or restore took",
            metrics.timeslice_operation_seconds.clone(),
        );
        metrics
    }
}

impl Metrics {
    pub fn served(&self, namespace: &str, workload: &str, how: How) {
        self.pods
            .get_or_create(&Served {
                namespace: namespace.to_string(),
                workload: workload.to_string(),
                how,
            })
            .inc();
    }

    pub fn failed(&self, namespace: &str, workload: &str) {
        self.pod_failures
            .get_or_create(&Workload {
                namespace: namespace.to_string(),
                workload: workload.to_string(),
            })
            .inc();
    }

    pub fn stepped(&self, step: Step, took: Duration) {
        self.step_seconds
            .get_or_create(&Stepped { step })
            .observe(took.as_secs_f64());
    }

    /// Replace what is known about the stores with a new look at each class:
    /// its name, what it holds and its free bytes.
    pub fn stored(&self, classes: &[(String, Vec<Usage>, u64)]) {
        for family in [&self.store_bytes, &self.store_revisions] {
            family.clear();
        }
        for (class, held, free_bytes) in classes {
            for usage in held {
                let stored = Stored {
                    class: class.clone(),
                    namespace: usage.namespace.clone(),
                    workload: usage.workload.clone(),
                };
                self.store_bytes
                    .get_or_create(&stored)
                    .set(clamped(usage.bytes));
                self.store_revisions
                    .get_or_create(&stored)
                    .set(clamped(usage.revisions));
            }
            self.store_free_bytes
                .get_or_create(&Class {
                    class: class.clone(),
                })
                .set(clamped(*free_bytes));
        }
    }

    pub fn swept(&self, class: &str, revisions: u64, bytes: u64) {
        let class = Class {
            class: class.to_string(),
        };
        self.swept_revisions.get_or_create(&class).inc_by(revisions);
        self.swept_bytes.get_or_create(&class).inc_by(bytes);
    }

    pub fn operated(&self, action: Action, outcome: Outcome, took: Duration) {
        self.timeslice_operations
            .get_or_create(&Operated { action, outcome })
            .inc();
        self.timeslice_operation_seconds
            .get_or_create(&Acted { action })
            .observe(took.as_secs_f64());
    }

    pub fn text(&self) -> Result<String> {
        let mut text = String::new();
        encode(&mut text, &self.registry).context("encode the metrics")?;
        Ok(text)
    }

    fn answer(&self, request: &Request<Incoming>) -> Response<Full<Bytes>> {
        let (status, kind, body) = match (request.uri().path(), self.text()) {
            ("/metrics", Ok(text)) => (StatusCode::OK, OPENMETRICS, text),
            ("/metrics", Err(error)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "text/plain",
                format!("{error:#}\n"),
            ),
            _ => (StatusCode::NOT_FOUND, "text/plain", String::new()),
        };
        let mut response = Response::new(Full::new(Bytes::from(body)));
        *response.status_mut() = status;
        if let Ok(kind) = kind.parse() {
            response.headers_mut().insert(CONTENT_TYPE, kind);
        }
        response
    }
}

/// Serve `/metrics` in the background.
pub async fn start(address: SocketAddr, metrics: Arc<Metrics>) -> Result<()> {
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("listen on {address}"))?;
    info!(%address, "serving metrics");
    tokio::spawn(async move {
        loop {
            let stream = match listener.accept().await {
                Ok((stream, _)) => stream,
                Err(error) => {
                    warn!(%error, "could not accept a metrics connection");
                    sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };
            let metrics = Arc::clone(&metrics);
            tokio::spawn(async move {
                let service = service_fn(move |request| {
                    let metrics = Arc::clone(&metrics);
                    async move { Ok::<_, Infallible>(metrics.answer(&request)) }
                });
                let served = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service);
                if let Err(error) = served.await {
                    debug!(%error, "a metrics connection ended badly");
                }
            });
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::metrics::{Action, How, Metrics, Outcome, Step};
    use crate::store::Usage;

    fn usage(namespace: &str, workload: &str, bytes: u64, revisions: u64) -> Usage {
        Usage {
            namespace: namespace.to_string(),
            workload: workload.to_string(),
            bytes,
            revisions,
        }
    }

    #[test]
    fn what_happened_is_written_in_the_prometheus_text_format() {
        let metrics = Metrics::default();
        metrics.served("serving", "qwen", How::Restore);
        metrics.served("serving", "qwen", How::Restore);
        metrics.served("serving", "qwen", How::ColdBoot);
        metrics.failed("serving", "qwen");
        metrics.stepped(Step::Restore, Duration::from_secs(3));
        metrics.swept("default", 2, 4096);
        metrics.operated(Action::Snapshot, Outcome::Failed, Duration::from_secs(20));
        let text = metrics.text().unwrap();
        for line in [
            r#"snapshotd_pods_total{namespace="serving",workload="qwen",how="restore"} 2"#,
            r#"snapshotd_pods_total{namespace="serving",workload="qwen",how="cold_boot"} 1"#,
            r#"snapshotd_pod_failures_total{namespace="serving",workload="qwen"} 1"#,
            r#"snapshotd_step_seconds_bucket{le="4.0",step="restore"} 1"#,
            r#"snapshotd_step_seconds_bucket{le="2.0",step="restore"} 0"#,
            r#"snapshotd_swept_revisions_total{class="default"} 2"#,
            r#"snapshotd_swept_bytes_total{class="default"} 4096"#,
            r#"snapshotd_timeslice_operations_total{action="snapshot",outcome="failed"} 1"#,
            r#"snapshotd_timeslice_operation_seconds_count{action="snapshot"} 1"#,
        ] {
            assert!(text.contains(line), "no {line:?} in:\n{text}");
        }
    }

    #[test]
    fn a_new_look_at_a_store_replaces_the_last() {
        let metrics = Metrics::default();
        let look = |held| {
            vec![
                ("default".to_string(), held, 70),
                ("shared".to_string(), vec![], 9),
            ]
        };
        metrics.stored(&look(vec![usage("serving", "qwen", 10, 2)]));
        metrics.stored(&look(vec![usage("serving", "glm", 30, 1)]));
        let text = metrics.text().unwrap();
        assert!(!text.contains(r#"workload="qwen""#), "{text}");
        for line in [
            r#"snapshotd_store_bytes{class="default",namespace="serving",workload="glm"} 30"#,
            r#"snapshotd_store_revisions{class="default",namespace="serving",workload="glm"} 1"#,
            r#"snapshotd_store_free_bytes{class="default"} 70"#,
            r#"snapshotd_store_free_bytes{class="shared"} 9"#,
        ] {
            assert!(text.contains(line), "no {line:?} in:\n{text}");
        }
    }
}
