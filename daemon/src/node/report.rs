use std::future::Future;

use anyhow::Result;
use tokio::time::Instant;
use tracing::{info, warn};

/// The part of a pod's name that tells it from its siblings. A pod synced
/// from a vCluster is named `<name>-x-<namespace>-x-<vcluster>` on the host.
pub fn short_id(pod: &str) -> &str {
    let name = pod.split("-x-").next().unwrap_or(pod);
    name.rsplit('-').next().unwrap_or(name)
}

pub fn seconds(since: Instant) -> String {
    format!("{:.2}", since.elapsed().as_secs_f64())
}

pub fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1u64 << 30) as f64)
}

pub async fn step<T>(name: &'static str, work: impl Future<Output = Result<T>>) -> Result<T> {
    let started = Instant::now();
    let outcome = work.await;
    match &outcome {
        Ok(_) => info!(seconds = %seconds(started), "{name}"),
        Err(_) => warn!(seconds = %seconds(started), "{name} failed"),
    }
    outcome
}

pub fn plain(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut rest = line.chars();
    while let Some(c) = rest.next() {
        if c == '\u{1b}' {
            for c in rest.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}

#[cfg(test)]
mod tests {
    use crate::node::report::{plain, short_id};

    #[test]
    fn colour_sequences_are_removed() {
        let line = "\u{1b}[0;36m(RustFrontend pid=421)\u{1b}[0m \u{1b}[32mINFO\u{1b}[0m ready";
        assert_eq!(plain(line), "(RustFrontend pid=421) INFO ready");
        assert_eq!(plain("no colours"), "no colours");
        assert_eq!(plain("cut off \u{1b}[3"), "cut off ");
    }

    #[test]
    fn a_pod_is_tagged_with_the_end_of_its_own_name() {
        assert_eq!(short_id("snapshot-serve-85f787b8cb-cjv9n"), "cjv9n");
        assert_eq!(
            short_id("qwen3-8b-kserve-6f79c6cd9c-zw7mr-x-kds-smoke-x-kds"),
            "zw7mr"
        );
        assert_eq!(short_id("pod"), "pod");
    }
}
