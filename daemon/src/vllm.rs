//! vLLM's process checkpoint lifecycle, over the Rust frontend's gRPC
//! `Control` service or the `/checkpoint` routes both frontends serve over
//! HTTP.
mod grpc;
mod http;

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Error, Result, anyhow};
use clap::ValueEnum;
use tokio::time::{Instant, sleep};

const POLL: Duration = Duration::from_millis(200);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Running,
    Preparing,
    Prepared,
    Restoring,
}

#[derive(Debug, thiserror::Error)]
pub enum VllmError {
    #[error("vLLM's checkpoint lifecycle failed: {reason}")]
    Failed { reason: String },
    #[error("vLLM reported no checkpoint state")]
    NoState,
    #[error("vLLM answered {code}: {body:?}")]
    Unexpected {
        code: hyper::StatusCode,
        body: String,
    },
    #[error("vLLM at {address} is still {state:?}")]
    NotUp {
        address: SocketAddr,
        state: Lifecycle,
    },
    #[error("vLLM is still {state:?} after {limit:?}, waiting for {wanted:?}")]
    TimedOut {
        state: Lifecycle,
        limit: Duration,
        wanted: Lifecycle,
    },
}

fn failed(reason: Option<&str>) -> Error {
    VllmError::Failed {
        reason: reason.unwrap_or("no reason given").to_string(),
    }
    .into()
}

/// How the daemon drives a server's checkpoint lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Transport {
    /// The Rust frontend's gRPC `Control` service.
    Grpc,
    /// The `/checkpoint` routes on the inference port, which the Python
    /// frontend serves too.
    Http,
}

impl Transport {
    /// The transport a pod names, or `default` when it names none.
    pub fn named(name: Option<&str>, default: Self) -> Result<Self> {
        match name {
            None => Ok(default),
            Some(name) => Self::from_str(name, false)
                .map_err(|_| anyhow!("no vLLM control transport {name:?}; grpc or http")),
        }
    }
}

/// How in-flight requests are drained before a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PauseMode {
    /// Let them finish.
    Wait,
    /// Abort them.
    Abort,
    /// Pause them and keep them for after the restore.
    Keep,
}

impl PauseMode {
    fn name(self) -> &'static str {
        match self {
            Self::Wait => "wait",
            Self::Abort => "abort",
            Self::Keep => "keep",
        }
    }
}

/// Where a pod's server answers, and over which transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub transport: Transport,
    pub address: SocketAddr,
}

pub enum Control {
    Grpc(grpc::Control),
    Http(http::Control),
}

impl Control {
    pub async fn connect(endpoint: Endpoint) -> Result<Self> {
        Ok(match endpoint.transport {
            Transport::Grpc => Self::Grpc(grpc::Control::connect(endpoint.address).await?),
            Transport::Http => Self::Http(http::Control::new(endpoint.address)),
        })
    }

    /// Wait for a starting server to answer: serving, or prepared for a
    /// checkpoint if it prepares itself at startup.
    pub async fn connect_when_up(endpoint: Endpoint, limit: Duration) -> Result<(Self, Lifecycle)> {
        let address = endpoint.address;
        let deadline = Instant::now() + limit;
        loop {
            let attempt = async {
                let mut control = Self::connect(endpoint).await?;
                control.state().await.map(|state| (control, state))
            };
            match attempt.await {
                Ok((control, state @ (Lifecycle::Running | Lifecycle::Prepared))) => {
                    return Ok((control, state));
                }
                Ok(_) | Err(_) if Instant::now() < deadline => sleep(POLL * 5).await,
                Ok((_, state)) => return Err(VllmError::NotUp { address, state }.into()),
                Err(error) => return Err(error.context("vLLM did not start serving")),
            }
        }
    }

    pub async fn state(&mut self) -> Result<Lifecycle> {
        match self {
            Self::Grpc(control) => control.state().await,
            Self::Http(control) => control.state().await,
        }
    }

    /// Returns once it is safe to capture.
    pub async fn prepare(&mut self, mode: PauseMode, limit: Duration) -> Result<()> {
        match self {
            Self::Grpc(control) => control.prepare(mode).await?,
            Self::Http(control) => control.prepare(mode).await?,
        }
        self.wait_for(Lifecycle::Prepared, limit).await
    }

    /// Returns once the server serves.
    pub async fn restore(&mut self, limit: Duration) -> Result<()> {
        match self {
            Self::Grpc(control) => control.restore().await?,
            Self::Http(control) => control.restore().await?,
        }
        self.wait_for(Lifecycle::Running, limit).await
    }

    async fn wait_for(&mut self, wanted: Lifecycle, limit: Duration) -> Result<()> {
        let deadline = Instant::now() + limit;
        loop {
            let state = self.state().await?;
            if state == wanted {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(VllmError::TimedOut {
                    state,
                    limit,
                    wanted,
                }
                .into());
            }
            sleep(POLL).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vllm::{PauseMode, Transport};

    #[test]
    fn a_pod_names_its_transport_or_takes_the_default() {
        assert_eq!(
            Transport::named(None, Transport::Grpc).unwrap(),
            Transport::Grpc
        );
        assert_eq!(
            Transport::named(None, Transport::Http).unwrap(),
            Transport::Http
        );
        assert_eq!(
            Transport::named(Some("http"), Transport::Grpc).unwrap(),
            Transport::Http
        );
        assert_eq!(
            Transport::named(Some("grpc"), Transport::Http).unwrap(),
            Transport::Grpc
        );
        for name in ["HTTP", "rest", ""] {
            let error = Transport::named(Some(name), Transport::Grpc).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("no vLLM control transport {name:?}; grpc or http")
            );
        }
    }

    #[test]
    fn pause_modes_are_named_as_the_http_api_names_them() {
        assert_eq!(PauseMode::Wait.name(), "wait");
        assert_eq!(PauseMode::Abort.name(), "abort");
        assert_eq!(PauseMode::Keep.name(), "keep");
    }
}
