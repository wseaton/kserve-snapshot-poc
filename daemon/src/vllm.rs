//! vLLM's gRPC control port: the checkpoint lifecycle of one server.
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::time::{Instant, sleep};
use tonic::transport::{Channel, Endpoint};

#[allow(clippy::all)]
pub mod pb {
    tonic::include_proto!("vllm");
}

use pb::control_client::ControlClient;

const POLL: Duration = Duration::from_millis(200);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// Where a server is in the checkpoint lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Running,
    Preparing,
    Prepared,
    Restoring,
}

/// Turn a state response into a lifecycle; a failed lifecycle is an error
/// carrying the server's reason.
pub fn lifecycle(response: &pb::CheckpointStateResponse) -> Result<Lifecycle> {
    match response.state() {
        pb::CheckpointState::Running => Ok(Lifecycle::Running),
        pb::CheckpointState::Preparing => Ok(Lifecycle::Preparing),
        pb::CheckpointState::Prepared => Ok(Lifecycle::Prepared),
        pb::CheckpointState::Restoring => Ok(Lifecycle::Restoring),
        pb::CheckpointState::Failed => Err(anyhow!(
            "vLLM's checkpoint lifecycle failed: {}",
            response.error.as_deref().unwrap_or("no reason given")
        )),
        pb::CheckpointState::Unspecified => bail!("vLLM reported no checkpoint state"),
    }
}

pub struct Control {
    client: ControlClient<Channel>,
}

impl Control {
    pub async fn connect(address: SocketAddr) -> Result<Self> {
        let channel = Endpoint::from_shared(format!("http://{address}"))
            .context("vLLM control address")?
            .connect_timeout(CONNECT_TIMEOUT)
            .connect()
            .await
            .with_context(|| format!("connect to vLLM's control port at {address}"))?;
        Ok(Self {
            client: ControlClient::new(channel),
        })
    }

    /// Wait for a server that is starting to listen and report `Running`.
    pub async fn connect_when_running(address: SocketAddr, limit: Duration) -> Result<Self> {
        let deadline = Instant::now() + limit;
        loop {
            let attempt = async {
                let mut control = Self::connect(address).await?;
                control.state().await.map(|state| (control, state))
            };
            match attempt.await {
                Ok((control, Lifecycle::Running)) => return Ok(control),
                Ok(_) | Err(_) if Instant::now() < deadline => sleep(POLL * 5).await,
                Ok((_, state)) => bail!("vLLM at {address} is still {state:?}"),
                Err(error) => return Err(error.context("vLLM did not start serving")),
            }
        }
    }

    pub async fn state(&mut self) -> Result<Lifecycle> {
        let response = self
            .client
            .get_checkpoint_state(pb::GetCheckpointStateRequest {})
            .await
            .context("GetCheckpointState")?;
        lifecycle(response.get_ref())
    }

    /// Drain the server and release what a checkpoint must not hold, and
    /// return once it is safe to capture.
    pub async fn prepare(&mut self, limit: Duration) -> Result<()> {
        self.client
            .checkpoint_prepare(pb::CheckpointPrepareRequest {
                mode: pb::PauseMode::Wait.into(),
            })
            .await
            .context("CheckpointPrepare")?;
        self.wait_for(Lifecycle::Prepared, limit).await
    }

    /// Bring back what preparing released, and return once the server serves.
    pub async fn restore(&mut self, limit: Duration) -> Result<()> {
        self.client
            .checkpoint_restore(pb::CheckpointRestoreRequest {})
            .await
            .context("CheckpointRestore")?;
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
                bail!("vLLM is still {state:?} after {limit:?}, waiting for {wanted:?}");
            }
            sleep(POLL).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vllm::{lifecycle, pb};

    fn response(state: pb::CheckpointState, error: Option<&str>) -> pb::CheckpointStateResponse {
        pb::CheckpointStateResponse {
            state: state.into(),
            error: error.map(str::to_string),
        }
    }

    #[test]
    fn a_failed_lifecycle_is_an_error_with_the_reason() {
        let error = lifecycle(&response(
            pb::CheckpointState::Failed,
            Some("could not release the weights"),
        ))
        .unwrap_err();
        assert!(error.to_string().contains("could not release the weights"));
        let error = lifecycle(&response(pb::CheckpointState::Failed, None)).unwrap_err();
        assert!(error.to_string().contains("no reason given"));
    }

    #[test]
    fn an_unset_or_unknown_state_is_an_error() {
        assert!(lifecycle(&response(pb::CheckpointState::Unspecified, None)).is_err());
        let unknown = pb::CheckpointStateResponse {
            state: 99,
            error: None,
        };
        assert!(lifecycle(&unknown).is_err());
    }
}
