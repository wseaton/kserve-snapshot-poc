use std::net::SocketAddr;

use anyhow::{Context, Result, bail};
use tonic::transport::{Channel, Endpoint};

use crate::vllm::{CONNECT_TIMEOUT, Lifecycle, PauseMode, failed};

#[allow(clippy::all)]
pub mod pb {
    tonic::include_proto!("vllm");
}

use pb::control_client::ControlClient;

/// A failed lifecycle is an error carrying the server's reason.
fn lifecycle(response: &pb::CheckpointStateResponse) -> Result<Lifecycle> {
    match response.state() {
        pb::CheckpointState::Running => Ok(Lifecycle::Running),
        pb::CheckpointState::Preparing => Ok(Lifecycle::Preparing),
        pb::CheckpointState::Prepared => Ok(Lifecycle::Prepared),
        pb::CheckpointState::Restoring => Ok(Lifecycle::Restoring),
        pb::CheckpointState::Failed => Err(failed(response.error.as_deref())),
        pb::CheckpointState::Unspecified => bail!("vLLM reported no checkpoint state"),
    }
}

impl From<PauseMode> for pb::PauseMode {
    fn from(mode: PauseMode) -> Self {
        match mode {
            PauseMode::Wait => Self::Wait,
            PauseMode::Abort => Self::Abort,
            PauseMode::Keep => Self::Keep,
        }
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

    pub async fn state(&mut self) -> Result<Lifecycle> {
        let response = self
            .client
            .get_checkpoint_state(pb::GetCheckpointStateRequest {})
            .await
            .context("GetCheckpointState")?;
        lifecycle(response.get_ref())
    }

    pub async fn prepare(&mut self, mode: PauseMode) -> Result<()> {
        self.client
            .checkpoint_prepare(pb::CheckpointPrepareRequest {
                mode: pb::PauseMode::from(mode).into(),
            })
            .await
            .context("CheckpointPrepare")?;
        Ok(())
    }

    pub async fn restore(&mut self) -> Result<()> {
        self.client
            .checkpoint_restore(pb::CheckpointRestoreRequest {})
            .await
            .context("CheckpointRestore")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::vllm::PauseMode;
    use crate::vllm::grpc::{lifecycle, pb};

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
        assert_eq!(
            error.to_string(),
            "vLLM's checkpoint lifecycle failed: could not release the weights"
        );
        let error = lifecycle(&response(pb::CheckpointState::Failed, None)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "vLLM's checkpoint lifecycle failed: no reason given"
        );
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

    #[test]
    fn pause_modes_map_onto_the_schema() {
        for (mode, wanted) in [
            (PauseMode::Wait, pb::PauseMode::Wait),
            (PauseMode::Abort, pb::PauseMode::Abort),
            (PauseMode::Keep, pb::PauseMode::Keep),
        ] {
            assert_eq!(pb::PauseMode::from(mode), wanted);
        }
    }
}
