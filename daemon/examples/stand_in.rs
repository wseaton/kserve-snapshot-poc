//! A stand-in for vLLM in end-to-end tests. It serves the checkpoint
//! lifecycle of vLLM's gRPC control port and reports an instance ID drawn
//! when the process starts, so a test can tell a restored process (same ID)
//! from one that booted (new ID).
use std::net::SocketAddr;
use std::sync::Mutex;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tonic::{Request, Response, Status};

#[allow(clippy::all)]
pub mod pb {
    tonic::include_proto!("vllm");
}

use pb::control_client::ControlClient;
use pb::control_server::{Control, ControlServer};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the control port until killed.
    Serve {
        #[arg(long, default_value = "0.0.0.0:50051")]
        listen: SocketAddr,
        /// Start prepared for a checkpoint, as vLLM does with
        /// --checkpoint-prepare-on-startup, and serve only after a restore.
        #[arg(long)]
        prepared: bool,
    },
    /// Print a server's instance ID and checkpoint state.
    Info { address: SocketAddr },
}

struct StandIn {
    instance: String,
    state: Mutex<pb::CheckpointState>,
}

type Rpc<T> = Result<Response<T>, Status>;

impl StandIn {
    fn step(
        &self,
        advance: impl FnOnce(pb::CheckpointState) -> pb::CheckpointState,
    ) -> Rpc<pb::CheckpointStateResponse> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Status::internal("state lock poisoned"))?;
        *state = advance(*state);
        Ok(Response::new(pb::CheckpointStateResponse {
            state: (*state).into(),
            error: None,
        }))
    }
}

/// The control service: the server info and checkpoint RPCs, and every other
/// RPC of the schema answered as unimplemented.
macro_rules! control_service {
    ($($name:ident($request:ident) -> $response:ident;)*) => {
        #[tonic::async_trait]
        impl Control for StandIn {
            async fn get_server_info(
                &self,
                _: Request<pb::GetServerInfoRequest>,
            ) -> Rpc<pb::ServerInfo> {
                Ok(Response::new(pb::ServerInfo {
                    instance_id: self.instance.clone(),
                    ..Default::default()
                }))
            }

            async fn checkpoint_prepare(
                &self,
                _: Request<pb::CheckpointPrepareRequest>,
            ) -> Rpc<pb::CheckpointStateResponse> {
                self.step(|_| pb::CheckpointState::Prepared)
            }

            async fn checkpoint_restore(
                &self,
                _: Request<pb::CheckpointRestoreRequest>,
            ) -> Rpc<pb::CheckpointStateResponse> {
                self.step(|_| pb::CheckpointState::Running)
            }

            async fn get_checkpoint_state(
                &self,
                _: Request<pb::GetCheckpointStateRequest>,
            ) -> Rpc<pb::CheckpointStateResponse> {
                self.step(|state| state)
            }

            $(async fn $name(&self, _: Request<pb::$request>) -> Rpc<pb::$response> {
                Err(Status::unimplemented(stringify!($name)))
            })*
        }
    };
}

control_service! {
    get_model_info(GetModelInfoRequest) -> ModelInfo;
    shutdown(ShutdownRequest) -> ShutdownResponse;
    abort(AbortRequest) -> AbortResponse;
    load_lora(LoadLoraRequest) -> LoadLoraResponse;
    unload_lora(UnloadLoraRequest) -> UnloadLoraResponse;
    list_loras(ListLorasRequest) -> ListLorasResponse;
    get_kv_event_sources(GetKvEventSourcesRequest) -> GetKvEventSourcesResponse;
    pause_generation(PauseGenerationRequest) -> PauseGenerationResponse;
    resume_generation(ResumeGenerationRequest) -> ResumeGenerationResponse;
    is_paused(IsPausedRequest) -> IsPausedResponse;
    sleep(SleepRequest) -> SleepResponse;
    wake_up(WakeUpRequest) -> WakeUpResponse;
    is_sleeping(IsSleepingRequest) -> IsSleepingResponse;
    init_weight_transfer_engine(InitWeightTransferEngineRequest) -> InitWeightTransferEngineResponse;
    start_weight_update(StartWeightUpdateRequest) -> StartWeightUpdateResponse;
    start_draft_weight_update(StartDraftWeightUpdateRequest) -> StartDraftWeightUpdateResponse;
    update_weights(UpdateWeightsRequest) -> UpdateWeightsResponse;
    finish_weight_update(FinishWeightUpdateRequest) -> FinishWeightUpdateResponse;
    update_weight_version(UpdateWeightVersionRequest) -> UpdateWeightVersionResponse;
    get_weight_version(GetWeightVersionRequest) -> GetWeightVersionResponse;
}

async fn serve(listen: SocketAddr, prepared: bool) -> Result<()> {
    let mut id = [0; 8];
    openssl::rand::rand_bytes(&mut id).context("draw an instance ID")?;
    let instance = hex::encode(id);
    println!("stand-in: instance {instance} on {listen}");
    tonic::transport::Server::builder()
        .add_service(ControlServer::new(StandIn {
            instance,
            state: Mutex::new(if prepared {
                pb::CheckpointState::Prepared
            } else {
                pb::CheckpointState::Running
            }),
        }))
        .serve(listen)
        .await
        .context("serve the control port")
}

async fn info(address: SocketAddr) -> Result<()> {
    let mut client = ControlClient::connect(format!("http://{address}"))
        .await
        .with_context(|| format!("connect to {address}"))?;
    let instance = client
        .get_server_info(pb::GetServerInfoRequest {})
        .await?
        .into_inner()
        .instance_id;
    let state = client
        .get_checkpoint_state(pb::GetCheckpointStateRequest {})
        .await?
        .into_inner()
        .state();
    println!("{instance} {state:?}");
    Ok(())
}

fn main() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("start the runtime")?;
    match Cli::parse().command {
        Command::Serve { listen, prepared } => runtime.block_on(serve(listen, prepared)),
        Command::Info { address } => runtime.block_on(info(address)),
    }
}
