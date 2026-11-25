//! A stand-in for vLLM in end-to-end tests. It serves the checkpoint
//! lifecycle of vLLM's gRPC control port and of its HTTP port, over one
//! state, and reports an instance ID drawn when the process starts, so a
//! test can tell a restored process (same ID) from one that booted (new ID).
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use http_body_util::{BodyExt, Empty, Full};
use hyper::body::{Bytes, Incoming};
use hyper::header::HOST;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::{TcpListener, TcpStream};
use tonic::Status;

#[allow(clippy::all)]
pub mod pb {
    tonic::include_proto!("vllm");
}

use pb::CheckpointState as State;
use pb::control_client::ControlClient;
use pb::control_server::{Control, ControlServer};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the control port and the HTTP port until killed.
    Serve {
        #[arg(long, default_value = "0.0.0.0:50051")]
        listen: SocketAddr,
        #[arg(long, default_value = "0.0.0.0:8000")]
        http_listen: SocketAddr,
        /// Serve only the HTTP port, as vLLM's Python frontend does.
        #[arg(long)]
        http_only: bool,
        /// Start prepared for a checkpoint, as vLLM does with
        /// --checkpoint-prepare-on-startup, and serve only after a restore.
        #[arg(long)]
        prepared: bool,
    },
    /// Print a server's instance ID and checkpoint state.
    Info {
        address: SocketAddr,
        /// Ask the HTTP port at `address`, not the control port.
        #[arg(long)]
        http: bool,
    },
}

fn name(state: State) -> &'static str {
    match state {
        State::Unspecified => "unspecified",
        State::Running => "running",
        State::Preparing => "preparing",
        State::Prepared => "prepared",
        State::Restoring => "restoring",
        State::Failed => "failed",
    }
}

fn named(state: &str) -> Option<State> {
    [
        State::Running,
        State::Preparing,
        State::Prepared,
        State::Restoring,
        State::Failed,
    ]
    .into_iter()
    .find(|candidate| name(*candidate) == state)
}

/// Preparing and restoring finish at once. Each returns the next state, or
/// nothing when it is refused in `state`.
fn prepare(state: State) -> Option<State> {
    match state {
        State::Running | State::Preparing | State::Prepared => Some(State::Prepared),
        State::Restoring | State::Failed | State::Unspecified => None,
    }
}

fn restore(state: State) -> Option<State> {
    match state {
        State::Prepared | State::Restoring | State::Running => Some(State::Running),
        State::Preparing | State::Failed | State::Unspecified => None,
    }
}

fn keep(state: State) -> Option<State> {
    Some(state)
}

struct StandIn {
    instance: String,
    state: Mutex<State>,
}

impl StandIn {
    fn new(instance: String, prepared: bool) -> Self {
        Self {
            instance,
            state: Mutex::new(if prepared {
                State::Prepared
            } else {
                State::Running
            }),
        }
    }

    /// The state afterwards, and whether the step was taken.
    fn step(&self, advance: fn(State) -> Option<State>) -> Result<(State, bool)> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("state lock poisoned"))?;
        Ok(match advance(*state) {
            Some(next) => {
                *state = next;
                (next, true)
            }
            None => (*state, false),
        })
    }

    fn rpc(&self, advance: fn(State) -> Option<State>) -> Rpc<pb::CheckpointStateResponse> {
        match self.step(advance) {
            Ok((state, true)) => Ok(tonic::Response::new(pb::CheckpointStateResponse {
                state: state.into(),
                error: None,
            })),
            Ok((state, false)) => Err(Status::failed_precondition(format!(
                "the stand-in is {}",
                name(state)
            ))),
            Err(error) => Err(Status::internal(error.to_string())),
        }
    }

    /// vLLM's HTTP routes: `/health` and the three `/checkpoint` routes, and
    /// `/instance` for the instance ID.
    fn answer(&self, method: &Method, path: &str) -> (StatusCode, String) {
        match (method, path) {
            (&Method::GET, "/health") => (StatusCode::OK, String::new()),
            (&Method::GET, "/instance") => (StatusCode::OK, self.instance.clone()),
            (&Method::GET, "/checkpoint/status") => self.http(keep, |state| match state {
                State::Prepared => StatusCode::OK,
                State::Failed => StatusCode::INTERNAL_SERVER_ERROR,
                _ => StatusCode::SERVICE_UNAVAILABLE,
            }),
            (&Method::POST, "/checkpoint/prepare") => self.http(prepare, |_| StatusCode::OK),
            (&Method::POST, "/checkpoint/restore") => self.http(restore, |_| StatusCode::OK),
            _ => (StatusCode::NOT_FOUND, String::new()),
        }
    }

    /// A taken step answers `accepted` of the state it leads to, a refused
    /// one 400.
    fn http(
        &self,
        advance: fn(State) -> Option<State>,
        accepted: fn(State) -> StatusCode,
    ) -> (StatusCode, String) {
        match self.step(advance) {
            Ok((state, taken)) => {
                let code = if taken {
                    accepted(state)
                } else {
                    StatusCode::BAD_REQUEST
                };
                (code, serde_json::json!({"state": name(state)}).to_string())
            }
            Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
        }
    }

    fn respond(&self, request: &Request<Incoming>) -> Response<Full<Bytes>> {
        let (code, body) = self.answer(request.method(), request.uri().path());
        let mut response = Response::new(Full::new(Bytes::from(body)));
        *response.status_mut() = code;
        response
    }
}

type Rpc<T> = Result<tonic::Response<T>, Status>;

/// The control service: the server info and checkpoint RPCs, and every other
/// RPC of the schema answered as unimplemented.
macro_rules! control_service {
    ($($name:ident($request:ident) -> $response:ident;)*) => {
        #[tonic::async_trait]
        impl Control for Arc<StandIn> {
            async fn get_server_info(
                &self,
                _: tonic::Request<pb::GetServerInfoRequest>,
            ) -> Rpc<pb::ServerInfo> {
                Ok(tonic::Response::new(pb::ServerInfo {
                    instance_id: self.instance.clone(),
                    ..Default::default()
                }))
            }

            async fn checkpoint_prepare(
                &self,
                _: tonic::Request<pb::CheckpointPrepareRequest>,
            ) -> Rpc<pb::CheckpointStateResponse> {
                self.rpc(prepare)
            }

            async fn checkpoint_restore(
                &self,
                _: tonic::Request<pb::CheckpointRestoreRequest>,
            ) -> Rpc<pb::CheckpointStateResponse> {
                self.rpc(restore)
            }

            async fn get_checkpoint_state(
                &self,
                _: tonic::Request<pb::GetCheckpointStateRequest>,
            ) -> Rpc<pb::CheckpointStateResponse> {
                self.rpc(keep)
            }

            $(async fn $name(&self, _: tonic::Request<pb::$request>) -> Rpc<pb::$response> {
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

async fn serve_http(listener: TcpListener, stand_in: Arc<StandIn>) -> Result<()> {
    loop {
        let (stream, _) = listener.accept().await.context("accept")?;
        let stand_in = Arc::clone(&stand_in);
        tokio::spawn(async move {
            let service = service_fn(move |request| {
                let stand_in = Arc::clone(&stand_in);
                async move { Ok::<_, Infallible>(stand_in.respond(&request)) }
            });
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await
            {
                eprintln!("stand-in: an HTTP connection ended badly: {error}");
            }
        });
    }
}

async fn serve(
    listen: SocketAddr,
    http_listen: SocketAddr,
    http_only: bool,
    prepared: bool,
) -> Result<()> {
    let mut id = [0; 8];
    openssl::rand::rand_bytes(&mut id).context("draw an instance ID")?;
    let instance = hex::encode(id);
    let stand_in = Arc::new(StandIn::new(instance.clone(), prepared));
    let listener = TcpListener::bind(http_listen)
        .await
        .with_context(|| format!("listen on {http_listen}"))?;
    let http = serve_http(listener, Arc::clone(&stand_in));
    if http_only {
        println!("stand-in: instance {instance} on HTTP {http_listen}");
        return http.await;
    }
    println!("stand-in: instance {instance} on {listen} and HTTP {http_listen}");
    let grpc = tonic::transport::Server::builder()
        .add_service(ControlServer::new(stand_in))
        .serve(listen);
    tokio::select! {
        served = grpc => served.context("serve the control port"),
        served = http => served.context("serve the HTTP port"),
    }
}

async fn get(address: SocketAddr, path: &str) -> Result<(StatusCode, String)> {
    let stream = TcpStream::connect(address)
        .await
        .with_context(|| format!("connect to {address}"))?;
    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .context("HTTP handshake")?;
    tokio::spawn(connection);
    let request = Request::builder()
        .uri(path)
        .header(HOST, address.to_string())
        .body(Empty::<Bytes>::new())
        .context("build the request")?;
    let response = sender
        .send_request(request)
        .await
        .with_context(|| format!("GET {path}"))?;
    let code = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .with_context(|| format!("read GET {path}"))?
        .to_bytes();
    Ok((code, String::from_utf8_lossy(&body).into_owned()))
}

async fn info_http(address: SocketAddr) -> Result<(String, State)> {
    let (_, instance) = get(address, "/instance").await?;
    let (_, status) = get(address, "/checkpoint/status").await?;
    let state = serde_json::from_str::<serde_json::Value>(&status)
        .ok()
        .and_then(|body| body.get("state")?.as_str().and_then(named))
        .with_context(|| format!("no checkpoint state in {status:?}"))?;
    Ok((instance, state))
}

async fn info_grpc(address: SocketAddr) -> Result<(String, State)> {
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
    Ok((instance, state))
}

async fn info(address: SocketAddr, http: bool) -> Result<()> {
    let (instance, state) = if http {
        info_http(address).await?
    } else {
        info_grpc(address).await?
    };
    println!("{instance} {state:?}");
    Ok(())
}

fn main() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("start the runtime")?;
    match Cli::parse().command {
        Command::Serve {
            listen,
            http_listen,
            http_only,
            prepared,
        } => runtime.block_on(serve(listen, http_listen, http_only, prepared)),
        Command::Info { address, http } => runtime.block_on(info(address, http)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hyper::{Method, StatusCode};
    use tokio::net::TcpListener;

    use crate::{StandIn, State, info_http, name, named, prepare, restore, serve_http};

    fn stand_in(state: State) -> StandIn {
        let stand_in = StandIn::new("abc".to_string(), false);
        *stand_in.state.lock().unwrap() = state;
        stand_in
    }

    fn body(state: &str) -> String {
        format!(r#"{{"state":"{state}"}}"#)
    }

    #[test]
    fn the_status_route_answers_as_vllm_does() {
        for (state, code) in [
            (State::Running, StatusCode::SERVICE_UNAVAILABLE),
            (State::Preparing, StatusCode::SERVICE_UNAVAILABLE),
            (State::Prepared, StatusCode::OK),
            (State::Restoring, StatusCode::SERVICE_UNAVAILABLE),
            (State::Failed, StatusCode::INTERNAL_SERVER_ERROR),
        ] {
            assert_eq!(
                stand_in(state).answer(&Method::GET, "/checkpoint/status"),
                (code, body(name(state))),
            );
        }
    }

    #[test]
    fn prepare_and_restore_move_the_state_or_are_refused() {
        for (path, from, wanted) in [
            ("/checkpoint/prepare", State::Running, Some(State::Prepared)),
            (
                "/checkpoint/prepare",
                State::Preparing,
                Some(State::Prepared),
            ),
            (
                "/checkpoint/prepare",
                State::Prepared,
                Some(State::Prepared),
            ),
            ("/checkpoint/prepare", State::Restoring, None),
            ("/checkpoint/prepare", State::Failed, None),
            ("/checkpoint/restore", State::Prepared, Some(State::Running)),
            (
                "/checkpoint/restore",
                State::Restoring,
                Some(State::Running),
            ),
            ("/checkpoint/restore", State::Running, Some(State::Running)),
            ("/checkpoint/restore", State::Preparing, None),
            ("/checkpoint/restore", State::Failed, None),
        ] {
            let stand_in = stand_in(from);
            let answered = stand_in.answer(&Method::POST, path);
            let after = *stand_in.state.lock().unwrap();
            match wanted {
                Some(next) => {
                    assert_eq!(
                        answered,
                        (StatusCode::OK, body(name(next))),
                        "{path} {from:?}"
                    );
                    assert_eq!(after, next);
                }
                None => {
                    assert_eq!(
                        answered,
                        (StatusCode::BAD_REQUEST, body(name(from))),
                        "{path} {from:?}"
                    );
                    assert_eq!(after, from);
                }
            }
        }
    }

    #[test]
    fn the_grpc_and_http_steps_agree() {
        let all = [
            State::Running,
            State::Preparing,
            State::Prepared,
            State::Restoring,
            State::Failed,
        ];
        for state in all {
            let refused = stand_in(state).rpc(prepare).is_err();
            assert_eq!(refused, prepare(state).is_none(), "{state:?}");
            let refused = stand_in(state).rpc(restore).is_err();
            assert_eq!(refused, restore(state).is_none(), "{state:?}");
            assert_eq!(named(name(state)), Some(state));
        }
        let error = stand_in(State::Failed).rpc(prepare).unwrap_err();
        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
        assert_eq!(error.message(), "the stand-in is failed");
    }

    #[test]
    fn health_instance_and_unknown_routes() {
        let stand_in = stand_in(State::Prepared);
        assert_eq!(
            stand_in.answer(&Method::GET, "/health"),
            (StatusCode::OK, String::new())
        );
        assert_eq!(
            stand_in.answer(&Method::GET, "/instance"),
            (StatusCode::OK, "abc".to_string())
        );
        for (method, path) in [
            (Method::GET, "/checkpoint/prepare"),
            (Method::POST, "/checkpoint/status"),
            (Method::GET, "/v1/models"),
        ] {
            assert_eq!(
                stand_in.answer(&method, path).0,
                StatusCode::NOT_FOUND,
                "{method} {path}"
            );
        }
    }

    #[tokio::test]
    async fn info_reads_the_http_port() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(serve_http(
            listener,
            Arc::new(StandIn::new("abc".to_string(), true)),
        ));
        assert_eq!(
            info_http(address).await.unwrap(),
            ("abc".to_string(), State::Prepared)
        );
    }
}
