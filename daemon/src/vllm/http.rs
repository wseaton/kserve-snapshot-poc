use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Context, Error, Result, anyhow, bail};
use http_body_util::{BodyExt, Empty, Limited};
use hyper::body::Bytes;
use hyper::header::HOST;
use hyper::{Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use tokio::net::TcpStream;
use tokio::time::timeout;
use tracing::debug;

use crate::vllm::{CONNECT_TIMEOUT, Lifecycle, PauseMode, failed};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const BODY_LIMIT: usize = 64 << 10;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum State {
    Running,
    Preparing,
    Prepared,
    Restoring,
    Failed,
}

impl State {
    fn name(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Preparing => "preparing",
            Self::Prepared => "prepared",
            Self::Restoring => "restoring",
            Self::Failed => "failed",
        }
    }
}

/// The body of every `/checkpoint` answer.
#[derive(Debug, Deserialize)]
struct Answer {
    state: State,
    error: Option<String>,
}

impl Answer {
    fn parse(code: StatusCode, body: &[u8]) -> Result<Self> {
        serde_json::from_slice(body).with_context(|| unexpected(code, body))
    }

    fn lifecycle(&self) -> Result<Lifecycle> {
        match self.state {
            State::Running => Ok(Lifecycle::Running),
            State::Preparing => Ok(Lifecycle::Preparing),
            State::Prepared => Ok(Lifecycle::Prepared),
            State::Restoring => Ok(Lifecycle::Restoring),
            State::Failed => Err(failed(self.error.as_deref())),
        }
    }

    fn refusal(self) -> Error {
        match (self.state, self.error) {
            (State::Failed, error) => failed(error.as_deref()),
            (_, Some(error)) => anyhow!("vLLM refused: {error}"),
            (state, None) => anyhow!("vLLM refused: it is {}", state.name()),
        }
    }
}

fn unexpected(code: StatusCode, body: &[u8]) -> String {
    format!(
        "vLLM answered {code}: {:?}",
        String::from_utf8_lossy(body).trim()
    )
}

/// `GET /checkpoint/status`: the body names the state whatever the status.
fn status(code: StatusCode, body: &[u8]) -> Result<Lifecycle> {
    Answer::parse(code, body)?.lifecycle()
}

/// `POST /checkpoint/prepare` and `POST /checkpoint/restore`.
fn started(code: StatusCode, body: &[u8]) -> Result<()> {
    match code {
        StatusCode::OK => Answer::parse(code, body)?.lifecycle().map(drop),
        StatusCode::BAD_REQUEST => Err(Answer::parse(code, body)?.refusal()),
        _ => bail!(unexpected(code, body)),
    }
}

pub struct Control {
    address: SocketAddr,
}

impl Control {
    pub fn new(address: SocketAddr) -> Self {
        Self { address }
    }

    pub async fn state(&mut self) -> Result<Lifecycle> {
        self.call(Method::GET, "/checkpoint/status", status).await
    }

    pub async fn prepare(&mut self, mode: PauseMode) -> Result<()> {
        let path = format!("/checkpoint/prepare?mode={}", mode.name());
        self.call(Method::POST, &path, started).await
    }

    pub async fn restore(&mut self) -> Result<()> {
        self.call(Method::POST, "/checkpoint/restore", started)
            .await
    }

    async fn call<T>(
        &self,
        method: Method,
        path: &str,
        answer: impl FnOnce(StatusCode, &[u8]) -> Result<T>,
    ) -> Result<T> {
        let exchanged = timeout(REQUEST_TIMEOUT, self.exchange(method.clone(), path))
            .await
            .unwrap_or_else(|_| Err(anyhow!("no answer within {REQUEST_TIMEOUT:?}")));
        exchanged
            .and_then(|(code, body)| answer(code, &body))
            .with_context(|| format!("{method} {path} at {}", self.address))
    }

    async fn exchange(&self, method: Method, path: &str) -> Result<(StatusCode, Bytes)> {
        let address = self.address;
        let stream = timeout(CONNECT_TIMEOUT, TcpStream::connect(address))
            .await
            .unwrap_or_else(|_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("no connection within {CONNECT_TIMEOUT:?}"),
                ))
            })
            .with_context(|| format!("connect to vLLM's HTTP port at {address}"))?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .context("HTTP handshake")?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                debug!(%error, "a vLLM HTTP connection ended badly");
            }
        });
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header(HOST, address.to_string())
            .body(Empty::<Bytes>::new())
            .context("build the request")?;
        let response = sender.send_request(request).await.context("send")?;
        let code = response.status();
        let body = Limited::new(response.into_body(), BODY_LIMIT)
            .collect()
            .await
            .map_err(|error| anyhow!("read the answer: {error}"))?
            .to_bytes();
        Ok((code, body))
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use http_body_util::Full;
    use hyper::body::{Bytes, Incoming};
    use hyper::service::service_fn;
    use hyper::{Method, Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    use crate::vllm::http::{started, status};
    use crate::vllm::{Control, Endpoint, Lifecycle, PauseMode, Transport};

    const STATES: [(&str, Option<Lifecycle>); 5] = [
        ("running", Some(Lifecycle::Running)),
        ("preparing", Some(Lifecycle::Preparing)),
        ("prepared", Some(Lifecycle::Prepared)),
        ("restoring", Some(Lifecycle::Restoring)),
        ("failed", None),
    ];

    fn body(state: &str, error: Option<&str>) -> Vec<u8> {
        match error {
            Some(error) => serde_json::json!({"state": state, "error": error}),
            None => serde_json::json!({"state": state}),
        }
        .to_string()
        .into_bytes()
    }

    #[test]
    fn the_status_route_is_read_from_its_body() {
        for (code, state, wanted) in [
            (StatusCode::OK, "prepared", Lifecycle::Prepared),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "running",
                Lifecycle::Running,
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "preparing",
                Lifecycle::Preparing,
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "restoring",
                Lifecycle::Restoring,
            ),
        ] {
            assert_eq!(status(code, &body(state, None)).unwrap(), wanted, "{state}");
        }
        for (state, wanted) in STATES {
            for code in [
                StatusCode::OK,
                StatusCode::SERVICE_UNAVAILABLE,
                StatusCode::INTERNAL_SERVER_ERROR,
            ] {
                assert_eq!(
                    status(code, &body(state, None)).ok(),
                    wanted,
                    "{code} {state}"
                );
            }
        }
    }

    #[test]
    fn a_failed_status_is_an_error_with_the_reason() {
        let error = status(
            StatusCode::INTERNAL_SERVER_ERROR,
            &body("failed", Some("could not release the weights")),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "vLLM's checkpoint lifecycle failed: could not release the weights"
        );
        let error = status(StatusCode::INTERNAL_SERVER_ERROR, &body("failed", None)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "vLLM's checkpoint lifecycle failed: no reason given"
        );
    }

    #[test]
    fn an_unreadable_answer_is_an_error_naming_the_status_and_body() {
        for (code, body, wanted) in [
            (
                StatusCode::NOT_FOUND,
                &br#"{"detail":"Not Found"}"#[..],
                r#"vLLM answered 404 Not Found: "{\"detail\":\"Not Found\"}""#,
            ),
            (
                StatusCode::OK,
                br#"{"state":"sleeping"}"#,
                r#"vLLM answered 200 OK: "{\"state\":\"sleeping\"}""#,
            ),
            (
                StatusCode::OK,
                br#"{"state":"Running"}"#,
                r#"vLLM answered 200 OK: "{\"state\":\"Running\"}""#,
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                b"",
                r#"vLLM answered 503 Service Unavailable: """#,
            ),
            (
                StatusCode::OK,
                b"not json\n",
                r#"vLLM answered 200 OK: "not json""#,
            ),
            (
                StatusCode::OK,
                br#"{"error":"x"}"#,
                r#"vLLM answered 200 OK: "{\"error\":\"x\"}""#,
            ),
            (
                StatusCode::OK,
                br#"{"state":3}"#,
                r#"vLLM answered 200 OK: "{\"state\":3}""#,
            ),
        ] {
            let error = status(code, body).unwrap_err();
            assert_eq!(error.to_string(), wanted);
            assert!(started(code, body).is_err(), "{wanted}");
        }
    }

    #[test]
    fn a_started_prepare_or_restore_is_ok() {
        for state in ["running", "preparing", "prepared", "restoring"] {
            started(StatusCode::OK, &body(state, None)).unwrap();
        }
        let error = started(StatusCode::OK, &body("failed", Some("gone"))).unwrap_err();
        assert_eq!(
            error.to_string(),
            "vLLM's checkpoint lifecycle failed: gone"
        );
    }

    #[test]
    fn a_refusal_carries_its_reason() {
        for (state, error, wanted) in [
            (
                "running",
                Some("engine 1 refused: sleep mode is off"),
                "vLLM refused: engine 1 refused: sleep mode is off",
            ),
            ("restoring", None, "vLLM refused: it is restoring"),
            ("preparing", None, "vLLM refused: it is preparing"),
            (
                "restoring",
                Some("a restore is in progress"),
                "vLLM refused: a restore is in progress",
            ),
            (
                "failed",
                Some("could not release the weights"),
                "vLLM's checkpoint lifecycle failed: could not release the weights",
            ),
            (
                "failed",
                None,
                "vLLM's checkpoint lifecycle failed: no reason given",
            ),
        ] {
            let refused = started(StatusCode::BAD_REQUEST, &body(state, error)).unwrap_err();
            assert_eq!(refused.to_string(), wanted);
        }
    }

    #[test]
    fn other_statuses_of_prepare_and_restore_are_errors() {
        for (code, wanted) in [
            (
                StatusCode::SERVICE_UNAVAILABLE,
                r#"vLLM answered 503 Service Unavailable: "{\"state\":\"running\"}""#,
            ),
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"vLLM answered 500 Internal Server Error: "{\"state\":\"running\"}""#,
            ),
            (
                StatusCode::NOT_FOUND,
                r#"vLLM answered 404 Not Found: "{\"state\":\"running\"}""#,
            ),
        ] {
            let error = started(code, &body("running", None)).unwrap_err();
            assert_eq!(error.to_string(), wanted);
        }
    }

    /// The checkpoint routes over a state machine that prepares and restores
    /// at once, and refuses a prepare with `refusal` when it is set.
    struct Server {
        state: Mutex<&'static str>,
        refusal: Option<&'static str>,
        asked: Mutex<Vec<String>>,
    }

    impl Server {
        fn answer(&self, request: &Request<Incoming>) -> Response<Full<Bytes>> {
            let mut state = self.state.lock().unwrap();
            self.asked
                .lock()
                .unwrap()
                .push(format!("{} {}", request.method(), request.uri()));
            let (code, error) = match (request.method(), request.uri().path()) {
                (&Method::GET, "/checkpoint/status") => (
                    match *state {
                        "prepared" => StatusCode::OK,
                        "failed" => StatusCode::INTERNAL_SERVER_ERROR,
                        _ => StatusCode::SERVICE_UNAVAILABLE,
                    },
                    None,
                ),
                (&Method::POST, "/checkpoint/prepare") => match self.refusal {
                    Some(refusal) => (StatusCode::BAD_REQUEST, Some(refusal)),
                    None => {
                        *state = "prepared";
                        (StatusCode::OK, None)
                    }
                },
                (&Method::POST, "/checkpoint/restore") => {
                    *state = "running";
                    (StatusCode::OK, None)
                }
                _ => {
                    let mut response = Response::new(Full::new(Bytes::new()));
                    *response.status_mut() = StatusCode::NOT_FOUND;
                    return response;
                }
            };
            let mut response = Response::new(Full::new(Bytes::from(body(*state, error))));
            *response.status_mut() = code;
            response
        }
    }

    async fn serve(listener: TcpListener, server: Arc<Server>) {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let server = Arc::clone(&server);
            tokio::spawn(async move {
                let service = service_fn(move |request| {
                    let server = Arc::clone(&server);
                    async move { Ok::<_, Infallible>(server.answer(&request)) }
                });
                hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await
            });
        }
    }

    fn server(state: &'static str, refusal: Option<&'static str>) -> Arc<Server> {
        Arc::new(Server {
            state: Mutex::new(state),
            refusal,
            asked: Mutex::new(Vec::new()),
        })
    }

    async fn started_server(server: &Arc<Server>) -> Endpoint {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(serve(listener, Arc::clone(server)));
        Endpoint {
            transport: Transport::Http,
            address,
        }
    }

    const LIMIT: Duration = Duration::from_secs(5);

    #[tokio::test]
    async fn a_server_is_prepared_and_restored_over_http() {
        let server = server("running", None);
        let endpoint = started_server(&server).await;
        let (mut control, state) = Control::connect_when_up(endpoint, LIMIT).await.unwrap();
        assert_eq!(state, Lifecycle::Running);
        control.prepare(PauseMode::Abort, LIMIT).await.unwrap();
        assert_eq!(control.state().await.unwrap(), Lifecycle::Prepared);
        control.restore(LIMIT).await.unwrap();
        assert_eq!(control.state().await.unwrap(), Lifecycle::Running);
        assert_eq!(
            *server.asked.lock().unwrap(),
            [
                "GET /checkpoint/status",
                "POST /checkpoint/prepare?mode=abort",
                "GET /checkpoint/status",
                "GET /checkpoint/status",
                "POST /checkpoint/restore",
                "GET /checkpoint/status",
                "GET /checkpoint/status",
            ]
        );
    }

    #[tokio::test]
    async fn a_refused_prepare_is_an_error_with_the_refusal() {
        let server = server("running", Some("engine 0 refused: sleep mode is off"));
        let endpoint = started_server(&server).await;
        let mut control = Control::connect(endpoint).await.unwrap();
        let error = control.prepare(PauseMode::Wait, LIMIT).await.unwrap_err();
        assert_eq!(
            format!("{error:#}"),
            format!(
                "POST /checkpoint/prepare?mode=wait at {}: vLLM refused: engine 0 refused: sleep mode is off",
                endpoint.address
            )
        );
    }

    #[tokio::test]
    async fn a_failed_server_is_an_error_with_its_reason() {
        let server = server("failed", None);
        let endpoint = started_server(&server).await;
        let mut control = Control::connect(endpoint).await.unwrap();
        let error = control.state().await.unwrap_err();
        assert_eq!(
            format!("{error:#}"),
            format!(
                "GET /checkpoint/status at {}: vLLM's checkpoint lifecycle failed: no reason given",
                endpoint.address
            )
        );
    }

    async fn free_address() -> SocketAddr {
        TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap()
    }

    #[tokio::test]
    async fn a_booting_server_is_waited_for_until_it_listens() {
        let address = free_address().await;
        let endpoint = Endpoint {
            transport: Transport::Http,
            address,
        };
        let server = server("prepared", None);
        let booting = Arc::clone(&server);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            serve(TcpListener::bind(address).await.unwrap(), booting).await;
        });
        let (_, state) = Control::connect_when_up(endpoint, LIMIT).await.unwrap();
        assert_eq!(state, Lifecycle::Prepared);
    }

    #[tokio::test]
    async fn a_server_that_never_listens_is_an_error() {
        let endpoint = Endpoint {
            transport: Transport::Http,
            address: free_address().await,
        };
        let Err(error) = Control::connect_when_up(endpoint, Duration::from_millis(100)).await
        else {
            panic!("connected to nothing");
        };
        let error = format!("{error:#}");
        assert!(
            error.starts_with(&format!(
                "vLLM did not start serving: GET /checkpoint/status at {}: connect to vLLM's HTTP port at {}: ",
                endpoint.address, endpoint.address
            )),
            "{error}"
        );
    }
}
