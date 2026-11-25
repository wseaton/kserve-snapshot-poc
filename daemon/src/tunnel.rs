//! An authenticated connection between the daemons of two nodes.
//!
//! ```text
//! asking daemon ══ TLS ══► kube-rbac-proxy ══ mutual TLS, inside the pod ══► holding daemon
//! ```
//!
//! A request names its revision in the path and the revision's namespace
//! and the asker's principal in the query:
//!
//! ```text
//! GET /v1/checkpoints/<namespace>/<workload>/<hash>?namespace=<namespace>&principal=controller:<uid>
//! ```
//!
//! The proxy authenticates the asker, checks that it may `get`
//! `checkpoints.snapshot.llm-d.ai` in the `namespace` of the query, and
//! passes the asker's user and groups on in headers. The daemon checks the
//! query names the path's namespace and asks the API server the same
//! question itself (see [`crate::access`]).
use std::collections::HashMap;
use std::convert::Infallible;
use std::fs;
use std::future::Future;
use std::net::SocketAddr;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use http_body_util::{BodyExt, Empty, Full};
use hyper::body::{Bytes, Incoming};
use hyper::header::{AUTHORIZATION, CONNECTION, HOST, HeaderMap, HeaderValue, UPGRADE};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use openssl::ssl::{Ssl, SslAcceptor, SslConnector, SslMethod, SslVersion};
use openssl::x509::X509;
use openssl::x509::store::X509StoreBuilder;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout};
use tokio_openssl::SslStream;
use tracing::{Instrument, debug, warn};

use crate::access::{AccessError, Subject, Verdict};
use crate::encrypt::TransferKey;
use crate::store::{Principal, RevisionKey};
use crate::topology::Place;

const PROTOCOL: &str = "snapshotd-transfer/1";
const ROUTE: &str = "/v1/checkpoints/";
const LAST_WORDS: Duration = Duration::from_secs(5);
const HANDSHAKE: Duration = Duration::from_secs(10);
const NAMESPACE_PARAMETER: &str = "namespace";
const PRINCIPAL_PARAMETER: &str = "principal";
/// kube-rbac-proxy's `--auth-header-*` defaults.
const USER_HEADER: &str = "x-remote-user";
const GROUPS_HEADER: &str = "x-remote-groups";
const GROUP_SEPARATOR: char = '|';

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoImage,
    Busy,
}

pub struct Tunnel {
    pub socket: UnixStream,
    pub key: TransferKey,
}

pub type Work = Box<dyn FnOnce(Tunnel) -> Result<()> + Send>;

/// What a daemon says about a revision without handing it over.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Status {
    pub held: bool,
    /// Transfers the daemon could start now.
    pub free_lanes: usize,
    pub place: Place,
}

pub trait Holder: Send + Sync + 'static {
    /// What this daemon holds of `revision` that pods of `wanted` may use.
    fn status(&self, revision: &RevisionKey, wanted: &Principal) -> Status;
    /// Decide whether to hand `revision` over for pods of `wanted`, and
    /// what to do with the connection if so.
    fn admit(&self, revision: &RevisionKey, wanted: &Principal) -> Result<Work, Refusal>;
    fn may_read(
        &self,
        caller: &Subject,
        revision: &RevisionKey,
    ) -> impl Future<Output = Result<Verdict, AccessError>> + Send;
}

/// Why a request is not answered.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum Unanswered {
    #[error("not a GET of a checkpoint")]
    NotFound,
    #[error("the query is not namespace=<namespace>&principal=<principal>: {0}")]
    BadQuery(String),
    #[error("it came without the proxy's {USER_HEADER} header")]
    NoCaller,
    #[error("the query names namespace {query:?} and the revision is in {revision:?}")]
    OtherNamespace { query: String, revision: String },
}

impl Unanswered {
    fn status(&self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::BadQuery(_) => StatusCode::BAD_REQUEST,
            Self::NoCaller | Self::OtherNamespace { .. } => StatusCode::FORBIDDEN,
        }
    }
}

/// What a peer asks for, as the proxy passed it on.
#[derive(Debug, PartialEq, Eq)]
struct Asked {
    revision: RevisionKey,
    wanted: Principal,
    caller: Subject,
}

/// Each parameter once, its value as sent: nothing in a namespace or a
/// principal needs escaping.
fn parameters(query: Option<&str>) -> Result<HashMap<&str, &str>, Unanswered> {
    let mut found = HashMap::new();
    for pair in query
        .unwrap_or_default()
        .split('&')
        .filter(|pair| !pair.is_empty())
    {
        let (name, value) = pair
            .split_once('=')
            .ok_or_else(|| Unanswered::BadQuery(format!("{pair:?} has no value")))?;
        if value.contains(['%', '+']) {
            return Err(Unanswered::BadQuery(format!("{name} is escaped")));
        }
        if found.insert(name, value).is_some() {
            return Err(Unanswered::BadQuery(format!("{name} is given twice")));
        }
    }
    Ok(found)
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
}

fn asked<B>(request: &Request<B>) -> Result<Asked, Unanswered> {
    let revision = request
        .uri()
        .path()
        .strip_prefix(ROUTE)
        .and_then(|name| name.parse::<RevisionKey>().ok())
        .filter(|_| request.method() == Method::GET)
        .ok_or(Unanswered::NotFound)?;
    let parameters = parameters(request.uri().query())?;
    let namespace = parameters
        .get(NAMESPACE_PARAMETER)
        .ok_or_else(|| Unanswered::BadQuery(format!("no {NAMESPACE_PARAMETER}")))?;
    let wanted = parameters
        .get(PRINCIPAL_PARAMETER)
        .ok_or_else(|| Unanswered::BadQuery(format!("no {PRINCIPAL_PARAMETER}")))?
        .parse::<Principal>()
        .map_err(|error| Unanswered::BadQuery(error.to_string()))?;
    let user = header(request.headers(), USER_HEADER).ok_or(Unanswered::NoCaller)?;
    if *namespace != revision.namespace() {
        return Err(Unanswered::OtherNamespace {
            query: namespace.to_string(),
            revision: revision.namespace().to_string(),
        });
    }
    let groups = header(request.headers(), GROUPS_HEADER)
        .map(|groups| {
            groups
                .split(GROUP_SEPARATOR)
                .filter(|group| !group.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Ok(Asked {
        revision,
        wanted,
        caller: Subject {
            user: user.to_string(),
            groups,
        },
    })
}

pub struct Identity {
    pub token_file: PathBuf,
    pub ca_file: PathBuf,
    pub server_name: String,
    pub connect_timeout: Duration,
    pub answer_timeout: Duration,
}

impl Identity {
    fn connector(&self) -> Result<SslConnector> {
        let pem =
            fs::read(&self.ca_file).with_context(|| format!("read {}", self.ca_file.display()))?;
        let authorities = X509::stack_from_pem(&pem)
            .with_context(|| format!("parse {}", self.ca_file.display()))?;
        ensure!(
            !authorities.is_empty(),
            "{} holds no certificate",
            self.ca_file.display()
        );
        let mut trusted = X509StoreBuilder::new()?;
        for authority in authorities {
            trusted.add_cert(authority)?;
        }
        let mut connector = SslConnector::builder(SslMethod::tls_client())?;
        connector.set_cert_store(trusted.build());
        connector.set_min_proto_version(Some(SslVersion::TLS1_2))?;
        connector.set_alpn_protos(b"\x08http/1.1")?;
        Ok(connector.build())
    }

    fn token(&self) -> Result<String> {
        let token = fs::read_to_string(&self.token_file)
            .with_context(|| format!("read {}", self.token_file.display()))?;
        ensure!(
            !token.trim().is_empty(),
            "{} is empty",
            self.token_file.display()
        );
        Ok(token.trim().to_string())
    }
}

async fn carry<T: Send + 'static>(
    remote: impl AsyncRead + AsyncWrite + Send + 'static,
    key: TransferKey,
    work: impl FnOnce(Tunnel) -> Result<T> + Send + 'static,
) -> Result<T> {
    let (ours, theirs) = tokio::net::UnixStream::pair().context("make a socket pair")?;
    let theirs = theirs.into_std().context("hand over a socket")?;
    theirs.set_nonblocking(false)?;
    let (mut from_work, mut to_work) = ours.into_split();
    let (mut from_remote, mut to_remote) = tokio::io::split(remote);
    let outbound = tokio::spawn(async move {
        tokio::io::copy(&mut from_work, &mut to_remote).await?;
        to_remote.shutdown().await
    });
    let inbound = tokio::spawn(async move {
        let _ = tokio::io::copy(&mut from_remote, &mut to_work).await;
        let _ = to_work.shutdown().await;
    });
    let outcome = tokio::task::spawn_blocking(move || {
        work(Tunnel {
            socket: theirs,
            key,
        })
    })
    .await;
    let _ = timeout(LAST_WORDS, outbound).await;
    inbound.abort();
    outcome.context("the work on a peer connection panicked")?
}

fn asks_to_upgrade(headers: &HeaderMap) -> bool {
    headers.get(UPGRADE).is_some_and(|protocol| {
        protocol
            .as_bytes()
            .eq_ignore_ascii_case(PROTOCOL.as_bytes())
    })
}

async fn answer(mut request: Request<Incoming>, holder: &impl Holder) -> Response<Full<Bytes>> {
    let status = |status| {
        let mut response = Response::new(Full::default());
        *response.status_mut() = status;
        response
    };
    let Asked {
        revision,
        wanted,
        caller,
    } = match asked(&request) {
        Ok(asked) => asked,
        Err(why) => {
            if why.status() == StatusCode::FORBIDDEN {
                warn!(
                    security = "peer-request-refused",
                    path = %request.uri(),
                    %why,
                    "refused a request on the peer port"
                );
            }
            return status(why.status());
        }
    };
    match holder.may_read(&caller, &revision).await {
        Ok(Verdict::Allowed) => {}
        Ok(Verdict::Denied(why)) => {
            warn!(
                security = "peer-access-denied",
                %revision,
                caller = %caller.user,
                why,
                "a peer may not read the revision's checkpoint"
            );
            return status(StatusCode::FORBIDDEN);
        }
        Err(error) => {
            warn!(
                %revision,
                caller = %caller.user,
                error = %format_args!("{:#}", anyhow::Error::from(error)),
                "could not ask whether a peer may read the revision's checkpoint"
            );
            return status(StatusCode::SERVICE_UNAVAILABLE);
        }
    }
    if !asks_to_upgrade(request.headers()) {
        return match serde_json::to_vec(&holder.status(&revision, &wanted)) {
            Ok(body) => Response::new(Full::from(body)),
            Err(_) => status(StatusCode::INTERNAL_SERVER_ERROR),
        };
    }
    let work = match holder.admit(&revision, &wanted) {
        Ok(work) => work,
        Err(Refusal::NoImage) => return status(StatusCode::NOT_FOUND),
        Err(Refusal::Busy) => return status(StatusCode::SERVICE_UNAVAILABLE),
    };
    tokio::spawn(
        async move {
            let outcome = async {
                let upgraded = hyper::upgrade::on(&mut request)
                    .await
                    .context("upgrade the connection")?;
                let mut upgraded = TokioIo::new(upgraded);
                let key = TransferKey::generate()?;
                upgraded
                    .write_all(key.as_bytes())
                    .await
                    .context("send the transfer key")?;
                carry(upgraded, key, work).await
            };
            if let Err(error) = outcome.await {
                warn!(%revision, error = %format_args!("{error:#}"), "a peer's request failed");
            }
        }
        .in_current_span(),
    );
    let mut response = status(StatusCode::SWITCHING_PROTOCOLS);
    let headers = response.headers_mut();
    headers.insert(CONNECTION, HeaderValue::from_static("upgrade"));
    headers.insert(UPGRADE, HeaderValue::from_static(PROTOCOL));
    response
}

async fn accept_link(link: &SslAcceptor, stream: TcpStream) -> Result<SslStream<TcpStream>> {
    let mut tls = SslStream::new(Ssl::new(link.context())?, stream)?;
    Pin::new(&mut tls).accept().await?;
    Ok(tls)
}

/// Answer the proxy on `listener`, over the mutual TLS of `link`.
pub async fn serve(listener: TcpListener, link: Arc<SslAcceptor>, holder: Arc<impl Holder>) {
    loop {
        let (stream, from) = match listener.accept().await {
            Ok((stream, from)) if stream.set_nodelay(true).is_ok() => (stream, from),
            Ok(_) => continue,
            Err(error) => {
                warn!(%error, "could not accept a peer");
                sleep(Duration::from_secs(1)).await;
                continue;
            }
        };
        let (link, holder) = (Arc::clone(&link), Arc::clone(&holder));
        tokio::spawn(
            async move {
                let tls = match timeout(HANDSHAKE, accept_link(&link, stream)).await {
                    Ok(Ok(tls)) => tls,
                    Ok(Err(error)) => {
                        warn!(
                            security = "peer-link-refused",
                            %from,
                            error = %format_args!("{error:#}"),
                            "a connection to the peer port did not come from this pod's proxy"
                        );
                        return;
                    }
                    Err(_) => {
                        warn!(
                            security = "peer-link-refused",
                            %from,
                            "a connection to the peer port did not finish its handshake"
                        );
                        return;
                    }
                };
                let service = service_fn(move |request| {
                    let holder = Arc::clone(&holder);
                    async move { Ok::<_, Infallible>(answer(request, holder.as_ref()).await) }
                });
                let served = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(tls), service)
                    .with_upgrades();
                if let Err(error) = served.await {
                    debug!(%error, "a peer connection ended badly");
                }
            }
            .in_current_span(),
        );
    }
}

type Sender = hyper::client::conn::http1::SendRequest<Empty<Bytes>>;

/// An HTTP connection to the proxy at `peer`, its certificate checked.
async fn open(peer: SocketAddr, identity: &Identity) -> Result<Sender> {
    let ssl = identity
        .connector()?
        .configure()?
        .into_ssl(&identity.server_name)?;
    let tcp = timeout(identity.connect_timeout, TcpStream::connect(peer))
        .await
        .context("connecting took too long")?
        .context("connect")?;
    tcp.set_nodelay(true)?;
    let mut tls = SslStream::new(ssl, tcp)?;
    Pin::new(&mut tls)
        .connect()
        .await
        .context("the peer's certificate or TLS")?;
    let (sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(tls))
        .await
        .context("start HTTP")?;
    tokio::spawn(connection.with_upgrades());
    Ok(sender)
}

fn request(
    identity: &Identity,
    revision: &RevisionKey,
    wanted: &Principal,
) -> Result<hyper::http::request::Builder> {
    let namespace = revision.namespace();
    Ok(Request::get(format!(
        "{ROUTE}{revision}?{NAMESPACE_PARAMETER}={namespace}&{PRINCIPAL_PARAMETER}={wanted}"
    ))
    .header(HOST, &identity.server_name)
    .header(AUTHORIZATION, format!("Bearer {}", identity.token()?)))
}

fn refused(status: StatusCode) -> anyhow::Error {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            anyhow!("the peer's proxy turned this daemon's token down ({status})")
        }
        status => anyhow!("the peer answered {status}"),
    }
}

pub async fn inquire(
    peer: SocketAddr,
    identity: &Identity,
    revision: &RevisionKey,
    wanted: &Principal,
) -> Result<Status> {
    let exchange = async {
        let mut sender = open(peer, identity).await?;
        let request = request(identity, revision, wanted)?.body(Empty::<Bytes>::new())?;
        let response = sender.send_request(request).await.context("ask")?;
        if response.status() != StatusCode::OK {
            return Err(refused(response.status()));
        }
        let body = response
            .into_body()
            .collect()
            .await
            .context("read the peer's answer")?
            .to_bytes();
        serde_json::from_slice(&body).context("parse the peer's answer")
    };
    timeout(identity.answer_timeout, exchange)
        .await
        .context("the peer took too long to answer")?
}

pub async fn ask<T: Send + 'static>(
    peer: SocketAddr,
    identity: &Identity,
    revision: &RevisionKey,
    wanted: &Principal,
    work: impl FnOnce(Tunnel) -> Result<T> + Send + 'static,
) -> Result<Result<T, Refusal>> {
    let exchange = async {
        let mut sender = open(peer, identity).await?;
        let request = request(identity, revision, wanted)?
            .header(CONNECTION, "upgrade")
            .header(UPGRADE, PROTOCOL)
            .body(Empty::<Bytes>::new())?;
        let response = sender.send_request(request).await.context("ask")?;
        match response.status() {
            StatusCode::SWITCHING_PROTOCOLS => {}
            StatusCode::NOT_FOUND => return Ok(Err(Refusal::NoImage)),
            StatusCode::SERVICE_UNAVAILABLE => return Ok(Err(Refusal::Busy)),
            status => return Err(refused(status)),
        }
        let upgraded = hyper::upgrade::on(response)
            .await
            .context("upgrade the connection")?;
        let mut upgraded = TokioIo::new(upgraded);
        let mut key = [0; TransferKey::BYTES];
        upgraded
            .read_exact(&mut key)
            .await
            .context("read the transfer key")?;
        Ok(Ok((upgraded, TransferKey::from_bytes(key))))
    };
    let admitted = timeout(identity.answer_timeout, exchange)
        .await
        .context("the peer took too long to answer")??;
    match admitted {
        Ok((upgraded, key)) => carry(upgraded, key, work).await.map(Ok),
        Err(refusal) => Ok(Err(refusal)),
    }
}

/// The order in which the daemon at `own` asks `peers`. It only spreads
/// askers over the peers, so they do not all start with the same one; a
/// ranking by distance or load belongs here.
pub fn ask_order(own: std::net::IpAddr, mut peers: Vec<SocketAddr>) -> Vec<SocketAddr> {
    use std::hash::{DefaultHasher, Hash, Hasher};
    peers.sort();
    peers.dedup();
    if !peers.is_empty() {
        let mut hasher = DefaultHasher::new();
        own.hash(&mut hasher);
        let start = hasher.finish() as usize % peers.len();
        peers.rotate_left(start);
    }
    peers
}

pub fn host(name: &str) -> Result<&str> {
    let (host, port) = name
        .rsplit_once(':')
        .with_context(|| format!("{name:?} is not host:port"))?;
    port.parse::<u16>()
        .with_context(|| format!("the port of {name:?}"))?;
    ensure!(!host.is_empty(), "{name:?} names no host");
    Ok(host)
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::io::{self, BufRead, BufReader, Read, Write};
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use openssl::asn1::Asn1Time;
    use openssl::bn::BigNum;
    use openssl::hash::MessageDigest;
    use openssl::pkey::{PKey, Private};
    use openssl::ssl::{SslAcceptor, SslMethod};
    use openssl::x509::extension::{BasicConstraints, SubjectAlternativeName};
    use openssl::x509::{X509, X509NameBuilder};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio_openssl::SslStream;
    use tracing_subscriber::fmt::MakeWriter;

    use crate::access::{AccessError, Subject, Verdict};
    use crate::link;
    use crate::store::{Principal, RevisionKey};
    use crate::topology::Place;
    use crate::tunnel::{
        Holder, Identity, Refusal, Status, Tunnel, Work, ask, ask_order, host, inquire, serve,
    };

    const NAME: &str = "snapshotd-peers.snapshotd.svc";
    const DAEMON: &str = "system:serviceaccount:snapshotd:snapshotd";
    const GROUPS: &str =
        "system:serviceaccounts|system:serviceaccounts:snapshotd|system:authenticated";

    fn qwen() -> Principal {
        Principal::Controller {
            uid: "d-1".to_string(),
        }
    }

    struct Authority {
        key: PKey<Private>,
        certificate: X509,
    }

    fn certificate(name: &str, serial: u32, signer: Option<&Authority>) -> (PKey<Private>, X509) {
        let key = PKey::ec_gen("prime256v1").unwrap();
        let mut subject = X509NameBuilder::new().unwrap();
        subject.append_entry_by_text("CN", name).unwrap();
        let subject = subject.build();
        let mut builder = X509::builder().unwrap();
        builder.set_version(2).unwrap();
        let serial = BigNum::from_u32(serial).unwrap().to_asn1_integer().unwrap();
        builder.set_serial_number(&serial).unwrap();
        builder.set_subject_name(&subject).unwrap();
        builder.set_pubkey(&key).unwrap();
        builder
            .set_not_before(&Asn1Time::days_from_now(0).unwrap())
            .unwrap();
        builder
            .set_not_after(&Asn1Time::days_from_now(1).unwrap())
            .unwrap();
        match signer {
            Some(authority) => {
                builder
                    .set_issuer_name(authority.certificate.subject_name())
                    .unwrap();
                let names = SubjectAlternativeName::new()
                    .dns(name)
                    .build(&builder.x509v3_context(Some(&authority.certificate), None))
                    .unwrap();
                builder.append_extension(names).unwrap();
                builder
                    .sign(&authority.key, MessageDigest::sha256())
                    .unwrap();
            }
            None => {
                builder.set_issuer_name(&subject).unwrap();
                builder
                    .append_extension(BasicConstraints::new().critical().ca().build().unwrap())
                    .unwrap();
                builder.sign(&key, MessageDigest::sha256()).unwrap();
            }
        }
        (key, builder.build())
    }

    fn authority(name: &str) -> Authority {
        let (key, certificate) = certificate(name, 1, None);
        Authority { key, certificate }
    }

    /// The request head `stream` sends, with the proxy's identity headers
    /// for `caller` in place of any it sent.
    async fn head(stream: &mut (impl AsyncReadExt + Unpin), caller: Option<&str>) -> Vec<u8> {
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            if stream.read(&mut byte).await.unwrap() == 0 {
                return Vec::new();
            }
            head.push(byte[0]);
        }
        let text = String::from_utf8(head).unwrap();
        let mut lines: Vec<&str> = text
            .trim_end()
            .split("\r\n")
            .filter(|line| {
                let line = line.to_ascii_lowercase();
                !line.starts_with("x-remote-")
            })
            .collect();
        let added;
        if let Some(caller) = caller {
            added = format!("x-remote-user: {caller}\r\nx-remote-groups: {GROUPS}");
            lines.push(&added);
        }
        format!("{}\r\n\r\n", lines.join("\r\n")).into_bytes()
    }

    /// What kube-rbac-proxy does in front of `daemon`: answer TLS as `name`
    /// signed by `authority`, pass each request on with the identity of
    /// `caller`, over the daemon's link.
    async fn tls_front_as(
        authority: &Authority,
        name: &str,
        daemon: &Daemon,
        caller: Option<&'static str>,
    ) -> SocketAddr {
        let (key, certificate) = certificate(name, 2, Some(authority));
        let mut acceptor = SslAcceptor::mozilla_intermediate_v5(SslMethod::tls_server()).unwrap();
        acceptor.set_private_key(&key).unwrap();
        acceptor.set_certificate(&certificate).unwrap();
        let acceptor = Arc::new(acceptor.build());
        let connector = Arc::new(link::tests::connector(&daemon.link, &daemon.link));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let upstream = daemon.address;
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let (acceptor, connector) = (Arc::clone(&acceptor), Arc::clone(&connector));
                tokio::spawn(async move {
                    let ssl = openssl::ssl::Ssl::new(acceptor.context()).unwrap();
                    let mut tls = SslStream::new(ssl, stream).unwrap();
                    if Pin::new(&mut tls).accept().await.is_err() {
                        return;
                    }
                    let head = head(&mut tls, caller).await;
                    if head.is_empty() {
                        return;
                    }
                    let mut upstream = linked(&connector, upstream).await.unwrap();
                    upstream.write_all(&head).await.unwrap();
                    let _ = tokio::io::copy_bidirectional(&mut tls, &mut upstream).await;
                });
            }
        });
        address
    }

    async fn tls_front(authority: &Authority, name: &str, daemon: &Daemon) -> SocketAddr {
        tls_front_as(authority, name, daemon, Some(DAEMON)).await
    }

    async fn linked(
        connector: &openssl::ssl::SslConnector,
        daemon: SocketAddr,
    ) -> Result<SslStream<TcpStream>> {
        let ssl = connector.configure()?.into_ssl("127.0.0.1")?;
        let mut tls = SslStream::new(ssl, TcpStream::connect(daemon).await?)?;
        Pin::new(&mut tls).connect().await?;
        Ok(tls)
    }

    struct Daemon {
        address: SocketAddr,
        admitted: Arc<AtomicUsize>,
        link: PathBuf,
    }

    /// Holds `serving/qwen/held` for the controller `d-1`, is busy with
    /// `serving/qwen/busy`, lets only the daemons' ServiceAccount read
    /// `serving`, and does `work` with whoever it admits.
    struct Fixture {
        work: fn(Tunnel) -> Result<()>,
        admitted: Arc<AtomicUsize>,
    }

    fn place() -> Place {
        let levels = ["rack".to_string()];
        Place::of(&levels, &[("rack".to_string(), "237".to_string())].into())
    }

    impl Holder for Fixture {
        fn status(&self, revision: &RevisionKey, wanted: &Principal) -> Status {
            let name = revision.to_string();
            Status {
                held: name != "serving/qwen/other" && *wanted == qwen(),
                free_lanes: usize::from(name == "serving/qwen/held"),
                place: place(),
            }
        }

        fn may_read(
            &self,
            caller: &Subject,
            revision: &RevisionKey,
        ) -> impl Future<Output = Result<Verdict, AccessError>> + Send {
            let verdict = if caller.user == DAEMON && revision.namespace() == "serving" {
                Verdict::Allowed
            } else {
                Verdict::Denied(format!("{} has no RoleBinding here", caller.user))
            };
            async move { Ok(verdict) }
        }

        fn admit(&self, revision: &RevisionKey, wanted: &Principal) -> Result<Work, Refusal> {
            if *wanted != qwen() {
                return Err(Refusal::NoImage);
            }
            match revision.to_string().as_str() {
                "serving/qwen/held" => {
                    self.admitted.fetch_add(1, Ordering::SeqCst);
                    Ok(Box::new(self.work))
                }
                "serving/qwen/busy" => Err(Refusal::Busy),
                _ => Err(Refusal::NoImage),
            }
        }
    }

    async fn holder(work: fn(Tunnel) -> Result<()>) -> Daemon {
        static LINKS: AtomicUsize = AtomicUsize::new(0);
        let link =
            link::tests::scratch(&format!("tunnel-{}", LINKS.fetch_add(1, Ordering::SeqCst)));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let admitted = Arc::new(AtomicUsize::new(0));
        let fixture = Fixture {
            work,
            admitted: Arc::clone(&admitted),
        };
        let acceptor = Arc::new(link::acceptor(&link).unwrap());
        tokio::spawn(serve(listener, acceptor, Arc::new(fixture)));
        Daemon {
            address,
            admitted,
            link,
        }
    }

    fn files(test: &str, authority: &Authority) -> Identity {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-tunnel-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (token_file, ca_file): (PathBuf, PathBuf) = (dir.join("token"), dir.join("ca.crt"));
        std::fs::write(&token_file, "a-service-account-token\n").unwrap();
        std::fs::write(&ca_file, authority.certificate.to_pem().unwrap()).unwrap();
        Identity {
            token_file,
            ca_file,
            server_name: NAME.to_string(),
            connect_timeout: Duration::from_secs(3),
            answer_timeout: Duration::from_secs(10),
        }
    }

    fn revision(name: &str) -> RevisionKey {
        name.parse().unwrap()
    }

    fn line(socket: &mut impl Read) -> String {
        let mut line = String::new();
        BufReader::new(socket).read_line(&mut line).unwrap();
        line
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn two_daemons_talk_both_ways_and_hold_one_key() {
        static HOLDER_KEY: std::sync::Mutex<Option<[u8; 32]>> = std::sync::Mutex::new(None);
        let ca = authority("peers");
        let daemon = holder(|mut tunnel| {
            *HOLDER_KEY.lock().unwrap() = Some(*tunnel.key.as_bytes());
            tunnel.socket.write_all(b"here it comes\n")?;
            if line(&mut tunnel.socket) != "got it\n" {
                return Err(anyhow!("the asker did not answer"));
            }
            tunnel.socket.write_all(b"done\n")?;
            Ok(())
        })
        .await;
        let front = tls_front(&ca, NAME, &daemon).await;
        let key = ask(
            front,
            &files("talk", &ca),
            &revision("serving/qwen/held"),
            &qwen(),
            |mut tunnel| {
                assert_eq!(line(&mut tunnel.socket), "here it comes\n");
                tunnel.socket.write_all(b"got it\n")?;
                assert_eq!(line(&mut tunnel.socket), "done\n");
                Ok(*tunnel.key.as_bytes())
            },
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(Some(key), *HOLDER_KEY.lock().unwrap());
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_refusal_reaches_the_asker_and_no_work_runs() {
        let ca = authority("peers");
        let daemon = holder(|_| Ok(())).await;
        let front = tls_front(&ca, NAME, &daemon).await;
        let identity = files("refusal", &ca);
        for (name, wanted) in [
            ("serving/qwen/busy", Refusal::Busy),
            ("serving/qwen/other", Refusal::NoImage),
        ] {
            let answer = ask(
                front,
                &identity,
                &revision(name),
                &qwen(),
                |_| -> Result<()> { panic!("work ran after a refusal") },
            )
            .await
            .unwrap();
            assert_eq!(answer, Err(wanted), "{name}");
        }
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_with_the_wrong_certificate_is_never_asked() {
        let (ca, other) = (authority("peers"), authority("someone else"));
        let daemon = holder(|_| Ok(())).await;
        let work = |_: Tunnel| -> Result<()> { panic!("work ran with an unverified peer") };
        let held = revision("serving/qwen/held");

        let unknown_signer = tls_front(&other, NAME, &daemon).await;
        assert!(
            ask(unknown_signer, &files("signer", &ca), &held, &qwen(), work)
                .await
                .is_err()
        );

        let another_name = tls_front(&ca, "metrics.snapshotd.svc", &daemon).await;
        assert!(
            ask(another_name, &files("name", &ca), &held, &qwen(), work)
                .await
                .is_err()
        );

        let mut no_authority = files("empty", &ca);
        std::fs::write(&no_authority.ca_file, "").unwrap();
        let front = tls_front(&ca, NAME, &daemon).await;
        assert!(
            ask(front, &no_authority, &held, &qwen(), work)
                .await
                .is_err()
        );
        no_authority.ca_file = no_authority.token_file.with_file_name("missing");
        assert!(
            ask(front, &no_authority, &held, &qwen(), work)
                .await
                .is_err()
        );

        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_daemon_without_a_token_does_not_ask() {
        let ca = authority("peers");
        let daemon = holder(|_| Ok(())).await;
        let front = tls_front(&ca, NAME, &daemon).await;
        let identity = files("token", &ca);
        std::fs::write(&identity.token_file, "\n").unwrap();
        let asked = ask(
            front,
            &identity,
            &revision("serving/qwen/held"),
            &qwen(),
            |_| Ok(()),
        )
        .await;
        assert!(asked.is_err());
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn when_one_side_stops_the_other_sees_the_connection_close() {
        static SEEN: std::sync::Mutex<Option<mpsc::Sender<usize>>> = std::sync::Mutex::new(None);
        let (seen, closed) = mpsc::channel();
        *SEEN.lock().unwrap() = Some(seen);
        let ca = authority("peers");
        let daemon = holder(|mut tunnel| {
            let mut rest = Vec::new();
            let read = tunnel.socket.read_to_end(&mut rest)?;
            SEEN.lock().unwrap().as_ref().unwrap().send(read)?;
            Ok(())
        })
        .await;
        let front = tls_front(&ca, NAME, &daemon).await;
        let failed = ask(
            front,
            &files("stop", &ca),
            &revision("serving/qwen/held"),
            &qwen(),
            |_| -> Result<()> { Err(anyhow!("out of room")) },
        )
        .await;
        assert!(format!("{:#}", failed.unwrap_err()).contains("out of room"));
        assert_eq!(closed.recv_timeout(Duration::from_secs(10)).unwrap(), 0);
    }

    /// The status line the daemon answers `request` with, sent over its
    /// link with `headers`.
    async fn raw(daemon: &Daemon, request: &str, headers: &str) -> String {
        let connector = link::tests::connector(&daemon.link, &daemon.link);
        let mut tls = linked(&connector, daemon.address).await.unwrap();
        let sent = format!("{request}\r\nHost: peer\r\n{headers}Connection: close\r\n\r\n");
        tls.write_all(sent.as_bytes()).await.unwrap();
        let mut answer = Vec::new();
        let _ = tls.read_to_end(&mut answer).await;
        String::from_utf8_lossy(&answer)
            .lines()
            .next()
            .unwrap_or_default()
            .to_string()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn only_a_get_of_a_checkpoint_path_from_an_allowed_caller_is_answered() {
        let daemon = holder(|_| Ok(())).await;
        let caller = format!("x-remote-user: {DAEMON}\r\nx-remote-groups: {GROUPS}\r\n");
        let stranger = "x-remote-user: system:serviceaccount:default:default\r\n";
        let query = "?namespace=serving&principal=controller:d-1";
        let held = format!("GET /v1/checkpoints/serving/qwen/held{query} HTTP/1.1");
        for (request, headers, wanted) in [
            (held.clone(), caller.as_str(), "HTTP/1.1 200"),
            (
                format!("POST /v1/checkpoints/serving/qwen/held{query} HTTP/1.1"),
                &caller,
                "HTTP/1.1 404",
            ),
            (
                format!("GET /v1/checkpoints/serving/../held{query} HTTP/1.1"),
                &caller,
                "HTTP/1.1 404",
            ),
            ("GET /metrics HTTP/1.1".to_string(), &caller, "HTTP/1.1 404"),
            (held.clone(), "", "HTTP/1.1 403"),
            (held.clone(), stranger, "HTTP/1.1 403"),
            (
                "GET /v1/checkpoints/serving/qwen/held HTTP/1.1".to_string(),
                &caller,
                "HTTP/1.1 400",
            ),
            (
                "GET /v1/checkpoints/serving/qwen/held?namespace=serving HTTP/1.1".to_string(),
                &caller,
                "HTTP/1.1 400",
            ),
            (
                "GET /v1/checkpoints/serving/qwen/held?namespace=batch&principal=controller:d-1 HTTP/1.1"
                    .to_string(),
                &caller,
                "HTTP/1.1 403",
            ),
            (
                "GET /v1/checkpoints/batch/qwen/held?namespace=batch&principal=controller:d-1 HTTP/1.1"
                    .to_string(),
                &caller,
                "HTTP/1.1 403",
            ),
            (
                format!("GET /v1/checkpoints/serving/qwen/held{query}&namespace=serving HTTP/1.1"),
                &caller,
                "HTTP/1.1 400",
            ),
            (
                "GET /v1/checkpoints/serving/qwen/held?namespace=serv%69ng&principal=controller:d-1 HTTP/1.1"
                    .to_string(),
                &caller,
                "HTTP/1.1 400",
            ),
            (
                "GET /v1/checkpoints/serving/qwen/held?namespace=serving&principal=owner:d-1 HTTP/1.1"
                    .to_string(),
                &caller,
                "HTTP/1.1 400",
            ),
        ] {
            let status = raw(&daemon, &request, headers).await;
            assert!(status.starts_with(wanted), "{request} {headers:?}: {status}");
        }
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[derive(Clone, Default)]
    struct Logged(Arc<Mutex<Vec<u8>>>);

    impl Write for Logged {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Logged {
        type Writer = Logged;

        fn make_writer(&'a self) -> Logged {
            self.clone()
        }
    }

    /// Whatever comes back on `stream` after `request`, if anything.
    async fn answered(mut stream: impl AsyncReadExt + AsyncWriteExt + Unpin) -> String {
        let request = "GET /v1/checkpoints/serving/qwen/held?namespace=serving&principal=controller:d-1 HTTP/1.1\r\n\
                       Host: peer\r\nx-remote-user: system:serviceaccount:snapshotd:snapshotd\r\n\
                       Connection: close\r\n\r\n";
        if stream.write_all(request.as_bytes()).await.is_err() {
            return String::new();
        }
        let mut answer = Vec::new();
        let _ = stream.read_to_end(&mut answer).await;
        String::from_utf8_lossy(&answer).into_owned()
    }

    #[tokio::test]
    async fn a_connection_without_the_proxys_certificate_is_refused_and_logged() {
        let logged = Logged::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logged.clone())
            .with_ansi(false)
            .finish();
        let _logging = tracing::subscriber::set_default(subscriber);
        let daemon = holder(|_| panic!("work ran without the proxy")).await;

        let plain = TcpStream::connect(daemon.address).await.unwrap();
        assert!(!answered(plain).await.contains("HTTP/1.1"), "plain HTTP");

        let mut anonymous = openssl::ssl::SslConnector::builder(SslMethod::tls_client()).unwrap();
        anonymous
            .set_ca_file(daemon.link.join(link::AUTHORITY))
            .unwrap();
        if let Ok(tls) = linked(&anonymous.build(), daemon.address).await {
            assert!(!answered(tls).await.contains("HTTP/1.1"), "no certificate");
        }

        let elsewhere = link::tests::scratch("another-pod");
        let stranger = link::tests::connector(&daemon.link, &elsewhere);
        if let Ok(tls) = linked(&stranger, daemon.address).await {
            assert!(
                !answered(tls).await.contains("HTTP/1.1"),
                "another pod's proxy"
            );
        }

        let ours = link::tests::connector(&daemon.link, &daemon.link);
        let tls = linked(&ours, daemon.address).await.unwrap();
        assert!(
            answered(tls).await.starts_with("HTTP/1.1 200"),
            "this pod's proxy"
        );

        let text = String::from_utf8(logged.0.lock().unwrap().clone()).unwrap();
        let refused = text
            .lines()
            .filter(|line| line.contains("security=\"peer-link-refused\""))
            .count();
        assert_eq!(refused, 3, "{text}");
        assert!(
            text.contains("a connection to the peer port did not come from this pod's proxy"),
            "{text}"
        );
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
        std::fs::remove_dir_all(elsewhere).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_caller_the_api_server_turns_down_gets_nothing() {
        let ca = authority("peers");
        let daemon = holder(|_| panic!("work ran for a refused caller")).await;
        let identity = files("denied", &ca);
        let held = revision("serving/qwen/held");
        let stranger = tls_front_as(
            &ca,
            NAME,
            &daemon,
            Some("system:serviceaccount:default:default"),
        )
        .await;
        assert!(inquire(stranger, &identity, &held, &qwen()).await.is_err());
        let failed = ask(stranger, &identity, &held, &qwen(), |_| Ok(())).await;
        assert!(
            format!("{:#}", failed.unwrap_err()).contains("403"),
            "a refused caller is told so"
        );
        let anonymous = tls_front_as(&ca, NAME, &daemon, None).await;
        assert!(inquire(anonymous, &identity, &held, &qwen()).await.is_err());

        let front = tls_front(&ca, NAME, &daemon).await;
        let elsewhere = revision("batch/qwen/held");
        assert!(
            inquire(front, &identity, &elsewhere, &qwen())
                .await
                .is_err()
        );
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_image_of_another_principal_is_neither_offered_nor_sent() {
        let ca = authority("peers");
        let daemon = holder(|_| panic!("work ran for another principal")).await;
        let front = tls_front(&ca, NAME, &daemon).await;
        let identity = files("principal", &ca);
        let held = revision("serving/qwen/held");
        let other = Principal::Controller {
            uid: "d-2".to_string(),
        };
        let status = inquire(front, &identity, &held, &other).await.unwrap();
        assert!(!status.held);
        let answer = ask(front, &identity, &held, &other, |_| -> Result<()> {
            panic!("work ran for another principal")
        })
        .await
        .unwrap();
        assert_eq!(answer, Err(Refusal::NoImage));
        assert!(
            inquire(front, &identity, &held, &qwen())
                .await
                .unwrap()
                .held
        );
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_daemon_says_what_it_holds_and_where_it_is_without_starting_work() {
        let (ca, other) = (authority("peers"), authority("someone else"));
        let daemon = holder(|_| panic!("work ran for a question")).await;
        let front = tls_front(&ca, NAME, &daemon).await;
        let identity = files("inquire", &ca);
        for (name, held, free_lanes) in [
            ("serving/qwen/held", true, 1),
            ("serving/qwen/busy", true, 0),
            ("serving/qwen/other", false, 0),
        ] {
            let status = inquire(front, &identity, &revision(name), &qwen())
                .await
                .unwrap();
            let wanted = Status {
                held,
                free_lanes,
                place: place(),
            };
            assert_eq!(status, wanted, "{name}");
        }
        let untrusted = tls_front(&other, NAME, &daemon).await;
        let held = revision("serving/qwen/held");
        assert!(inquire(untrusted, &identity, &held, &qwen()).await.is_err());
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn askers_start_with_different_peers_and_each_asks_every_peer() {
        let peers: Vec<SocketAddr> = (1..=8)
            .map(|n| format!("10.0.0.{n}:8443").parse().unwrap())
            .collect();
        let mut firsts = std::collections::HashSet::new();
        for asker in 1..=32 {
            let own = format!("10.0.1.{asker}").parse().unwrap();
            let mut shuffled = peers.clone();
            shuffled.reverse();
            shuffled.push(peers[0]);
            let order = ask_order(own, shuffled);
            assert_eq!(order, ask_order(own, peers.clone()));
            firsts.insert(order[0]);
            let mut sorted = order;
            sorted.sort();
            assert_eq!(sorted, peers);
        }
        assert!(
            firsts.len() >= 4,
            "32 askers started at {} peers",
            firsts.len()
        );
        assert!(ask_order("10.0.1.1".parse().unwrap(), Vec::new()).is_empty());
    }

    #[test]
    fn the_peer_name_is_the_host_of_the_peers_address() {
        assert_eq!(host("snapshotd-peers.snapshotd.svc:8443").unwrap(), NAME);
        for name in ["snapshotd-peers", ":8443", "snapshotd-peers:https", ""] {
            assert!(host(name).is_err(), "{name:?}");
        }
    }
}
