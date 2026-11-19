//! An authenticated connection between the daemons of two nodes.
//!
//! ```text
//! asking daemon ══ TLS ══► kube-rbac-proxy ── HTTP, inside the pod ──► holding daemon
//! ```
use std::convert::Infallible;
use std::fs;
use std::net::SocketAddr;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use http_body_util::Empty;
use hyper::body::{Bytes, Incoming};
use hyper::header::{AUTHORIZATION, CONNECTION, HOST, HeaderMap, HeaderValue, UPGRADE};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use openssl::ssl::{SslConnector, SslMethod, SslVersion};
use openssl::x509::X509;
use openssl::x509::store::X509StoreBuilder;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout};
use tokio_openssl::SslStream;
use tracing::{Instrument, debug, warn};

use crate::encrypt::TransferKey;
use crate::store::RevisionKey;

const PROTOCOL: &str = "snapshotd-transfer/1";
const ROUTE: &str = "/v1/checkpoints/";
const LAST_WORDS: Duration = Duration::from_secs(5);

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

fn answer<A>(mut request: Request<Incoming>, admit: &A) -> Response<Empty<Bytes>>
where
    A: Fn(&RevisionKey) -> Result<Work, Refusal>,
{
    let status = |status| {
        let mut response = Response::new(Empty::new());
        *response.status_mut() = status;
        response
    };
    let revision = request
        .uri()
        .path()
        .strip_prefix(ROUTE)
        .and_then(|name| name.parse::<RevisionKey>().ok());
    let Some(revision) = revision.filter(|_| request.method() == Method::GET) else {
        return status(StatusCode::NOT_FOUND);
    };
    if !asks_to_upgrade(request.headers()) {
        return status(StatusCode::UPGRADE_REQUIRED);
    }
    let work = match admit(&revision) {
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

pub async fn serve<A>(listener: TcpListener, admit: A)
where
    A: Fn(&RevisionKey) -> Result<Work, Refusal> + Send + Sync + 'static,
{
    let admit = Arc::new(admit);
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) if stream.set_nodelay(true).is_ok() => stream,
            Ok(_) => continue,
            Err(error) => {
                warn!(%error, "could not accept a peer");
                sleep(Duration::from_secs(1)).await;
                continue;
            }
        };
        let admit = Arc::clone(&admit);
        tokio::spawn(async move {
            let service = service_fn(move |request| {
                let admit = Arc::clone(&admit);
                async move { Ok::<_, Infallible>(answer(request, admit.as_ref())) }
            });
            let served = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .with_upgrades();
            if let Err(error) = served.await {
                debug!(%error, "a peer connection ended badly");
            }
        });
    }
}

pub async fn ask<T: Send + 'static>(
    peer: SocketAddr,
    identity: &Identity,
    revision: &RevisionKey,
    work: impl FnOnce(Tunnel) -> Result<T> + Send + 'static,
) -> Result<Result<T, Refusal>> {
    let token = identity.token()?;
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
    let exchange = async {
        Pin::new(&mut tls)
            .connect()
            .await
            .context("the peer's certificate or TLS")?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(tls))
            .await
            .context("start HTTP")?;
        tokio::spawn(connection.with_upgrades());
        let request = Request::get(format!("{ROUTE}{revision}"))
            .header(HOST, &identity.server_name)
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .header(CONNECTION, "upgrade")
            .header(UPGRADE, PROTOCOL)
            .body(Empty::<Bytes>::new())?;
        let response = sender.send_request(request).await.context("ask")?;
        match response.status() {
            StatusCode::SWITCHING_PROTOCOLS => {}
            StatusCode::NOT_FOUND => return Ok(Err(Refusal::NoImage)),
            StatusCode::SERVICE_UNAVAILABLE => return Ok(Err(Refusal::Busy)),
            status @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => {
                bail!("the peer's proxy turned this daemon's token down ({status})")
            }
            status => bail!("the peer answered {status}"),
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
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use anyhow::{Result, bail};
    use openssl::asn1::Asn1Time;
    use openssl::bn::BigNum;
    use openssl::hash::MessageDigest;
    use openssl::pkey::{PKey, Private};
    use openssl::ssl::{SslAcceptor, SslMethod};
    use openssl::x509::extension::{BasicConstraints, SubjectAlternativeName};
    use openssl::x509::{X509, X509NameBuilder};
    use tokio::net::{TcpListener, TcpStream};
    use tokio_openssl::SslStream;

    use crate::store::RevisionKey;
    use crate::tunnel::{Identity, Refusal, Tunnel, Work, ask, host, serve};

    const NAME: &str = "snapshotd-peers.snapshotd.svc";

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

    async fn tls_front(authority: &Authority, name: &str, upstream: SocketAddr) -> SocketAddr {
        let (key, certificate) = certificate(name, 2, Some(authority));
        let mut acceptor = SslAcceptor::mozilla_intermediate_v5(SslMethod::tls_server()).unwrap();
        acceptor.set_private_key(&key).unwrap();
        acceptor.set_certificate(&certificate).unwrap();
        let acceptor = Arc::new(acceptor.build());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let acceptor = Arc::clone(&acceptor);
                tokio::spawn(async move {
                    let ssl = openssl::ssl::Ssl::new(acceptor.context()).unwrap();
                    let mut tls = SslStream::new(ssl, stream).unwrap();
                    if Pin::new(&mut tls).accept().await.is_err() {
                        return;
                    }
                    let mut upstream = TcpStream::connect(upstream).await.unwrap();
                    let _ = tokio::io::copy_bidirectional(&mut tls, &mut upstream).await;
                });
            }
        });
        address
    }

    struct Daemon {
        address: SocketAddr,
        admitted: Arc<AtomicUsize>,
    }

    async fn holder(work: fn(Tunnel) -> Result<()>) -> Daemon {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let admitted = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&admitted);
        tokio::spawn(serve(
            listener,
            move |revision: &RevisionKey| match revision.to_string().as_str() {
                "serving/qwen/held" => {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(Box::new(work) as Work)
                }
                "serving/qwen/busy" => Err(Refusal::Busy),
                _ => Err(Refusal::NoImage),
            },
        ));
        Daemon { address, admitted }
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
                bail!("the asker did not answer");
            }
            tunnel.socket.write_all(b"done\n")?;
            Ok(())
        })
        .await;
        let front = tls_front(&ca, NAME, daemon.address).await;
        let key = ask(
            front,
            &files("talk", &ca),
            &revision("serving/qwen/held"),
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
        let front = tls_front(&ca, NAME, daemon.address).await;
        let identity = files("refusal", &ca);
        for (name, wanted) in [
            ("serving/qwen/busy", Refusal::Busy),
            ("serving/qwen/other", Refusal::NoImage),
        ] {
            let answer = ask(front, &identity, &revision(name), |_| -> Result<()> {
                panic!("work ran after a refusal")
            })
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

        let unknown_signer = tls_front(&other, NAME, daemon.address).await;
        assert!(
            ask(unknown_signer, &files("signer", &ca), &held, work)
                .await
                .is_err()
        );

        let another_name = tls_front(&ca, "metrics.snapshotd.svc", daemon.address).await;
        assert!(
            ask(another_name, &files("name", &ca), &held, work)
                .await
                .is_err()
        );

        let mut no_authority = files("empty", &ca);
        std::fs::write(&no_authority.ca_file, "").unwrap();
        let front = tls_front(&ca, NAME, daemon.address).await;
        assert!(ask(front, &no_authority, &held, work).await.is_err());
        no_authority.ca_file = no_authority.token_file.with_file_name("missing");
        assert!(ask(front, &no_authority, &held, work).await.is_err());

        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_daemon_without_a_token_does_not_ask() {
        let ca = authority("peers");
        let daemon = holder(|_| Ok(())).await;
        let front = tls_front(&ca, NAME, daemon.address).await;
        let identity = files("token", &ca);
        std::fs::write(&identity.token_file, "\n").unwrap();
        let asked = ask(front, &identity, &revision("serving/qwen/held"), |_| Ok(())).await;
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
        let front = tls_front(&ca, NAME, daemon.address).await;
        let failed = ask(
            front,
            &files("stop", &ca),
            &revision("serving/qwen/held"),
            |_| -> Result<()> { bail!("out of room") },
        )
        .await;
        assert!(format!("{:#}", failed.unwrap_err()).contains("out of room"));
        assert_eq!(closed.recv_timeout(Duration::from_secs(10)).unwrap(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn only_an_upgrade_of_a_checkpoint_path_is_answered() {
        let daemon = holder(|_| Ok(())).await;
        for (request, wanted) in [
            (
                "GET /v1/checkpoints/serving/qwen/held HTTP/1.1",
                "HTTP/1.1 426",
            ),
            (
                "POST /v1/checkpoints/serving/qwen/held HTTP/1.1",
                "HTTP/1.1 404",
            ),
            (
                "GET /v1/checkpoints/serving/../held HTTP/1.1",
                "HTTP/1.1 404",
            ),
            ("GET /metrics HTTP/1.1", "HTTP/1.1 404"),
        ] {
            let mut stream = std::net::TcpStream::connect(daemon.address).unwrap();
            write!(
                stream,
                "{request}\r\nHost: peer\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            let status = line(&mut stream);
            assert!(status.starts_with(wanted), "{request}: {status}");
        }
        assert_eq!(daemon.admitted.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn the_peer_name_is_the_host_of_the_peers_address() {
        assert_eq!(host("snapshotd-peers.snapshotd.svc:8443").unwrap(), NAME);
        for name in ["snapshotd-peers", ":8443", "snapshotd-peers:https", ""] {
            assert!(host(name).is_err(), "{name:?}");
        }
    }
}
