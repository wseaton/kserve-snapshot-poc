//! Checkpoints between the daemons of different nodes: answer a peer that
//! asks for a revision this node holds, and ask peers for one it lacks.
//!
//! Peers are the addresses one DNS name resolves to, a headless Service over
//! the DaemonSet. A revision is named `<namespace>/<workload>/<hash>`, the
//! same path it has under each node's store.
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::handshake::{self, Key, Refusal, Revision};
use crate::store::{RevisionKey, RevisionStore, directory_bytes};
use crate::transfer::{self, Pool};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const BUSY_RETRY: Duration = Duration::from_secs(2);
const BUSY_WAIT: Duration = Duration::from_secs(300);

#[derive(clap::Args)]
pub struct Args {
    /// Name and port that resolve to the daemons of the other nodes. Without
    /// it, checkpoints stay on the node that took them.
    #[arg(long, requires_all = ["own_ip", "key_file"])]
    peers: Option<String>,
    /// Where peers ask this daemon for checkpoints.
    #[arg(long, default_value = "0.0.0.0:7700")]
    peer_listen: String,
    /// This daemon's address, left out of the peers.
    #[arg(long)]
    own_ip: Option<IpAddr>,
    /// File holding the key the daemons share.
    #[arg(long)]
    key_file: Option<PathBuf>,
    #[command(flatten)]
    pool: Pool,
}

pub struct Peers {
    name: String,
    own_ip: IpAddr,
    key: Key,
    store: PathBuf,
    pool: Pool,
    sending: Mutex<()>,
}

fn bound(stream: &TcpStream, limit: Option<Duration>) -> Result<()> {
    stream.set_read_timeout(limit)?;
    stream.set_write_timeout(limit)?;
    Ok(())
}

impl Peers {
    /// Start answering peers, if the daemon was told about any.
    pub fn start(args: &Args, store: &Path) -> Result<Option<Arc<Self>>> {
        let (Some(name), Some(own_ip), Some(key_file)) = (&args.peers, args.own_ip, &args.key_file)
        else {
            return Ok(None);
        };
        let peers = Arc::new(Self {
            name: name.clone(),
            own_ip,
            key: Key::load(key_file)?,
            store: store.to_path_buf(),
            pool: args.pool,
            sending: Mutex::new(()),
        });
        let listener = TcpListener::bind(&args.peer_listen)
            .with_context(|| format!("listen on {}", args.peer_listen))?;
        info!(address = %listener.local_addr()?, name = %name, "answering peers");
        let answering = Arc::clone(&peers);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let peers = Arc::clone(&answering);
                        std::thread::spawn(move || {
                            let peer = stream.peer_addr();
                            if let Err(error) = peers.answer(stream) {
                                warn!(?peer, error = %format_args!("{error:#}"), "peer request failed");
                            }
                        });
                    }
                    Err(error) => warn!(%error, "could not accept a peer"),
                }
            }
        });
        Ok(Some(peers))
    }

    fn answer(&self, mut stream: TcpStream) -> Result<()> {
        let peer = stream.peer_addr()?;
        bound(&stream, Some(HANDSHAKE_TIMEOUT))?;
        let asked = handshake::authenticate(&mut stream, &self.key)?;
        let image = asked
            .to_string()
            .parse::<RevisionKey>()
            .map(|revision| revision.image_dir(&self.store));
        let turn = self.sending.try_lock();
        let verdict = match &image {
            Ok(image) if image.is_dir() && turn.is_ok() => Ok(()),
            Ok(image) if image.is_dir() => Err(Refusal::Busy),
            _ => Err(Refusal::NoImage),
        };
        handshake::answer(&mut stream, verdict)?;
        let (Ok(()), Ok(image)) = (verdict, image) else {
            return Ok(());
        };
        bound(&stream, None)?;
        let start = Instant::now();
        transfer::send_over(stream, &image, self.pool, None)?;
        let (seconds, bytes) = (
            start.elapsed().as_secs_f64(),
            directory_bytes(&image) as f64,
        );
        info!(
            revision = %asked,
            %peer,
            seconds = %format_args!("{seconds:.2}"),
            gb = %format_args!("{:.1}", bytes / 1e9),
            gbit_per_s = %format_args!("{:.1}", bytes * 8.0 / 1e9 / seconds),
            "sent a checkpoint to a peer"
        );
        Ok(())
    }

    fn addresses(&self) -> Vec<SocketAddr> {
        match self.name.to_socket_addrs() {
            Ok(addresses) => {
                let mut peers: Vec<SocketAddr> =
                    addresses.filter(|a| a.ip() != self.own_ip).collect();
                peers.sort();
                peers.dedup();
                peers
            }
            Err(error) => {
                debug!(name = %self.name, %error, "the peer name does not resolve");
                Vec::new()
            }
        }
    }

    /// Get the checkpoint of `revision` from the first peer that has it and
    /// publish it in `store`. Returns the peer it came from, if any.
    pub fn pull(
        &self,
        revision: &RevisionKey,
        store: &RevisionStore,
    ) -> Result<Option<SocketAddr>> {
        let revision = &Revision::new(revision.to_string())?;
        let deadline = Instant::now() + BUSY_WAIT;
        loop {
            let mut busy = false;
            for peer in self.addresses() {
                store.clear_staging()?;
                match self.pull_from(peer, revision, store) {
                    Ok(Ok(())) => {
                        store.publish()?;
                        return Ok(Some(peer));
                    }
                    Ok(Err(Refusal::Busy)) => busy = true,
                    Ok(Err(Refusal::NoImage)) => debug!(%peer, "the peer has no checkpoint of it"),
                    Err(error) => warn!(%peer, error = %format_args!("{error:#}"), "pull failed"),
                }
            }
            store.clear_staging()?;
            if !busy || Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::sleep(BUSY_RETRY);
        }
    }

    fn pull_from(
        &self,
        peer: SocketAddr,
        revision: &Revision,
        store: &RevisionStore,
    ) -> Result<Result<(), Refusal>> {
        let mut stream = TcpStream::connect_timeout(&peer, CONNECT_TIMEOUT).context("connect")?;
        bound(&stream, Some(HANDSHAKE_TIMEOUT))?;
        if let Err(reason) = handshake::ask(&mut stream, &self.key, revision)? {
            return Ok(Err(reason));
        }
        bound(&stream, None)?;
        transfer::receive_over(stream, store.staging_dir(), true, false, None, None)?;
        Ok(Ok(()))
    }
}
