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

use crate::handshake::{self, Key, Refusal, Revision};
use crate::state::{State, revision_dir};
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
        println!("answering peers on {}", listener.local_addr()?);
        let answering = Arc::clone(&peers);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let peers = Arc::clone(&answering);
                        std::thread::spawn(move || {
                            let peer = stream.peer_addr();
                            if let Err(error) = peers.answer(stream) {
                                println!("peer {peer:?}: {error:#}");
                            }
                        });
                    }
                    Err(error) => println!("accept a peer: {error}"),
                }
            }
        });
        Ok(Some(peers))
    }

    fn answer(&self, mut stream: TcpStream) -> Result<()> {
        let peer = stream.peer_addr()?;
        bound(&stream, Some(HANDSHAKE_TIMEOUT))?;
        let asked = handshake::authenticate(&mut stream, &self.key)?;
        let image = revision_dir(&self.store, &asked.to_string()).map(|dir| dir.join("image"));
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
        println!(
            "sent {asked} to {peer} in {:.1} s",
            start.elapsed().as_secs_f64()
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
                println!("no peers at {}: {error}", self.name);
                Vec::new()
            }
        }
    }

    /// Get the checkpoint of `revision` from the first peer that has it and
    /// publish it in `state`'s store. Returns whether one arrived.
    pub fn pull(&self, revision: &Revision, state: &State) -> Result<bool> {
        let deadline = Instant::now() + BUSY_WAIT;
        loop {
            let mut busy = false;
            for peer in self.addresses() {
                state.clear_staging()?;
                match self.pull_from(peer, revision, state) {
                    Ok(Ok(())) => {
                        state.publish()?;
                        return Ok(true);
                    }
                    Ok(Err(Refusal::Busy)) => busy = true,
                    Ok(Err(Refusal::NoImage)) => {}
                    Err(error) => println!("peer {peer}: {error:#}"),
                }
            }
            state.clear_staging()?;
            if !busy || Instant::now() >= deadline {
                return Ok(false);
            }
            std::thread::sleep(BUSY_RETRY);
        }
    }

    fn pull_from(
        &self,
        peer: SocketAddr,
        revision: &Revision,
        state: &State,
    ) -> Result<Result<(), Refusal>> {
        let mut stream = TcpStream::connect_timeout(&peer, CONNECT_TIMEOUT).context("connect")?;
        bound(&stream, Some(HANDSHAKE_TIMEOUT))?;
        if let Err(reason) = handshake::ask(&mut stream, &self.key, revision)? {
            return Ok(Err(reason));
        }
        bound(&stream, None)?;
        let start = Instant::now();
        transfer::receive_over(stream, state.staging_dir(), true, false, None, None)?;
        println!(
            "pulled {revision} from {peer} in {:.1} s",
            start.elapsed().as_secs_f64()
        );
        Ok(Ok(()))
    }
}
