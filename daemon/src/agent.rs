//! The transfer container of a serving pod: get a checkpoint from a replica
//! that has one, then offer this pod's to the replicas that come after.
//!
//! ```text
//! start ─► image on this node? ─► else ask each peer ─► pull.status
//!   └────► answer peers for as long as the pod lives, once the node has an image
//! ```
//!
//! The image lives in a store directory on the node's disk, one per revision
//! (see `state.rs`), so only the first pod of a revision on a node transfers.
//!
//! Peers are the addresses one DNS name resolves to, a headless Service over
//! the ready replicas. A replica is ready only once it serves from an image,
//! so every address is a candidate.
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};

use crate::handshake::{self, Key, Refusal, Revision};
use crate::state::{PullOutcome, State};
use crate::transfer::{self, Pool};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const BUSY_RETRY: Duration = Duration::from_secs(2);

/// Where the image directory lives.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Medium {
    /// A disk; chunks are written past the page cache.
    Disk,
    /// A RAM-backed filesystem, which has no page cache to bypass.
    Memory,
}

#[derive(clap::Args)]
pub struct Args {
    /// Directory shared with the serving container.
    #[arg(long)]
    state_dir: PathBuf,
    /// Directory on the node's disk for this revision's image, shared with
    /// the serving container.
    #[arg(long)]
    store_dir: PathBuf,
    /// Where peers ask for this pod's image.
    #[arg(long, default_value = "0.0.0.0:7700")]
    listen: String,
    /// Name and port that resolve to the agents of the ready replicas.
    #[arg(long)]
    peers: String,
    /// This pod's address, left out of the peers.
    #[arg(long)]
    own_ip: IpAddr,
    /// What this pod runs; only an image of the same revision is pulled or sent.
    #[arg(long)]
    revision: Revision,
    /// File holding the key the replicas share.
    #[arg(long)]
    key_file: PathBuf,
    #[arg(long, value_enum, default_value_t = Medium::Disk)]
    medium: Medium,
    /// Seconds to keep asking while every peer with an image is sending it elsewhere.
    #[arg(long, default_value_t = 300)]
    busy_wait_s: u64,
    #[command(flatten)]
    pool: Pool,
}

struct Agent {
    args: Args,
    key: Key,
    state: State,
    sending: Mutex<()>,
}

pub fn run(args: Args) -> Result<()> {
    let agent = Arc::new(Agent {
        key: Key::load(&args.key_file)?,
        state: State::open(&args.state_dir, &args.store_dir, &args.own_ip.to_string())?,
        sending: Mutex::new(()),
        args,
    });
    let listener = TcpListener::bind(&agent.args.listen)
        .with_context(|| format!("listen on {}", agent.args.listen))?;
    println!("listening on {}", listener.local_addr()?);
    let answering = {
        let agent = Arc::clone(&agent);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let agent = Arc::clone(&agent);
                        std::thread::spawn(move || {
                            let peer = stream.peer_addr();
                            if let Err(error) = agent.answer(stream) {
                                println!("peer {peer:?}: {error:#}");
                            }
                        });
                    }
                    Err(error) => println!("accept: {error}"),
                }
            }
        })
    };
    if agent.state.pull_outcome()?.is_none() {
        let outcome = if agent.state.image_ready() {
            PullOutcome::Local
        } else {
            agent.pull()?
        };
        agent.state.record_pull(outcome)?;
        println!("PULL {outcome:?}");
    }
    answering
        .join()
        .map_err(|_| anyhow!("the thread answering peers panicked"))
}

fn bound(stream: &TcpStream, limit: Option<Duration>) -> Result<()> {
    stream.set_read_timeout(limit)?;
    stream.set_write_timeout(limit)?;
    Ok(())
}

impl Agent {
    fn answer(&self, mut stream: TcpStream) -> Result<()> {
        let peer = stream.peer_addr()?;
        bound(&stream, Some(HANDSHAKE_TIMEOUT))?;
        let asked = handshake::authenticate(&mut stream, &self.key)?;
        let turn = self.sending.try_lock();
        let verdict = if asked != self.args.revision {
            Err(Refusal::WrongRevision)
        } else if !self.state.image_ready() {
            Err(Refusal::NoImage)
        } else if turn.is_err() {
            Err(Refusal::Busy)
        } else {
            Ok(())
        };
        handshake::answer(&mut stream, verdict)?;
        if let Err(reason) = verdict {
            println!("peer {peer}: refused, {reason:?}");
            return Ok(());
        }
        bound(&stream, None)?;
        let start = Instant::now();
        transfer::send_over(stream, &self.state.image_dir(), self.args.pool, None)?;
        println!("SENT to {peer} in {:.1} s", start.elapsed().as_secs_f64());
        Ok(())
    }

    fn pull(&self) -> Result<PullOutcome> {
        let deadline = Instant::now() + Duration::from_secs(self.args.busy_wait_s);
        loop {
            let mut busy = false;
            for peer in peers(&self.args.peers, self.args.own_ip) {
                self.state.clear_staging()?;
                match self.pull_from(peer) {
                    Ok(Ok(())) => {
                        self.state.publish()?;
                        return Ok(PullOutcome::Pulled);
                    }
                    Ok(Err(Refusal::Busy)) => busy = true,
                    Ok(Err(reason)) => println!("peer {peer}: {reason:?}"),
                    Err(error) => println!("peer {peer}: {error:#}"),
                }
            }
            self.state.clear_staging()?;
            if !busy || Instant::now() >= deadline {
                return Ok(PullOutcome::None);
            }
            std::thread::sleep(BUSY_RETRY);
        }
    }

    fn pull_from(&self, peer: SocketAddr) -> Result<Result<(), Refusal>> {
        let mut stream = TcpStream::connect_timeout(&peer, CONNECT_TIMEOUT).context("connect")?;
        bound(&stream, Some(HANDSHAKE_TIMEOUT))?;
        if let Err(reason) = handshake::ask(&mut stream, &self.key, &self.args.revision)? {
            return Ok(Err(reason));
        }
        bound(&stream, None)?;
        let start = Instant::now();
        let direct = matches!(self.args.medium, Medium::Disk);
        transfer::receive_over(stream, self.state.staging_dir(), direct, false, None, None)?;
        println!(
            "PULLED from {peer} in {:.1} s",
            start.elapsed().as_secs_f64()
        );
        Ok(Ok(()))
    }
}

fn peers(name: &str, own_ip: IpAddr) -> Vec<SocketAddr> {
    match name.to_socket_addrs() {
        Ok(addresses) => {
            let mut peers: Vec<SocketAddr> = addresses.filter(|a| a.ip() != own_ip).collect();
            peers.sort();
            peers.dedup();
            peers
        }
        Err(error) => {
            println!("no peers at {name}: {error}");
            Vec::new()
        }
    }
}
