//! Checkpoints between the daemons of different nodes: answer a peer that
//! asks for a revision this node holds, and ask peers for one it lacks.
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::time::{Instant, sleep};
use tracing::{debug, info, warn};

use crate::store::{RevisionKey, RevisionStore, directory_bytes};
use crate::transfer::{self, Endpoint, Pool};
use crate::tunnel::{self, Identity, Refusal, Tunnel, Work};

const BUSY_RETRY: Duration = Duration::from_secs(2);

#[derive(clap::Args)]
pub struct Args {
    /// Name and port that resolve to the kube-rbac-proxy of every node's
    /// daemon. Without it, checkpoints stay on the node that took them.
    #[arg(long, env = "SNAPSHOTD_PEERS", requires_all = ["own_ip", "peer_ca_file"])]
    peers: Option<String>,
    /// Where this daemon answers its own kube-rbac-proxy.
    #[arg(long, env = "SNAPSHOTD_PEER_LISTEN", default_value = "127.0.0.1:7700")]
    peer_listen: SocketAddr,
    /// This daemon's address, left out of the peers.
    #[arg(long, env = "SNAPSHOTD_OWN_IP")]
    own_ip: Option<IpAddr>,
    /// The ServiceAccount token this daemon shows its peers.
    #[arg(
        long,
        env = "SNAPSHOTD_PEER_TOKEN_FILE",
        default_value = "/var/run/secrets/snapshotd/peers/token"
    )]
    peer_token_file: PathBuf,
    /// The authority that signs the certificate of the peers' proxies.
    #[arg(long, env = "SNAPSHOTD_PEER_CA_FILE")]
    peer_ca_file: Option<PathBuf>,
    /// Seconds a peer may take to accept a connection.
    #[arg(long, env = "SNAPSHOTD_PEER_CONNECT_TIMEOUT_S", default_value_t = 3)]
    peer_connect_timeout_s: u64,
    /// Seconds a peer may take to answer a request.
    #[arg(long, env = "SNAPSHOTD_PEER_ANSWER_TIMEOUT_S", default_value_t = 10)]
    peer_answer_timeout_s: u64,
    /// Seconds to keep asking while the peers that hold a checkpoint are
    /// busy sending it.
    #[arg(long, env = "SNAPSHOTD_PEER_BUSY_TIMEOUT_S", default_value_t = 300)]
    peer_busy_timeout_s: u64,
    #[command(flatten)]
    pool: Pool,
}

type Job = Box<dyn FnOnce(Result<&Endpoint>) -> bool + Send>;

/// The thread every transfer of this daemon runs on, in either direction,
/// with the one registered pool they share. The pool is made by the first
/// transfer and kept: making one costs seconds, or far longer on a node
/// whose memory is fragmented.
#[derive(Clone)]
struct Transfers {
    jobs: mpsc::Sender<Job>,
}

impl Transfers {
    fn start(name: String, pool: Pool) -> Result<Self> {
        let (jobs, queue) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("transfers".to_string())
            .spawn(move || {
                let mut endpoint = None;
                for job in queue {
                    if endpoint.is_none() {
                        match Endpoint::new(&name, pool, false) {
                            Ok(made) => endpoint = Some(made),
                            Err(error) => {
                                job(Err(error));
                                continue;
                            }
                        }
                    }
                    if let Some(ready) = &endpoint
                        && !job(Ok(ready))
                    {
                        endpoint = None;
                    }
                }
            })
            .context("start the transfer thread")?;
        Ok(Self { jobs })
    }

    /// Run `work` with the pool and wait for it. A pool that a transfer
    /// failed on is not used again.
    fn run<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Endpoint) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (done, outcome) = mpsc::channel();
        let job: Job = Box::new(move |endpoint| {
            let result = endpoint.and_then(work);
            let fine = result.is_ok();
            let _ = done.send(result);
            fine
        });
        self.jobs
            .send(job)
            .map_err(|_| anyhow!("the transfer thread stopped"))?;
        outcome
            .recv()
            .context("the transfer thread dropped a transfer")?
    }
}

pub struct Peers {
    name: String,
    own_ip: IpAddr,
    identity: Identity,
    busy_timeout: Duration,
    store: PathBuf,
    transfers: Transfers,
    /// Held by the one transfer this daemon takes part in, sending or
    /// receiving.
    turn: Arc<Semaphore>,
}

impl Peers {
    pub async fn start(args: &Args, store: &Path) -> Result<Option<Arc<Self>>> {
        let (Some(name), Some(own_ip), Some(ca_file)) =
            (&args.peers, args.own_ip, &args.peer_ca_file)
        else {
            return Ok(None);
        };
        let peers = Arc::new(Self {
            name: name.clone(),
            own_ip,
            identity: Identity {
                token_file: args.peer_token_file.clone(),
                ca_file: ca_file.clone(),
                server_name: tunnel::host(name)?.to_string(),
                connect_timeout: Duration::from_secs(args.peer_connect_timeout_s),
                answer_timeout: Duration::from_secs(args.peer_answer_timeout_s),
            },
            busy_timeout: Duration::from_secs(args.peer_busy_timeout_s),
            store: store.to_path_buf(),
            transfers: Transfers::start(
                format!("snapshotd-{own_ip}-{}", std::process::id()),
                args.pool,
            )?,
            turn: Arc::new(Semaphore::new(1)),
        });
        let listener = TcpListener::bind(args.peer_listen)
            .await
            .with_context(|| format!("listen on {}", args.peer_listen))?;
        info!(address = %args.peer_listen, name = %name, "answering peers");
        let holder = Arc::clone(&peers);
        tokio::spawn(tunnel::serve(listener, move |revision: &RevisionKey| {
            holder.admit(revision)
        }));
        Ok(Some(peers))
    }

    fn admit(&self, revision: &RevisionKey) -> Result<Work, Refusal> {
        let image = revision.image_dir(&self.store);
        if !image.is_dir() {
            return Err(Refusal::NoImage);
        }
        let turn = Arc::clone(&self.turn)
            .try_acquire_owned()
            .map_err(|_| Refusal::Busy)?;
        let (revision, transfers) = (revision.clone(), self.transfers.clone());
        Ok(Box::new(move |tunnel: Tunnel| {
            let _turn = turn;
            let start = std::time::Instant::now();
            let sent = image.clone();
            transfers.run(move |endpoint| {
                let Tunnel { socket, key } = tunnel;
                transfer::send_over(socket, &sent, endpoint, Some(&key), None).map(drop)
            })?;
            let (seconds, bytes) = (
                start.elapsed().as_secs_f64(),
                directory_bytes(&image) as f64,
            );
            info!(
                %revision,
                seconds = %format_args!("{seconds:.2}"),
                gb = %format_args!("{:.1}", bytes / 1e9),
                gbit_per_s = %format_args!("{:.1}", bytes * 8.0 / 1e9 / seconds),
                "sent a checkpoint to a peer"
            );
            Ok(())
        }))
    }

    async fn addresses(&self) -> Vec<SocketAddr> {
        match tokio::net::lookup_host(&self.name).await {
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

    /// Returns the peer the checkpoint came from, if any had it.
    pub async fn pull(
        &self,
        revision: &RevisionKey,
        store: &RevisionStore,
    ) -> Result<Option<SocketAddr>> {
        let deadline = Instant::now() + self.busy_timeout;
        loop {
            let mut busy = None;
            {
                let _turn = self
                    .turn
                    .acquire()
                    .await
                    .map_err(|_| anyhow!("the transfer turn is closed"))?;
                for peer in self.addresses().await {
                    store.clear_staging()?;
                    let staging = store.staging_dir().to_path_buf();
                    let transfers = self.transfers.clone();
                    let receive = move |tunnel: Tunnel| {
                        transfers.run(move |endpoint| {
                            let Tunnel { socket, key } = tunnel;
                            let endpoint = Some(endpoint);
                            let key = Some(&key);
                            transfer::receive_over(
                                socket, &staging, true, false, None, endpoint, key, None,
                            )
                            .map(drop)
                        })
                    };
                    match tunnel::ask(peer, &self.identity, revision, receive).await {
                        Ok(Ok(())) => {
                            store.publish()?;
                            return Ok(Some(peer));
                        }
                        Ok(Err(Refusal::Busy)) => busy = Some(peer),
                        Ok(Err(Refusal::NoImage)) => {
                            debug!(%peer, "the peer has no checkpoint of it")
                        }
                        Err(error) => {
                            warn!(%peer, error = %format_args!("{error:#}"), "pull failed")
                        }
                    }
                }
                store.clear_staging()?;
            }
            let Some(peer) = busy else {
                return Ok(None);
            };
            if Instant::now() >= deadline {
                return Ok(None);
            }
            sleep(if self.own_ip < peer.ip() {
                BUSY_RETRY
            } else {
                BUSY_RETRY * 3
            })
            .await;
        }
    }
}
