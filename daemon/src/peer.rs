//! Checkpoints between the daemons of different nodes: answer a peer that
//! asks for a revision this node holds, and ask peers for one it lacks.
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::time::{Instant, sleep};
use tracing::{debug, info, warn};

use crate::store::{RevisionKey, RevisionStore, directory_bytes};
use crate::transfer::{self, Pool};
use crate::tunnel::{self, Identity, Refusal, Tunnel, Work};

const BUSY_RETRY: Duration = Duration::from_secs(2);
const BUSY_WAIT: Duration = Duration::from_secs(300);

#[derive(clap::Args)]
pub struct Args {
    /// Name and port that resolve to the kube-rbac-proxy of every node's
    /// daemon. Without it, checkpoints stay on the node that took them.
    #[arg(long, requires_all = ["own_ip", "peer_ca_file"])]
    peers: Option<String>,
    /// Where this daemon answers its own kube-rbac-proxy.
    #[arg(long, default_value = "127.0.0.1:7700")]
    peer_listen: SocketAddr,
    /// This daemon's address, left out of the peers.
    #[arg(long)]
    own_ip: Option<IpAddr>,
    /// The ServiceAccount token this daemon shows its peers.
    #[arg(long, default_value = "/var/run/secrets/snapshotd/peers/token")]
    peer_token_file: PathBuf,
    /// The authority that signs the certificate of the peers' proxies.
    #[arg(long)]
    peer_ca_file: Option<PathBuf>,
    #[command(flatten)]
    pool: Pool,
}

pub struct Peers {
    name: String,
    own_ip: IpAddr,
    identity: Identity,
    store: PathBuf,
    pool: Pool,
    sending: Arc<Semaphore>,
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
            },
            store: store.to_path_buf(),
            pool: args.pool,
            sending: Arc::new(Semaphore::new(1)),
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
        let turn = Arc::clone(&self.sending)
            .try_acquire_owned()
            .map_err(|_| Refusal::Busy)?;
        let (revision, pool) = (revision.clone(), self.pool);
        Ok(Box::new(move |tunnel: Tunnel| {
            let _turn = turn;
            let start = std::time::Instant::now();
            transfer::send_over(tunnel.socket, &image, pool, Some(&tunnel.key), None)?;
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
        let deadline = Instant::now() + BUSY_WAIT;
        loop {
            let mut busy = false;
            for peer in self.addresses().await {
                store.clear_staging()?;
                let staging = store.staging_dir().to_path_buf();
                let receive = move |tunnel: Tunnel| {
                    let key = Some(&tunnel.key);
                    transfer::receive_over(tunnel.socket, &staging, true, false, None, key, None)
                        .map(drop)
                };
                match tunnel::ask(peer, &self.identity, revision, receive).await {
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
            sleep(BUSY_RETRY).await;
        }
    }
}
