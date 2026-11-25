//! Checkpoints between the daemons of different nodes: answer a peer that
//! asks for a revision this node holds, and ask peers for one it lacks.
//!
//! ```text
//! pull:  resolve the peers ─► ask each what it holds, at once ─► nearest holder first
//!        ─► take a lane ─► upgrade and receive ─► publish
//! ```
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use k8s_openapi::api::core::v1::Node;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep};
use tracing::{debug, info, warn};

use crate::lanes::Lanes;
use crate::node::environment::Environment;
use crate::node::manifest::Manifest;
use crate::store::{Classes, RevisionKey, RevisionStore, directory_bytes};
use crate::topology::{Candidate, Place, nearest_first};
use crate::transfer::{self, Endpoint, Pool};
use crate::tunnel::{self, Holder, Identity, Refusal, Status, Tunnel, Work};

const BUSY_RETRY: Duration = Duration::from_secs(2);

#[derive(clap::Args)]
#[group(id = "peers")]
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
    /// Transfers this daemon takes part in at once, sending or receiving.
    /// The pool's slots are split between them.
    #[arg(long, env = "SNAPSHOTD_TRANSFER_LANES", default_value_t = 2)]
    transfer_lanes: usize,
    /// Node labels that say where a node is, widest domain first, as in a
    /// Kueue Topology. A checkpoint is pulled from the holder that shares
    /// the most levels with this node. Needs --node-name.
    #[arg(
        long,
        env = "SNAPSHOTD_TOPOLOGY_LEVELS",
        value_delimiter = ',',
        requires = "node_name"
    )]
    topology_levels: Vec<String>,
    /// The node this daemon runs on.
    #[arg(long, env = "SNAPSHOTD_NODE_NAME")]
    node_name: Option<String>,
    #[command(flatten)]
    pool: Pool,
}

/// This node's value for each of `levels`, read from its labels.
async fn place_of(node: &str, levels: &[String]) -> Result<Place> {
    let client = kube::Client::try_default()
        .await
        .context("reach the Kubernetes API")?;
    let labels = kube::Api::<Node>::all(client)
        .get(node)
        .await
        .with_context(|| format!("read node {node}"))?
        .metadata
        .labels
        .unwrap_or_default();
    Ok(Place::of(levels, &labels))
}

pub struct Peers {
    name: String,
    own_ip: IpAddr,
    identity: Identity,
    busy_timeout: Duration,
    place: Place,
    classes: Classes,
    environment: Environment,
    /// One lane for each transfer this daemon can take part in at once,
    /// sending or receiving, each with its share of the registered pool.
    transfers: Lanes<Endpoint>,
    /// A permit for each lane. A pull takes one before it asks, so a lane
    /// is free for what the peer sends back.
    turns: Arc<Semaphore>,
}

impl Peers {
    pub async fn start(
        args: &Args,
        classes: Classes,
        environment: Environment,
    ) -> Result<Option<Arc<Self>>> {
        let (Some(name), Some(own_ip), Some(ca_file)) =
            (&args.peers, args.own_ip, &args.peer_ca_file)
        else {
            return Ok(None);
        };
        let lanes = args.transfer_lanes;
        ensure!(lanes > 0, "--transfer-lanes must be at least 1");
        let pool = Pool {
            slots: (args.pool.slots / lanes).max(1),
            slot_mib: args.pool.slot_mib,
        };
        let agent = format!("snapshotd-{own_ip}-{}", std::process::id());
        let place = match &args.node_name {
            Some(node) if !args.topology_levels.is_empty() => {
                place_of(node, &args.topology_levels).await?
            }
            _ => Place::default(),
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
            place,
            classes,
            environment,
            transfers: Lanes::start(lanes, move |lane| {
                Endpoint::new(&format!("{agent}-{lane}"), pool, false)
            })?,
            turns: Arc::new(Semaphore::new(lanes)),
        });
        let listener = TcpListener::bind(args.peer_listen)
            .await
            .with_context(|| format!("listen on {}", args.peer_listen))?;
        info!(
            address = %args.peer_listen,
            name = %name,
            place = %peers.place,
            "answering peers"
        );
        tokio::spawn(tunnel::serve(listener, Arc::clone(&peers)));
        Ok(Some(peers))
    }
}

impl Holder for Peers {
    fn status(&self, revision: &RevisionKey) -> Status {
        let image = self.classes.image_of(revision);
        let environment = image.as_deref().and_then(|image| match Manifest::load(image) {
            Ok(manifest) => manifest.environment,
            Err(error) => {
                debug!(%revision, error = %format_args!("{error:#}"), "the held manifest is unreadable");
                None
            }
        });
        Status {
            held: image.is_some(),
            free_lanes: self.turns.available_permits(),
            place: self.place.clone(),
            environment,
        }
    }

    fn admit(&self, revision: &RevisionKey) -> Result<Work, Refusal> {
        let image = self.classes.image_of(revision).ok_or(Refusal::NoImage)?;
        let turn = Arc::clone(&self.turns)
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
}

impl Peers {
    async fn addresses(&self) -> Vec<SocketAddr> {
        match tokio::net::lookup_host(&self.name).await {
            Ok(addresses) => tunnel::ask_order(
                self.own_ip,
                addresses.filter(|a| a.ip() != self.own_ip).collect(),
            ),
            Err(error) => {
                debug!(name = %self.name, %error, "the peer name does not resolve");
                Vec::new()
            }
        }
    }

    /// The peers that hold `revision`, the nearest first.
    async fn holders(self: &Arc<Self>, revision: &RevisionKey) -> Vec<Candidate> {
        let addresses = self.addresses().await;
        let mut asked = JoinSet::new();
        for peer in addresses.iter().copied() {
            let (peers, revision) = (Arc::clone(self), revision.clone());
            asked.spawn(async move {
                (
                    peer,
                    tunnel::inquire(peer, &peers.identity, &revision).await,
                )
            });
        }
        let mut holding = Vec::new();
        while let Some(answer) = asked.join_next().await {
            match answer {
                Ok((address, Ok(status))) if status.held => {
                    match self
                        .environment
                        .incompatibility(status.environment.as_ref())
                    {
                        None => holding.push(Candidate {
                            address,
                            place: status.place,
                            free_lanes: status.free_lanes,
                        }),
                        Some(reason) => {
                            debug!(peer = %address, reason, "the peer's checkpoint cannot be restored here")
                        }
                    }
                }
                Ok((peer, Ok(_))) => debug!(%peer, "the peer has no checkpoint of it"),
                Ok((peer, Err(error))) => {
                    debug!(%peer, error = %format_args!("{error:#}"), "the peer did not answer")
                }
                Err(error) => debug!(%error, "asking a peer was cut short"),
            }
        }
        holding.sort_by_key(|candidate| addresses.iter().position(|a| *a == candidate.address));
        nearest_first(&self.place, holding)
    }

    /// Returns the peer the checkpoint came from, if any had it.
    pub async fn pull(
        self: &Arc<Self>,
        revision: &RevisionKey,
        store: &RevisionStore,
    ) -> Result<Option<SocketAddr>> {
        let deadline = Instant::now() + self.busy_timeout;
        loop {
            let holders = self.holders(revision).await;
            let mut busy = None;
            {
                let _turn = self
                    .turns
                    .acquire()
                    .await
                    .map_err(|_| anyhow!("the transfer turn is closed"))?;
                for holder in holders {
                    let peer = holder.address;
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
                    info!(
                        %peer,
                        place = %holder.place,
                        shared_levels = self.place.shared_levels(&holder.place),
                        of = self.place.levels(),
                        "asking a peer that has it"
                    );
                    match tunnel::ask(peer, &self.identity, revision, receive).await {
                        Ok(Ok(())) => {
                            store.publish()?;
                            return Ok(Some(peer));
                        }
                        Ok(Err(Refusal::Busy)) => busy = Some(peer),
                        Ok(Err(Refusal::NoImage)) => {
                            debug!(%peer, "the peer no longer has it")
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
