//! The node daemon. For every opted-in pod that starts on this node it
//! either restores the pod's revision from a checkpoint, or lets the server
//! cold boot, checkpoints it once and restores it from that.
//!
//! ```text
//! CRI ──► new pod ──► shim.sock ──► plan
//!           restore: the node's checkpoint, or one pulled from a peer ──► serving
//!           cold boot, checkpoint, restore ─────────────────────────────► serving
//!           cold boot (no room, or it failed here before) ──────────────► serving
//! ```
//!
//! The pod stays unprivileged. The daemon reaches vLLM on its gRPC control
//! port, the driver's checkpoint API by host PID, and runs CRIU inside the
//! pod's namespaces with its own tools and the image directory mounted in.
mod criu;
mod layout;
mod log;
mod manifest;
mod pod;
mod process;
mod shim;

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};
use tracing::{Instrument, debug, error, info, info_span, warn};

use crate::cri::{Runtime, Sandbox};
use crate::node::log::{seconds, short_id};
use crate::node::manifest::Manifest;
use crate::node::pod::Pod;
use crate::node::process::{HostPid, PodPid};
use crate::node::shim::ShimClient;
#[cfg(feature = "nixl")]
use crate::peer::{self, Peers};
use crate::podresources;
#[cfg(feature = "nixl")]
use crate::store::directory_bytes;
use crate::store::{RevisionKey, RevisionStore};
use crate::vllm::Control;

/// The label a ReplicaSet puts on its pods; it names their revision.
const REVISION_LABEL: &str = "pod-template-hash";
const CONTAINER_WAIT: Duration = Duration::from_secs(600);
const BOOT: Duration = Duration::from_secs(3600);
const RETRY: Duration = Duration::from_secs(10);

#[derive(clap::Args)]
pub struct Args {
    /// The container runtime's CRI socket.
    #[arg(long, default_value = "/run/containerd/containerd.sock")]
    runtime_socket: PathBuf,
    /// The kubelet's PodResources socket.
    #[arg(long, default_value = "/var/lib/kubelet/pod-resources/kubelet.sock")]
    pod_resources_socket: PathBuf,
    /// Directory on the node's data disk where checkpoints are kept.
    #[arg(long)]
    store_dir: PathBuf,
    /// Directory holding criu, its lib/ and its criu-plugins/.
    #[arg(long)]
    tools_dir: PathBuf,
    /// Pod label that opts a pod in; its value names the workload.
    #[arg(long, default_value = "snapshot.llm-d.ai/workload")]
    label: String,
    /// Name of the container that runs the server.
    #[arg(long, default_value = "main")]
    container: String,
    /// vLLM's gRPC control port.
    #[arg(long, default_value_t = 50051)]
    control_port: u16,
    /// Free space the store must have before a checkpoint is taken.
    #[arg(long, default_value_t = 256)]
    min_free_gib: u64,
    /// Seconds between looks at the runtime's pods.
    #[arg(long, default_value_t = 2)]
    poll_s: u64,
    #[cfg(feature = "nixl")]
    #[command(flatten)]
    peers: peer::Args,
}

/// What every pod's task shares.
struct Node {
    args: Args,
    #[cfg(feature = "nixl")]
    peers: Option<Arc<Peers>>,
}

/// The node's checkpoint of a pod's revision, as far as that pod is concerned.
#[derive(Debug, PartialEq, Eq)]
enum Image {
    Absent,
    Usable,
    /// There is one and it cannot be restored into this pod.
    Unusable(String),
}

impl Image {
    fn of(store: &RevisionStore, pod: &Pod) -> Result<Self> {
        if !store.image_ready() {
            return Ok(Self::Absent);
        }
        let image = store.image_dir();
        Ok(match Manifest::load(&image)?.refusal(&image, pod.gpus())? {
            Some(reason) => Self::Unusable(reason),
            None => Self::Usable,
        })
    }
}

/// What to do with a pod that has not started its server.
#[derive(Debug, PartialEq, Eq)]
enum Plan {
    Restore,
    /// Cold boot, and take the revision's checkpoint from this pod.
    Capture,
    /// Cold boot and nothing more, and why.
    ColdBoot(String),
}

impl Plan {
    /// `room` is whether the store can take a new checkpoint.
    fn choose(failed_before: bool, image: Image, room: Result<()>) -> Self {
        if failed_before {
            return Self::ColdBoot("a checkpoint or restore failed in this pod before".to_string());
        }
        match (image, room) {
            (Image::Usable, _) => Self::Restore,
            (Image::Unusable(reason), _) => Self::ColdBoot(format!(
                "the node's checkpoint cannot be restored into it: {reason}"
            )),
            (Image::Absent, Ok(())) => Self::Capture,
            (Image::Absent, Err(error)) => Self::ColdBoot(format!("{error:#}")),
        }
    }
}

/// How a pod came to serve.
#[derive(Clone, Copy)]
enum Served {
    StartedEarlier,
    Restored,
    ColdBooted,
    Captured,
}

impl fmt::Display for Served {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StartedEarlier => "started before this daemon",
            Self::Restored => "restore",
            Self::ColdBooted => "cold boot",
            Self::Captured => "cold boot, checkpoint, restore",
        })
    }
}

/// Wait for the pod's server container and return its PID 1.
async fn container_init(runtime: &mut Runtime, sandbox: &Sandbox, name: &str) -> Result<HostPid> {
    let deadline = Instant::now() + CONTAINER_WAIT;
    loop {
        if let Some(pid) = runtime.pid(sandbox, name).await? {
            return Ok(HostPid(pid));
        }
        ensure!(Instant::now() < deadline, "container {name} did not start");
        sleep(Duration::from_secs(1)).await;
    }
}

fn revision_of(sandbox: &Sandbox, label: &str) -> Result<RevisionKey> {
    let workload = sandbox
        .labels
        .get(label)
        .context("the opt-in label is gone")?;
    let hash = sandbox
        .labels
        .get(REVISION_LABEL)
        .with_context(|| format!("no {REVISION_LABEL} label; is the pod from a Deployment?"))?;
    RevisionKey::new(&sandbox.namespace, workload, hash)
}

#[cfg(feature = "nixl")]
async fn pull_from_peers(
    peers: &Arc<Peers>,
    revision: &RevisionKey,
    store: &RevisionStore,
) -> Result<()> {
    info!("this node has no checkpoint of the revision; asking the other nodes");
    let asked = Instant::now();
    let (peers, revision, into) = (Arc::clone(peers), revision.clone(), store.clone());
    match process::blocking(move || peers.pull(&revision, &into)).await? {
        Some(peer) => info!(
            %peer,
            seconds = %seconds(asked),
            gib = %log::gib(directory_bytes(&store.image_dir())),
            "pulled the checkpoint from a peer"
        ),
        None => info!(seconds = %seconds(asked), "no other node has it"),
    }
    Ok(())
}

async fn cold_boot(pod: &Pod, shim: &mut ShimClient) -> Result<(PodPid, Control)> {
    let asked = Instant::now();
    let server = shim.start().await?;
    let control = Control::connect_when_running(pod.control(), BOOT).await?;
    info!(seconds = %seconds(asked), "the server is running");
    Ok((server, control))
}

/// Hand a restored server to the shim. If there is none, the server the pod
/// had is gone or in an unknown state: the container starts over and this
/// pod does not try again.
async fn adopt(
    restored: Result<PodPid>,
    pod: &Pod,
    store: &RevisionStore,
    shim: &mut ShimClient,
) -> Result<()> {
    let error = match restored {
        Ok(server) => return shim.adopt(server).await,
        Err(error) => error,
    };
    pod.log_server_tail();
    if let Err(error) = store.record_failure(&error) {
        warn!(error = %format_args!("{error:#}"), "the failure is not recorded");
    }
    if let Err(error) = shim.fail().await {
        warn!(error = %format_args!("{error:#}"), "the shim was not told to start over");
    }
    Err(error.context("the container starts over cold"))
}

/// Get a pod whose server has not started to a running server.
async fn bring_up(
    node: &Node,
    sandbox: &Sandbox,
    pod: &Pod,
    shim: &mut ShimClient,
) -> Result<Served> {
    let args = &node.args;
    let revision = revision_of(sandbox, &args.label)?;
    let store = RevisionStore::open(&args.store_dir, &revision, &sandbox.uid)?;
    info!(
        pod = %format_args!("{}/{}", sandbox.namespace, sandbox.name),
        %revision,
        address = %pod.address(),
        gpus = pod.gpus().len(),
        visible_gpus = pod.visible_gpus(),
        pid = %pod.init(),
        "new pod"
    );
    pod.mount_tools().await?;
    #[cfg(feature = "nixl")]
    if let Some(peers) = &node.peers
        && !store.image_ready()
        && !store.failed_before()
    {
        pull_from_peers(peers, &revision, &store).await?;
    }
    let plan = Plan::choose(
        store.failed_before(),
        Image::of(&store, pod)?,
        store.ensure_room(args.min_free_gib << 30),
    );
    match plan {
        Plan::Restore => {
            info!("restoring the revision's checkpoint");
            let restored = pod.restore(&store.image_dir()).await;
            adopt(restored, pod, &store, shim).await?;
            Ok(Served::Restored)
        }
        Plan::ColdBoot(because) => {
            warn!(because, "cold booting, with no checkpoint taken");
            cold_boot(pod, shim).await?;
            Ok(Served::ColdBooted)
        }
        Plan::Capture => {
            info!("cold booting; this pod gives the revision its checkpoint");
            let (server, control) = cold_boot(pod, shim).await?;
            store.clear_staging()?;
            let staging = store.staging_dir();
            let cycled = async {
                pod.checkpoint(control, shim, server, staging).await?;
                pod.restore(staging).await
            }
            .await;
            adopt(cycled, pod, &store, shim).await?;
            store.publish()?;
            info!(image = %store.image_dir().display(), "checkpoint published on this node");
            Ok(Served::Captured)
        }
    }
}

async fn look_after(node: &Node, sandbox: &Sandbox) -> Result<()> {
    let args = &node.args;
    let seen = Instant::now();
    let mut runtime = Runtime::connect(&args.runtime_socket).await?;
    let init = container_init(&mut runtime, sandbox, &args.container).await?;
    let mut shim = ShimClient::connect(init).await?;
    let progress = shim.progress().await?;
    if progress.serving {
        debug!("already serving, nothing to do");
        return Ok(());
    }
    let allotted = podresources::gpus(
        &args.pod_resources_socket,
        &sandbox.namespace,
        &sandbox.name,
        &args.container,
    )
    .await?;
    let control = SocketAddr::new(runtime.address(sandbox).await?, args.control_port);
    let pod = Pod::new(init, control, &args.tools_dir, &allotted)?;
    let served = if progress.started {
        info!("the server was started before this daemon; waiting for it to run");
        Control::connect_when_running(pod.control(), BOOT).await?;
        Served::StartedEarlier
    } else {
        bring_up(node, sandbox, &pod, &mut shim).await?
    };
    shim.serving().await?;
    info!(seconds = %seconds(seen), how = %served, "serving");
    Ok(())
}

/// Look after one pod until it serves, trying again for as long as it fails.
fn watch(node: Arc<Node>, sandbox: Sandbox) -> JoinHandle<()> {
    let span = info_span!("pod", id = short_id(&sandbox.name));
    tokio::spawn(
        async move {
            while let Err(error) = look_after(&node, &sandbox).await {
                error!(
                    error = %format!("{error:#}").replace('\n', "; "),
                    "gave up on this pod for now"
                );
                sleep(RETRY).await;
            }
        }
        .instrument(span),
    )
}

async fn serve(args: Args) -> Result<()> {
    std::fs::create_dir_all(&args.store_dir)
        .with_context(|| format!("create {}", args.store_dir.display()))?;
    ensure!(
        args.tools_dir.join("criu").is_file(),
        "no criu in {}",
        args.tools_dir.display()
    );
    let node = Arc::new(Node {
        #[cfg(feature = "nixl")]
        peers: Peers::start(&args.peers, &args.store_dir)?,
        args,
    });
    let args = &node.args;
    let mut runtime = Runtime::connect(&args.runtime_socket).await?;
    info!(
        label = %args.label,
        store = %args.store_dir.display(),
        tools = %args.tools_dir.display(),
        runtime = %args.runtime_socket.display(),
        "watching this node's pods"
    );
    let mut watched: HashMap<String, JoinHandle<()>> = HashMap::new();
    loop {
        match runtime.sandboxes(&args.label).await {
            Ok(sandboxes) => {
                watched.retain(|id, task| {
                    let here = sandboxes.iter().any(|sandbox| &sandbox.id == id);
                    if !here {
                        task.abort();
                    }
                    here
                });
                for sandbox in sandboxes {
                    watched
                        .entry(sandbox.id.clone())
                        .or_insert_with(|| watch(Arc::clone(&node), sandbox));
                }
            }
            Err(error) => warn!(error = %format_args!("{error:#}"), "could not list pods"),
        }
        sleep(Duration::from_secs(args.poll_s)).await;
    }
}

pub fn run(args: Args) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("start the runtime")?
        .block_on(serve(args))
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;

    use crate::node::{Image, Plan};

    #[test]
    fn a_pod_is_restored_captured_or_only_cold_booted() {
        let unusable = || Image::Unusable("other GPUs".to_string());
        let full = || Err(anyhow!("12 bytes free"));
        let cold = |because: &str| Plan::ColdBoot(because.to_string());
        let failed = "a checkpoint or restore failed in this pod before";
        let other = "the node's checkpoint cannot be restored into it: other GPUs";
        let cases = [
            (false, Image::Usable, Ok(()), Plan::Restore),
            (false, Image::Usable, full(), Plan::Restore),
            (false, Image::Absent, Ok(()), Plan::Capture),
            (false, Image::Absent, full(), cold("12 bytes free")),
            (false, unusable(), Ok(()), cold(other)),
            (false, unusable(), full(), cold(other)),
            (true, Image::Usable, Ok(()), cold(failed)),
            (true, Image::Absent, Ok(()), cold(failed)),
            (true, unusable(), Ok(()), cold(failed)),
        ];
        for (number, (failed_before, image, room, wanted)) in cases.into_iter().enumerate() {
            assert_eq!(
                Plan::choose(failed_before, image, room),
                wanted,
                "case {number}"
            );
        }
    }
}
