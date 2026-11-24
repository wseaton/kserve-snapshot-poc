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
mod criu;
mod layout;
mod manifest;
mod pod;
mod process;
mod report;
mod shim;

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, ensure};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};
use tracing::{Instrument, debug, error, info, info_span, warn};

use crate::cri::{Runtime, Sandbox};
use crate::metrics::{self, How, Metrics, Step};
use crate::node::manifest::Manifest;
use crate::node::pod::Pod;
use crate::node::process::{HostPid, PodPid};
use crate::node::report::{seconds, short_id};
use crate::node::shim::ShimClient;
#[cfg(feature = "nixl")]
use crate::peer::{self, Peers};
use crate::podresources;
#[cfg(feature = "nixl")]
use crate::store::directory_bytes;
use crate::store::{Class, Classes, RevisionKey, RevisionStore};
use crate::timeslice;
use crate::vllm::{Control, Lifecycle};

#[derive(clap::Args)]
pub struct Args {
    /// The container runtime's CRI socket.
    #[arg(
        long,
        env = "SNAPSHOTD_RUNTIME_SOCKET",
        default_value = "/run/containerd/containerd.sock"
    )]
    runtime_socket: PathBuf,
    /// The kubelet's PodResources socket.
    #[arg(
        long,
        env = "SNAPSHOTD_POD_RESOURCES_SOCKET",
        default_value = "/var/lib/kubelet/pod-resources/kubelet.sock"
    )]
    pod_resources_socket: PathBuf,
    /// Directory on the node's data disk where checkpoints are kept.
    #[arg(long, env = "SNAPSHOTD_STORE_DIR")]
    store_dir: PathBuf,
    /// Pod label that opts a pod in; its value names the workload.
    #[arg(
        long,
        env = "SNAPSHOTD_LABEL",
        default_value = "snapshot.llm-d.ai/workload"
    )]
    label: String,
    /// Name of the container that runs the server.
    #[arg(long, env = "SNAPSHOTD_CONTAINER", default_value = "main")]
    container: String,
    /// vLLM's gRPC control port.
    #[arg(long, env = "SNAPSHOTD_CONTROL_PORT", default_value_t = 50051)]
    control_port: u16,
    /// Free space the store must have before a checkpoint is taken.
    #[arg(long, env = "SNAPSHOTD_MIN_FREE_GIB", default_value_t = 256)]
    min_free_gib: u64,
    /// GiB one namespace may hold in the store before its checkpoints stop
    /// being taken.
    #[arg(long, env = "SNAPSHOTD_NAMESPACE_LIMIT_GIB")]
    namespace_limit_gib: Option<u64>,
    /// Hours the store keeps a revision after a pod of it last started.
    #[arg(long, env = "SNAPSHOTD_RETAIN_HOURS")]
    retain_hours: Option<u64>,
    /// JSON file of more store classes: an object of name to `dir` and,
    /// optionally, `min_free_gib`, `namespace_limit_gib` and `retain_hours`.
    /// The options above describe the class `default`.
    #[arg(long, env = "SNAPSHOTD_CLASSES_FILE")]
    classes_file: Option<PathBuf>,
    /// Pod annotation that names the store class a pod's checkpoints go to.
    #[arg(
        long,
        env = "SNAPSHOTD_CLASS_ANNOTATION",
        default_value = "snapshot.llm-d.ai/class"
    )]
    class_annotation: String,
    /// Seconds between looks at the stores, which measure them and remove
    /// what is past its retention.
    #[arg(long, env = "SNAPSHOTD_SWEEP_S", default_value_t = 600)]
    sweep_s: u64,
    /// Address to serve Prometheus metrics on, at /metrics.
    #[arg(long, env = "SNAPSHOTD_METRICS_LISTEN", default_value = "0.0.0.0:9090")]
    metrics_listen: SocketAddr,
    /// Pod label that names a pod's revision.
    #[arg(
        long,
        env = "SNAPSHOTD_REVISION_LABEL",
        default_value = "pod-template-hash"
    )]
    revision_label: String,
    /// Seconds between looks at the runtime's pods.
    #[arg(long, env = "SNAPSHOTD_POLL_S", default_value_t = 2)]
    poll_s: u64,
    /// Seconds a pod's server container may take to start.
    #[arg(long, env = "SNAPSHOTD_CONTAINER_TIMEOUT_S", default_value_t = 600)]
    container_timeout_s: u64,
    /// Seconds a pod's shim may take to listen.
    #[arg(long, env = "SNAPSHOTD_SHIM_TIMEOUT_S", default_value_t = 60)]
    shim_timeout_s: u64,
    /// Seconds a server may take to cold boot.
    #[arg(long, env = "SNAPSHOTD_BOOT_TIMEOUT_S", default_value_t = 3600)]
    boot_timeout_s: u64,
    /// Seconds before a pod that could not be handled is tried again.
    #[arg(long, env = "SNAPSHOTD_RETRY_S", default_value_t = 10)]
    retry_s: u64,
    #[command(flatten)]
    pod: pod::Options,
    #[command(flatten)]
    timeslice: timeslice::Args,
    #[cfg(feature = "nixl")]
    #[command(flatten)]
    peers: peer::Args,
}

struct Node {
    args: Args,
    classes: Classes,
    metrics: Arc<Metrics>,
    #[cfg(feature = "nixl")]
    peers: Option<Arc<Peers>>,
    /// A lock for each revision, held while its checkpoint is pulled, so
    /// pods of one revision that start together pull it once.
    #[cfg(feature = "nixl")]
    pulling: std::sync::Mutex<HashMap<RevisionKey, Arc<tokio::sync::Mutex<()>>>>,
}

#[cfg(feature = "nixl")]
impl Node {
    fn pull_turn(&self, revision: &RevisionKey) -> Result<Arc<tokio::sync::Mutex<()>>> {
        let mut pulling = self
            .pulling
            .lock()
            .map_err(|_| anyhow::anyhow!("pull lock poisoned"))?;
        Ok(Arc::clone(pulling.entry(revision.clone()).or_default()))
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Image {
    Absent,
    Usable,
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

#[derive(Debug, PartialEq, Eq)]
enum Plan {
    Restore,
    Capture,
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

#[derive(Clone, Copy)]
enum Served {
    StartedEarlier,
    Restored,
    ColdBooted,
    Captured,
}

impl Served {
    fn how(self) -> How {
        match self {
            Self::StartedEarlier => How::StartedEarlier,
            Self::Restored => How::Restore,
            Self::ColdBooted => How::ColdBoot,
            Self::Captured => How::Capture,
        }
    }
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

async fn container_init(
    runtime: &mut Runtime,
    sandbox: &Sandbox,
    name: &str,
    within: Duration,
) -> Result<HostPid> {
    let deadline = Instant::now() + within;
    loop {
        if let Some(pid) = runtime.pid(sandbox, name).await? {
            return Ok(HostPid(pid));
        }
        ensure!(Instant::now() < deadline, "container {name} did not start");
        sleep(Duration::from_secs(1)).await;
    }
}

fn revision_of(sandbox: &Sandbox, label: &str, revision_label: &str) -> Result<RevisionKey> {
    let workload = sandbox
        .labels
        .get(label)
        .context("the opt-in label is gone")?;
    let hash = sandbox
        .labels
        .get(revision_label)
        .with_context(|| format!("no {revision_label} label; is the pod from a Deployment?"))?;
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
    match peers.pull(revision, store).await? {
        Some(peer) => info!(
            %peer,
            seconds = %seconds(asked),
            gib = %report::gib(directory_bytes(&store.image_dir())),
            "pulled the checkpoint from a peer"
        ),
        None => info!(seconds = %seconds(asked), "no other node has it"),
    }
    Ok(())
}

async fn cold_boot(
    node: &Node,
    pod: &Pod,
    shim: &mut ShimClient,
) -> Result<(PodPid, Control, Lifecycle)> {
    let asked = Instant::now();
    let server = shim.start().await?;
    let within = Duration::from_secs(node.args.boot_timeout_s);
    let (control, state) = Control::connect_when_up(pod.control(), within).await?;
    info!(seconds = %seconds(asked), ?state, "the server is up");
    node.metrics.stepped(Step::ColdBoot, asked.elapsed());
    Ok((server, control, state))
}

async fn cold_boot_only(
    node: &Node,
    pod: &Pod,
    shim: &mut ShimClient,
    because: String,
) -> Result<Served> {
    warn!(because, "cold booting, with no checkpoint taken");
    let (_, control, state) = cold_boot(node, pod, shim).await?;
    if state == Lifecycle::Prepared {
        pod.resume(control).await?;
    }
    Ok(Served::ColdBooted)
}

async fn adopt_or_give_up(
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

async fn bring_up(
    node: &Node,
    sandbox: &Sandbox,
    pod: &Pod,
    shim: &mut ShimClient,
) -> Result<Served> {
    let args = &node.args;
    let revision = revision_of(sandbox, &args.label, &args.revision_label)?;
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
    let named = sandbox.annotations.get(&args.class_annotation);
    let class = match node.classes.get(named.map(String::as_str)) {
        Ok(class) => class,
        Err(error) => return cold_boot_only(node, pod, shim, format!("{error:#}")).await,
    };
    let store = RevisionStore::open(&class.dir, &revision, &sandbox.uid)?;
    #[cfg(feature = "nixl")]
    if let Some(peers) = &node.peers {
        let turn = node.pull_turn(&revision)?;
        let _turn = turn.lock().await;
        if !store.image_ready() && !store.failed_before() {
            let asked = Instant::now();
            pull_from_peers(peers, &revision, &store).await?;
            if store.image_ready() {
                node.metrics.stepped(Step::Pull, asked.elapsed());
            }
        }
    }
    let plan = Plan::choose(
        store.failed_before(),
        Image::of(&store, pod)?,
        class.room(&revision),
    );
    match plan {
        Plan::Restore => {
            info!("restoring the revision's checkpoint");
            let asked = Instant::now();
            let restored = pod.restore(&store.image_dir()).await;
            adopt_or_give_up(restored, pod, &store, shim).await?;
            node.metrics.stepped(Step::Restore, asked.elapsed());
            Ok(Served::Restored)
        }
        Plan::ColdBoot(because) => cold_boot_only(node, pod, shim, because).await,
        Plan::Capture => {
            info!("cold booting; this pod gives the revision its checkpoint");
            let (server, control, _) = cold_boot(node, pod, shim).await?;
            store.clear_staging()?;
            let staging = store.staging_dir();
            let cycled = async {
                let asked = Instant::now();
                pod.checkpoint(control, shim, server, staging).await?;
                node.metrics.stepped(Step::Checkpoint, asked.elapsed());
                let asked = Instant::now();
                let restored = pod.restore(staging).await?;
                node.metrics.stepped(Step::Restore, asked.elapsed());
                Ok(restored)
            }
            .await;
            adopt_or_give_up(cycled, pod, &store, shim).await?;
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
    let init = container_init(
        &mut runtime,
        sandbox,
        &args.container,
        Duration::from_secs(args.container_timeout_s),
    )
    .await?;
    let mut shim = ShimClient::connect(init, Duration::from_secs(args.shim_timeout_s)).await?;
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
    let pod = Pod::new(init, control, &args.pod, &allotted)?;
    let served = if progress.started {
        info!("the server was started before this daemon; waiting for it to run");
        let boot = Duration::from_secs(args.boot_timeout_s);
        let (control, state) = Control::connect_when_up(pod.control(), boot).await?;
        if state == Lifecycle::Prepared {
            pod.resume(control).await?;
        }
        Served::StartedEarlier
    } else {
        bring_up(node, sandbox, &pod, &mut shim).await?
    };
    shim.serving().await?;
    info!(seconds = %seconds(seen), how = %served, "serving");
    node.metrics
        .served(&sandbox.namespace, workload_of(node, sandbox), served.how());
    Ok(())
}

fn workload_of<'a>(node: &Node, sandbox: &'a Sandbox) -> &'a str {
    sandbox
        .labels
        .get(&node.args.label)
        .map_or("", String::as_str)
}

/// Remove what is past its retention and measure what is left.
async fn look_at_stores(node: &Arc<Node>) {
    let node = Arc::clone(node);
    let looked = tokio::task::spawn_blocking(move || {
        let mut seen = Vec::new();
        for (name, class) in node.classes.iter() {
            match class.sweep(SystemTime::now()) {
                Ok(removed) => {
                    for (revision, bytes) in &removed {
                        info!(
                            class = name,
                            %revision,
                            gib = %report::gib(*bytes),
                            "removed a revision no pod has started within its retention"
                        );
                    }
                    let bytes = removed.iter().map(|(_, bytes)| bytes).sum();
                    node.metrics.swept(name, removed.len() as u64, bytes);
                }
                Err(error) => {
                    warn!(class = name, error = %format_args!("{error:#}"), "could not sweep a store")
                }
            }
            seen.push((
                name.to_string(),
                class.usage(),
                class.free_bytes().unwrap_or(0),
            ));
        }
        node.metrics.stored(&seen);
    })
    .await;
    if looked.is_err() {
        warn!("a look at the stores panicked");
    }
}

fn watch(node: Arc<Node>, sandbox: Sandbox) -> JoinHandle<()> {
    let span = info_span!("pod", id = short_id(&sandbox.name));
    tokio::spawn(
        async move {
            while let Err(error) = look_after(&node, &sandbox).await {
                error!(
                    error = %format!("{error:#}").replace('\n', "; "),
                    "gave up on this pod for now"
                );
                node.metrics
                    .failed(&sandbox.namespace, workload_of(&node, &sandbox));
                sleep(Duration::from_secs(node.args.retry_s)).await;
            }
        }
        .instrument(span),
    )
}

async fn serve(args: Args) -> Result<()> {
    let classes = Classes::load(
        Class {
            dir: args.store_dir.clone(),
            min_free_gib: args.min_free_gib,
            namespace_limit_gib: args.namespace_limit_gib,
            retain_hours: args.retain_hours,
        },
        args.classes_file.as_deref(),
    )?;
    ensure!(
        args.pod.tools_dir.join("criu").is_file(),
        "no criu in {}",
        args.pod.tools_dir.display()
    );
    let metrics = Arc::new(Metrics::default());
    metrics::start(args.metrics_listen, Arc::clone(&metrics)).await?;
    let node = Arc::new(Node {
        #[cfg(feature = "nixl")]
        peers: Peers::start(&args.peers, classes.clone()).await?,
        classes,
        metrics,
        #[cfg(feature = "nixl")]
        pulling: std::sync::Mutex::default(),
        args,
    });
    let args = &node.args;
    let mut runtime = Runtime::connect(&args.runtime_socket).await?;
    timeslice::start(
        &args.timeslice,
        &args.runtime_socket,
        Arc::clone(&node.metrics),
    )?;
    info!(
        label = %args.label,
        classes = %node.classes,
        tools = %args.pod.tools_dir.display(),
        runtime = %args.runtime_socket.display(),
        "watching this node's pods"
    );
    let mut watched: HashMap<String, JoinHandle<()>> = HashMap::new();
    let mut looked = None::<Instant>;
    loop {
        if looked.is_none_or(|at| at.elapsed() >= Duration::from_secs(args.sweep_s)) {
            look_at_stores(&node).await;
            looked = Some(Instant::now());
        }
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
