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
use std::net::{IpAddr, SocketAddr};
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
use crate::store::{Class, Classes, RevisionKey, RevisionStore, Workload};
use crate::timeslice;
use crate::vllm::{Control, Endpoint, Lifecycle, Transport};

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
    /// vLLM's HTTP port, which serves the `/checkpoint` routes in the Python
    /// and the Rust frontend.
    #[arg(long, env = "SNAPSHOTD_HTTP_PORT", default_value_t = 8000)]
    http_port: u16,
    /// How a pod's checkpoint lifecycle is driven when the pod does not name
    /// a transport: over the gRPC control port or the HTTP port.
    #[arg(
        long,
        env = "SNAPSHOTD_CONTROL_TRANSPORT",
        value_enum,
        default_value_t = Transport::Grpc
    )]
    control_transport: Transport,
    /// Pod annotation that names a pod's transport, `grpc` or `http`.
    #[arg(
        long,
        env = "SNAPSHOTD_CONTROL_ANNOTATION",
        default_value = "snapshot.llm-d.ai/control"
    )]
    control_annotation: String,
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
    /// Pod labels that name a pod's revision, in order; the first a pod
    /// carries is used.
    #[arg(
        long,
        env = "SNAPSHOTD_REVISION_LABEL",
        value_delimiter = ',',
        default_value = "disaggregatedset.x-k8s.io/revision,leaderworkerset.sigs.k8s.io/template-revision-hash,pod-template-hash"
    )]
    revision_label: Vec<String>,
    /// Pod label that names a workload's role, such as prefill or decode. Pods
    /// of one workload in different roles keep separate checkpoints. Empty to
    /// ignore roles.
    #[arg(
        long,
        env = "SNAPSHOTD_ROLE_LABEL",
        default_value = "disaggregatedset.x-k8s.io/role"
    )]
    role_label: String,
    /// Pod label that names which pod of a multi-pod group a pod is. Empty
    /// to treat every pod as a whole replica.
    #[arg(
        long,
        env = "SNAPSHOTD_MEMBER_LABEL",
        default_value = "leaderworkerset.sigs.k8s.io/worker-index"
    )]
    member_label: String,
    /// Pod annotation that says how many pods a group has. The member label
    /// counts only when it is above 1.
    #[arg(
        long,
        env = "SNAPSHOTD_GROUP_SIZE_ANNOTATION",
        default_value = "leaderworkerset.sigs.k8s.io/size"
    )]
    group_size_annotation: String,
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

impl Args {
    /// Where and how to reach the server of a pod at `address`. An annotation
    /// that names no transport falls back to the node's, with a warning.
    fn control(&self, sandbox: &Sandbox, address: IpAddr) -> Endpoint {
        let named = sandbox.annotations.get(&self.control_annotation);
        let transport = Transport::named(named.map(String::as_str), self.control_transport)
            .unwrap_or_else(|error| {
                warn!(
                    annotation = %self.control_annotation,
                    error = %format_args!("{error:#}"),
                    fallback = ?self.control_transport,
                    "using the node's control transport"
                );
                self.control_transport
            });
        let port = match transport {
            Transport::Grpc => self.control_port,
            Transport::Http => self.http_port,
        };
        Endpoint {
            transport,
            address: SocketAddr::new(address, port),
        }
    }

    fn labels(&self) -> Labels<'_> {
        Labels {
            workload: &self.label,
            role: &self.role_label,
            member: &self.member_label,
            group_size: &self.group_size_annotation,
            revision: &self.revision_label,
        }
    }
}

/// Which labels name a pod's workload, role, group member and revision.
struct Labels<'a> {
    workload: &'a str,
    role: &'a str,
    member: &'a str,
    group_size: &'a str,
    revision: &'a [String],
}

impl Labels<'_> {
    fn workload(&self, sandbox: &Sandbox) -> Result<Workload> {
        let name = sandbox
            .labels
            .get(self.workload)
            .context("the opt-in label is gone")?;
        let role = configured(self.role)
            .and_then(|label| sandbox.labels.get(label))
            .map(String::as_str);
        Workload::new(name, role, self.member(sandbox)?)
    }

    fn member(&self, sandbox: &Sandbox) -> Result<Option<u32>> {
        let (Some(label), Some(annotation)) =
            (configured(self.member), configured(self.group_size))
        else {
            return Ok(None);
        };
        let Some(size) = sandbox.annotations.get(annotation) else {
            return Ok(None);
        };
        let size: u32 = size
            .parse()
            .with_context(|| format!("{annotation} is {size:?}, not a pod count"))?;
        if size <= 1 {
            return Ok(None);
        }
        let member = sandbox
            .labels
            .get(label)
            .with_context(|| format!("a pod of a group of {size} has no {label} label"))?;
        member
            .parse()
            .map(Some)
            .with_context(|| format!("{label} is {member:?}, not a pod index"))
    }

    fn revision_of(&self, sandbox: &Sandbox) -> Result<RevisionKey> {
        let hash = self
            .revision
            .iter()
            .find_map(|label| sandbox.labels.get(label))
            .with_context(|| {
                format!(
                    "the pod has none of the revision labels {}",
                    self.revision.join(", ")
                )
            })?;
        RevisionKey::new(&sandbox.namespace, self.workload(sandbox)?, hash)
    }

    /// The workload as metrics name it, which is how the store names it.
    fn workload_of(&self, sandbox: &Sandbox) -> String {
        match self.workload(sandbox) {
            Ok(workload) => workload.to_string(),
            Err(_) => sandbox
                .labels
                .get(self.workload)
                .cloned()
                .unwrap_or_default(),
        }
    }
}

fn configured(label: &str) -> Option<&str> {
    (!label.is_empty()).then_some(label)
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
    info!(
        pod = %format_args!("{}/{}", sandbox.namespace, sandbox.name),
        address = %pod.address(),
        transport = ?pod.control().transport,
        gpus = pod.gpus().len(),
        visible_gpus = pod.visible_gpus(),
        pid = %pod.init(),
        "new pod"
    );
    pod.mount_tools().await?;
    let revision = match args.labels().revision_of(sandbox) {
        Ok(revision) => revision,
        Err(error) => return cold_boot_only(node, pod, shim, format!("{error:#}")).await,
    };
    info!(%revision, "keyed the pod's revision");
    let named = sandbox.annotations.get(&args.class_annotation);
    let class = match node.classes.get(named.map(String::as_str)) {
        Ok(class) => class,
        Err(error) => return cold_boot_only(node, pod, shim, format!("{error:#}")).await,
    };
    if let Some(member) = revision.workload().member() {
        let because = format!(
            "pod {member} of a multi-pod group; its group would have to be captured and restored together"
        );
        return cold_boot_only(node, pod, shim, because).await;
    }
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
    let control = args.control(sandbox, runtime.address(sandbox).await?);
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
    node.metrics.served(
        &sandbox.namespace,
        &workload_of(node, sandbox),
        served.how(),
    );
    Ok(())
}

fn workload_of(node: &Node, sandbox: &Sandbox) -> String {
    node.args.labels().workload_of(sandbox)
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
                    .failed(&sandbox.namespace, &workload_of(&node, &sandbox));
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
    use std::collections::HashMap;
    use std::io::{self, Write};
    use std::net::{IpAddr, SocketAddr};
    use std::sync::{Arc, Mutex};

    use anyhow::anyhow;
    use clap::Parser;
    use tracing_subscriber::fmt::MakeWriter;

    use crate::cri::Sandbox;
    use crate::node::{Args, Image, Labels, Plan};
    use crate::vllm::{Endpoint, Transport};

    fn sandbox(labels: &[(&str, &str)]) -> Sandbox {
        grouped(labels, None)
    }

    fn grouped(labels: &[(&str, &str)], size: Option<&str>) -> Sandbox {
        Sandbox {
            id: "id".to_string(),
            uid: "uid".to_string(),
            namespace: "serving".to_string(),
            name: "pod".to_string(),
            labels: labels
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect::<HashMap<_, _>>(),
            annotations: size
                .map(|size| HashMap::from([(GROUP_SIZE.to_string(), size.to_string())]))
                .unwrap_or_default(),
        }
    }

    const WORKLOAD: (&str, &str) = ("snapshot.llm-d.ai/workload", "qwen");
    const DS_REVISION: &str = "disaggregatedset.x-k8s.io/revision";
    const DS_ROLE: &str = "disaggregatedset.x-k8s.io/role";
    const MEMBER: &str = "leaderworkerset.sigs.k8s.io/worker-index";
    const LWS_REVISION: &str = "leaderworkerset.sigs.k8s.io/template-revision-hash";
    const GROUP_SIZE: &str = "leaderworkerset.sigs.k8s.io/size";

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: Args,
    }

    fn parse(argv: &[&str]) -> Args {
        Cli::try_parse_from(argv).unwrap().args
    }

    #[derive(Clone, Default)]
    struct Logged(Arc<Mutex<Vec<u8>>>);

    impl Logged {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

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

    fn defaults() -> Args {
        parse(&["node", "--store-dir", "/store", "--tools-dir", "/tools"])
    }

    #[test]
    fn the_default_labels_cover_deployments_and_disaggregated_sets() {
        let args = defaults();
        assert_eq!(
            args.revision_label,
            [DS_REVISION, LWS_REVISION, "pod-template-hash"]
        );
        assert_eq!(args.role_label, DS_ROLE);
        assert_eq!(args.member_label, MEMBER);
        assert_eq!(args.group_size_annotation, GROUP_SIZE);
        let listed = parse(&[
            "node",
            "--store-dir",
            "/store",
            "--tools-dir",
            "/tools",
            "--revision-label",
            "a,b",
            "--role-label",
            "",
        ]);
        assert_eq!(listed.revision_label, ["a", "b"]);
        assert_eq!(listed.role_label, "");
    }

    #[test]
    fn a_pod_annotation_picks_the_transport_and_its_port() {
        const CONTROL: &str = "snapshot.llm-d.ai/control";
        let address: IpAddr = "10.0.0.7".parse().unwrap();
        let pod = |named: Option<&str>| Sandbox {
            annotations: named
                .map(|named| HashMap::from([(CONTROL.to_string(), named.to_string())]))
                .unwrap_or_default(),
            ..sandbox(&[WORKLOAD])
        };
        let endpoint = |transport, port| Endpoint {
            transport,
            address: SocketAddr::new(address, port),
        };
        let args = defaults();
        assert_eq!(args.control_transport, Transport::Grpc);
        for (named, wanted) in [
            (None, endpoint(Transport::Grpc, 50051)),
            (Some("grpc"), endpoint(Transport::Grpc, 50051)),
            (Some("http"), endpoint(Transport::Http, 8000)),
        ] {
            assert_eq!(args.control(&pod(named), address), wanted);
        }
        let logged = Logged::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logged.clone())
            .with_ansi(false)
            .finish();
        let fallback = tracing::subscriber::with_default(subscriber, || {
            args.control(&pod(Some("rest")), address)
        });
        assert_eq!(
            fallback,
            endpoint(Transport::Grpc, 50051),
            "an unknown transport falls back to the node's"
        );
        let logged = logged.text();
        assert!(logged.contains("WARN"), "{logged}");
        assert!(
            logged.contains("using the node's control transport"),
            "{logged}"
        );
        assert!(
            logged.contains(r#"no vLLM control transport "rest""#),
            "{logged}"
        );
        assert!(logged.contains(CONTROL), "{logged}");
        assert!(logged.contains("fallback=Grpc"), "{logged}");

        let http = parse(&[
            "node",
            "--store-dir",
            "/store",
            "--tools-dir",
            "/tools",
            "--control-transport",
            "http",
            "--http-port",
            "9000",
            "--control-port",
            "6000",
        ]);
        for (named, wanted) in [
            (None, endpoint(Transport::Http, 9000)),
            (Some("http"), endpoint(Transport::Http, 9000)),
            (Some("grpc"), endpoint(Transport::Grpc, 6000)),
        ] {
            assert_eq!(http.control(&pod(named), address), wanted);
        }
    }

    #[test]
    fn roles_of_one_disaggregated_revision_get_separate_checkpoints() {
        let args = defaults();
        let labels = args.labels();
        let pod = |role: &str, slice: &str| {
            sandbox(&[
                WORKLOAD,
                (DS_REVISION, "6f9b7c"),
                (DS_ROLE, role),
                ("disaggregatedset.x-k8s.io/slice", slice),
                ("leaderworkerset.sigs.k8s.io/template-revision-hash", "aa11"),
            ])
        };
        let prefill = labels.revision_of(&pod("prefill", "0")).unwrap();
        let decode = labels.revision_of(&pod("decode", "0")).unwrap();
        assert_eq!(prefill.to_string(), "serving/qwen@prefill/6f9b7c");
        assert_eq!(decode.to_string(), "serving/qwen@decode/6f9b7c");
        assert_eq!(
            labels.revision_of(&pod("prefill", "3")).unwrap(),
            prefill,
            "slices are copies of one template and share its checkpoint"
        );
        assert_eq!(
            labels.workload_of(&pod("decode", "0")),
            decode.workload().to_string()
        );

        let next = sandbox(&[WORKLOAD, (DS_REVISION, "0c41e2"), (DS_ROLE, "prefill")]);
        assert_ne!(labels.revision_of(&next).unwrap(), prefill);
    }

    #[test]
    fn a_deployment_pod_keeps_its_key() {
        let args = defaults();
        let labels = args.labels();
        let pod = sandbox(&[WORKLOAD, ("pod-template-hash", "7b9f694bf8")]);
        assert_eq!(
            labels.revision_of(&pod).unwrap().to_string(),
            "serving/qwen/7b9f694bf8"
        );
        assert_eq!(labels.workload_of(&pod), "qwen");

        let both = sandbox(&[
            WORKLOAD,
            ("pod-template-hash", "7b9f694bf8"),
            (DS_REVISION, "6f9b7c"),
        ]);
        assert_eq!(
            labels.revision_of(&both).unwrap().to_string(),
            "serving/qwen/6f9b7c",
            "the first listed revision label wins"
        );
    }

    #[test]
    fn a_pod_without_a_revision_or_with_a_bad_role_is_refused() {
        let args = defaults();
        let labels = args.labels();
        let error = labels
            .revision_of(&sandbox(&[WORKLOAD, (DS_ROLE, "prefill")]))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "the pod has none of the revision labels disaggregatedset.x-k8s.io/revision, leaderworkerset.sigs.k8s.io/template-revision-hash, pod-template-hash"
        );
        assert!(
            labels
                .revision_of(&sandbox(&[(DS_REVISION, "6f9b7c")]))
                .is_err()
        );
        assert!(
            labels
                .revision_of(&sandbox(&[
                    WORKLOAD,
                    (DS_REVISION, "6f9b7c"),
                    (DS_ROLE, "")
                ]))
                .is_err()
        );
    }

    #[test]
    fn pods_of_a_multi_pod_group_are_told_apart_by_member() {
        let args = defaults();
        let labels = args.labels();
        let pod = |member: &str, group: &str, size: Option<&str>| {
            grouped(
                &[
                    WORKLOAD,
                    (DS_REVISION, "6f9b7c"),
                    (DS_ROLE, "decode"),
                    (MEMBER, member),
                    ("leaderworkerset.sigs.k8s.io/group-index", group),
                ],
                size,
            )
        };
        let leader = labels.revision_of(&pod("0", "0", Some("2"))).unwrap();
        let worker = labels.revision_of(&pod("1", "0", Some("2"))).unwrap();
        assert_eq!(leader.to_string(), "serving/qwen@decode+0/6f9b7c");
        assert_eq!(worker.to_string(), "serving/qwen@decode+1/6f9b7c");
        assert_eq!(worker.workload().member(), Some(1));
        assert_eq!(
            labels.revision_of(&pod("1", "5", Some("2"))).unwrap(),
            worker,
            "groups are copies of one template and share their members' checkpoints"
        );
        for single in [Some("1"), None] {
            let whole = labels.revision_of(&pod("0", "0", single)).unwrap();
            assert_eq!(whole.to_string(), "serving/qwen@decode/6f9b7c");
            assert_eq!(whole.workload().member(), None);
        }
        for (member, size, wanted) in [
            (
                "x",
                "2",
                "leaderworkerset.sigs.k8s.io/worker-index is \"x\", not a pod index",
            ),
            (
                "1",
                "two",
                "leaderworkerset.sigs.k8s.io/size is \"two\", not a pod count",
            ),
        ] {
            let error = labels
                .revision_of(&pod(member, "0", Some(size)))
                .unwrap_err();
            assert_eq!(error.to_string(), wanted);
        }
        let lws = grouped(
            &[WORKLOAD, (LWS_REVISION, "5b6d"), (MEMBER, "3")],
            Some("4"),
        );
        assert_eq!(
            labels.revision_of(&lws).unwrap().to_string(),
            "serving/qwen+3/5b6d",
            "a LeaderWorkerSet outside a DisaggregatedSet is keyed by its own revision"
        );
        let unlabelled = grouped(&[WORKLOAD, (DS_REVISION, "6f9b7c")], Some("4"));
        assert_eq!(
            labels.revision_of(&unlabelled).unwrap_err().to_string(),
            "a pod of a group of 4 has no leaderworkerset.sigs.k8s.io/worker-index label"
        );
    }

    #[test]
    fn roles_are_ignored_when_no_role_label_is_set() {
        let labels = Labels {
            workload: WORKLOAD.0,
            role: "",
            member: "",
            group_size: "",
            revision: &[DS_REVISION.to_string()],
        };
        let pod = sandbox(&[WORKLOAD, (DS_REVISION, "6f9b7c"), (DS_ROLE, "prefill")]);
        assert_eq!(
            labels.revision_of(&pod).unwrap().to_string(),
            "serving/qwen/6f9b7c"
        );
        assert_eq!(labels.workload_of(&pod), "qwen");
    }

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
