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
//! The revision is named after the controller that owns the pod's template,
//! read from the API server, and a pod restores it only if its
//! ServiceAccount may `get` it (see [`crate::access`]) and the image was
//! captured by a pod of the same controller.
mod criu;
mod identity;
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

use crate::access::{self, AccessError, Read, Subject, Verdict};
use crate::cri::{Runtime, Sandbox};
use crate::cuda::Gpu;
use crate::metrics::{self, How, Metrics, Step};
use crate::node::identity::{IdentityError, PodIdentity};
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
use crate::store::{
    Attestation, Class, Classes, Owner, Published, RevisionKey, RevisionStore, Workload,
};
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
    /// `None` when the daemon could not configure a client, and every pod
    /// cold boots.
    api: Option<kube::Client>,
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
    /// Captured by a pod of another principal, named here.
    Foreign(String),
}

impl Image {
    /// `ours` is what this pod would attest to if it captured the image.
    fn of(store: &RevisionStore, gpus: &[Gpu], ours: &Attestation) -> Result<Self> {
        if !store.image_ready() {
            return Ok(Self::Absent);
        }
        let image = store.image_dir();
        let manifest = Manifest::load(&image)?;
        match &manifest.attestation {
            Some(by) if !by.same_principal(ours) => {
                warn!(
                    security = "foreign-checkpoint",
                    revision = %ours.revision,
                    pod = %format_args!("{}/{}", ours.namespace, ours.pod),
                    captured_by = %by,
                    pod_is = %ours,
                    "refused the pod a checkpoint of its revision that another principal captured"
                );
                return Ok(Self::Foreign(by.to_string()));
            }
            Some(_) => {}
            None => info!("the checkpoint predates attestation; it is restored as owned by no one"),
        }
        Ok(match manifest.refusal(&image, gpus)? {
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
            (Image::Foreign(by), _) => Self::ColdBoot(format!(
                "the node's checkpoint of the revision was captured by {by}, \
                 which is not the pod's controller"
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
    /// Named after `owner`, or after the opt-in label for a pod that no
    /// controller owns.
    fn workload(&self, sandbox: &Sandbox, owner: Option<&Owner>) -> Result<Workload> {
        let name = match owner {
            Some(owner) => &owner.name,
            None => sandbox
                .labels
                .get(self.workload)
                .context("the opt-in label is gone")?,
        };
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

    fn revision_of(&self, sandbox: &Sandbox, owner: Option<&Owner>) -> Result<RevisionKey> {
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
        RevisionKey::new(&sandbox.namespace, self.workload(sandbox, owner)?, hash)
    }

    /// The workload as metrics name it, which is how the store names it.
    fn workload_of(&self, sandbox: &Sandbox, owner: Option<&Owner>) -> String {
        match self.workload(sandbox, owner) {
            Ok(workload) => workload.to_string(),
            Err(_) => owner
                .map(|owner| owner.name.clone())
                .or_else(|| sandbox.labels.get(self.workload).cloned())
                .unwrap_or_default(),
        }
    }
}

fn configured(label: &str) -> Option<&str> {
    (!label.is_empty()).then_some(label)
}

async fn identify(node: &Node, sandbox: &Sandbox) -> Result<PodIdentity, IdentityError> {
    let api = node.api.as_ref().ok_or(IdentityError::NoClient)?;
    PodIdentity::of(api, sandbox).await
}

async fn may_read(
    node: &Node,
    identity: &PodIdentity,
    revision: &RevisionKey,
) -> Result<Verdict, AccessError> {
    let api = node.api.as_ref().ok_or(AccessError::NoClient)?;
    let subject = Subject::service_account(&identity.namespace, &identity.service_account);
    access::may_read(api, &subject, revision, Read::Restore).await
}

/// Why the pod may not read `revision`, if it may not.
fn refusal(
    asked: Result<Verdict, AccessError>,
    identity: &PodIdentity,
    revision: &RevisionKey,
) -> Option<String> {
    match asked {
        Ok(Verdict::Allowed) => None,
        Ok(Verdict::Denied(why)) => {
            warn!(
                security = "checkpoint-access-denied",
                %revision,
                pod = %format_args!("{}/{}", identity.namespace, identity.name),
                service_account = %identity.service_account,
                why,
                "the pod may not read its revision's checkpoint"
            );
            Some(format!(
                "ServiceAccount {} may not get checkpoints.{} {} in {}: {why}",
                identity.service_account,
                access::GROUP,
                revision.workload().name(),
                revision.namespace()
            ))
        }
        Err(error) => Some(format!(
            "could not ask whether the pod may read its checkpoint: {:#}",
            anyhow::Error::from(error)
        )),
    }
}

#[cfg(feature = "nixl")]
async fn pull_from_peers(
    peers: &Arc<Peers>,
    revision: &RevisionKey,
    ours: &Attestation,
    store: &RevisionStore,
) -> Result<()> {
    info!("this node has no checkpoint of the revision; asking the other nodes");
    let asked = Instant::now();
    match peers.pull(revision, &ours.principal(), store).await? {
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
    identity: Result<PodIdentity, IdentityError>,
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
    let identity = match identity {
        Ok(identity) => identity,
        Err(error) => {
            let because = format!(
                "could not tell which controller the pod belongs to: {:#}",
                anyhow::Error::from(error)
            );
            return cold_boot_only(node, pod, shim, because).await;
        }
    };
    let revision = match args.labels().revision_of(sandbox, identity.owner.as_ref()) {
        Ok(revision) => revision,
        Err(error) => return cold_boot_only(node, pod, shim, format!("{error:#}")).await,
    };
    info!(
        %revision,
        owner = %identity
            .owner
            .as_ref()
            .map_or_else(|| "none".to_string(), ToString::to_string),
        service_account = %identity.service_account,
        "keyed the pod's revision"
    );
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
    let asked = may_read(node, &identity, &revision).await;
    if let Some(because) = refusal(asked, &identity, &revision) {
        return cold_boot_only(node, pod, shim, because).await;
    }
    let ours = identity.attestation(&revision);
    let store = RevisionStore::open(&class.dir, &revision, &sandbox.uid)?;
    #[cfg(feature = "nixl")]
    if let Some(peers) = &node.peers {
        let turn = node.pull_turn(&revision)?;
        let _turn = turn.lock().await;
        if !store.image_ready() && !store.failed_before() {
            let asked = Instant::now();
            pull_from_peers(peers, &revision, &ours, &store).await?;
            if store.image_ready() {
                node.metrics.stepped(Step::Pull, asked.elapsed());
            }
        }
    }
    let plan = Plan::choose(
        store.failed_before(),
        Image::of(&store, pod.gpus(), &ours)?,
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
                pod.checkpoint(control, shim, server, staging, &ours)
                    .await?;
                node.metrics.stepped(Step::Checkpoint, asked.elapsed());
                let asked = Instant::now();
                let restored = pod.restore(staging).await?;
                node.metrics.stepped(Step::Restore, asked.elapsed());
                Ok(restored)
            }
            .await;
            adopt_or_give_up(cycled, pod, &store, shim).await?;
            match store.publish()? {
                Published::Ours => info!(
                    image = %store.image_dir().display(),
                    "checkpoint published on this node"
                ),
                Published::Kept => info!(
                    image = %store.image_dir().display(),
                    "another pod of the workload published first; its checkpoint stays"
                ),
                Published::Collided(_) => {}
            }
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
    let (served, owner) = if progress.started {
        info!("the server was started before this daemon; waiting for it to run");
        let boot = Duration::from_secs(args.boot_timeout_s);
        let (control, state) = Control::connect_when_up(pod.control(), boot).await?;
        if state == Lifecycle::Prepared {
            pod.resume(control).await?;
        }
        (Served::StartedEarlier, None)
    } else {
        let identity = identify(node, sandbox).await;
        let owner = identity
            .as_ref()
            .ok()
            .and_then(|identity| identity.owner.clone());
        let served = bring_up(node, sandbox, &pod, &mut shim, identity).await?;
        (served, owner)
    };
    shim.serving().await?;
    info!(seconds = %seconds(seen), how = %served, "serving");
    node.metrics.served(
        &sandbox.namespace,
        &workload_of(node, sandbox, owner.as_ref()),
        served.how(),
    );
    Ok(())
}

fn workload_of(node: &Node, sandbox: &Sandbox, owner: Option<&Owner>) -> String {
    node.args.labels().workload_of(sandbox, owner)
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
                    .failed(&sandbox.namespace, &workload_of(&node, &sandbox, None));
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
    let api = match kube::Client::try_default().await {
        Ok(api) => Some(api),
        Err(error) => {
            warn!(
                %error,
                "no Kubernetes API client; every pod cold boots without a checkpoint"
            );
            None
        }
    };
    let node = Arc::new(Node {
        #[cfg(feature = "nixl")]
        peers: Peers::start(&args.peers, classes.clone(), api.clone()).await?,
        api,
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
    use std::fs;
    use std::io::{self, Write};
    use std::net::{IpAddr, SocketAddr};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use anyhow::anyhow;
    use clap::Parser;
    use tracing_subscriber::fmt::MakeWriter;

    use crate::access::{AccessError, Verdict};
    use crate::cri::Sandbox;
    use crate::node::identity::PodIdentity;
    use crate::node::manifest::Manifest;
    use crate::node::process::PodPid;
    use crate::node::{Args, Image, Labels, Plan, refusal};
    use crate::store::{Attestation, Owner, RevisionKey, RevisionStore};
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
        let prefill = labels.revision_of(&pod("prefill", "0"), None).unwrap();
        let decode = labels.revision_of(&pod("decode", "0"), None).unwrap();
        assert_eq!(prefill.to_string(), "serving/qwen@prefill/6f9b7c");
        assert_eq!(decode.to_string(), "serving/qwen@decode/6f9b7c");
        assert_eq!(
            labels.revision_of(&pod("prefill", "3"), None).unwrap(),
            prefill,
            "slices are copies of one template and share its checkpoint"
        );
        assert_eq!(
            labels.workload_of(&pod("decode", "0"), None),
            decode.workload().to_string()
        );

        let next = sandbox(&[WORKLOAD, (DS_REVISION, "0c41e2"), (DS_ROLE, "prefill")]);
        assert_ne!(labels.revision_of(&next, None).unwrap(), prefill);
    }

    #[test]
    fn a_deployment_pod_keeps_its_key() {
        let args = defaults();
        let labels = args.labels();
        let pod = sandbox(&[WORKLOAD, ("pod-template-hash", "7b9f694bf8")]);
        assert_eq!(
            labels.revision_of(&pod, None).unwrap().to_string(),
            "serving/qwen/7b9f694bf8"
        );
        assert_eq!(labels.workload_of(&pod, None), "qwen");

        let both = sandbox(&[
            WORKLOAD,
            ("pod-template-hash", "7b9f694bf8"),
            (DS_REVISION, "6f9b7c"),
        ]);
        assert_eq!(
            labels.revision_of(&both, None).unwrap().to_string(),
            "serving/qwen/6f9b7c",
            "the first listed revision label wins"
        );
    }

    #[test]
    fn a_pod_without_a_revision_or_with_a_bad_role_is_refused() {
        let args = defaults();
        let labels = args.labels();
        let error = labels
            .revision_of(&sandbox(&[WORKLOAD, (DS_ROLE, "prefill")]), None)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "the pod has none of the revision labels disaggregatedset.x-k8s.io/revision, leaderworkerset.sigs.k8s.io/template-revision-hash, pod-template-hash"
        );
        assert!(
            labels
                .revision_of(&sandbox(&[(DS_REVISION, "6f9b7c")]), None)
                .is_err()
        );
        assert!(
            labels
                .revision_of(
                    &sandbox(&[WORKLOAD, (DS_REVISION, "6f9b7c"), (DS_ROLE, "")]),
                    None
                )
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
        let leader = labels.revision_of(&pod("0", "0", Some("2")), None).unwrap();
        let worker = labels.revision_of(&pod("1", "0", Some("2")), None).unwrap();
        assert_eq!(leader.to_string(), "serving/qwen@decode+0/6f9b7c");
        assert_eq!(worker.to_string(), "serving/qwen@decode+1/6f9b7c");
        assert_eq!(worker.workload().member(), Some(1));
        assert_eq!(
            labels.revision_of(&pod("1", "5", Some("2")), None).unwrap(),
            worker,
            "groups are copies of one template and share their members' checkpoints"
        );
        for single in [Some("1"), None] {
            let whole = labels.revision_of(&pod("0", "0", single), None).unwrap();
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
                .revision_of(&pod(member, "0", Some(size)), None)
                .unwrap_err();
            assert_eq!(error.to_string(), wanted);
        }
        let lws = grouped(
            &[WORKLOAD, (LWS_REVISION, "5b6d"), (MEMBER, "3")],
            Some("4"),
        );
        assert_eq!(
            labels.revision_of(&lws, None).unwrap().to_string(),
            "serving/qwen+3/5b6d",
            "a LeaderWorkerSet outside a DisaggregatedSet is keyed by its own revision"
        );
        let unlabelled = grouped(&[WORKLOAD, (DS_REVISION, "6f9b7c")], Some("4"));
        assert_eq!(
            labels
                .revision_of(&unlabelled, None)
                .unwrap_err()
                .to_string(),
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
            labels.revision_of(&pod, None).unwrap().to_string(),
            "serving/qwen/6f9b7c"
        );
        assert_eq!(labels.workload_of(&pod, None), "qwen");
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

    const HASH: (&str, &str) = ("pod-template-hash", "7b9f694bf8");

    fn owner(kind: &str, name: &str, uid: &str) -> Owner {
        Owner {
            api_version: "apps/v1".to_string(),
            kind: kind.to_string(),
            name: name.to_string(),
            uid: uid.to_string(),
        }
    }

    fn identity(pod: &str, owner: Option<Owner>, service_account: &str) -> PodIdentity {
        PodIdentity {
            namespace: "serving".to_string(),
            name: pod.to_string(),
            uid: format!("{pod}-uid"),
            service_account: service_account.to_string(),
            owner,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-node-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// A store for `revision` whose image was captured by `by`.
    fn captured(dir: &Path, revision: &RevisionKey, by: Option<Attestation>) -> RevisionStore {
        let store = RevisionStore::open(dir, revision, "restoring-pod").unwrap();
        fs::create_dir_all(store.image_dir()).unwrap();
        Manifest {
            root: PodPid(1),
            gpus: Vec::new(),
            address: "10.0.0.7".parse().unwrap(),
            attestation: by,
        }
        .save(&store.image_dir())
        .unwrap();
        store
    }

    fn with_log<T>(run: impl FnOnce() -> T) -> (T, String) {
        let logged = Logged::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logged.clone())
            .with_ansi(false)
            .finish();
        let value = tracing::subscriber::with_default(subscriber, run);
        (value, logged.text())
    }

    #[test]
    fn a_controller_names_the_workload_whatever_the_label_says() {
        let args = defaults();
        let labels = args.labels();
        let qwen = owner("Deployment", "qwen", "d-1");
        let labelled = sandbox(&[WORKLOAD, HASH]);
        assert_eq!(
            labels.revision_of(&labelled, Some(&qwen)).unwrap(),
            labels.revision_of(&labelled, None).unwrap(),
            "a Deployment named as its label keeps its key"
        );
        let evil = owner("Deployment", "evil", "d-2");
        assert_eq!(
            labels
                .revision_of(&labelled, Some(&evil))
                .unwrap()
                .to_string(),
            "serving/evil/7b9f694bf8",
            "a pod of another Deployment cannot claim qwen's key by its label"
        );
        assert_eq!(labels.workload_of(&labelled, Some(&evil)), "evil");
        let kserve = owner("Deployment", "qwen3-8b-kserve", "d-3");
        assert_eq!(
            labels
                .revision_of(
                    &sandbox(&[("snapshot.llm-d.ai/workload", "qwen3-8b"), HASH]),
                    Some(&kserve)
                )
                .unwrap()
                .to_string(),
            "serving/qwen3-8b-kserve/7b9f694bf8"
        );

        let set = Owner {
            api_version: "disaggregatedset.x-k8s.io/v1".to_string(),
            ..owner("DisaggregatedSet", "qwen", "s-1")
        };
        let prefill = sandbox(&[WORKLOAD, (DS_REVISION, "6f9b7c"), (DS_ROLE, "prefill")]);
        let keyed = labels.revision_of(&prefill, Some(&set)).unwrap();
        assert_eq!(keyed, labels.revision_of(&prefill, None).unwrap());
        assert_eq!(keyed.to_string(), "serving/qwen@prefill/6f9b7c");
        assert_eq!(labels.workload_of(&prefill, Some(&set)), "qwen@prefill");

        let unlabelled = sandbox(&[HASH]);
        assert_eq!(
            labels
                .revision_of(&unlabelled, Some(&qwen))
                .unwrap()
                .to_string(),
            "serving/qwen/7b9f694bf8",
            "an owned pod needs the label only to opt in"
        );
        assert!(labels.revision_of(&unlabelled, None).is_err());
        assert_eq!(labels.workload_of(&unlabelled, None), "");
    }

    #[test]
    fn a_pod_labelled_as_another_workloads_revision_cold_boots_instead_of_restoring() {
        let dir = scratch("foreign");
        let args = defaults();
        let labels = args.labels();
        let qwen = owner("Deployment", "qwen", "d-1");
        let qwen_pod = sandbox(&[WORKLOAD, HASH]);
        let revision = labels.revision_of(&qwen_pod, Some(&qwen)).unwrap();
        let store = captured(
            &dir,
            &revision,
            Some(
                identity("qwen-7b9f694bf8-aaaaa", Some(qwen.clone()), "qwen")
                    .attestation(&revision),
            ),
        );

        let stranger = sandbox(&[WORKLOAD, HASH]);
        let stranger_key = labels.revision_of(&stranger, None).unwrap();
        assert_eq!(
            stranger_key, revision,
            "an unowned pod still keys by its label"
        );
        let ours = identity("stranger", None, "qwen").attestation(&stranger_key);
        let (image, log) = with_log(|| Image::of(&store, &[], &ours).unwrap());
        let Image::Foreign(by) = &image else {
            panic!("{image:?}");
        };
        assert!(by.contains("Deployment qwen (d-1)"), "{by}");
        for wanted in [
            "WARN",
            "security=\"foreign-checkpoint\"",
            "revision=serving/qwen/7b9f694bf8",
            "pod=serving/stranger",
            "captured_by=Deployment qwen (d-1)",
            "owned by no controller, as ServiceAccount qwen",
            "refused the pod a checkpoint of its revision that another principal captured",
        ] {
            assert!(log.contains(wanted), "{wanted:?} not in {log}");
        }
        let Plan::ColdBoot(because) = Plan::choose(false, image, Ok(())) else {
            panic!("the stranger was not cold booted");
        };
        assert!(
            because.contains("captured by Deployment qwen (d-1)"),
            "{because}"
        );

        let lookalike = identity(
            "lookalike",
            Some(owner("Deployment", "qwen", "d-9")),
            "qwen",
        );
        assert!(
            matches!(
                Image::of(&store, &[], &lookalike.attestation(&revision)).unwrap(),
                Image::Foreign(_)
            ),
            "a recreated Deployment of the same name is another principal"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_replica_of_the_capturing_controller_still_restores() {
        let dir = scratch("sibling");
        let args = defaults();
        let labels = args.labels();
        let set = Owner {
            api_version: "disaggregatedset.x-k8s.io/v1".to_string(),
            ..owner("DisaggregatedSet", "qwen", "s-1")
        };
        let cases = [
            (
                owner("Deployment", "qwen", "d-1"),
                sandbox(&[WORKLOAD, HASH]),
            ),
            (
                set,
                sandbox(&[WORKLOAD, (DS_REVISION, "6f9b7c"), (DS_ROLE, "decode")]),
            ),
        ];
        for (controller, pod) in cases {
            let revision = labels.revision_of(&pod, Some(&controller)).unwrap();
            assert_eq!(
                revision,
                labels.revision_of(&pod, None).unwrap(),
                "{controller}"
            );
            let first = identity("first", Some(controller.clone()), "qwen");
            let store = captured(&dir, &revision, Some(first.attestation(&revision)));
            let later = identity("later", Some(controller.clone()), "other");
            let image = Image::of(&store, &[], &later.attestation(&revision)).unwrap();
            assert_eq!(image, Image::Usable, "{controller}");
            assert_eq!(
                Plan::choose(false, image, Ok(())),
                Plan::Restore,
                "{controller}"
            );
            fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn an_image_captured_before_attestation_is_restored_as_owned_by_no_one() {
        let dir = scratch("legacy");
        let revision: RevisionKey = "serving/qwen/7b9f694bf8".parse().unwrap();
        let store = captured(&dir, &revision, None);
        let ours = identity("later", Some(owner("Deployment", "qwen", "d-1")), "qwen");
        let (image, log) =
            with_log(|| Image::of(&store, &[], &ours.attestation(&revision)).unwrap());
        assert_eq!(image, Image::Usable);
        assert!(log.contains("the checkpoint predates attestation"), "{log}");
        assert_eq!(Plan::choose(false, image, Ok(())), Plan::Restore);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_pod_whose_service_account_may_not_read_its_revision_cold_boots() {
        let revision: RevisionKey = "serving/qwen@prefill/6f9b7c".parse().unwrap();
        let pod = identity("stranger", None, "other-sa");
        assert_eq!(refusal(Ok(Verdict::Allowed), &pod, &revision), None);

        let denied = Verdict::Denied("no RoleBinding grants it".to_string());
        let (because, log) = with_log(|| refusal(Ok(denied), &pod, &revision));
        assert_eq!(
            because.as_deref(),
            Some(
                "ServiceAccount other-sa may not get checkpoints.snapshot.llm-d.ai qwen in serving: \
                 no RoleBinding grants it"
            )
        );
        for wanted in [
            "WARN",
            "security=\"checkpoint-access-denied\"",
            "revision=serving/qwen@prefill/6f9b7c",
            "pod=serving/stranger",
            "service_account=other-sa",
            "why=\"no RoleBinding grants it\"",
            "the pod may not read its revision's checkpoint",
        ] {
            assert!(log.contains(wanted), "{wanted:?} not in {log}");
        }

        let unreachable = refusal(Err(AccessError::NoClient), &pod, &revision);
        assert_eq!(
            unreachable.as_deref(),
            Some(
                "could not ask whether the pod may read its checkpoint: \
                 this daemon has no Kubernetes API client"
            )
        );
    }

    #[test]
    fn a_foreign_image_cold_boots_without_a_capture() {
        let foreign = || Image::Foreign("Deployment qwen (d-1)".to_string());
        for (failed_before, room) in [
            (false, Ok(())),
            (false, Err(anyhow!("full"))),
            (true, Ok(())),
        ] {
            assert!(matches!(
                Plan::choose(failed_before, foreign(), room),
                Plan::ColdBoot(_)
            ));
        }
    }
}
