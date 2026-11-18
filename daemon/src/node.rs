//! The node daemon. For every opted-in pod that starts on this node it
//! either restores the pod's revision from the checkpoint this node holds, or
//! lets the server cold boot, checkpoints it once and restores it from that.
//!
//! ```text
//! CRI ──► new pod ──► shim.sock ──► checkpoint here?
//!                                     yes: restore into the pod ──────────────► Serving
//!                                     no:  Start, wait, checkpoint, restore ──► Serving
//! ```
//!
//! The pod stays unprivileged. The daemon reaches vLLM on its gRPC control
//! port, the driver's checkpoint API by host PID, and runs CRIU inside the
//! pod's namespaces with its own tools and the image directory mounted in.
use std::collections::HashSet;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufStream};
use tokio::net::UnixStream;
use tokio::process::Command;
use tokio::time::{Instant, sleep};
use tracing::{Instrument, debug, error, info, info_span, warn};

use crate::cri::{Runtime, Sandbox};
use crate::cuda::{self, Driver, GpuMap, GpuUuid};
use crate::cuinterpose;
#[cfg(feature = "nixl")]
use crate::handshake::Revision;
#[cfg(feature = "nixl")]
use crate::peer::{self, Peers};
use crate::podresources;
use crate::shim::{Reply, Request};
use crate::state::{State, directory_bytes, ensure_room};
use crate::vllm::Control;

const REVISION_LABEL: &str = "pod-template-hash";
const TOOLS: &str = "/run/snapshotd/tools";
const IMAGE: &str = "/run/snapshotd/image";
const WORK: &str = "/run/snapshotd/work";
const SERVER_LOG: &str = "/run/snapshotd/server.log";
const SHIM_SOCKET: &str = "/run/snapshotd/shim.sock";
const FAR_PID: &str = "4000000";

const CONTAINER_WAIT: Duration = Duration::from_secs(600);
const SHIM_WAIT: Duration = Duration::from_secs(60);
const BOOT: Duration = Duration::from_secs(3600);
const LIFECYCLE: Duration = Duration::from_secs(900);
const CUDA_LOCK: Duration = Duration::from_secs(60);
const RETRY: Duration = Duration::from_secs(10);
const REAP_WAIT: Duration = Duration::from_secs(30);
const SERVER_TAIL: usize = 12;

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

/// What a restore needs to know about where the checkpoint came from.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The server's PID inside its pod.
    pub root_pid: i32,
    pub gpus: Vec<String>,
    /// The device minor of each GPU.
    #[serde(default)]
    pub minors: Vec<u32>,
    pub address: IpAddr,
}

impl Manifest {
    fn path(image: &Path) -> PathBuf {
        image.join("manifest.json")
    }

    pub fn load(image: &Path) -> Result<Self> {
        let path = Self::path(image);
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn save(&self, image: &Path) -> Result<()> {
        let path = Self::path(image);
        fs::write(&path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("write {}", path.display()))
    }
}

/// The host PID of the child of `parent` that is `pid` inside its PID
/// namespace.
pub fn host_pid(parent: i32, pid: i32) -> Result<i32> {
    let tasks = fs::read_dir(format!("/proc/{parent}/task"))
        .with_context(|| format!("process {parent} is gone"))?;
    for task in tasks.flatten() {
        let Ok(children) = fs::read_to_string(task.path().join("children")) else {
            continue;
        };
        for child in children.split_ascii_whitespace() {
            let Ok(status) = fs::read_to_string(format!("/proc/{child}/status")) else {
                continue;
            };
            let inner = status
                .lines()
                .find_map(|line| line.strip_prefix("NSpid:"))
                .and_then(|pids| pids.split_ascii_whitespace().last());
            if inner == Some(pid.to_string().as_str()) {
                return child.parse().context("child PID");
            }
        }
    }
    bail!("process {parent} has no child that is {pid} in its PID namespace")
}

/// Why the checkpoint in `image` cannot be restored onto GPUs with these
/// device minors, if it cannot. Processes that share GPU memory can import it
/// again only on devices they had initialized when they were checkpointed, so
/// such a checkpoint needs GPUs with the same minors.
pub fn unrestorable(image: &Path, minors: &[u32]) -> Result<Option<String>> {
    let manifest = Manifest::load(image)?;
    if manifest.minors.len() != minors.len() {
        return Ok(Some(format!(
            "it ran on {} GPUs and the pod has {}",
            manifest.minors.len(),
            minors.len()
        )));
    }
    if manifest.minors != minors && cuinterpose::Manifest::load(&image.join("img"))?.is_some() {
        return Ok(Some(format!(
            "its processes share GPU memory and it was taken on GPUs {:?}; the pod has {:?}",
            manifest.minors, minors
        )));
    }
    Ok(None)
}

fn in_pod(pid: i32, path: &str) -> PathBuf {
    PathBuf::from(format!("/proc/{pid}/root{path}"))
}

async fn run(command: &mut Command, what: &str) -> Result<()> {
    let output = command
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("start {what}"))?;
    ensure!(
        output.status.success(),
        "{what} failed ({}): {}{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// The lines of a CRIU log that say why it failed.
pub fn criu_errors(log: &str) -> String {
    let errors: Vec<&str> = log.lines().filter(|line| line.contains("Error")).collect();
    let shown = if errors.is_empty() {
        log.lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    } else {
        errors
    };
    shown.join("\n")
}

/// Run CRIU; if it fails, log what its own log says about why.
async fn run_criu(command: &mut Command, what: &str, log: &Path) -> Result<()> {
    let outcome = run(command, what).await;
    if outcome.is_err() {
        for line in criu_errors(&fs::read_to_string(log).unwrap_or_default()).lines() {
            warn!("criu: {}", line.trim());
        }
    }
    outcome
}

/// `line` without terminal colour sequences.
pub fn plain(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut rest = line.chars();
    while let Some(c) = rest.next() {
        if c == '\u{1b}' {
            for c in rest.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}

/// Wait for a process CRIU has dumped and killed to be reaped. Until then
/// its PID is taken and a restore into the same pod cannot have it back.
async fn gone(pid: i32) -> Result<()> {
    let deadline = Instant::now() + REAP_WAIT;
    while Path::new(&format!("/proc/{pid}")).exists() {
        ensure!(
            Instant::now() < deadline,
            "process {pid} is still there {REAP_WAIT:?} after its dump"
        );
        sleep(Duration::from_millis(20)).await;
    }
    Ok(())
}

/// What a pod's log lines are tagged with: the part of its name that tells it
/// from the other pods of its workload.
pub fn short_id(pod: &str) -> &str {
    pod.rsplit('-').next().unwrap_or(pod)
}

fn seconds(since: Instant) -> String {
    format!("{:.2}", since.elapsed().as_secs_f64())
}

fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1u64 << 30) as f64)
}

/// Run one step of a checkpoint or restore and log how long it took.
async fn step<T>(name: &'static str, work: impl Future<Output = Result<T>>) -> Result<T> {
    let started = Instant::now();
    let outcome = work.await;
    match &outcome {
        Ok(_) => info!(seconds = %seconds(started), "{name}"),
        Err(_) => warn!(seconds = %seconds(started), "{name} failed"),
    }
    outcome
}

/// Run blocking work off the runtime, inside the caller's span.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || span.in_scope(work)).await?
}

async fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to).with_context(|| format!("create {}", to.display()))?;
    let mut source = from.as_os_str().to_owned();
    source.push("/.");
    run(
        Command::new("cp").arg("-a").arg(source).arg(to),
        &format!("copy {} to {}", from.display(), to.display()),
    )
    .await
}

struct Shim {
    stream: BufStream<UnixStream>,
}

impl Shim {
    async fn connect(pid: i32) -> Result<Self> {
        let socket = in_pod(pid, SHIM_SOCKET);
        let deadline = Instant::now() + SHIM_WAIT;
        loop {
            match UnixStream::connect(&socket).await {
                Ok(stream) => {
                    return Ok(Self {
                        stream: BufStream::new(stream),
                    });
                }
                Err(error) if Instant::now() >= deadline => {
                    return Err(error).with_context(|| format!("connect {}", socket.display()));
                }
                Err(_) => sleep(Duration::from_millis(200)).await,
            }
        }
    }

    async fn call(&mut self, request: Request) -> Result<Reply> {
        let mut line = serde_json::to_vec(&request)?;
        line.push(b'\n');
        self.stream.write_all(&line).await?;
        self.stream.flush().await?;
        let mut answer = String::new();
        ensure!(
            self.stream.read_line(&mut answer).await? > 0,
            "the shim closed the connection"
        );
        match serde_json::from_str(&answer).context("parse the shim's reply")? {
            Reply::Refused { reason } => bail!("the shim refused {request:?}: {reason}"),
            reply => Ok(reply),
        }
    }

    async fn start(&mut self) -> Result<i32> {
        match self.call(Request::Start).await? {
            Reply::Started { pid } => Ok(pid),
            other => bail!("unexpected {other:?}"),
        }
    }

    async fn state(&mut self) -> Result<(bool, bool)> {
        match self.call(Request::State).await? {
            Reply::State { started, serving } => Ok((started, serving)),
            other => bail!("unexpected {other:?}"),
        }
    }
}

const CUINTERPOSE: &str = "cuinterpose";

struct Pod<'a> {
    args: &'a Args,
    shim_pid: i32,
    address: IpAddr,
    gpus: Vec<GpuUuid>,
    minors: Vec<u32>,
    /// Every GPU the container can open. More than its own when it is
    /// privileged; the driver wants a restore map over all of them.
    visible: Vec<GpuUuid>,
}

impl Pod<'_> {
    fn control(&self) -> SocketAddr {
        SocketAddr::new(self.address, self.args.control_port)
    }

    /// CRIU, run in every namespace of the pod and in its cgroup, so what it
    /// restores is accounted and confined like the pod's own processes. A
    /// process can only be moved into the pod's cgroup from a cgroup
    /// namespace that sees both, which is the node's.
    ///
    /// ```text
    /// nsenter (node's cgroup namespace) ─► unshare (mounts) ─► mount cgroup2,
    ///   join the pod's cgroup ─► nsenter (all of the pod's namespaces) ─► criu
    /// ```
    fn criu(&self, action: &str) -> Command {
        let pid = self.shim_pid.to_string();
        let enter = format!(
            "mount -t cgroup2 none /mnt \
             && echo $$ > \"/mnt$(sed -n 's/^0:://p' /proc/{pid}/cgroup)/cgroup.procs\" \
             && exec nsenter -t {pid} -a -- \"$@\""
        );
        let mut command = Command::new("nsenter");
        command
            .args([
                "-t", "1", "-C", "--", "unshare", "-m", "--", "sh", "-c", &enter, "sh",
            ])
            .arg(format!("{TOOLS}/criu"))
            .arg(action)
            .env("LD_LIBRARY_PATH", format!("{TOOLS}/lib"))
            .args(["--libdir", &format!("{TOOLS}/criu-plugins")])
            .args([
                "--manage-cgroups=ignore",
                "--network-lock",
                "skip",
                "--tcp-established",
                "--ext-unix-sk",
            ])
            .args(["--file-locks", "--link-remap", "-v4"]);
        command
    }

    async fn mount(&self, source: &Path, target: &str) -> Result<()> {
        let exe = std::env::current_exe().context("this program's path")?;
        run(
            Command::new(exe)
                .arg("inject")
                .args(["--pid", &self.shim_pid.to_string()])
                .arg("--source")
                .arg(source)
                .args(["--target", target]),
            &format!("mount {} in the pod", source.display()),
        )
        .await
    }

    /// Log the end of the server's own output, after a failure.
    fn log_server_tail(&self) {
        let log = fs::read_to_string(in_pod(self.shim_pid, SERVER_LOG)).unwrap_or_default();
        let lines: Vec<&str> = log.lines().collect();
        for line in &lines[lines.len().saturating_sub(SERVER_TAIL)..] {
            warn!("server: {}", plain(line));
        }
    }

    fn cuinterpose(&self) -> cuinterpose::Config {
        cuinterpose::Config {
            bundle_dir: self.args.tools_dir.join(CUINTERPOSE),
            socket_dir: in_pod(self.shim_pid, "/tmp"),
            loaded_from: Some(PathBuf::from(format!("{TOOLS}/{CUINTERPOSE}"))),
        }
    }

    /// The CUDA processes of the server; none for a pod without GPUs, which
    /// needs no driver at all.
    async fn cuda_pids(&self, server: i32) -> Result<Vec<i32>> {
        if self.gpus.is_empty() {
            return Ok(Vec::new());
        }
        tokio::task::spawn_blocking(move || -> Result<Vec<i32>> {
            let driver = Driver::load()?;
            Ok(cuda::process_tree(server)
                .into_iter()
                .filter(|&pid| driver.is_cuda_process(pid))
                .collect())
        })
        .await?
    }

    /// Checkpoint the server, whose PID in the pod is `root_pid`, into
    /// `image`. Once the shim is told to release the server it is not coming
    /// back except by a restore. The control connection is closed before the
    /// dump, so CRIU does not carry it into the image.
    async fn checkpoint(
        &self,
        mut control: Control,
        shim: &mut Shim,
        root_pid: i32,
        image: &Path,
    ) -> Result<()> {
        let started = Instant::now();
        fs::create_dir_all(image.join("img"))
            .with_context(|| format!("create {}", image.display()))?;
        step("vLLM prepare", control.prepare(LIFECYCLE)).await?;
        drop(control);
        let server = host_pid(self.shim_pid, root_pid)?;
        let pids = self.cuda_pids(server).await?;
        shim.call(Request::Release).await?;
        if !pids.is_empty() {
            let (config, img, sharing) = (self.cuinterpose(), image.join("img"), pids.clone());
            step(
                "cuInterpose prepare",
                blocking(move || {
                    match cuinterpose::prepare_checkpoint(&config, &img, &sharing)? {
                        cuinterpose::SharedMemory::NotShared => {
                            info!("no CUDA memory is shared between processes");
                        }
                        cuinterpose::SharedMemory::Prepared(manifest) => {
                            info!(processes = manifest.pids.len(), "shared CUDA memory saved");
                        }
                    }
                    Ok(())
                }),
            )
            .await?;
            step(
                "CUDA checkpoint",
                blocking(move || cuda::checkpoint_job(&Driver::load()?, &pids, CUDA_LOCK)),
            )
            .await?;
        }
        self.mount(image, IMAGE).await?;
        step("CRIU dump", async {
            run_criu(
                self.criu("dump")
                    .args(["-t", &root_pid.to_string()])
                    .args(["-D", &format!("{IMAGE}/img"), "-W", &format!("{IMAGE}/img")])
                    .args(["--ghost-limit", "1G", "-o", "dump.log"]),
                "criu dump",
                &image.join("img/dump.log"),
            )
            .await?;
            gone(server).await
        })
        .await?;
        step("copy shared memory out", async {
            copy_tree(&in_pod(self.shim_pid, "/dev/shm"), &image.join("shm")).await?;
            fs::copy(in_pod(self.shim_pid, SERVER_LOG), image.join("server.log"))
                .context("copy the server's log")?;
            Ok(())
        })
        .await?;
        Manifest {
            root_pid,
            gpus: self.gpus.iter().map(GpuUuid::to_string).collect(),
            minors: self.minors.clone(),
            address: self.address,
        }
        .save(image)?;
        info!(
            seconds = %seconds(started),
            gib = %gib(directory_bytes(image)),
            "checkpointed"
        );
        Ok(())
    }

    /// Restore the checkpoint in `image` into this pod and bring vLLM back to
    /// serving. Returns the server's PID in the pod.
    async fn restore(&self, image: &Path) -> Result<i32> {
        let started = Instant::now();
        let manifest = Manifest::load(image)?;
        let source: Vec<GpuUuid> = manifest
            .gpus
            .iter()
            .map(|gpu| gpu.parse())
            .collect::<Result<_>>()
            .context("GPUs in the manifest")?;
        ensure!(
            source.len() == self.gpus.len(),
            "the checkpoint ran on {} GPUs, the pod has {}",
            source.len(),
            self.gpus.len()
        );
        step("copy shared memory in", async {
            copy_tree(&image.join("shm"), &in_pod(self.shim_pid, "/dev/shm")).await?;
            fs::copy(image.join("server.log"), in_pod(self.shim_pid, SERVER_LOG))
                .context("put the server's log back")?;
            fs::create_dir_all(in_pod(self.shim_pid, WORK)).context("create the work directory")
        })
        .await?;
        let mut criu = self.criu("restore");
        if manifest.address != self.address {
            info!(from = %manifest.address, to = %self.address, "the pod address differs, sockets are remapped");
            fs::write(
                in_pod(self.shim_pid, &format!("{WORK}/inet-remap.map")),
                format!("{} {}\n", manifest.address, self.address),
            )
            .context("write the address map")?;
            criu.env(
                "SNAPSHOT_CRIU_INET_REMAP_FILE",
                format!("{WORK}/inet-remap.map"),
            );
        }
        run(
            Command::new("nsenter")
                .args(["-t", &self.shim_pid.to_string(), "-p", "--", "sh", "-c"])
                .arg(format!("echo {FAR_PID} > /proc/sys/kernel/ns_last_pid")),
            "move the pod's PID counter",
        )
        .await?;
        self.mount(image, IMAGE).await?;
        step(
            "CRIU restore",
            run_criu(
                criu.args(["-D", &format!("{IMAGE}/img"), "-W", WORK])
                    .args(["--restore-detached", "-o", "restore.log"]),
                "criu restore",
                &in_pod(self.shim_pid, &format!("{WORK}/restore.log")),
            ),
        )
        .await?;
        let server = host_pid(self.shim_pid, manifest.root_pid)?;
        let pids = self.cuda_pids(server).await?;
        if !pids.is_empty() {
            let map = GpuMap::new(&source, &self.gpus, &self.visible)?;
            if !map.is_identity() {
                info!(from = ?manifest.minors, to = ?self.minors, "GPUs remapped, by device minor");
            }
            step(
                "CUDA restore",
                blocking(move || cuda::restore_job(&Driver::load()?, &pids, &map)),
            )
            .await?;
            let (config, img) = (self.cuinterpose(), image.join("img"));
            step(
                "cuInterpose restore",
                blocking(move || match cuinterpose::Manifest::load(&img)? {
                    Some(manifest) => {
                        manifest.verify_libraries(&config)?;
                        cuinterpose::restore(&config, &img, &manifest)
                    }
                    None => Ok(()),
                }),
            )
            .await?;
        }
        step("vLLM restore", async {
            Control::connect(self.control())
                .await?
                .restore(LIFECYCLE)
                .await
        })
        .await?;
        info!(seconds = %seconds(started), from = %image.display(), "restored");
        Ok(manifest.root_pid)
    }
}

async fn look_after(node: &Node, sandbox: &Sandbox) -> Result<()> {
    let args = &node.args;
    let seen = Instant::now();
    let mut runtime = Runtime::connect(&args.runtime_socket).await?;
    let deadline = Instant::now() + CONTAINER_WAIT;
    let shim_pid = loop {
        if let Some(pid) = runtime.pid(sandbox, &args.container).await? {
            break pid;
        }
        ensure!(
            Instant::now() < deadline,
            "container {} did not start",
            args.container
        );
        sleep(Duration::from_secs(1)).await;
    };
    let mut shim = Shim::connect(shim_pid).await?;
    let (started, serving) = shim.state().await?;
    if serving {
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
    let ordered = if allotted.is_empty() {
        Vec::new()
    } else {
        cuda::in_enumeration_order(&allotted)?
    };
    let pod = Pod {
        args,
        shim_pid,
        address: runtime.address(sandbox).await?,
        gpus: ordered.iter().map(|&(gpu, _)| gpu).collect(),
        minors: ordered.iter().map(|&(_, minor)| minor).collect(),
        visible: if ordered.is_empty() {
            Vec::new()
        } else {
            cuda::node_gpus()?
                .into_iter()
                .filter(|(_, minor)| in_pod(shim_pid, &format!("/dev/nvidia{minor}")).exists())
                .map(|(gpu, _)| gpu)
                .collect()
        },
    };
    if started {
        info!("the server was started before this daemon; waiting for it to run");
        Control::connect_when_running(pod.control(), BOOT).await?;
        shim.call(Request::Serving).await?;
        info!(seconds = %seconds(seen), how = "already started", "serving");
        return Ok(());
    }

    let workload = sandbox
        .labels
        .get(&args.label)
        .context("the opt-in label is gone")?;
    let revision = sandbox
        .labels
        .get(REVISION_LABEL)
        .with_context(|| format!("no {REVISION_LABEL} label; is the pod from a Deployment?"))?;
    let revision_dir = args
        .store_dir
        .join(&sandbox.namespace)
        .join(workload)
        .join(revision);
    let store = State::open(&revision_dir, &sandbox.uid)?;
    let gave_up = revision_dir.join(format!("failed-{}", sandbox.uid));
    info!(
        pod = %format_args!("{}/{}", sandbox.namespace, sandbox.name),
        workload,
        revision,
        address = %pod.address,
        gpus = pod.gpus.len(),
        visible_gpus = pod.visible.len(),
        pid = shim_pid,
        "new pod"
    );
    pod.mount(&args.tools_dir, TOOLS).await?;
    if gave_up.exists() {
        warn!("a checkpoint or restore failed in this pod before; it cold boots");
    } else if store.image_ready() {
        info!("this node holds the revision's checkpoint");
    }

    #[cfg(feature = "nixl")]
    if let Some(peers) = node
        .peers
        .clone()
        .filter(|_| !store.image_ready() && !gave_up.exists())
    {
        info!("this node has no checkpoint of the revision; asking the other nodes");
        let revision = Revision::new(format!("{}/{workload}/{revision}", sandbox.namespace))?;
        let (directory, uid) = (revision_dir.clone(), sandbox.uid.clone());
        let asked = Instant::now();
        let pulled =
            blocking(move || peers.pull(&revision, &State::open(&directory, &uid)?)).await?;
        match pulled {
            Some(peer) => info!(
                %peer,
                seconds = %seconds(asked),
                gib = %gib(directory_bytes(&store.image_dir())),
                "pulled the checkpoint from a peer"
            ),
            None => info!(seconds = %seconds(asked), "no other node has it"),
        }
    }

    let refused = if store.image_ready() && !gave_up.exists() {
        unrestorable(&store.image_dir(), &pod.minors)?
    } else {
        None
    };
    if let Some(reason) = &refused {
        warn!(
            reason,
            "the checkpoint cannot be restored into this pod; it cold boots"
        );
    }
    if store.image_ready() && !gave_up.exists() && refused.is_none() {
        match pod.restore(&store.image_dir()).await {
            Ok(pid) => {
                shim.call(Request::Adopt { pid }).await?;
                shim.call(Request::Serving).await?;
                info!(seconds = %seconds(seen), how = "restored", "serving");
                return Ok(());
            }
            Err(error) => {
                pod.log_server_tail();
                fs::write(&gave_up, format!("{error:#}\n")).context("record the failure")?;
                shim.call(Request::Fail).await?;
                return Err(error.context("restore failed; the container starts over cold"));
            }
        }
    }

    info!("cold booting the server");
    let boot = Instant::now();
    let root_pid = shim.start().await?;
    let control = Control::connect_when_running(pod.control(), BOOT).await?;
    info!(seconds = %seconds(boot), "the server is running");
    if store.image_ready() || gave_up.exists() {
        shim.call(Request::Serving).await?;
        info!(seconds = %seconds(seen), how = "cold boot", "serving");
        return Ok(());
    }
    if let Err(error) = ensure_room(&revision_dir, args.min_free_gib << 30) {
        warn!(error = %format_args!("{error:#}"), "no checkpoint is taken");
        shim.call(Request::Serving).await?;
        info!(seconds = %seconds(seen), how = "cold boot", "serving");
        return Ok(());
    }
    info!("taking the revision's checkpoint");
    store.clear_staging()?;
    let staging = store.staging_dir().to_path_buf();
    let outcome = async {
        pod.checkpoint(control, &mut shim, root_pid, &staging)
            .await?;
        pod.restore(&staging).await
    }
    .await;
    match outcome {
        Ok(pid) => {
            shim.call(Request::Adopt { pid }).await?;
            store.publish()?;
            info!(image = %store.image_dir().display(), "checkpoint published on this node");
            shim.call(Request::Serving).await?;
            info!(seconds = %seconds(seen), how = "cold boot, checkpoint, restore", "serving");
            Ok(())
        }
        Err(error) => {
            pod.log_server_tail();
            fs::write(&gave_up, format!("{error:#}\n")).context("record the failure")?;
            shim.call(Request::Fail).await?;
            Err(error.context("checkpoint failed; the container starts over cold"))
        }
    }
}

async fn serve(args: Args) -> Result<()> {
    fs::create_dir_all(&args.store_dir)
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
    let handled = Arc::new(Mutex::new(HashSet::<String>::new()));
    info!(
        label = %args.label,
        store = %args.store_dir.display(),
        tools = %args.tools_dir.display(),
        runtime = %args.runtime_socket.display(),
        "watching this node's pods"
    );
    loop {
        match runtime.sandboxes(&args.label).await {
            Ok(sandboxes) => {
                let mut seen = handled
                    .lock()
                    .map_err(|_| anyhow!("pod set lock poisoned"))?;
                seen.retain(|uid| sandboxes.iter().any(|sandbox| &sandbox.uid == uid));
                for sandbox in sandboxes {
                    if !seen.insert(sandbox.uid.clone()) {
                        continue;
                    }
                    let (node, handled) = (Arc::clone(&node), Arc::clone(&handled));
                    let span = info_span!("pod", id = short_id(&sandbox.name));
                    tokio::spawn(
                        async move {
                            if let Err(error) = look_after(&node, &sandbox).await {
                                error!(
                                    error = %format!("{error:#}").replace('\n', "; "),
                                    "gave up on this pod for now"
                                );
                                sleep(RETRY).await;
                                if let Ok(mut seen) = handled.lock() {
                                    seen.remove(&sandbox.uid);
                                }
                            }
                        }
                        .instrument(span),
                    );
                }
            }
            Err(error) => warn!(error = %format_args!("{error:#}"), "could not list pods"),
        }
        sleep(Duration::from_secs(args.poll_s)).await;
    }
}

pub fn run_daemon(args: Args) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("start the runtime")?
        .block_on(serve(args))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::{Command, Stdio};

    use crate::cuinterpose;
    use crate::node::{
        Manifest, criu_errors, gone, host_pid, in_pod, plain, short_id, unrestorable,
    };

    #[test]
    fn a_child_is_found_by_its_pid() {
        let mut child = Command::new("sleep")
            .arg("30")
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let own = std::process::id() as i32;
        let pid = child.id() as i32;
        assert_eq!(host_pid(own, pid).unwrap(), pid);
        assert!(host_pid(own, 0).is_err());
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(host_pid(own, pid).is_err());
    }

    #[test]
    fn a_missing_parent_is_an_error() {
        assert!(host_pid(i32::MAX, 1).is_err());
    }

    #[test]
    fn a_manifest_round_trips() {
        let dir = std::env::temp_dir().join(format!("snapshotd-manifest-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let manifest = Manifest {
            root_pid: 57,
            gpus: vec!["GPU-03e97d3f-21d4-1c89-f142-657edeaf3566".to_string()],
            minors: vec![3],
            address: "10.0.2.165".parse().unwrap(),
        };
        manifest.save(&dir).unwrap();
        assert_eq!(Manifest::load(&dir).unwrap(), manifest);
        fs::write(dir.join("manifest.json"), "{").unwrap();
        assert!(Manifest::load(&dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
        assert!(Manifest::load(&dir).is_err());
    }

    #[tokio::test]
    async fn a_reaped_process_is_gone() {
        let mut child = Command::new("sleep")
            .arg("30")
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id() as i32;
        child.kill().unwrap();
        let reaper = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            child.wait().unwrap();
        });
        gone(pid).await.unwrap();
        reaper.join().unwrap();
        gone(i32::MAX).await.unwrap();
    }

    #[test]
    fn a_criu_log_is_cut_down_to_its_errors() {
        let log = "(00.001) Warn  (criu/x.c:1): a\n(00.002) Error (criu/y.c:2): first\n(00.003) fine\n(00.004) Error (criu/z.c:3): second\n";
        assert_eq!(
            criu_errors(log),
            "(00.002) Error (criu/y.c:2): first\n(00.004) Error (criu/z.c:3): second"
        );
        let quiet = "one\ntwo\nthree\nfour\nfive\nsix\n";
        assert_eq!(criu_errors(quiet), "two\nthree\nfour\nfive\nsix");
        assert_eq!(criu_errors(""), "");
    }

    #[test]
    fn shared_gpu_memory_ties_a_checkpoint_to_its_device_minors() {
        let dir = std::env::temp_dir().join(format!("snapshotd-minors-{}", std::process::id()));
        fs::create_dir_all(dir.join("img")).unwrap();
        Manifest {
            root_pid: 57,
            gpus: vec![
                "GPU-60432197-b540-ea7e-4d3f-d4688626f64d".to_string(),
                "GPU-e97d4405-df15-56a3-8ae5-08ec4c1a4b14".to_string(),
            ],
            minors: vec![4, 5],
            address: "10.0.2.165".parse().unwrap(),
        }
        .save(&dir)
        .unwrap();
        assert_eq!(unrestorable(&dir, &[4, 5]).unwrap(), None);
        assert_eq!(unrestorable(&dir, &[6, 7]).unwrap(), None);
        assert!(
            unrestorable(&dir, &[4])
                .unwrap()
                .unwrap()
                .contains("2 GPUs")
        );

        cuinterpose::Manifest {
            libraries: Default::default(),
            pids: vec![675, 676],
        }
        .save(&dir.join("img"))
        .unwrap();
        assert_eq!(unrestorable(&dir, &[4, 5]).unwrap(), None);
        let reason = unrestorable(&dir, &[6, 7]).unwrap().unwrap();
        assert!(
            reason.contains("[4, 5]") && reason.contains("[6, 7]"),
            "{reason}"
        );
        fs::remove_dir_all(&dir).unwrap();
        assert!(unrestorable(&dir, &[4, 5]).is_err());
    }

    #[test]
    fn colour_sequences_are_removed() {
        let line = "\u{1b}[0;36m(RustFrontend pid=421)\u{1b}[0m \u{1b}[32mINFO\u{1b}[0m ready";
        assert_eq!(plain(line), "(RustFrontend pid=421) INFO ready");
        assert_eq!(plain("no colours"), "no colours");
        assert_eq!(plain("cut off \u{1b}[3"), "cut off ");
    }

    #[test]
    fn a_pod_is_tagged_with_the_end_of_its_name() {
        assert_eq!(short_id("snapshot-serve-db6c5d479-ltmws"), "ltmws");
        assert_eq!(short_id("standalone"), "standalone");
    }

    #[test]
    fn pod_paths_go_through_the_process_root() {
        assert_eq!(
            in_pod(4242, "/dev/shm").to_str(),
            Some("/proc/4242/root/dev/shm")
        );
    }
}
