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

use crate::cri::{Runtime, Sandbox};
use crate::cuda::{self, Driver, GpuMap, GpuUuid};
use crate::podresources;
use crate::shim::{Reply, Request};
use crate::state::{State, ensure_room};
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
}

/// What a restore needs to know about where the checkpoint came from.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The server's PID inside its pod.
    pub root_pid: i32,
    pub gpus: Vec<String>,
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

struct Pod<'a> {
    args: &'a Args,
    name: String,
    shim_pid: i32,
    address: IpAddr,
    gpus: Vec<GpuUuid>,
}

impl Pod<'_> {
    fn control(&self) -> SocketAddr {
        SocketAddr::new(self.address, self.args.control_port)
    }

    fn criu(&self, action: &str) -> Command {
        let mut command = Command::new("nsenter");
        command
            .args(["-t", &self.shim_pid.to_string(), "-a", "--"])
            .arg(format!("{TOOLS}/criu"))
            .arg(action)
            .env("LD_LIBRARY_PATH", format!("{TOOLS}/lib"))
            .args(["--libdir", &format!("{TOOLS}/criu-plugins")])
            .args([
                "--tcp-established",
                "--ext-unix-sk",
                "--file-locks",
                "--link-remap",
                "-v4",
            ]);
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
            &format!("mount {} in {}", source.display(), self.name),
        )
        .await
    }

    async fn cuda_pids(&self, server: i32) -> Result<Vec<i32>> {
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
    /// dump: CRIU will not dump a process with a TCP connection it cannot lock.
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
        control.prepare(LIFECYCLE).await?;
        drop(control);
        let server = host_pid(self.shim_pid, root_pid)?;
        let pids = self.cuda_pids(server).await?;
        shim.call(Request::Release).await?;
        tokio::task::spawn_blocking(move || {
            cuda::checkpoint_job(&Driver::load()?, &pids, CUDA_LOCK)
        })
        .await??;
        self.mount(&self.args.tools_dir, TOOLS).await?;
        self.mount(image, IMAGE).await?;
        run(
            self.criu("dump")
                .args(["-t", &root_pid.to_string()])
                .args(["-D", &format!("{IMAGE}/img"), "-W", &format!("{IMAGE}/img")])
                .args(["--ghost-limit", "1G", "-o", "dump.log"]),
            "criu dump",
        )
        .await?;
        copy_tree(&in_pod(self.shim_pid, "/dev/shm"), &image.join("shm")).await?;
        fs::copy(in_pod(self.shim_pid, SERVER_LOG), image.join("server.log"))
            .context("copy the server's log")?;
        Manifest {
            root_pid,
            gpus: self.gpus.iter().map(GpuUuid::to_string).collect(),
            address: self.address,
        }
        .save(image)?;
        println!(
            "{}: checkpointed in {:.1} s",
            self.name,
            started.elapsed().as_secs_f64()
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
        copy_tree(&image.join("shm"), &in_pod(self.shim_pid, "/dev/shm")).await?;
        fs::copy(image.join("server.log"), in_pod(self.shim_pid, SERVER_LOG))
            .context("put the server's log back")?;
        fs::create_dir_all(in_pod(self.shim_pid, WORK)).context("create the work directory")?;
        let mut criu = self.criu("restore");
        if manifest.address != self.address {
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
        self.mount(&self.args.tools_dir, TOOLS).await?;
        self.mount(image, IMAGE).await?;
        run(
            criu.args(["-D", &format!("{IMAGE}/img"), "-W", WORK])
                .args(["--restore-detached", "-o", "restore.log"]),
            "criu restore",
        )
        .await?;
        let server = host_pid(self.shim_pid, manifest.root_pid)?;
        let pids = self.cuda_pids(server).await?;
        let map = GpuMap::new(&source, &self.gpus, &self.gpus)?;
        tokio::task::spawn_blocking(move || cuda::restore_job(&Driver::load()?, &pids, &map))
            .await??;
        Control::connect(self.control())
            .await?
            .restore(LIFECYCLE)
            .await?;
        println!(
            "{}: restored from {} in {:.1} s",
            self.name,
            image.display(),
            started.elapsed().as_secs_f64()
        );
        Ok(manifest.root_pid)
    }
}

async fn look_after(args: &Args, sandbox: &Sandbox) -> Result<()> {
    let name = format!("{}/{}", sandbox.namespace, sandbox.name);
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
        return Ok(());
    }
    let pod = Pod {
        args,
        name: name.clone(),
        shim_pid,
        address: runtime.address(sandbox).await?,
        gpus: podresources::gpus(
            &args.pod_resources_socket,
            &sandbox.namespace,
            &sandbox.name,
            &args.container,
        )
        .await?,
    };
    if started {
        Control::connect_when_running(pod.control(), BOOT).await?;
        shim.call(Request::Serving).await?;
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
    let store = State::open(&revision_dir, &revision_dir, &sandbox.uid)?;
    let gave_up = revision_dir.join(format!("failed-{}", sandbox.uid));

    if store.image_ready() && !gave_up.exists() {
        match pod.restore(&store.image_dir()).await {
            Ok(pid) => {
                shim.call(Request::Adopt { pid }).await?;
                shim.call(Request::Serving).await?;
                return Ok(());
            }
            Err(error) => {
                fs::write(&gave_up, format!("{error:#}\n")).context("record the failure")?;
                shim.call(Request::Fail).await?;
                return Err(error.context("restore failed; the container starts over cold"));
            }
        }
    }

    let root_pid = shim.start().await?;
    let control = Control::connect_when_running(pod.control(), BOOT).await?;
    println!("{name}: cold booted");
    if store.image_ready() || gave_up.exists() {
        shim.call(Request::Serving).await?;
        return Ok(());
    }
    if let Err(error) = ensure_room(&revision_dir, args.min_free_gib << 30) {
        println!("{name}: no checkpoint: {error:#}");
        shim.call(Request::Serving).await?;
        return Ok(());
    }
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
            shim.call(Request::Serving).await?;
            Ok(())
        }
        Err(error) => {
            fs::write(&gave_up, format!("{error:#}\n")).context("record the failure")?;
            shim.call(Request::Fail).await?;
            Err(error.context("checkpoint failed; the container starts over cold"))
        }
    }
}

async fn serve(args: Arc<Args>) -> Result<()> {
    fs::create_dir_all(&args.store_dir)
        .with_context(|| format!("create {}", args.store_dir.display()))?;
    ensure!(
        args.tools_dir.join("criu").is_file(),
        "no criu in {}",
        args.tools_dir.display()
    );
    let mut runtime = Runtime::connect(&args.runtime_socket).await?;
    let handled = Arc::new(Mutex::new(HashSet::<String>::new()));
    println!("watching pods labelled {}", args.label);
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
                    let (args, handled) = (Arc::clone(&args), Arc::clone(&handled));
                    tokio::spawn(async move {
                        if let Err(error) = look_after(&args, &sandbox).await {
                            println!("{}/{}: {error:#}", sandbox.namespace, sandbox.name);
                            sleep(RETRY).await;
                            if let Ok(mut seen) = handled.lock() {
                                seen.remove(&sandbox.uid);
                            }
                        }
                    });
                }
            }
            Err(error) => println!("list pods: {error:#}"),
        }
        sleep(Duration::from_secs(args.poll_s)).await;
    }
}

pub fn run_daemon(args: Args) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("start the runtime")?
        .block_on(serve(Arc::new(args)))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::{Command, Stdio};

    use crate::node::{Manifest, host_pid, in_pod};

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
            address: "10.0.2.165".parse().unwrap(),
        };
        manifest.save(&dir).unwrap();
        assert_eq!(Manifest::load(&dir).unwrap(), manifest);
        fs::write(dir.join("manifest.json"), "{").unwrap();
        assert!(Manifest::load(&dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
        assert!(Manifest::load(&dir).is_err());
    }

    #[test]
    fn pod_paths_go_through_the_process_root() {
        assert_eq!(
            in_pod(4242, "/dev/shm").to_str(),
            Some("/proc/4242/root/dev/shm")
        );
    }
}
