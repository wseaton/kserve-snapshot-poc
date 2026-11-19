use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::time::Instant;
use tracing::{info, warn};

use crate::cuda::{self, Driver, Gpu, GpuMap, GpuUuid};
use crate::cuinterpose::{self, SharedMemory};
use crate::node::criu;
use crate::node::layout::{CUINTERPOSE, IMAGE, TOOLS, WORK};
use crate::node::log::{gib, plain, seconds, step};
use crate::node::manifest::Manifest;
use crate::node::process::{self, HostPid, PodPid, blocking, copy_tree};
use crate::node::shim::ShimClient;
use crate::store::directory_bytes;
use crate::vllm::Control;
use snapshotd_shim::SERVER_LOG;

const FAR_PID: &str = "4000000";
const LIFECYCLE: Duration = Duration::from_secs(900);
const CUDA_LOCK: Duration = Duration::from_secs(60);
const SERVER_TAIL: usize = 12;

pub struct Pod {
    init: HostPid,
    control: SocketAddr,
    tools: PathBuf,
    /// The pod's GPUs, in the driver's order.
    gpus: Vec<Gpu>,
    /// Every GPU the container can open.
    visible: Vec<GpuUuid>,
}

impl Pod {
    pub fn new(
        init: HostPid,
        control: SocketAddr,
        tools: &Path,
        allotted: &[GpuUuid],
    ) -> Result<Self> {
        let (gpus, visible) = if allotted.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let node = cuda::node_gpus()?;
            let visible = node
                .iter()
                .filter(|gpu| init.root(&format!("/dev/nvidia{}", gpu.minor)).exists())
                .map(|gpu| gpu.uuid)
                .collect();
            (cuda::pick(&node, allotted)?, visible)
        };
        Ok(Self {
            init,
            control,
            tools: tools.to_path_buf(),
            gpus,
            visible,
        })
    }

    pub fn init(&self) -> HostPid {
        self.init
    }

    pub fn address(&self) -> IpAddr {
        self.control.ip()
    }

    pub fn control(&self) -> SocketAddr {
        self.control
    }

    pub fn gpus(&self) -> &[Gpu] {
        &self.gpus
    }

    pub fn visible_gpus(&self) -> usize {
        self.visible.len()
    }

    async fn mount(&self, source: &Path, target: &str) -> Result<()> {
        let exe = std::env::current_exe().context("this program's path")?;
        process::run(
            Command::new(exe)
                .arg("inject")
                .args(["--pid", &self.init.to_string()])
                .arg("--source")
                .arg(source)
                .args(["--target", target]),
            &format!("mount {} in the pod", source.display()),
        )
        .await
    }

    pub async fn mount_tools(&self) -> Result<()> {
        self.mount(&self.tools, TOOLS).await
    }

    pub fn log_server_tail(&self) {
        let log = fs::read_to_string(self.init.root(SERVER_LOG)).unwrap_or_default();
        let lines: Vec<&str> = log.lines().collect();
        for line in &lines[lines.len().saturating_sub(SERVER_TAIL)..] {
            warn!("server: {}", plain(line));
        }
    }

    fn cuinterpose(&self) -> cuinterpose::Config {
        cuinterpose::Config {
            bundle_dir: self.tools.join(CUINTERPOSE),
            socket_dir: self.init.root("/tmp"),
            loaded_from: Some(PathBuf::from(format!("{TOOLS}/{CUINTERPOSE}"))),
        }
    }

    async fn cuda_processes(&self, server: HostPid) -> Result<Vec<i32>> {
        if self.gpus.is_empty() {
            return Ok(Vec::new());
        }
        blocking(move || {
            let driver = Driver::load()?;
            Ok(cuda::process_tree(server.0)
                .into_iter()
                .filter(|&pid| driver.is_cuda_process(pid))
                .collect())
        })
        .await
    }

    pub async fn checkpoint(
        &self,
        mut control: Control,
        shim: &mut ShimClient,
        root: PodPid,
        image: &Path,
    ) -> Result<()> {
        let started = Instant::now();
        fs::create_dir_all(image.join("img"))
            .with_context(|| format!("create {}", image.display()))?;
        step("vLLM prepare", control.prepare(LIFECYCLE)).await?;
        drop(control);
        let server = self.init.child(root)?;
        let processes = self.cuda_processes(server).await?;
        shim.release().await?;
        if !processes.is_empty() {
            let (config, img, sharing) = (self.cuinterpose(), image.join("img"), processes.clone());
            step(
                "cuInterpose prepare",
                blocking(move || {
                    match cuinterpose::prepare_checkpoint(&config, &img, &sharing)? {
                        SharedMemory::NotShared => {
                            info!("no CUDA memory is shared between processes");
                        }
                        SharedMemory::Prepared(manifest) => {
                            info!(processes = manifest.pids.len(), "shared CUDA memory saved");
                        }
                    }
                    Ok(())
                }),
            )
            .await?;
            step(
                "CUDA checkpoint",
                blocking(move || cuda::checkpoint_job(&Driver::load()?, &processes, CUDA_LOCK)),
            )
            .await?;
        }
        self.mount(image, IMAGE).await?;
        step("CRIU dump", async {
            criu::run(
                criu::command(self.init, "dump")
                    .args(["-t", &root.to_string()])
                    .args(["-D", &format!("{IMAGE}/img"), "-W", &format!("{IMAGE}/img")])
                    .args(["--ghost-limit", "1G", "-o", "dump.log"]),
                "criu dump",
                &image.join("img/dump.log"),
            )
            .await?;
            server.reaped().await
        })
        .await?;
        step("copy shared memory out", async {
            copy_tree(&self.init.root("/dev/shm"), &image.join("shm")).await?;
            fs::copy(self.init.root(SERVER_LOG), image.join("server.log"))
                .context("copy the server's log")?;
            Ok(())
        })
        .await?;
        Manifest {
            root,
            gpus: self.gpus.clone(),
            address: self.address(),
        }
        .save(image)?;
        info!(
            seconds = %seconds(started),
            gib = %gib(directory_bytes(image)),
            "checkpointed"
        );
        Ok(())
    }

    pub async fn restore(&self, image: &Path) -> Result<PodPid> {
        let started = Instant::now();
        let manifest = Manifest::load(image)?;
        step("copy shared memory in", async {
            let shm = image.join("shm");
            if shm.is_dir() {
                copy_tree(&shm, &self.init.root("/dev/shm")).await?;
            }
            fs::copy(image.join("server.log"), self.init.root(SERVER_LOG))
                .context("put the server's log back")?;
            fs::create_dir_all(self.init.root(WORK)).context("create the work directory")
        })
        .await?;
        let mut restore = criu::command(self.init, "restore");
        if manifest.address != self.address() {
            info!(
                from = %manifest.address,
                to = %self.address(),
                "the pod address differs, sockets are remapped"
            );
            let map = format!("{WORK}/inet-remap.map");
            fs::write(
                self.init.root(&map),
                format!("{} {}\n", manifest.address, self.address()),
            )
            .context("write the address map")?;
            restore.env("SNAPSHOT_CRIU_INET_REMAP_FILE", map);
        }
        process::run(
            Command::new("nsenter")
                .args(["-t", &self.init.to_string(), "-p", "--", "sh", "-c"])
                .arg(format!("echo {FAR_PID} > /proc/sys/kernel/ns_last_pid")),
            "move the pod's PID counter",
        )
        .await?;
        self.mount(image, IMAGE).await?;
        step(
            "CRIU restore",
            criu::run(
                restore
                    .args(["-D", &format!("{IMAGE}/img"), "-W", WORK])
                    .args(["--restore-detached", "-o", "restore.log"]),
                "criu restore",
                &self.init.root(&format!("{WORK}/restore.log")),
            ),
        )
        .await?;
        let server = self.init.child(manifest.root)?;
        let processes = self.cuda_processes(server).await?;
        if !processes.is_empty() {
            let uuids = |gpus: &[Gpu]| gpus.iter().map(|gpu| gpu.uuid).collect::<Vec<_>>();
            let map = GpuMap::new(&uuids(&manifest.gpus), &uuids(&self.gpus), &self.visible)?;
            if !map.is_identity() {
                let minors = |gpus: &[Gpu]| gpus.iter().map(|gpu| gpu.minor).collect::<Vec<_>>();
                info!(
                    from = ?minors(&manifest.gpus),
                    to = ?minors(&self.gpus),
                    "GPUs remapped, by device minor"
                );
            }
            step(
                "CUDA restore",
                blocking(move || cuda::restore_job(&Driver::load()?, &processes, &map)),
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
            Control::connect(self.control)
                .await?
                .restore(LIFECYCLE)
                .await
        })
        .await?;
        info!(seconds = %seconds(started), from = %image.display(), "restored");
        Ok(manifest.root)
    }
}
