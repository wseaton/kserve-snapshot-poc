use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::time::Instant;
use tracing::{info, warn};

use crate::cuda::{self, Driver, Gpu, GpuMap, GpuUuid};
use crate::cuinterpose::{self, SharedMemory};
use crate::node::criu;
use crate::node::environment::Environment;
use crate::node::layout::{CUINTERPOSE, IMAGE, TOOLS, WORK};
use crate::node::manifest::{ImageId, Manifest};
use crate::node::process::{self, HostPid, PodPid, blocking, copy_tree};
use crate::node::report::{self, gib, plain, seconds};
use crate::node::shim::ShimClient;
use crate::node::step::{self, Step};
use crate::store::{RESTORE_LOG_TAIL, directory_bytes};
use crate::vllm::{Control, Endpoint, PauseMode};
use snapshotd_shim::SERVER_LOG;

#[derive(Clone, clap::Args)]
pub struct Options {
    /// Directory holding criu, its lib/ and its criu-plugins/.
    #[arg(long, env = "SNAPSHOTD_TOOLS_DIR")]
    pub tools_dir: PathBuf,
    /// Seconds vLLM may take to prepare for a checkpoint, or to come back
    /// from one.
    #[arg(long, env = "SNAPSHOTD_LIFECYCLE_TIMEOUT_S", default_value_t = 900)]
    lifecycle_timeout_s: u64,
    /// How vLLM drains in-flight requests before a checkpoint.
    #[arg(
        long,
        env = "SNAPSHOTD_PREPARE_MODE",
        value_enum,
        default_value_t = PauseMode::Wait
    )]
    prepare_mode: PauseMode,
    /// Seconds a process may take to reach a point where the driver can
    /// checkpoint it.
    #[arg(long, env = "SNAPSHOTD_CUDA_LOCK_TIMEOUT_S", default_value_t = 60)]
    cuda_lock_timeout_s: u64,
    /// Seconds a dumped server may take to be reaped.
    #[arg(long, env = "SNAPSHOTD_REAP_TIMEOUT_S", default_value_t = 30)]
    reap_timeout_s: u64,
    /// Where a pod's PID counter is moved before a restore; above every PID
    /// a checkpoint holds.
    #[arg(long, env = "SNAPSHOTD_RESTORE_FIRST_PID", default_value_t = 4_000_000)]
    restore_first_pid: u32,
    /// Lines of the server's log repeated in the daemon's when a checkpoint
    /// or restore fails.
    #[arg(long, env = "SNAPSHOTD_SERVER_LOG_LINES", default_value_t = 12)]
    server_log_lines: usize,
}

impl Options {
    fn lifecycle_timeout(&self) -> Duration {
        Duration::from_secs(self.lifecycle_timeout_s)
    }
}

pub struct Pod {
    init: HostPid,
    control: Endpoint,
    options: Options,
    /// The pod's GPUs, in the driver's order.
    gpus: Vec<Gpu>,
    /// Every GPU the container can open.
    visible: Vec<GpuUuid>,
}

impl Pod {
    pub fn new(
        init: HostPid,
        control: Endpoint,
        options: &Options,
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
            options: options.clone(),
            gpus,
            visible,
        })
    }

    pub fn init(&self) -> HostPid {
        self.init
    }

    pub fn address(&self) -> IpAddr {
        self.control.address.ip()
    }

    pub fn control(&self) -> Endpoint {
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
        self.mount(&self.options.tools_dir, TOOLS).await
    }

    pub fn log_server_tail(&self) {
        let log = fs::read_to_string(self.init.root(SERVER_LOG)).unwrap_or_default();
        let lines: Vec<&str> = log.lines().collect();
        for line in &lines[lines.len().saturating_sub(self.options.server_log_lines)..] {
            warn!("server: {}", plain(line));
        }
    }

    fn restore_log(&self) -> PathBuf {
        self.init.root(&format!("{WORK}/restore.log"))
    }

    /// The end of CRIU's log of the last restore into this pod.
    pub fn restore_log_tail(&self) -> Vec<u8> {
        report::tail(&self.restore_log(), RESTORE_LOG_TAIL)
    }

    fn cuinterpose(&self) -> cuinterpose::Config {
        cuinterpose::Config {
            bundle_dir: self.options.tools_dir.join(CUINTERPOSE),
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
        environment: &Environment,
    ) -> Result<()> {
        let started = Instant::now();
        fs::create_dir_all(image.join("img"))
            .with_context(|| format!("create {}", image.display()))?;
        step::run(
            Step::VllmPrepare,
            control.prepare(self.options.prepare_mode, self.options.lifecycle_timeout()),
        )
        .await?;
        drop(control);
        let server = self.init.child(root)?;
        let processes = self.cuda_processes(server).await?;
        shim.release().await?;
        if !processes.is_empty() {
            let (config, img, sharing) = (self.cuinterpose(), image.join("img"), processes.clone());
            step::run(
                Step::CuinterposePrepare,
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
            let lock = Duration::from_secs(self.options.cuda_lock_timeout_s);
            step::run(
                Step::CudaCheckpoint,
                blocking(move || cuda::checkpoint_job(&Driver::load()?, &processes, lock)),
            )
            .await?;
        }
        self.mount(image, IMAGE).await?;
        fs::create_dir_all(self.init.root(WORK)).context("create the work directory")?;
        let log = self.init.root(&format!("{WORK}/dump.log"));
        step::run(Step::CriuDump, async {
            criu::run(
                criu::command(self.init, "dump")
                    .args(["-t", &root.to_string()])
                    .args(["-D", &format!("{IMAGE}/img"), "-W", WORK])
                    .args(["--ghost-limit", "1G", "-o", "dump.log"]),
                "criu dump",
                &log,
            )
            .await?;
            server
                .reaped(Duration::from_secs(self.options.reap_timeout_s))
                .await
        })
        .await?;
        step::run(Step::CopySharedMemoryOut, async {
            copy_tree(&self.init.root("/dev/shm"), &image.join("shm")).await?;
            fs::copy(self.init.root(SERVER_LOG), image.join("server.log"))
                .context("copy the server's log")?;
            fs::copy(&log, image.join("dump.log")).context("copy CRIU's log")?;
            Ok(())
        })
        .await?;
        Manifest {
            image: Some(ImageId::draw()),
            root,
            gpus: self.gpus.clone(),
            address: self.address(),
            environment: Some(environment.clone()),
        }
        .save(image)?;
        info!(
            seconds = %seconds(started),
            gib = %gib(directory_bytes(image)),
            "checkpointed"
        );
        Ok(())
    }

    /// Put a server that prepared itself at startup into service.
    pub async fn resume(&self, mut control: Control) -> Result<()> {
        step::run(
            Step::VllmRestore,
            control.restore(self.options.lifecycle_timeout()),
        )
        .await
    }

    pub async fn restore(&self, image: &Path) -> Result<PodPid> {
        let started = Instant::now();
        let manifest = Manifest::load(image)?;
        step::run(Step::CopySharedMemoryIn, async {
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
        if let Some(context) = self.init.selinux_context() {
            restore.args(["--lsm-profile", &format!("selinux:{context}")]);
        }
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
                .arg(format!(
                    "echo {} > /proc/sys/kernel/ns_last_pid",
                    self.options.restore_first_pid
                )),
            "move the pod's PID counter",
        )
        .await?;
        self.mount(image, IMAGE).await?;
        step::run(
            Step::CriuRestore,
            criu::run(
                restore
                    .args(["-D", &format!("{IMAGE}/img"), "-W", WORK])
                    .args(["--restore-detached", "-o", "restore.log"]),
                "criu restore",
                &self.restore_log(),
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
            step::run(
                Step::CudaRestore,
                blocking(move || cuda::restore_job(&Driver::load()?, &processes, &map)),
            )
            .await?;
            let (config, img) = (self.cuinterpose(), image.join("img"));
            step::run(
                Step::CuinterposeRestore,
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
        step::run(Step::VllmRestore, async {
            Control::connect(self.control)
                .await?
                .restore(self.options.lifecycle_timeout())
                .await
        })
        .await?;
        info!(seconds = %seconds(started), from = %image.display(), "restored");
        Ok(manifest.root)
    }
}
