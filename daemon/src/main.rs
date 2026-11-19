//! snapshotd: the privileged node path for snapshot scale-out. See docs/design.md.
//!
//! `node` is the daemon. The other subcommands are its building blocks,
//! usable on their own from a privileged container.
mod cri;
mod cuda;
mod cuinterpose;
#[cfg(any(test, feature = "nixl"))]
mod encrypt;
mod image;
mod inject;
#[cfg(any(test, feature = "nixl"))]
mod lanes;
mod node;
#[cfg(feature = "nixl")]
mod peer;
mod podresources;
mod store;
#[cfg(feature = "nixl")]
mod trace;
#[cfg(feature = "nixl")]
mod transfer;
#[cfg(any(test, feature = "nixl"))]
mod tunnel;
mod uds;
mod vllm;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rustix::process::{Resource, getrlimit, setrlimit};
use tracing_subscriber::EnvFilter;

use crate::cuda::{Driver, GpuMap, GpuUuid};

#[derive(Parser)]
#[command(name = "snapshotd")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// CUDA process checkpoint through the host driver.
    #[command(subcommand)]
    Cuda(CudaCommand),
    /// Mount a directory into a container's mount namespace.
    Inject(inject::Args),
    /// Checkpoint and restore the opted-in pods of this node.
    Node(node::Args),
    /// cuInterpose shared-memory tracking.
    #[command(subcommand)]
    Cuinterpose(CuinterposeCommand),
    /// Hold a checkpoint in huge-page memory.
    #[command(subcommand)]
    Image(ImageCommand),
    /// Copy a checkpoint directory to another node over RDMA.
    #[cfg(feature = "nixl")]
    #[command(subcommand)]
    Transfer(TransferCommand),
}

#[derive(Subcommand)]
enum ImageCommand {
    /// Take one `criu dump --stream` into memory, then answer every
    /// `criu restore --stream` from it until killed.
    Hold {
        /// The directory given to CRIU as its images directory.
        #[arg(long)]
        images_dir: PathBuf,
        /// Size of each huge-page segment.
        #[arg(long, default_value_t = 4)]
        segment_gib: usize,
        /// Memory to map and fault in before CRIU starts; the expected
        /// checkpoint size.
        #[arg(long, default_value_t = 0)]
        reserve_gib: usize,
    },
}

#[cfg(feature = "nixl")]
#[derive(Subcommand)]
enum TransferCommand {
    /// Wait for one sender and write what it sends under a directory.
    Receive {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value = "0.0.0.0:7700")]
        listen: String,
        /// Write with O_DIRECT, past the page cache and its dirty limits.
        #[arg(long)]
        direct: bool,
        #[command(flatten)]
        hold: transfer::Hold,
        /// Write a per-chunk timeline here, in the Chrome trace event format.
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// Take one `criu dump --stream` and send it to a receiver as it arrives.
    Stream {
        /// The directory given to CRIU as its images directory.
        #[arg(long)]
        images_dir: PathBuf,
        /// The receiver's address and port.
        #[arg(long)]
        to: String,
        #[command(flatten)]
        pool: transfer::Pool,
        /// Write a per-chunk timeline here, in the Chrome trace event format.
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// Send every file under a directory to a receiver.
    Send {
        #[arg(long)]
        dir: PathBuf,
        /// The receiver's address and port.
        #[arg(long)]
        to: String,
        #[command(flatten)]
        pool: transfer::Pool,
        /// Write a per-chunk timeline here, in the Chrome trace event format.
        #[arg(long)]
        trace: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum CudaCommand {
    /// Print the CUDA processes in the tree rooted at a PID.
    Pids { root: i32 },
    /// Print each process's checkpoint state.
    State { pids: Vec<i32> },
    /// Lock every process, then checkpoint every process.
    Checkpoint {
        #[arg(long, default_value_t = 30_000)]
        lock_timeout_ms: u64,
        #[arg(required = true)]
        pids: Vec<i32>,
    },
    /// Restore every process, onto other GPUs if given, then unlock every process.
    Restore {
        /// GPUs the checkpoint ran on, in order.
        #[arg(long, value_delimiter = ',', requires = "visible")]
        source: Vec<GpuUuid>,
        /// GPUs to restore onto, in the same order.
        #[arg(long, value_delimiter = ',')]
        target: Vec<GpuUuid>,
        /// Every GPU the processes can see; the driver needs a full bijection.
        #[arg(long, value_delimiter = ',')]
        visible: Vec<GpuUuid>,
        #[arg(required = true)]
        pids: Vec<i32>,
    },
}

#[derive(Subcommand)]
enum CuinterposeCommand {
    /// Print the shared CUDA memory each preloaded process holds.
    Inspect {
        #[arg(long, default_value = "/tmp")]
        socket_dir: PathBuf,
        #[arg(required = true)]
        namespace_pids: Vec<u32>,
    },
    /// Inspect the CUDA processes and, if any shares memory, save it with the
    /// checkpoint and unmap it. The processes cannot resume afterwards.
    Prepare {
        #[command(flatten)]
        config: cuinterpose::Config,
        #[arg(long)]
        checkpoint_dir: PathBuf,
        #[arg(required = true)]
        pids: Vec<i32>,
    },
    /// Re-import the shared memory a checkpoint saved, after CUDA restore.
    Restore {
        #[command(flatten)]
        config: cuinterpose::Config,
        #[arg(long)]
        checkpoint_dir: PathBuf,
    },
}

fn raise_open_file_limit() -> Result<()> {
    let mut limit = getrlimit(Resource::Nofile);
    limit.current = limit.maximum;
    setrlimit(Resource::Nofile, limit).context("raise the open file limit to its hard limit")
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("SNAPSHOTD_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_ansi(std::io::stdout().is_terminal())
        .init();
    raise_open_file_limit()?;
    Cli::parse().command.run()
}

impl Command {
    fn run(self) -> Result<()> {
        match self {
            Self::Inject(args) => inject::run(&args),
            Self::Node(args) => node::run(args),
            Self::Cuda(command) => cuda_command(command),
            Self::Cuinterpose(command) => cuinterpose_command(command),
            Self::Image(ImageCommand::Hold {
                images_dir,
                segment_gib,
                reserve_gib,
            }) => image::hold(&images_dir, segment_gib << 30, reserve_gib << 30),
            #[cfg(feature = "nixl")]
            Self::Transfer(command) => command.run(),
        }
    }
}

#[cfg(feature = "nixl")]
impl TransferCommand {
    fn run(self) -> Result<()> {
        match self {
            Self::Receive {
                dir,
                listen,
                direct,
                hold,
                trace,
            } => transfer::receive_directory(&dir, &listen, direct, hold, trace),
            Self::Stream {
                images_dir,
                to,
                pool,
                trace,
            } => transfer::stream_capture(&images_dir, &to, pool, trace),
            Self::Send {
                dir,
                to,
                pool,
                trace,
            } => transfer::send_directory(&dir, &to, pool, trace),
        }
    }
}

fn cuinterpose_command(command: CuinterposeCommand) -> Result<()> {
    match command {
        CuinterposeCommand::Inspect {
            socket_dir,
            namespace_pids,
        } => {
            for pid in namespace_pids {
                let sharing = cuinterpose::Sharing::of(&cuinterpose::inspect(&socket_dir, pid)?);
                println!("{pid}: {sharing:?}");
            }
        }
        CuinterposeCommand::Prepare {
            config,
            checkpoint_dir,
            pids,
        } => match cuinterpose::prepare_checkpoint(&config, &checkpoint_dir, &pids)? {
            cuinterpose::SharedMemory::NotShared => println!("no shared CUDA memory"),
            cuinterpose::SharedMemory::Prepared(manifest) => println!(
                "saved the shared CUDA memory of {} processes",
                manifest.pids.len()
            ),
        },
        CuinterposeCommand::Restore {
            config,
            checkpoint_dir,
        } => match cuinterpose::Manifest::load(&checkpoint_dir)? {
            None => println!("no shared CUDA memory in the checkpoint"),
            Some(manifest) => {
                manifest.verify_libraries(&config)?;
                cuinterpose::restore(&config, &checkpoint_dir, &manifest)?;
            }
        },
    }
    Ok(())
}

fn cuda_command(command: CudaCommand) -> Result<()> {
    let driver = Driver::load()?;
    match command {
        CudaCommand::Pids { root } => {
            let pids: Vec<String> = cuda::process_tree(root)
                .into_iter()
                .filter(|&pid| driver.is_cuda_process(pid))
                .map(|pid| pid.to_string())
                .collect();
            println!("{}", pids.join(" "));
        }
        CudaCommand::State { pids } => {
            for pid in pids {
                println!("{pid}: {:?}", driver.state(pid)?);
            }
        }
        CudaCommand::Checkpoint {
            lock_timeout_ms,
            pids,
        } => cuda::checkpoint_job(&driver, &pids, Duration::from_millis(lock_timeout_ms))?,
        CudaCommand::Restore {
            source,
            target,
            visible,
            pids,
        } => {
            let map = GpuMap::new(&source, &target, &visible)?;
            if !map.is_identity() {
                println!("remap {map}");
            }
            cuda::restore_job(&driver, &pids, &map)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rustix::process::{Resource, getrlimit};

    use crate::raise_open_file_limit;

    #[test]
    fn open_file_limit_is_raised_to_the_hard_limit() {
        raise_open_file_limit().unwrap();
        let limit = getrlimit(Resource::Nofile);
        assert_eq!(limit.current, limit.maximum);
    }
}
