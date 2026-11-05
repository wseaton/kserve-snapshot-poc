//! snapshotd: the privileged node path for snapshot scale-out. See DESIGN.md.
//!
//! The subcommands are the daemon's building blocks, usable on their own from
//! a privileged container while the control loop is written.
mod cuda;
mod cuinterpose;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};

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
    /// cuInterpose shared-memory tracking.
    #[command(subcommand)]
    Cuinterpose(CuinterposeCommand),
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
        #[arg(long, value_delimiter = ',')]
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
        paths: CuinterposePaths,
        #[arg(required = true)]
        pids: Vec<i32>,
    },
    /// Re-import the shared memory a checkpoint saved, after CUDA restore.
    Restore {
        #[command(flatten)]
        paths: CuinterposePaths,
    },
}

#[derive(clap::Args)]
struct CuinterposePaths {
    #[arg(long)]
    bundle_dir: PathBuf,
    #[arg(long, default_value = "/tmp")]
    socket_dir: PathBuf,
    #[arg(long)]
    checkpoint_dir: PathBuf,
}

impl CuinterposePaths {
    fn config(&self) -> cuinterpose::Config {
        cuinterpose::Config {
            bundle_dir: self.bundle_dir.clone(),
            socket_dir: self.socket_dir.clone(),
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Cuda(command) => cuda_command(command),
        Command::Cuinterpose(CuinterposeCommand::Prepare { paths, pids }) => {
            let shared =
                cuinterpose::prepare_checkpoint(&paths.config(), &paths.checkpoint_dir, &pids)?;
            println!("{shared:?}");
            Ok(())
        }
        Command::Cuinterpose(CuinterposeCommand::Restore { paths }) => {
            match cuinterpose::Manifest::load(&paths.checkpoint_dir)? {
                None => println!("no shared CUDA memory in the checkpoint"),
                Some(manifest) => {
                    manifest.verify_libraries(&paths.config())?;
                    cuinterpose::restore(&paths.config(), &paths.checkpoint_dir, &manifest)?;
                }
            }
            Ok(())
        }
        Command::Cuinterpose(CuinterposeCommand::Inspect {
            socket_dir,
            namespace_pids,
        }) => {
            let config = cuinterpose::Config {
                bundle_dir: PathBuf::new(),
                socket_dir,
            };
            for pid in namespace_pids {
                let sharing = cuinterpose::Sharing::of(&cuinterpose::inspect(&config, pid)?);
                println!("{pid}: {sharing:?}");
            }
            Ok(())
        }
    }
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
            let map = if source.is_empty() && target.is_empty() {
                GpuMap::default()
            } else {
                ensure!(
                    !visible.is_empty(),
                    "--visible is required with --source and --target"
                );
                GpuMap::new(&source, &target, &visible)?
            };
            if !map.is_identity() {
                println!("remap {map}");
            }
            cuda::restore_job(&driver, &pids, &map)?;
        }
    }
    Ok(())
}
