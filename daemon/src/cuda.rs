//! CUDA process checkpoint through the driver API.
//!
//! The daemon calls `cuCheckpointProcess*` on host PIDs with the host's
//! `libcuda`, instead of running the `cuda-checkpoint` CLI inside the pod.
//! A multi-process job is suspended and resumed phase by phase, every
//! process in parallel within a phase:
//!
//! ```text
//! checkpoint: lock all ─► checkpoint all
//! restore:    restore all (GPU pairs) ─► unlock all
//! ```
//!
//! Locking every process before any of them checkpoints is the order the
//! driver requires when processes share memory.
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{
    CUcheckpointGpuPair, CUcheckpointLockArgs, CUcheckpointRestoreArgs, CUprocessState, CUresult,
    CUuuid,
};
use libloading::Library;
use uuid::Uuid;

const LIBCUDA: &str = "libcuda.so.1";

type Init = unsafe extern "C" fn(u32) -> CUresult;
type GetState = unsafe extern "C" fn(i32, *mut CUprocessState) -> CUresult;
type GetRestoreThreadId = unsafe extern "C" fn(i32, *mut i32) -> CUresult;
type Lock = unsafe extern "C" fn(i32, *mut CUcheckpointLockArgs) -> CUresult;
type Checkpoint = unsafe extern "C" fn(i32, *mut libc::c_void) -> CUresult;
type Restore = unsafe extern "C" fn(i32, *mut CUcheckpointRestoreArgs) -> CUresult;
type Unlock = unsafe extern "C" fn(i32, *mut libc::c_void) -> CUresult;

/// The checkpoint entry points of the host driver.
pub struct Driver {
    get_state: GetState,
    get_restore_thread_id: GetRestoreThreadId,
    lock: Lock,
    checkpoint: Checkpoint,
    restore: Restore,
    unlock: Unlock,
    _library: Library,
}

fn check(call: &str, pid: i32, result: CUresult) -> Result<()> {
    ensure!(
        result == CUresult::CUDA_SUCCESS,
        "{call} for pid {pid}: {result:?}"
    );
    Ok(())
}

impl Driver {
    pub fn load() -> Result<Self> {
        // SAFETY: libcuda's initializers have no preconditions.
        let library =
            unsafe { Library::new(LIBCUDA) }.with_context(|| format!("load {LIBCUDA}"))?;
        // SAFETY: each symbol is resolved with the signature cuda.h declares for it.
        unsafe {
            let init = *library.get::<Init>(b"cuInit\0")?;
            let driver = Self {
                get_state: *library.get(b"cuCheckpointProcessGetState\0")?,
                get_restore_thread_id: *library.get(b"cuCheckpointProcessGetRestoreThreadId\0")?,
                lock: *library.get(b"cuCheckpointProcessLock\0")?,
                checkpoint: *library.get(b"cuCheckpointProcessCheckpoint\0")?,
                restore: *library.get(b"cuCheckpointProcessRestore\0")?,
                unlock: *library.get(b"cuCheckpointProcessUnlock\0")?,
                _library: library,
            };
            check("cuInit", 0, init(0))?;
            Ok(driver)
        }
    }

    pub fn state(&self, pid: i32) -> Result<CUprocessState> {
        let mut state = CUprocessState::CU_PROCESS_STATE_FAILED;
        // SAFETY: `state` is a valid out pointer.
        check("cuCheckpointProcessGetState", pid, unsafe {
            (self.get_state)(pid, &mut state)
        })?;
        Ok(state)
    }

    /// Whether the driver tracks `pid` as a CUDA process it can checkpoint.
    pub fn is_cuda_process(&self, pid: i32) -> bool {
        let mut tid = 0;
        // SAFETY: `tid` is a valid out pointer.
        let result = unsafe { (self.get_restore_thread_id)(pid, &mut tid) };
        result == CUresult::CUDA_SUCCESS
    }

    fn lock(&self, pid: i32, timeout: Duration) -> Result<()> {
        // SAFETY: the reserved fields must be zero.
        let mut args: CUcheckpointLockArgs = unsafe { std::mem::zeroed() };
        args.timeoutMs = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
        // SAFETY: `args` outlives the call.
        check("cuCheckpointProcessLock", pid, unsafe {
            (self.lock)(pid, &mut args)
        })
    }

    fn checkpoint(&self, pid: i32) -> Result<()> {
        // SAFETY: a null argument selects the defaults.
        check("cuCheckpointProcessCheckpoint", pid, unsafe {
            (self.checkpoint)(pid, std::ptr::null_mut())
        })
    }

    fn restore(&self, pid: i32, map: &GpuMap) -> Result<()> {
        let mut pairs: Vec<CUcheckpointGpuPair> = map.pairs().collect();
        // SAFETY: the reserved fields must be zero.
        let mut args: CUcheckpointRestoreArgs = unsafe { std::mem::zeroed() };
        let args_ptr = if pairs.is_empty() {
            std::ptr::null_mut()
        } else {
            args.gpuPairs = pairs.as_mut_ptr();
            args.gpuPairsCount = u32::try_from(pairs.len()).context("too many GPU pairs")?;
            &mut args as *mut CUcheckpointRestoreArgs
        };
        // SAFETY: `args` and `pairs` outlive the call.
        check("cuCheckpointProcessRestore", pid, unsafe {
            (self.restore)(pid, args_ptr)
        })
    }

    fn unlock(&self, pid: i32) -> Result<()> {
        // SAFETY: a null argument selects the defaults.
        check("cuCheckpointProcessUnlock", pid, unsafe {
            (self.unlock)(pid, std::ptr::null_mut())
        })
    }
}

/// Run `op` for every pid at once and report every failure.
fn each(pids: &[i32], op: impl Fn(i32) -> Result<()> + Sync) -> Result<()> {
    let failures: Vec<String> = std::thread::scope(|scope| {
        let jobs: Vec<_> = pids
            .iter()
            .map(|&pid| {
                let op = &op;
                scope.spawn(move || op(pid))
            })
            .collect();
        jobs.into_iter()
            .filter_map(|job| match job.join() {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("{error:#}")),
                Err(_) => Some("checkpoint thread panicked".to_string()),
            })
            .collect()
    });
    ensure!(failures.is_empty(), "{}", failures.join("; "));
    Ok(())
}

fn timed(phase: &str, run: impl FnOnce() -> Result<()>) -> Result<()> {
    let start = Instant::now();
    run()?;
    println!("PHASE cuda_{phase} {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

/// Lock every process, then checkpoint every process.
pub fn checkpoint_job(driver: &Driver, pids: &[i32], lock_timeout: Duration) -> Result<()> {
    timed("lock", || each(pids, |pid| driver.lock(pid, lock_timeout)))?;
    timed("checkpoint", || each(pids, |pid| driver.checkpoint(pid)))
}

/// Restore every process onto the GPUs `map` names, then unlock every process.
pub fn restore_job(driver: &Driver, pids: &[i32], map: &GpuMap) -> Result<()> {
    timed("restore", || each(pids, |pid| driver.restore(pid, map)))?;
    timed("unlock", || each(pids, |pid| driver.unlock(pid)))
}

/// Every process in the tree rooted at `root`. vLLM forks workers from
/// non-main threads, so children are read from every thread.
pub fn process_tree(root: i32) -> Vec<i32> {
    let mut pids = vec![root];
    let mut next = 0;
    while let Some(&pid) = pids.get(next) {
        next += 1;
        let Ok(tasks) = fs::read_dir(format!("/proc/{pid}/task")) else {
            continue;
        };
        for task in tasks.flatten() {
            let Ok(children) = fs::read_to_string(task.path().join("children")) else {
                continue;
            };
            pids.extend(
                children
                    .split_ascii_whitespace()
                    .filter_map(|c| c.parse::<i32>().ok()),
            );
        }
    }
    pids
}

/// A GPU UUID as `nvidia-smi` and `NVIDIA_VISIBLE_DEVICES` print it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuUuid(Uuid);

impl FromStr for GpuUuid {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        let raw = text
            .strip_prefix("GPU-")
            .with_context(|| format!("{text:?} is not a GPU UUID"))?;
        Ok(Self(
            Uuid::parse_str(raw).with_context(|| format!("{text:?}"))?,
        ))
    }
}

impl fmt::Display for GpuUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU-{}", self.0.hyphenated())
    }
}

impl GpuUuid {
    fn cuuuid(self) -> CUuuid {
        CUuuid {
            bytes: self.0.into_bytes().map(|b| b as libc::c_char),
        }
    }
}

/// Where each GPU a checkpoint used is restored. The R580 driver needs a
/// bijection over every GPU the processes can see, so the map is complete
/// and one-to-one or empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuMap(Vec<(GpuUuid, GpuUuid)>);

impl GpuMap {
    /// Map `source[i]` to `target[i]` and complete the bijection over
    /// `visible` with the displaced GPUs, keeping every other GPU in place.
    pub fn new(source: &[GpuUuid], target: &[GpuUuid], visible: &[GpuUuid]) -> Result<Self> {
        ensure!(
            source.len() == target.len(),
            "{} source GPUs but {} target GPUs",
            source.len(),
            target.len()
        );
        let visible_set: BTreeSet<_> = visible.iter().copied().collect();
        for gpu in source.iter().chain(target) {
            ensure!(visible_set.contains(gpu), "{gpu} is not visible");
        }
        ensure!(
            source.iter().collect::<BTreeSet<_>>().len() == source.len()
                && target.iter().collect::<BTreeSet<_>>().len() == target.len(),
            "duplicate GPU in the source or target list"
        );
        if source == target {
            return Ok(Self::default());
        }
        let mut pairs: Vec<(GpuUuid, GpuUuid)> =
            source.iter().copied().zip(target.iter().copied()).collect();
        let (unmoved, displaced): (Vec<GpuUuid>, Vec<GpuUuid>) = visible_set
            .iter()
            .filter(|gpu| !source.contains(gpu))
            .partition(|gpu| !target.contains(gpu));
        let freed = visible_set
            .iter()
            .filter(|gpu| source.contains(gpu) && !target.contains(gpu));
        pairs.extend(displaced.into_iter().zip(freed.copied()));
        pairs.extend(unmoved.into_iter().map(|gpu| (gpu, gpu)));
        pairs.sort();
        Ok(Self(pairs))
    }

    fn pairs(&self) -> impl Iterator<Item = CUcheckpointGpuPair> + '_ {
        self.0.iter().map(|&(old, new)| CUcheckpointGpuPair {
            oldUuid: old.cuuuid(),
            newUuid: new.cuuuid(),
        })
    }

    pub fn is_identity(&self) -> bool {
        self.0.iter().all(|(old, new)| old == new)
    }
}

impl fmt::Display for GpuMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pairs: Vec<String> = self
            .0
            .iter()
            .filter(|(old, new)| old != new)
            .map(|(old, new)| format!("{old}->{new}"))
            .collect();
        write!(f, "{}", pairs.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda::{GpuMap, GpuUuid, process_tree};

    fn gpus(n: usize) -> Vec<GpuUuid> {
        (0..n)
            .map(|i| {
                format!("GPU-{i:08x}-8867-7cb7-7ffd-202baf8a3566")
                    .parse()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    #[allow(
        clippy::unnecessary_cast,
        reason = "c_char is i8 on x86_64 and u8 on aarch64"
    )]
    fn gpu_uuid_round_trips_nvidia_smi_form() {
        let text = "GPU-461036bd-8867-7cb7-7ffd-202baf8a3566";
        let gpu: GpuUuid = text.parse().unwrap();
        assert_eq!(gpu.to_string(), text);
        assert_eq!(
            gpu.cuuuid().bytes.map(|b| b as u8)[..4],
            [0x46, 0x10, 0x36, 0xbd]
        );
        assert!(
            "461036bd-8867-7cb7-7ffd-202baf8a3566"
                .parse::<GpuUuid>()
                .is_err()
        );
        assert!("GPU-nothex".parse::<GpuUuid>().is_err());
    }

    #[test]
    fn same_gpus_need_no_map() {
        let all = gpus(8);
        let map = GpuMap::new(&all[0..2], &all[0..2], &all).unwrap();
        assert!(map.is_identity());
        assert_eq!(map.pairs().count(), 0);
    }

    #[test]
    fn disjoint_move_swaps_with_the_displaced_gpus() {
        let all = gpus(8);
        let map = GpuMap::new(&all[0..2], &all[2..4], &all).unwrap();
        assert_eq!(map.pairs().count(), 8);
        let moved: Vec<_> = map.0.iter().filter(|(o, n)| o != n).copied().collect();
        assert_eq!(
            moved,
            vec![
                (all[0], all[2]),
                (all[1], all[3]),
                (all[2], all[0]),
                (all[3], all[1])
            ]
        );
        assert_eq!(map.to_string().matches("->").count(), 4);
    }

    #[test]
    fn overlapping_move_stays_a_bijection() {
        let all = gpus(4);
        let map = GpuMap::new(&all[0..2], &all[1..3], &all).unwrap();
        let olds: std::collections::BTreeSet<_> = map.0.iter().map(|p| p.0).collect();
        let news: std::collections::BTreeSet<_> = map.0.iter().map(|p| p.1).collect();
        assert_eq!(olds.len(), 4);
        assert_eq!(news.len(), 4);
        assert!(map.0.contains(&(all[0], all[1])));
        assert!(map.0.contains(&(all[1], all[2])));
        assert!(map.0.contains(&(all[2], all[0])));
        assert!(map.0.contains(&(all[3], all[3])));
    }

    #[test]
    fn rejects_bad_selections() {
        let all = gpus(4);
        let other = gpus(5);
        assert!(GpuMap::new(&all[0..2], &all[2..3], &all).is_err());
        assert!(GpuMap::new(&all[0..1], &other[4..5], &all).is_err());
        assert!(GpuMap::new(&[all[0], all[0]], &all[1..3], &all).is_err());
    }

    #[test]
    fn process_tree_finds_children_of_this_process() {
        let mut child = std::process::Command::new("sleep")
            .arg("5")
            .spawn()
            .unwrap();
        let child_pid = child.id() as i32;
        // The children files are unreliable while this process's own threads
        // come and go, which the parallel test harness does.
        let tree = (0..100)
            .map(|_| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                process_tree(std::process::id() as i32)
            })
            .find(|tree| tree.contains(&child_pid))
            .unwrap_or_default();
        child.kill().unwrap();
        child.wait().unwrap();
        assert_eq!(tree.first(), Some(&(std::process::id() as i32)));
        assert!(tree.contains(&child_pid), "{tree:?}");
    }
}
