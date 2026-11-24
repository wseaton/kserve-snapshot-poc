//! The node agent API of llm-d-rl-time-slicing, served by this daemon, so
//! its orchestrator can take a job's GPUs away and give them back.
//!
//! ```text
//! orchestrator ── Status ─────────► every labelled job here: idle, running, saved
//!              ── Snapshot(job) ──► operation: CUDA lock and checkpoint,
//!                                   every process of the job's pods at once
//!              ── Restore(job) ───► operation: CUDA restore and unlock
//!              ── GetOperation ───► pending | complete | failed
//! ```
//!
//! A snapshot leaves the processes alive with their device memory in host
//! memory. A job's state is read from the driver and NVML each time it is
//! asked for: running while a process of it holds device memory, saved
//! while its processes are checkpointed, idle otherwise.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::UsedGpuMemory;
use tonic::{Request, Response, Status};
use tracing::{info, warn};
use uuid::Uuid;

use crate::cri::Runtime;
use crate::cuda::{self, Driver, GpuMap, Phase};
use crate::metrics::{Action, Metrics, Outcome};

#[allow(clippy::all)]
pub mod pb {
    tonic::include_proto!("snapshot_agent.v1alpha1");
}

use pb::backend_config::Backend;
use pb::snapshot_agent_service_server::{SnapshotAgentService, SnapshotAgentServiceServer};
use pb::{JobState, OperationStatus};

#[derive(Clone, clap::Args)]
#[group(id = "timeslice")]
pub struct Args {
    /// Address to serve the time-slicing snapshot agent API on. Not served
    /// when unset.
    #[arg(long, env = "SNAPSHOTD_TIMESLICE_LISTEN")]
    timeslice_listen: Option<SocketAddr>,
    /// Pod label whose value is a time-sliced job's id.
    #[arg(
        long,
        env = "SNAPSHOTD_TIMESLICE_JOB_LABEL",
        default_value = "timeslice.io/job-id"
    )]
    timeslice_job_label: String,
    /// Seconds a job's processes may take to reach a point where the driver
    /// can checkpoint them.
    #[arg(long, env = "SNAPSHOTD_TIMESLICE_LOCK_TIMEOUT_S", default_value_t = 60)]
    timeslice_lock_timeout_s: u64,
    /// Finished operations kept for GetOperation.
    #[arg(long, env = "SNAPSHOTD_TIMESLICE_OPERATIONS", default_value_t = 1024)]
    timeslice_operations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Process {
    pid: i32,
    phase: Phase,
    device_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Seen {
    Idle,
    /// The processes that hold device memory.
    Running(Vec<i32>),
    /// The processes that are checkpointed.
    Saved(Vec<i32>),
    /// Paused in part, or a process is stuck between the two.
    Torn,
}

impl Seen {
    /// A process that has loaded CUDA and holds no device memory counts for
    /// nothing, as it does not for the upstream agent.
    fn of(processes: &[Process]) -> Self {
        let pids = |wanted: fn(&Process) -> bool| -> Vec<i32> {
            processes
                .iter()
                .filter(|process| wanted(process))
                .map(|process| process.pid)
                .collect()
        };
        let stuck = pids(|process| matches!(process.phase, Phase::Locked | Phase::Failed));
        let paused = pids(|process| process.phase == Phase::Checkpointed);
        let holding = pids(|process| process.phase == Phase::Running && process.device_bytes > 0);
        match (stuck.is_empty(), paused.is_empty(), holding.is_empty()) {
            (false, _, _) | (true, false, false) => Self::Torn,
            (true, false, true) => Self::Saved(paused),
            (true, true, false) => Self::Running(holding),
            (true, true, true) => Self::Idle,
        }
    }

    fn state(&self) -> JobState {
        match self {
            Self::Idle => JobState::Idle,
            Self::Running(_) => JobState::Running,
            Self::Saved(_) => JobState::Saved,
            Self::Torn => JobState::Faulted,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Step {
    Nothing,
    Pause(Vec<i32>),
    Resume(Vec<i32>),
}

fn step(action: Action, seen: Seen) -> Result<Step> {
    Ok(match (action, seen) {
        (_, Seen::Torn) => bail!("some of the job's processes are paused and some are not"),
        (Action::Snapshot, Seen::Running(pids)) => Step::Pause(pids),
        (Action::Restore, Seen::Saved(pids)) => Step::Resume(pids),
        (Action::Snapshot, Seen::Idle | Seen::Saved(_))
        | (Action::Restore, Seen::Idle | Seen::Running(_)) => Step::Nothing,
    })
}

struct Operation {
    started: Instant,
    /// Milliseconds it took and its error, once it has finished.
    finished: Option<(i64, Option<String>)>,
}

#[derive(Default)]
struct Book {
    busy: HashSet<String>,
    operations: HashMap<String, Operation>,
    finished: Vec<String>,
}

struct Agent {
    args: Args,
    runtime_socket: PathBuf,
    nvml: Arc<Nvml>,
    metrics: Arc<Metrics>,
    book: Mutex<Book>,
}

/// Device memory by host PID, over every GPU of the node.
fn device_bytes(nvml: &Nvml) -> Result<HashMap<u32, u64>> {
    let mut bytes: HashMap<u32, u64> = HashMap::new();
    for index in 0..nvml.device_count().context("count the GPUs")? {
        let device = nvml.device_by_index(index).context("open a GPU")?;
        let processes = device
            .running_compute_processes()
            .context("list a GPU's processes")?;
        for process in processes {
            if let UsedGpuMemory::Used(used) = process.used_gpu_memory {
                *bytes.entry(process.pid).or_default() += used;
            }
        }
    }
    Ok(bytes)
}

fn accelerators(nvml: &Nvml) -> Result<Vec<pb::AcceleratorStatus>> {
    (0..nvml.device_count().context("count the GPUs")?)
        .map(|index| {
            let device = nvml.device_by_index(index).context("open a GPU")?;
            let memory = device.memory_info().context("read a GPU's memory")?;
            Ok(pb::AcceleratorStatus {
                id: device.uuid().context("read a GPU's UUID")?,
                memory_used_bytes: i64::try_from(memory.used).unwrap_or(i64::MAX),
                memory_total_bytes: i64::try_from(memory.total).unwrap_or(i64::MAX),
            })
        })
        .collect()
}

fn milliseconds(since: Instant) -> i64 {
    i64::try_from(since.elapsed().as_millis()).unwrap_or(i64::MAX)
}

impl Agent {
    fn book(&self) -> Result<MutexGuard<'_, Book>, Status> {
        self.book
            .lock()
            .map_err(|_| Status::internal("the agent's state is poisoned"))
    }

    async fn jobs(&self) -> Result<BTreeMap<String, Seen>> {
        let label = &self.args.timeslice_job_label;
        let mut runtime = Runtime::connect(&self.runtime_socket).await?;
        let mut roots: BTreeMap<String, Vec<i32>> = BTreeMap::new();
        for sandbox in runtime.sandboxes(label).await? {
            let Some(job) = sandbox.labels.get(label) else {
                continue;
            };
            roots
                .entry(job.clone())
                .or_default()
                .extend(runtime.pids(&sandbox).await?);
        }
        let nvml = Arc::clone(&self.nvml);
        tokio::task::spawn_blocking(move || {
            let driver = Driver::load()?;
            let held = device_bytes(&nvml)?;
            roots
                .into_iter()
                .map(|(job, roots)| {
                    let processes = roots
                        .into_iter()
                        .flat_map(cuda::process_tree)
                        .filter(|&pid| driver.is_cuda_process(pid))
                        .map(|pid| {
                            Ok(Process {
                                pid,
                                phase: driver.phase(pid)?,
                                device_bytes: u32::try_from(pid)
                                    .ok()
                                    .and_then(|pid| held.get(&pid).copied())
                                    .unwrap_or(0),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok((job, Seen::of(&processes)))
                })
                .collect()
        })
        .await
        .context("the process search panicked")?
    }

    async fn seen(&self, job: &str) -> Result<Seen> {
        Ok(self.jobs().await?.remove(job).unwrap_or(Seen::Idle))
    }

    async fn run(&self, action: Action, job: &str) -> Result<()> {
        let lock = Duration::from_secs(self.args.timeslice_lock_timeout_s);
        let step = step(action, self.seen(job).await?)?;
        let pausing = matches!(step, Step::Pause(_));
        let done = tokio::task::spawn_blocking(move || match step {
            Step::Nothing => Ok(()),
            Step::Pause(pids) => cuda::checkpoint_job(&Driver::load()?, &pids, lock),
            Step::Resume(pids) => cuda::restore_job(&Driver::load()?, &pids, &GpuMap::default()),
        })
        .await
        .context("the driver call panicked")?;
        match done {
            Err(error) if pausing && self.seen(job).await? == Seen::Idle => {
                info!(job, error = %format_args!("{error:#}"), "the job exited while it was being paused");
                Ok(())
            }
            done => done,
        }
    }

    fn begin(self: &Arc<Self>, action: Action, job: String) -> Result<String, Status> {
        let id = Uuid::new_v4().to_string();
        let mut book = self.book()?;
        if !book.busy.insert(job.clone()) {
            return Err(Status::aborted(format!(
                "another operation on job {job} is running"
            )));
        }
        book.operations.insert(
            id.clone(),
            Operation {
                started: Instant::now(),
                finished: None,
            },
        );
        drop(book);
        let (agent, operation) = (Arc::clone(self), id.clone());
        tokio::spawn(async move {
            let outcome = agent.run(action, &job).await;
            agent.finish(action, &job, &operation, outcome);
        });
        Ok(id)
    }

    fn finish(&self, action: Action, job: &str, operation: &str, outcome: Result<()>) {
        let Ok(mut book) = self.book.lock() else {
            return;
        };
        book.busy.remove(job);
        let Some(entry) = book.operations.get_mut(operation) else {
            return;
        };
        let ms = milliseconds(entry.started);
        let error = outcome.err().map(|error| format!("{error:#}"));
        let how = if error.is_none() {
            Outcome::Complete
        } else {
            Outcome::Failed
        };
        self.metrics.operated(action, how, entry.started.elapsed());
        match &error {
            None => info!(job, ?action, ms, "time-slicing operation done"),
            Some(error) => warn!(job, ?action, ms, error, "time-slicing operation failed"),
        }
        entry.finished = Some((ms, error));
        book.finished.push(operation.to_string());
        while book.finished.len() > self.args.timeslice_operations {
            let oldest = book.finished.remove(0);
            book.operations.remove(&oldest);
        }
    }
}

fn supported(config: Option<&pb::BackendConfig>) -> Result<(), Status> {
    match config.and_then(|config| config.backend.as_ref()) {
        None => Ok(()),
        Some(Backend::Cuda(cuda)) if cuda.explicit_target.is_none() => Ok(()),
        Some(Backend::Cuda(_)) => Err(Status::invalid_argument(
            "explicit PIDs are not taken; the job's pods are found by their label",
        )),
        Some(_) => Err(Status::unimplemented(
            "this agent offers the cuda backend only",
        )),
    }
}

fn named(job_id: String) -> Result<String, Status> {
    if job_id.is_empty() {
        return Err(Status::invalid_argument("job_id is required"));
    }
    Ok(job_id)
}

struct Service(Arc<Agent>);

#[tonic::async_trait]
impl SnapshotAgentService for Service {
    async fn snapshot(
        &self,
        request: Request<pb::SnapshotRequest>,
    ) -> Result<Response<pb::SnapshotResponse>, Status> {
        let request = request.into_inner();
        supported(request.backend_config.as_ref())?;
        let operation_id = self.0.begin(Action::Snapshot, named(request.job_id)?)?;
        Ok(Response::new(pb::SnapshotResponse { operation_id }))
    }

    async fn restore(
        &self,
        request: Request<pb::RestoreRequest>,
    ) -> Result<Response<pb::RestoreResponse>, Status> {
        let request = request.into_inner();
        supported(request.backend_config.as_ref())?;
        let operation_id = self.0.begin(Action::Restore, named(request.job_id)?)?;
        Ok(Response::new(pb::RestoreResponse { operation_id }))
    }

    async fn get_operation(
        &self,
        request: Request<pb::GetOperationRequest>,
    ) -> Result<Response<pb::GetOperationResponse>, Status> {
        let id = request.into_inner().operation_id;
        let book = self.0.book()?;
        let operation = book
            .operations
            .get(&id)
            .ok_or_else(|| Status::not_found(format!("no operation {id}")))?;
        let (status, elapsed_ms, error) = match &operation.finished {
            None => (
                OperationStatus::Pending,
                milliseconds(operation.started),
                None,
            ),
            Some((ms, None)) => (OperationStatus::Complete, *ms, None),
            Some((ms, error)) => (OperationStatus::Failed, *ms, error.clone()),
        };
        Ok(Response::new(pb::GetOperationResponse {
            status: status.into(),
            elapsed_ms,
            error,
            ..Default::default()
        }))
    }

    async fn status(
        &self,
        _: Request<pb::StatusRequest>,
    ) -> Result<Response<pb::StatusResponse>, Status> {
        let jobs = self
            .0
            .jobs()
            .await
            .map_err(|error| Status::unavailable(format!("{error:#}")))?;
        let nvml = Arc::clone(&self.0.nvml);
        let accelerator_statuses = tokio::task::spawn_blocking(move || accelerators(&nvml))
            .await
            .map_err(|_| Status::internal("reading the GPUs panicked"))?
            .map_err(|error| Status::unavailable(format!("{error:#}")))?;
        let book = self.0.book()?;
        let job_statuses = jobs
            .into_iter()
            .map(|(job_id, seen)| {
                let state = if book.busy.contains(&job_id) {
                    JobState::Transitioning
                } else {
                    seen.state()
                };
                pb::JobStatus {
                    job_id,
                    state: state.into(),
                }
            })
            .collect();
        Ok(Response::new(pb::StatusResponse {
            job_statuses,
            accelerator_statuses,
        }))
    }

    type WorkloadChannelStream = tonic::codegen::BoxStream<pb::AgentCommand>;

    async fn workload_channel(
        &self,
        _: Request<tonic::Streaming<pb::WorkloadMessage>>,
    ) -> Result<Response<Self::WorkloadChannelStream>, Status> {
        Err(Status::unimplemented(
            "this agent offers the cuda backend only",
        ))
    }
}

pub fn start(args: &Args, runtime_socket: &Path, metrics: Arc<Metrics>) -> Result<()> {
    let Some(address) = args.timeslice_listen else {
        return Ok(());
    };
    let agent = Arc::new(Agent {
        args: args.clone(),
        runtime_socket: runtime_socket.to_path_buf(),
        nvml: Arc::new(cuda::nvml()?),
        metrics,
        book: Mutex::default(),
    });
    info!(%address, label = %args.timeslice_job_label, "serving the time-slicing agent API");
    tokio::spawn(async move {
        let (reporter, health) = tonic_health::server::health_reporter();
        reporter
            .set_serving::<SnapshotAgentServiceServer<Service>>()
            .await;
        let served = tonic::transport::Server::builder()
            .add_service(health)
            .add_service(SnapshotAgentServiceServer::new(Service(agent)))
            .serve(address)
            .await;
        if let Err(error) = served.map_err(|error| anyhow!(error)) {
            warn!(error = %format_args!("{error:#}"), "the time-slicing agent API stopped");
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::cuda::Phase;
    use crate::timeslice::{Process, Seen};

    const GIB: u64 = 1 << 30;

    fn process(pid: i32, phase: Phase, device_bytes: u64) -> Process {
        Process {
            pid,
            phase,
            device_bytes,
        }
    }

    #[test]
    fn a_job_runs_while_it_holds_device_memory_and_is_saved_while_paused() {
        let loaded = process(5, Phase::Running, 0);
        assert_eq!(Seen::of(&[]), Seen::Idle);
        assert_eq!(Seen::of(&[loaded]), Seen::Idle);
        assert_eq!(
            Seen::of(&[
                loaded,
                process(7, Phase::Running, GIB),
                process(9, Phase::Running, 1)
            ]),
            Seen::Running(vec![7, 9])
        );
        assert_eq!(
            Seen::of(&[
                loaded,
                process(7, Phase::Checkpointed, 0),
                process(9, Phase::Checkpointed, 0)
            ]),
            Seen::Saved(vec![7, 9])
        );
        assert_eq!(
            Seen::of(&[
                process(7, Phase::Running, GIB),
                process(9, Phase::Checkpointed, 0)
            ]),
            Seen::Torn
        );
        for stuck in [Phase::Locked, Phase::Failed] {
            assert_eq!(Seen::of(&[process(7, stuck, 0)]), Seen::Torn);
            assert_eq!(
                Seen::of(&[process(7, Phase::Running, GIB), process(9, stuck, 0)]),
                Seen::Torn
            );
        }
    }
}
