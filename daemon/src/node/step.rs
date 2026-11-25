use std::error::Error;
use std::fmt;
use std::future::Future;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tracing::{info, warn};

use crate::node::report::seconds;

/// One step of taking or restoring a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    VllmPrepare,
    CuinterposePrepare,
    CudaCheckpoint,
    CriuDump,
    CopySharedMemoryOut,
    CopySharedMemoryIn,
    CriuRestore,
    CudaRestore,
    CuinterposeRestore,
    VllmRestore,
}

impl Step {
    /// Whether a failure here points at the checkpoint itself rather than
    /// at the pod or the server it restores into.
    pub fn implicates_image(self) -> bool {
        matches!(
            self,
            Self::CopySharedMemoryIn
                | Self::CriuRestore
                | Self::CudaRestore
                | Self::CuinterposeRestore
        )
    }
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::VllmPrepare => "vLLM prepare",
            Self::CuinterposePrepare => "cuInterpose prepare",
            Self::CudaCheckpoint => "CUDA checkpoint",
            Self::CriuDump => "CRIU dump",
            Self::CopySharedMemoryOut => "copy shared memory out",
            Self::CopySharedMemoryIn => "copy shared memory in",
            Self::CriuRestore => "CRIU restore",
            Self::CudaRestore => "CUDA restore",
            Self::CuinterposeRestore => "cuInterpose restore",
            Self::VllmRestore => "vLLM restore",
        })
    }
}

/// An error that names the step it came from. It reads as the error it
/// wraps.
#[derive(Debug)]
pub struct StepFailed {
    pub step: Step,
    error: anyhow::Error,
}

impl StepFailed {
    /// The step `error` came from, if it came from one.
    pub fn of(error: &anyhow::Error) -> Option<Step> {
        error.downcast_ref::<Self>().map(|failed| failed.step)
    }
}

impl fmt::Display for StepFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl Error for StepFailed {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.error.source()
    }
}

pub async fn run<T>(step: Step, work: impl Future<Output = Result<T>>) -> Result<T> {
    let started = Instant::now();
    match work.await {
        Ok(done) => {
            info!(seconds = %seconds(started), "{step}");
            Ok(done)
        }
        Err(error) => {
            warn!(seconds = %seconds(started), "{step} failed");
            Err(StepFailed { step, error }.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;

    use crate::node::step::{Step, StepFailed, run};

    #[test]
    fn only_steps_that_read_the_image_implicate_it() {
        let implicating = [
            Step::CopySharedMemoryIn,
            Step::CriuRestore,
            Step::CudaRestore,
            Step::CuinterposeRestore,
        ];
        for step in [
            Step::VllmPrepare,
            Step::CuinterposePrepare,
            Step::CudaCheckpoint,
            Step::CriuDump,
            Step::CopySharedMemoryOut,
            Step::CopySharedMemoryIn,
            Step::CriuRestore,
            Step::CudaRestore,
            Step::CuinterposeRestore,
            Step::VllmRestore,
        ] {
            assert_eq!(
                step.implicates_image(),
                implicating.contains(&step),
                "{step}"
            );
        }
    }

    #[test]
    fn a_step_keeps_the_name_it_is_logged_and_recorded_under() {
        for (step, logged, recorded) in [
            (Step::VllmPrepare, "vLLM prepare", "vllm_prepare"),
            (
                Step::CuinterposePrepare,
                "cuInterpose prepare",
                "cuinterpose_prepare",
            ),
            (Step::CudaCheckpoint, "CUDA checkpoint", "cuda_checkpoint"),
            (Step::CriuDump, "CRIU dump", "criu_dump"),
            (
                Step::CopySharedMemoryOut,
                "copy shared memory out",
                "copy_shared_memory_out",
            ),
            (
                Step::CopySharedMemoryIn,
                "copy shared memory in",
                "copy_shared_memory_in",
            ),
            (Step::CriuRestore, "CRIU restore", "criu_restore"),
            (Step::CudaRestore, "CUDA restore", "cuda_restore"),
            (
                Step::CuinterposeRestore,
                "cuInterpose restore",
                "cuinterpose_restore",
            ),
            (Step::VllmRestore, "vLLM restore", "vllm_restore"),
        ] {
            assert_eq!(step.to_string(), logged);
            let json = serde_json::to_value(step).unwrap();
            assert_eq!(json, recorded);
            assert_eq!(serde_json::from_value::<Step>(json).unwrap(), step);
        }
    }

    #[tokio::test]
    async fn a_failed_step_is_named_by_its_error_which_reads_as_before() {
        let failed = run(Step::CriuRestore, async {
            Err::<(), _>(anyhow!("exit status 1").context("criu restore"))
        })
        .await
        .unwrap_err();
        assert_eq!(StepFailed::of(&failed), Some(Step::CriuRestore));
        assert_eq!(format!("{failed:#}"), "criu restore: exit status 1");
        let wrapped = failed.context("the container starts over cold");
        assert_eq!(StepFailed::of(&wrapped), Some(Step::CriuRestore));
        assert_eq!(
            format!("{wrapped:#}"),
            "the container starts over cold: criu restore: exit status 1"
        );

        assert_eq!(StepFailed::of(&anyhow!("mount the image")), None);
        assert_eq!(
            run(Step::VllmRestore, async { Ok(7) }).await.unwrap(),
            7,
            "a step that works hands its result back"
        );
    }
}
