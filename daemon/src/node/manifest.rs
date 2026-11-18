//! What a checkpoint records about the pod it was taken from.
use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cuda::Gpu;
use crate::cuinterpose;
use crate::node::process::PodPid;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The server's process.
    pub root: PodPid,
    /// The pod's GPUs, in the driver's order.
    pub gpus: Vec<Gpu>,
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

    /// Why this checkpoint, kept in `image`, cannot be restored onto `gpus`,
    /// if it cannot. Processes that share GPU memory can import it again
    /// only on devices they had initialized when they were checkpointed, so
    /// such a checkpoint needs GPUs with the same device minors.
    pub fn refusal(&self, image: &Path, gpus: &[Gpu]) -> Result<Option<String>> {
        if self.gpus.len() != gpus.len() {
            return Ok(Some(format!(
                "it ran on {} GPUs and the pod has {}",
                self.gpus.len(),
                gpus.len()
            )));
        }
        let minors = |gpus: &[Gpu]| gpus.iter().map(|gpu| gpu.minor).collect::<Vec<_>>();
        let (taken_on, offered) = (minors(&self.gpus), minors(gpus));
        if taken_on != offered && cuinterpose::Manifest::load(&image.join("img"))?.is_some() {
            return Ok(Some(format!(
                "its processes share GPU memory and it was taken on GPUs {taken_on:?}; \
                 the pod has {offered:?}"
            )));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::cuda::Gpu;
    use crate::cuinterpose;
    use crate::node::manifest::Manifest;
    use crate::node::process::PodPid;

    fn gpus(minors: &[u32]) -> Vec<Gpu> {
        minors
            .iter()
            .map(|&minor| Gpu {
                uuid: format!("GPU-{minor:08x}-8867-7cb7-7ffd-202baf8a3566")
                    .parse()
                    .unwrap(),
                minor,
            })
            .collect()
    }

    #[test]
    fn shared_gpu_memory_ties_a_checkpoint_to_its_device_minors() {
        let dir = std::env::temp_dir().join(format!("snapshotd-manifest-{}", std::process::id()));
        fs::create_dir_all(dir.join("img")).unwrap();
        let manifest = Manifest {
            root: PodPid(57),
            gpus: gpus(&[4, 5]),
            address: "10.0.2.165".parse().unwrap(),
        };
        manifest.save(&dir).unwrap();
        let manifest = Manifest::load(&dir).unwrap();
        assert_eq!(manifest.refusal(&dir, &gpus(&[4, 5])).unwrap(), None);
        assert_eq!(manifest.refusal(&dir, &gpus(&[6, 7])).unwrap(), None);
        let fewer = manifest.refusal(&dir, &gpus(&[4])).unwrap().unwrap();
        assert!(fewer.contains("2 GPUs"), "{fewer}");

        cuinterpose::Manifest {
            libraries: Default::default(),
            pids: vec![675, 676],
        }
        .save(&dir.join("img"))
        .unwrap();
        assert_eq!(manifest.refusal(&dir, &gpus(&[4, 5])).unwrap(), None);
        let moved = manifest.refusal(&dir, &gpus(&[6, 7])).unwrap().unwrap();
        assert!(
            moved.contains("[4, 5]") && moved.contains("[6, 7]"),
            "{moved}"
        );

        fs::write(dir.join("manifest.json"), "{").unwrap();
        assert!(Manifest::load(&dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
        assert!(Manifest::load(&dir).is_err());
    }
}
