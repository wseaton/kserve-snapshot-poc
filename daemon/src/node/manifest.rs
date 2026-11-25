use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cuda::Gpu;
use crate::cuinterpose;
use crate::node::process::PodPid;
use crate::store::{Attestation, MANIFEST};

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub root: PodPid,
    /// The pod's GPUs, in the driver's order.
    pub gpus: Vec<Gpu>,
    pub address: IpAddr,
    /// The pod that captured the image. `None` in images captured before
    /// images were attested, which are restored as owned by no one.
    #[serde(default)]
    pub attestation: Option<Attestation>,
}

impl Manifest {
    fn path(image: &Path) -> PathBuf {
        image.join(MANIFEST)
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
    use crate::store::{Attestation, Owner};

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
            attestation: None,
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

    #[test]
    fn the_capturing_pod_round_trips_and_an_older_manifest_has_none() {
        let dir = std::env::temp_dir().join(format!("snapshotd-attested-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let attestation = Attestation {
            namespace: "serving".to_string(),
            pod: "qwen3-8b-7b9f694bf8-zw7mr".to_string(),
            uid: "5b1f0c3e-2f1d-4b8e-9a51-0c2e7d4f1a9b".to_string(),
            service_account: "qwen".to_string(),
            owner: Some(Owner {
                api_version: "apps/v1".to_string(),
                kind: "Deployment".to_string(),
                name: "qwen3-8b".to_string(),
                uid: "0f6b1c2d-6f7e-4a1b-8c9d-1e2f3a4b5c6d".to_string(),
            }),
            revision: "serving/qwen3-8b/7b9f694bf8".to_string(),
        };
        let manifest = Manifest {
            root: PodPid(57),
            gpus: gpus(&[0]),
            address: "10.0.2.165".parse().unwrap(),
            attestation: Some(attestation.clone()),
        };
        manifest.save(&dir).unwrap();
        assert_eq!(Manifest::load(&dir).unwrap(), manifest);
        assert_eq!(
            Attestation::of_image(&dir).unwrap(),
            Some(attestation),
            "the store reads the same field"
        );

        let older = r#"{"root": 57, "gpus": [], "address": "10.0.2.165"}"#;
        fs::write(dir.join("manifest.json"), older).unwrap();
        let loaded = Manifest::load(&dir).unwrap();
        assert_eq!(loaded.attestation, None);
        assert_eq!(loaded.root, PodPid(57));
        assert_eq!(Attestation::of_image(&dir).unwrap(), None);
        fs::remove_dir_all(&dir).unwrap();
    }
}
