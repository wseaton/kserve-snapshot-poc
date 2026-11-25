use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::cuda::Gpu;
use crate::cuinterpose;
use crate::node::environment::Environment;
use crate::node::process::PodPid;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub root: PodPid,
    /// The pod's GPUs, in the driver's order.
    pub gpus: Vec<Gpu>,
    pub address: IpAddr,
    /// The node's environment when the checkpoint was taken; `None` in a
    /// manifest written before it was recorded.
    #[serde(default)]
    pub environment: Option<Environment>,
}

/// Why a checkpoint cannot be restored into a pod.
#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    /// This pod cannot take it; another pod of the revision may.
    Pod(String),
    /// No pod on this node can take it.
    Node(String),
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

    pub fn refusal(
        &self,
        image: &Path,
        gpus: &[Gpu],
        node: &Environment,
    ) -> Result<Option<Refusal>> {
        if let Some(reason) = node.incompatibility(self.environment.as_ref()) {
            return Ok(Some(Refusal::Node(reason)));
        }
        if self.gpus.len() != gpus.len() {
            return Ok(Some(Refusal::Pod(format!(
                "it ran on {} GPUs and the pod has {}",
                self.gpus.len(),
                gpus.len()
            ))));
        }
        let minors = |gpus: &[Gpu]| gpus.iter().map(|gpu| gpu.minor).collect::<Vec<_>>();
        let (taken_on, offered) = (minors(&self.gpus), minors(gpus));
        if taken_on != offered && cuinterpose::Manifest::load(&image.join("img"))?.is_some() {
            return Ok(Some(Refusal::Pod(format!(
                "its processes share GPU memory and it was taken on GPUs {taken_on:?}; \
                 the pod has {offered:?}"
            ))));
        }
        for drift in self
            .environment
            .as_ref()
            .map(|taken| node.drift(taken))
            .unwrap_or_default()
        {
            warn!(drift, "restoring a checkpoint taken in another environment");
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::cuda::Gpu;
    use crate::cuinterpose;
    use crate::node::environment::tests::{environment, with_driver, with_kernel};
    use crate::node::manifest::{Manifest, Refusal};
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

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-manifest-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("img")).unwrap();
        dir
    }

    fn manifest(minors: &[u32]) -> Manifest {
        Manifest {
            root: PodPid(57),
            gpus: gpus(minors),
            address: "10.0.2.165".parse().unwrap(),
            environment: Some(environment()),
        }
    }

    fn pod_refusal(refusal: Option<Refusal>) -> String {
        match refusal {
            Some(Refusal::Pod(reason)) => reason,
            other => panic!("wanted a pod refusal, got {other:?}"),
        }
    }

    #[test]
    fn shared_gpu_memory_ties_a_checkpoint_to_its_device_minors() {
        let dir = scratch("minors");
        let node = environment();
        let manifest = manifest(&[4, 5]);
        manifest.save(&dir).unwrap();
        let manifest = Manifest::load(&dir).unwrap();
        assert_eq!(manifest.refusal(&dir, &gpus(&[4, 5]), &node).unwrap(), None);
        assert_eq!(manifest.refusal(&dir, &gpus(&[6, 7]), &node).unwrap(), None);
        let fewer = pod_refusal(manifest.refusal(&dir, &gpus(&[4]), &node).unwrap());
        assert!(fewer.contains("2 GPUs"), "{fewer}");

        cuinterpose::Manifest {
            libraries: Default::default(),
            pids: vec![675, 676],
        }
        .save(&dir.join("img"))
        .unwrap();
        assert_eq!(manifest.refusal(&dir, &gpus(&[4, 5]), &node).unwrap(), None);
        let moved = pod_refusal(manifest.refusal(&dir, &gpus(&[6, 7]), &node).unwrap());
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
    fn a_manifest_round_trips_with_its_environment() {
        let dir = scratch("round-trip");
        let manifest = manifest(&[0, 1]);
        manifest.save(&dir).unwrap();
        let text = fs::read_to_string(dir.join("manifest.json")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["environment"]["nvidia"]["driver"], "580.65.06");
        assert_eq!(Manifest::load(&dir).unwrap(), manifest);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_manifest_from_before_environments_loads_and_is_refused_for_the_node() {
        let dir = scratch("legacy");
        fs::write(
            dir.join("manifest.json"),
            r#"{
                "root": 57,
                "gpus": [{"uuid": "GPU-00000004-8867-7cb7-7ffd-202baf8a3566", "minor": 4}],
                "address": "10.0.2.165"
            }"#,
        )
        .unwrap();
        let legacy = Manifest::load(&dir).unwrap();
        assert_eq!(legacy.environment, None);
        assert_eq!(legacy.gpus, gpus(&[4]));
        assert_eq!(
            legacy.refusal(&dir, &gpus(&[4]), &environment()).unwrap(),
            Some(Refusal::Node("it records no environment".to_string()))
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_node_refusal_comes_before_a_pod_refusal() {
        let dir = scratch("driver");
        let manifest = manifest(&[4, 5]);
        let upgraded = with_driver("580.82.07");
        assert_eq!(
            manifest.refusal(&dir, &gpus(&[4, 5]), &upgraded).unwrap(),
            Some(Refusal::Node(
                "it was taken under NVIDIA driver 580.65.06 and the node runs 580.82.07"
                    .to_string()
            ))
        );
        assert!(matches!(
            manifest.refusal(&dir, &gpus(&[4]), &upgraded).unwrap(),
            Some(Refusal::Node(_))
        ));
        assert_eq!(
            manifest
                .refusal(&dir, &gpus(&[4, 5]), &with_kernel("6.12.0-55.el10.x86_64"))
                .unwrap(),
            None,
            "a kernel change alone does not refuse"
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
