#[cfg(any(test, feature = "nixl"))]
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tracing::warn;
use uuid::Uuid;

use crate::cuda::Gpu;
use crate::cuinterpose;
use crate::node::environment::Environment;
use crate::node::process::PodPid;

/// Names one checkpoint, drawn when it is taken. A recapture of the same
/// revision gets a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageId(Uuid);

impl ImageId {
    pub fn draw() -> Self {
        Self(Uuid::new_v4())
    }
}

impl FromStr for ImageId {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        Ok(Self(
            Uuid::parse_str(text).with_context(|| format!("{text:?} is not an image id"))?,
        ))
    }
}

impl fmt::Display for ImageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.hyphenated())
    }
}

impl Serialize for ImageId {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ImageId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// `None` in a manifest written before images were named.
    #[serde(default)]
    pub image: Option<ImageId>,
    pub root: PodPid,
    /// The pod's GPUs, in the driver's order.
    pub gpus: Vec<Gpu>,
    pub address: IpAddr,
    /// The node's environment when the checkpoint was taken; `None` in a
    /// manifest written before it was recorded.
    #[serde(default)]
    pub environment: Option<Environment>,
}

pub const NO_IMAGE_ID: &str = "it records no image id";

/// Why a checkpoint cannot be restored into a pod.
#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    /// This pod cannot take it; another pod of the revision may.
    Pod(String),
    /// No pod on this node can take it.
    Node(String),
}

impl Manifest {
    pub fn path(image: &Path) -> PathBuf {
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

    /// Why a node in `node` that condemned `condemned` should not take
    /// this checkpoint from a peer, or `None` if it may.
    #[cfg(any(test, feature = "nixl"))]
    pub fn pull_refusal(
        &self,
        node: &Environment,
        condemned: &BTreeSet<ImageId>,
    ) -> Option<String> {
        pull_refusal(self.environment.as_ref(), self.image, node, condemned)
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
        if self.image.is_none() {
            return Ok(Some(Refusal::Node(NO_IMAGE_ID.to_string())));
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

/// Why a checkpoint taken in `environment` and named `image` is not worth
/// pulling to a node in `node` that condemned `condemned`. A peer's
/// answer is judged by it before a pull, the pulled manifest after.
#[cfg(any(test, feature = "nixl"))]
pub fn pull_refusal(
    environment: Option<&Environment>,
    image: Option<ImageId>,
    node: &Environment,
    condemned: &BTreeSet<ImageId>,
) -> Option<String> {
    if let Some(reason) = node.incompatibility(environment) {
        return Some(reason);
    }
    match image {
        None => Some(NO_IMAGE_ID.to_string()),
        Some(image) if condemned.contains(&image) => {
            Some(format!("this node condemned image {image}"))
        }
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::cuda::Gpu;
    use crate::cuinterpose;
    use crate::node::environment::tests::{environment, with_driver, with_kernel};
    use crate::node::manifest::{ImageId, Manifest, NO_IMAGE_ID, Refusal};
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

    const ID: &str = "4f1c7d1e-2b54-4c8e-9a51-0e7b1f0c2d3a";

    fn manifest(minors: &[u32]) -> Manifest {
        Manifest {
            image: Some(ID.parse().unwrap()),
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
    fn a_manifest_round_trips_with_its_environment_and_image_id() {
        let dir = scratch("round-trip");
        let manifest = manifest(&[0, 1]);
        manifest.save(&dir).unwrap();
        let text = fs::read_to_string(dir.join("manifest.json")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["environment"]["nvidia"]["driver"], "580.65.06");
        assert_eq!(json["image"], ID);
        assert_eq!(Manifest::load(&dir).unwrap(), manifest);

        let drawn = Manifest {
            image: Some(ImageId::draw()),
            ..manifest
        };
        drawn.save(&dir).unwrap();
        assert_eq!(Manifest::load(&dir).unwrap(), drawn);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn image_ids_are_drawn_apart_and_parse_back() {
        let (a, b) = (ImageId::draw(), ImageId::draw());
        assert_ne!(a, b);
        assert_eq!(a.to_string().parse::<ImageId>().unwrap(), a);
        let id: ImageId = ID.parse().unwrap();
        assert_eq!(id.to_string(), ID);
        for text in ["", "image", "GPU-4f1c7d1e-2b54-4c8e-9a51-0e7b1f0c2d3a"] {
            assert!(text.parse::<ImageId>().is_err(), "{text:?}");
        }
        assert!(serde_json::from_str::<ImageId>(r#""not-an-id""#).is_err());
    }

    #[test]
    fn a_manifest_with_an_environment_and_no_image_id_is_refused_for_the_node() {
        let dir = scratch("no-id");
        let unnamed = Manifest {
            image: None,
            ..manifest(&[4])
        };
        unnamed.save(&dir).unwrap();
        let loaded = Manifest::load(&dir).unwrap();
        assert_eq!(loaded.image, None);
        assert_eq!(
            loaded.refusal(&dir, &gpus(&[4]), &environment()).unwrap(),
            Some(Refusal::Node(NO_IMAGE_ID.to_string()))
        );
        assert_eq!(
            loaded
                .refusal(&dir, &gpus(&[4, 5]), &environment())
                .unwrap(),
            Some(Refusal::Node(NO_IMAGE_ID.to_string())),
            "it is refused for the node before it is refused for the pod"
        );
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
        assert_eq!(legacy.image, None);
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
