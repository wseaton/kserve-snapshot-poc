//! What a checkpoint depends on outside its own image: the NVIDIA driver,
//! the GPU models, the CPU's features, the tools that dump and restore it,
//! and the kernel.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cuda;
use crate::cuinterpose::{self, Sha256Digest};
use crate::node::layout::CUINTERPOSE;

const OSRELEASE: &str = "/proc/sys/kernel/osrelease";
const NVIDIA_DRIVER: &str = "/proc/driver/nvidia/version";
const CPUINFO: &str = "/proc/cpuinfo";

macro_rules! text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

text_newtype!(DaemonVersion);
text_newtype!(KernelRelease);
text_newtype!(DriverVersion);
text_newtype!(GpuName);
text_newtype!(CpuFlag);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ComputeCapability {
    pub major: i32,
    pub minor: i32,
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GpuModel {
    pub name: GpuName,
    pub compute_capability: ComputeCapability,
}

impl fmt::Display for GpuModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (sm {})", self.name, self.compute_capability)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nvidia {
    pub driver: DriverVersion,
    /// Every distinct model among the node's GPUs.
    pub gpus: BTreeSet<GpuModel>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tools {
    pub criu: Sha256Digest,
    /// `None` when the tools carry no cuInterpose bundle.
    pub cuinterpose: Option<BTreeMap<String, Sha256Digest>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Environment {
    /// Recorded, not compared.
    pub daemon: DaemonVersion,
    /// A change is logged, not refused.
    pub kernel: KernelRelease,
    /// `None` on a node without an NVIDIA driver.
    pub nvidia: Option<Nvidia>,
    /// The CPU features the kernel reports. `None` in a fingerprint taken
    /// before they were recorded.
    #[serde(default)]
    pub cpu_flags: Option<BTreeSet<CpuFlag>>,
    pub tools: Tools,
}

impl Environment {
    pub fn of_node(tools_dir: &Path) -> Result<Self> {
        let kernel = fs::read_to_string(OSRELEASE).with_context(|| format!("read {OSRELEASE}"))?;
        let bundle_dir = tools_dir.join(CUINTERPOSE);
        let cuinterpose = if bundle_dir.is_dir() {
            Some(cuinterpose::hash_libraries(&cuinterpose::Config {
                bundle_dir,
                socket_dir: Default::default(),
                loaded_from: None,
            })?)
        } else {
            None
        };
        Ok(Self {
            daemon: DaemonVersion(env!("CARGO_PKG_VERSION").to_string()),
            kernel: KernelRelease(kernel.trim().to_string()),
            nvidia: nvidia()?,
            cpu_flags: Some(cpu_flags(
                &fs::read_to_string(CPUINFO).with_context(|| format!("read {CPUINFO}"))?,
            )),
            tools: Tools {
                criu: Sha256Digest::of_file(&tools_dir.join("criu"))?,
                cuinterpose,
            },
        })
    }

    /// Why a checkpoint taken in `taken` cannot be restored on this node,
    /// or `None` if it can. A checkpoint that records no environment is
    /// refused.
    pub fn incompatibility(&self, taken: Option<&Self>) -> Option<String> {
        let Some(taken) = taken else {
            return Some("it records no environment".to_string());
        };
        let mut reasons = Vec::new();
        match (&taken.nvidia, &self.nvidia) {
            (Some(then), Some(now)) => {
                if then.driver != now.driver {
                    reasons.push(format!(
                        "it was taken under NVIDIA driver {} and the node runs {}",
                        then.driver, now.driver
                    ));
                }
                if then.gpus != now.gpus {
                    reasons.push(format!(
                        "it was taken on {} and the node has {}",
                        models(&then.gpus),
                        models(&now.gpus)
                    ));
                }
            }
            (Some(then), None) => reasons.push(format!(
                "it was taken under NVIDIA driver {} and the node has none",
                then.driver
            )),
            (None, Some(now)) => reasons.push(format!(
                "it was taken without an NVIDIA driver and the node runs {}",
                now.driver
            )),
            (None, None) => {}
        }
        if let (Some(then), Some(now)) = (&taken.cpu_flags, &self.cpu_flags) {
            let lacking: Vec<&str> = then.difference(now).map(|flag| flag.0.as_str()).collect();
            if !lacking.is_empty() {
                reasons.push(format!(
                    "it was taken on a CPU with flags the node lacks: {}",
                    lacking.join(" ")
                ));
            }
        }
        if taken.tools.criu != self.tools.criu {
            reasons.push(format!(
                "it was dumped by CRIU {} and the node has {}",
                taken.tools.criu, self.tools.criu
            ));
        }
        if taken.tools.cuinterpose != self.tools.cuinterpose {
            reasons.push(format!(
                "it was taken with cuInterpose {} and the node has {}",
                bundle(taken.tools.cuinterpose.as_ref()),
                bundle(self.tools.cuinterpose.as_ref())
            ));
        }
        (!reasons.is_empty()).then(|| reasons.join("; "))
    }

    /// What differs from `taken` without stopping a restore.
    pub fn drift(&self, taken: &Self) -> Vec<String> {
        let mut drift = Vec::new();
        if taken.kernel != self.kernel {
            drift.push(format!(
                "it was taken on kernel {} and the node runs {}",
                taken.kernel, self.kernel
            ));
        }
        drift
    }
}

impl fmt::Display for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "snapshotd {}, kernel {}", self.daemon, self.kernel)?;
        match &self.nvidia {
            Some(nvidia) => write!(
                f,
                ", NVIDIA driver {} on {}",
                nvidia.driver,
                models(&nvidia.gpus)
            )?,
            None => f.write_str(", no NVIDIA driver")?,
        }
        if let Some(flags) = &self.cpu_flags {
            write!(f, ", {} CPU flags", flags.len())?;
        }
        write!(
            f,
            ", CRIU {}, cuInterpose {}",
            self.tools.criu,
            bundle(self.tools.cuinterpose.as_ref())
        )
    }
}

fn models(gpus: &BTreeSet<GpuModel>) -> String {
    if gpus.is_empty() {
        return "no GPUs".to_string();
    }
    gpus.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn bundle(libraries: Option<&BTreeMap<String, Sha256Digest>>) -> String {
    match libraries {
        None => "absent".to_string(),
        Some(libraries) => libraries
            .iter()
            .map(|(name, digest)| format!("{name}={digest}"))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// The flags of the first processor in `/proc/cpuinfo`: `flags` on x86,
/// `Features` on Arm.
fn cpu_flags(cpuinfo: &str) -> BTreeSet<CpuFlag> {
    cpuinfo
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            matches!(key.trim(), "flags" | "Features").then_some(value)
        })
        .map(|value| {
            value
                .split_whitespace()
                .map(|flag| CpuFlag(flag.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn nvidia() -> Result<Option<Nvidia>> {
    if !Path::new(NVIDIA_DRIVER).exists() {
        return Ok(None);
    }
    let nvml = cuda::nvml()?;
    let driver = DriverVersion(
        nvml.sys_driver_version()
            .context("read the NVIDIA driver version")?,
    );
    let gpus = (0..nvml.device_count().context("count the GPUs")?)
        .map(|index| {
            let device = nvml.device_by_index(index).context("open a GPU")?;
            let capability = device
                .cuda_compute_capability()
                .context("read a GPU's compute capability")?;
            Ok(GpuModel {
                name: GpuName(device.name().context("read a GPU's name")?),
                compute_capability: ComputeCapability {
                    major: capability.major,
                    minor: capability.minor,
                },
            })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    Ok(Some(Nvidia { driver, gpus }))
}

#[cfg(test)]
pub mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use crate::cuinterpose::Sha256Digest;
    use crate::node::environment::{
        ComputeCapability, CpuFlag, DaemonVersion, DriverVersion, Environment, GpuModel, GpuName,
        KernelRelease, Nvidia, Tools, cpu_flags,
    };

    fn digest(byte: u8) -> Sha256Digest {
        serde_json::from_value(serde_json::Value::String(hex::encode([byte; 32]))).unwrap()
    }

    fn h100() -> GpuModel {
        GpuModel {
            name: GpuName("NVIDIA H100 80GB HBM3".to_string()),
            compute_capability: ComputeCapability { major: 9, minor: 0 },
        }
    }

    /// A node with H100s under driver 580.65.06.
    pub fn environment() -> Environment {
        Environment {
            daemon: DaemonVersion("0.1.0".to_string()),
            kernel: KernelRelease("5.14.0-570.el9_6.x86_64".to_string()),
            nvidia: Some(Nvidia {
                driver: DriverVersion("580.65.06".to_string()),
                gpus: BTreeSet::from([h100()]),
            }),
            cpu_flags: Some(flags(&["avx2", "avx512f", "fpu", "sse4_2"])),
            tools: Tools {
                criu: digest(1),
                cuinterpose: Some(BTreeMap::from([
                    ("libcuinterpose.so".to_string(), digest(2)),
                    ("libcuinterpose_core.so".to_string(), digest(3)),
                ])),
            },
        }
    }

    pub fn with_driver(driver: &str) -> Environment {
        let mut environment = environment();
        if let Some(nvidia) = environment.nvidia.as_mut() {
            nvidia.driver = DriverVersion(driver.to_string());
        }
        environment
    }

    pub fn flags(names: &[&str]) -> BTreeSet<CpuFlag> {
        names.iter().map(|name| CpuFlag(name.to_string())).collect()
    }

    pub fn with_cpu_flags(names: Option<&[&str]>) -> Environment {
        Environment {
            cpu_flags: names.map(flags),
            ..environment()
        }
    }

    pub fn with_kernel(release: &str) -> Environment {
        Environment {
            kernel: KernelRelease(release.to_string()),
            ..environment()
        }
    }

    #[test]
    fn the_same_environment_is_compatible_and_shows_no_drift() {
        let here = environment();
        assert_eq!(here.incompatibility(Some(&environment())), None);
        assert!(here.drift(&environment()).is_empty());
        let mut bare = environment();
        bare.nvidia = None;
        bare.tools.cuinterpose = None;
        assert_eq!(bare.incompatibility(Some(&bare.clone())), None);
    }

    #[test]
    fn no_recorded_environment_is_incompatible() {
        assert_eq!(
            environment().incompatibility(None).as_deref(),
            Some("it records no environment")
        );
    }

    #[test]
    fn a_driver_upgrade_is_incompatible() {
        let here = with_driver("580.82.07");
        assert_eq!(
            here.incompatibility(Some(&environment())).as_deref(),
            Some("it was taken under NVIDIA driver 580.65.06 and the node runs 580.82.07")
        );
    }

    #[test]
    fn other_gpu_models_are_incompatible() {
        let mut here = environment();
        let a100 = GpuModel {
            name: GpuName("NVIDIA A100-SXM4-80GB".to_string()),
            compute_capability: ComputeCapability { major: 8, minor: 0 },
        };
        here.nvidia.as_mut().unwrap().gpus = BTreeSet::from([a100.clone()]);
        assert_eq!(
            here.incompatibility(Some(&environment())).as_deref(),
            Some(
                "it was taken on NVIDIA H100 80GB HBM3 (sm 9.0) and the node has \
                 NVIDIA A100-SXM4-80GB (sm 8.0)"
            )
        );
        here.nvidia.as_mut().unwrap().gpus = BTreeSet::from([a100, h100()]);
        assert!(here.incompatibility(Some(&environment())).is_some());
        let mut renamed_capability = environment();
        renamed_capability.nvidia.as_mut().unwrap().gpus = BTreeSet::from([GpuModel {
            compute_capability: ComputeCapability { major: 9, minor: 1 },
            ..h100()
        }]);
        assert!(
            renamed_capability
                .incompatibility(Some(&environment()))
                .is_some()
        );
    }

    #[test]
    fn a_driver_that_appears_or_disappears_is_incompatible() {
        let mut bare = environment();
        bare.nvidia = None;
        assert_eq!(
            bare.incompatibility(Some(&environment())).as_deref(),
            Some("it was taken under NVIDIA driver 580.65.06 and the node has none")
        );
        assert_eq!(
            environment().incompatibility(Some(&bare)).as_deref(),
            Some("it was taken without an NVIDIA driver and the node runs 580.65.06")
        );
    }

    #[test]
    fn other_tools_are_incompatible() {
        let mut criu = environment();
        criu.tools.criu = digest(9);
        let reason = criu.incompatibility(Some(&environment())).unwrap();
        assert!(reason.starts_with("it was dumped by CRIU 0101"), "{reason}");
        assert!(
            reason.ends_with(&format!("node has {}", digest(9))),
            "{reason}"
        );

        let mut library = environment();
        library
            .tools
            .cuinterpose
            .as_mut()
            .unwrap()
            .insert("libcuinterpose_core.so".to_string(), digest(4));
        let reason = library.incompatibility(Some(&environment())).unwrap();
        assert!(
            reason.starts_with("it was taken with cuInterpose"),
            "{reason}"
        );

        let mut absent = environment();
        absent.tools.cuinterpose = None;
        let reason = absent.incompatibility(Some(&environment())).unwrap();
        assert!(reason.ends_with("the node has absent"), "{reason}");
    }

    #[test]
    fn a_checkpoint_needs_only_cpu_flags_the_node_has() {
        let here = environment();
        let fewer = with_cpu_flags(Some(&["avx2", "fpu"]));
        assert_eq!(
            here.incompatibility(Some(&fewer)),
            None,
            "a checkpoint taken on a CPU with a subset of the node's flags restores"
        );
        assert_eq!(here.incompatibility(Some(&with_cpu_flags(Some(&[])))), None);
        let more = with_cpu_flags(Some(&[
            "amx_tile", "avx2", "avx512f", "fpu", "sse4_2", "vaes",
        ]));
        assert_eq!(
            here.incompatibility(Some(&more)).as_deref(),
            Some("it was taken on a CPU with flags the node lacks: amx_tile vaes")
        );
        assert!(
            fewer.incompatibility(Some(&here)).is_some(),
            "the node lacks avx512f and sse4_2"
        );
        let legacy = with_cpu_flags(None);
        assert_eq!(
            here.incompatibility(Some(&legacy)),
            None,
            "a fingerprint from before CPU flags were recorded is accepted"
        );
        assert_eq!(legacy.incompatibility(Some(&here)), None);
    }

    #[test]
    fn the_first_processors_flags_are_read_from_cpuinfo() {
        let x86 = "processor\t: 0\nvendor_id\t: GenuineIntel\nflags\t\t: fpu sse4_2 avx2\n\
                   bugs\t\t: spectre_v1\n\nprocessor\t: 1\nflags\t\t: fpu\n";
        assert_eq!(cpu_flags(x86), flags(&["avx2", "fpu", "sse4_2"]));
        let arm = "processor\t: 0\nBogoMIPS\t: 50.00\nFeatures\t: fp asimd aes\n";
        assert_eq!(cpu_flags(arm), flags(&["aes", "asimd", "fp"]));
        assert!(cpu_flags("processor\t: 0\n").is_empty());
        assert!(cpu_flags("").is_empty());
        assert_eq!(
            cpu_flags("vmx flags\t: ept vpid\nflags\t: fpu\n"),
            flags(&["fpu"]),
            "only the line keyed flags counts"
        );
    }

    #[test]
    fn every_incompatibility_is_reported_at_once() {
        let mut here = with_driver("590.1");
        here.tools.criu = digest(7);
        let reason = here.incompatibility(Some(&environment())).unwrap();
        assert_eq!(reason.matches("; ").count(), 1, "{reason}");
        assert!(
            reason.contains("590.1") && reason.contains("CRIU"),
            "{reason}"
        );
    }

    #[test]
    fn a_kernel_or_daemon_change_is_compatible_and_the_kernel_drifts() {
        let mut here = with_kernel("5.14.0-611.el9_7.x86_64");
        here.daemon = DaemonVersion("0.2.0".to_string());
        assert_eq!(here.incompatibility(Some(&environment())), None);
        assert_eq!(
            here.drift(&environment()),
            [
                "it was taken on kernel 5.14.0-570.el9_6.x86_64 and the node runs \
                 5.14.0-611.el9_7.x86_64"
            ]
        );
    }

    #[test]
    fn an_environment_round_trips_through_json() {
        let environment = environment();
        let json = serde_json::to_value(&environment).unwrap();
        assert_eq!(json["daemon"], "0.1.0");
        assert_eq!(json["kernel"], "5.14.0-570.el9_6.x86_64");
        assert_eq!(json["nvidia"]["driver"], "580.65.06");
        assert_eq!(
            json["nvidia"]["gpus"][0],
            serde_json::json!({
                "name": "NVIDIA H100 80GB HBM3",
                "compute_capability": {"major": 9, "minor": 0}
            })
        );
        assert_eq!(json["tools"]["criu"], hex::encode([1; 32]));
        assert_eq!(
            json["cpu_flags"],
            serde_json::json!(["avx2", "avx512f", "fpu", "sse4_2"])
        );
        assert_eq!(
            serde_json::from_value::<Environment>(json).unwrap(),
            environment
        );
        let mut bare = environment.clone();
        bare.nvidia = None;
        bare.tools.cuinterpose = None;
        let json = serde_json::to_string(&bare).unwrap();
        assert_eq!(serde_json::from_str::<Environment>(&json).unwrap(), bare);
        assert!(serde_json::from_str::<Environment>(r#"{"daemon":"0.1.0"}"#).is_err());
        let mut legacy = serde_json::to_value(&environment).unwrap();
        legacy.as_object_mut().unwrap().remove("cpu_flags");
        assert_eq!(
            serde_json::from_value::<Environment>(legacy).unwrap(),
            with_cpu_flags(None)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_node_environment_reads_the_kernel_and_hashes_the_tools() {
        let tools =
            std::env::temp_dir().join(format!("snapshotd-environment-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tools);
        std::fs::create_dir_all(&tools).unwrap();
        assert!(Environment::of_node(&tools).is_err(), "criu is missing");

        std::fs::write(tools.join("criu"), "criu 4.1").unwrap();
        let read = Environment::of_node(&tools).unwrap();
        let osrelease = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap();
        assert_eq!(read.kernel.to_string(), osrelease.trim());
        assert_eq!(read.daemon.to_string(), env!("CARGO_PKG_VERSION"));
        assert_eq!(
            read.tools.criu,
            Sha256Digest::of_file(&tools.join("criu")).unwrap()
        );
        assert_eq!(read.tools.cuinterpose, None);
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap();
        assert_eq!(read.cpu_flags, Some(cpu_flags(&cpuinfo)));
        assert_eq!(
            read.nvidia.is_some(),
            std::path::Path::new("/proc/driver/nvidia/version").exists()
        );

        let bundle = tools.join("cuinterpose");
        std::fs::create_dir_all(&bundle).unwrap();
        assert!(Environment::of_node(&tools).is_err(), "the bundle is empty");
        for name in ["libcuinterpose.so", "libcuinterpose_core.so"] {
            std::fs::write(bundle.join(name), name).unwrap();
        }
        let bundled = Environment::of_node(&tools).unwrap();
        let libraries = bundled.tools.cuinterpose.clone().unwrap();
        assert_eq!(
            libraries.keys().collect::<Vec<_>>(),
            ["libcuinterpose.so", "libcuinterpose_core.so"]
        );
        assert!(bundled.incompatibility(Some(&read)).is_some());

        std::fs::write(tools.join("criu"), "criu 4.2").unwrap();
        let rebuilt = Environment::of_node(&tools).unwrap();
        assert!(
            rebuilt
                .incompatibility(Some(&bundled))
                .unwrap()
                .starts_with("it was dumped by CRIU")
        );
        std::fs::remove_dir_all(&tools).unwrap();
    }

    #[test]
    fn the_environment_names_what_it_holds() {
        let text = environment().to_string();
        for part in [
            "snapshotd 0.1.0",
            "kernel 5.14.0-570.el9_6.x86_64",
            "NVIDIA driver 580.65.06 on NVIDIA H100 80GB HBM3 (sm 9.0)",
            "libcuinterpose.so=0202",
            "4 CPU flags",
        ] {
            assert!(text.contains(part), "{text}");
        }
    }
}
