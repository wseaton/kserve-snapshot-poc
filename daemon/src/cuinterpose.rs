//! cuInterpose integration for TP>1 checkpoints.
//!
//! cuInterpose (NVIDIA, Apache-2.0) is an `LD_PRELOAD` library that tracks
//! CUDA VMM and IPC memory so allocations shared between processes can be
//! unmapped before cuda-checkpoint and re-imported after restore. The agent
//! uses it in three places:
//!
//! ```text
//! pod start   vLLM starts under cuinterpose-launch from the daemon's
//!             read-only hostPath bundle, at the same path on every node
//! checkpoint  inspect ─► shared? ─no─► plain cuda-checkpoint
//!                           │yes
//!                           ▼
//!             write cuinterpose.json ─► coordinator --prepare ─► cuda lock+checkpoint ─► CRIU dump
//! restore     verify library hashes ─► CRIU restore ─► cuda restore+unlock ─► coordinator --restore
//! ```
//!
//! Inspection talks to each process's control socket directly with
//! `cuinterpose-protocol`. The prepare and restore phases still run NVIDIA's
//! coordinator binary, which validates the cross-process topology.
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use cuinterpose_protocol::{NamespacePid, Record, Reply, Request, Response};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FRONTEND_LIBRARY: &str = "libcuinterpose.so";
pub const CORE_LIBRARY: &str = "libcuinterpose_core.so";
const LIBRARIES: [&str; 2] = [FRONTEND_LIBRARY, CORE_LIBRARY];
const COORDINATOR: &str = "cuinterpose-coordinator";
const MANIFEST_FILE: &str = "cuinterpose.json";
const STATE_FILE: &str = "cuinterpose.state";

/// Where the cuInterpose bundle and the workload's control sockets live, as
/// seen from inside the workload's mount namespace.
#[derive(Clone, Debug, clap::Args)]
pub struct Config {
    /// Directory holding the cuInterpose libraries and coordinator.
    #[arg(long)]
    pub bundle_dir: PathBuf,
    /// Directory holding the processes' control sockets.
    #[arg(long, default_value = "/tmp")]
    pub socket_dir: PathBuf,
}

impl Config {
    fn library(&self, name: &str) -> PathBuf {
        self.bundle_dir.join(name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Sha256Digest(#[serde(with = "hex_digest")] [u8; 32]);

impl Sha256Digest {
    fn of_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        ensure!(
            file.metadata()?.is_file(),
            "{} is not a regular file",
            path.display()
        );
        let mut digest = Sha256::new();
        io::copy(&mut file, &mut digest).with_context(|| format!("read {}", path.display()))?;
        Ok(Self(digest.finalize().into()))
    }
}

mod hex_digest {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 32], D::Error> {
        let text = String::deserialize(deserializer)?;
        let mut bytes = [0; 32];
        hex::decode_to_slice(text, &mut bytes).map_err(D::Error::custom)?;
        Ok(bytes)
    }
}

/// What a checkpoint needs to find the same cuInterpose build at restore.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub libraries: BTreeMap<String, Sha256Digest>,
    /// Namespace PIDs of the participants, as the coordinator recorded them.
    pub pids: Vec<NamespacePid>,
}

impl Manifest {
    pub fn path(ckpt_dir: &Path) -> PathBuf {
        ckpt_dir.join(MANIFEST_FILE)
    }

    pub fn load(ckpt_dir: &Path) -> Result<Option<Self>> {
        let path = Self::path(ckpt_dir);
        match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .with_context(|| format!("parse {}", path.display())),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
        }
    }

    fn save(&self, ckpt_dir: &Path) -> Result<()> {
        let path = Self::path(ckpt_dir);
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))
    }

    /// Refuse a restore whose bundle is not the build the checkpoint loaded.
    pub fn verify_libraries(&self, config: &Config) -> Result<()> {
        let actual = hash_libraries(config)?;
        ensure!(
            actual == self.libraries,
            "cuInterpose libraries in {} do not match the checkpoint: expected {:?}, found {:?}",
            config.bundle_dir.display(),
            self.libraries,
            actual
        );
        Ok(())
    }
}

pub fn hash_libraries(config: &Config) -> Result<BTreeMap<String, Sha256Digest>> {
    LIBRARIES
        .iter()
        .map(|name| {
            Ok((
                name.to_string(),
                Sha256Digest::of_file(&config.library(name))?,
            ))
        })
        .collect()
}

/// A shared object mapped into a process: device as in `/proc/PID/maps`
/// (`major:minor` in hex) and inode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MappedFile {
    major: u32,
    minor: u32,
    inode: u64,
}

/// Find `path` in a `/proc/PID/maps` listing. A replaced file shows up with a
/// ` (deleted)` suffix, which does not match.
fn find_mapping(maps: &str, path: &Path) -> Result<Option<MappedFile>> {
    let path = path.to_string_lossy();
    for line in maps.lines() {
        let mut fields = line.split_ascii_whitespace();
        let (Some(_), Some(_), Some(_), Some(device), Some(inode)) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            continue;
        };
        let rest: Vec<&str> = fields.collect();
        if rest.join(" ") != path {
            continue;
        }
        let (major, minor) = device
            .split_once(':')
            .with_context(|| format!("malformed maps device {device:?}"))?;
        return Ok(Some(MappedFile {
            major: u32::from_str_radix(major, 16)?,
            minor: u32::from_str_radix(minor, 16)?,
            inode: inode.parse()?,
        }));
    }
    Ok(None)
}

/// Require that every participant loaded the frontend from the bundle and that
/// the bundle files on disk are still the ones mapped, so the recorded hashes
/// describe what is running.
fn check_loaded_libraries(config: &Config, pids: &[i32]) -> Result<()> {
    for name in LIBRARIES {
        let path = config.library(name);
        let metadata = fs::metadata(&path).with_context(|| format!("stat {}", path.display()))?;
        for &pid in pids {
            let maps = fs::read_to_string(format!("/proc/{pid}/maps"))
                .with_context(|| format!("read /proc/{pid}/maps"))?;
            match find_mapping(&maps, &path)? {
                Some(mapped) => ensure!(
                    libc::major(metadata.dev()) == mapped.major
                        && libc::minor(metadata.dev()) == mapped.minor
                        && metadata.ino() == mapped.inode,
                    "pid {pid}: {} was replaced after it was loaded",
                    path.display()
                ),
                None if name == FRONTEND_LIBRARY => {
                    bail!("pid {pid} did not preload {}", path.display())
                }
                None => {}
            }
        }
    }
    Ok(())
}

/// The innermost PID of a process, which names its control socket.
pub fn namespace_pid(pid: i32) -> Result<NamespacePid> {
    let status = fs::read_to_string(format!("/proc/{pid}/status"))
        .with_context(|| format!("read /proc/{pid}/status"))?;
    parse_namespace_pid(&status).with_context(|| format!("pid {pid}: no NSpid line"))
}

fn parse_namespace_pid(status: &str) -> Option<NamespacePid> {
    status
        .lines()
        .find_map(|line| line.strip_prefix("NSpid:"))
        .and_then(|pids| pids.split_ascii_whitespace().last())
        .and_then(|pid| pid.parse().ok())
}

/// Ask one participant for the CUDA memory it tracks.
pub fn inspect(socket_dir: &Path, pid: NamespacePid) -> Result<Vec<Record>> {
    let endpoint = cuinterpose_protocol::socket_path(socket_dir, pid);
    let socket = cuinterpose_protocol::connect(&endpoint, cuinterpose_protocol::control_timeout())
        .with_context(|| format!("connect {}", endpoint.display()))?;
    socket.set_read_timeout(Some(cuinterpose_protocol::control_timeout()))?;
    let request = Request::Inspect { namespace_pid: pid };
    cuinterpose_protocol::send(&socket, &request, None)
        .with_context(|| format!("{}: send inspect", endpoint.display()))?;
    let (response, fd): (Response, _) = cuinterpose_protocol::receive(&socket)
        .with_context(|| format!("{}: receive inspection", endpoint.display()))?;
    ensure!(
        fd.is_none(),
        "{}: unexpected descriptor",
        endpoint.display()
    );
    ensure!(
        response.namespace_pid == pid,
        "{}: answered for pid {}",
        endpoint.display(),
        response.namespace_pid
    );
    match response.result.map_err(anyhow::Error::msg)? {
        Reply::Inspection { records } => Ok(records),
        other => bail!(
            "{}: expected an inspection, got {other:?}",
            endpoint.display()
        ),
    }
}

/// Cross-process CUDA memory one participant still holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sharing {
    pub shared_allocations: usize,
    pub shared_bytes: u64,
    pub multicast_objects: usize,
}

impl Sharing {
    pub fn of(records: &[Record]) -> Self {
        let mut sharing = Self::default();
        for record in records {
            match record {
                Record::Allocation {
                    shared: true, size, ..
                } => {
                    sharing.shared_allocations += 1;
                    sharing.shared_bytes += size;
                }
                Record::Multicast { .. } => sharing.multicast_objects += 1,
                _ => {}
            }
        }
        sharing
    }

    pub fn is_empty(&self) -> bool {
        self.shared_allocations == 0 && self.multicast_objects == 0
    }
}

/// How the checkpoint handles cross-process CUDA memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SharedMemory {
    /// No participant shares CUDA memory, so cuda-checkpoint alone is enough.
    NotShared,
    /// The coordinator unmapped shared memory and saved it with the checkpoint.
    /// The source processes cannot resume without a coordinator restore.
    Prepared(Manifest),
}

/// Inspect every CUDA process that preloaded cuInterpose and, if any of them
/// shares memory, run the coordinator's prepare phase. The application must
/// already be idle, which vLLM's `/checkpoint/prepare` guarantees.
pub fn prepare_checkpoint(
    config: &Config,
    ckpt_dir: &Path,
    cuda_pids: &[i32],
) -> Result<SharedMemory> {
    let mut host_pids = Vec::new();
    let mut namespace_pids = Vec::new();
    let mut shared = false;
    for &pid in cuda_pids {
        let namespace_pid = namespace_pid(pid)?;
        if !cuinterpose_protocol::socket_path(&config.socket_dir, namespace_pid).exists() {
            continue;
        }
        let sharing = Sharing::of(&inspect(&config.socket_dir, namespace_pid)?);
        println!("cuinterpose pid {pid} (ns {namespace_pid}): {sharing:?}");
        shared |= !sharing.is_empty();
        host_pids.push(pid);
        namespace_pids.push(namespace_pid);
    }
    if !shared {
        return Ok(SharedMemory::NotShared);
    }
    check_loaded_libraries(config, &host_pids)?;
    let manifest = Manifest {
        libraries: hash_libraries(config)?,
        pids: namespace_pids,
    };
    manifest.save(ckpt_dir)?;
    run_coordinator(config, "--prepare", ckpt_dir, &manifest.pids)?;
    ensure!(
        ckpt_dir.join(STATE_FILE).is_file(),
        "coordinator did not write {STATE_FILE}"
    );
    Ok(SharedMemory::Prepared(manifest))
}

/// Re-import shared memory after CRIU and cuda-checkpoint have restored every
/// participant. The PIDs are the checkpoint's namespace PIDs, which CRIU keeps.
pub fn restore(config: &Config, ckpt_dir: &Path, manifest: &Manifest) -> Result<()> {
    run_coordinator(config, "--restore", ckpt_dir, &manifest.pids)
}

fn run_coordinator(
    config: &Config,
    action: &str,
    ckpt_dir: &Path,
    pids: &[NamespacePid],
) -> Result<()> {
    let binary = config.bundle_dir.join(COORDINATOR);
    let mut command = Command::new(&binary);
    command
        .arg(action)
        .arg("--checkpoint-dir")
        .arg(ckpt_dir)
        .arg("--socket-dir")
        .arg(&config.socket_dir);
    for pid in pids {
        command.arg("--process").arg(pid.to_string());
    }
    let output = command
        .output()
        .with_context(|| format!("run {}", binary.display()))?;
    ensure!(
        output.status.success(),
        "{} {action} exited with {}: {}",
        binary.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cuinterpose_protocol::{AllocationReference, MemoryLocation, MulticastProperties, Record};

    use crate::cuinterpose::{
        CORE_LIBRARY, Config, FRONTEND_LIBRARY, Manifest, MappedFile, Sha256Digest, Sharing,
        find_mapping, hash_libraries, parse_namespace_pid,
    };

    fn bundle(label: &str, frontend: &[u8], core: &[u8]) -> Config {
        let dir = std::env::temp_dir().join(format!("cuinterpose-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(FRONTEND_LIBRARY), frontend).unwrap();
        std::fs::write(dir.join(CORE_LIBRARY), core).unwrap();
        Config {
            bundle_dir: dir,
            socket_dir: PathBuf::from("/tmp"),
        }
    }

    fn allocation(id: u8, shared: bool, size: u64) -> Record {
        Record::Allocation {
            allocation: AllocationReference {
                id: [id; 16],
                creator_pid: 7,
            },
            shared,
            size,
            allocation_type: 1,
            handle_types: 1,
            location: MemoryLocation {
                location_type: 1,
                id: 0,
            },
            virtual_allocation_handle_count: 1,
        }
    }

    #[test]
    fn hashes_match_sha256_of_each_library() {
        let config = bundle("hash", b"frontend", b"core");
        let hashes = hash_libraries(&config).unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(
            hex::encode(hashes[FRONTEND_LIBRARY].0),
            "1cf387c012cd2d02f1923e589f86df4a4802427057dbc90f47b87f04d29d54bc"
        );
        assert_eq!(
            hashes[CORE_LIBRARY],
            Sha256Digest(<sha2::Sha256 as sha2::Digest>::digest(b"core").into())
        );
    }

    #[test]
    fn verify_rejects_a_bundle_built_differently() {
        let checkpointed = bundle("verify-a", b"frontend", b"core");
        let manifest = Manifest {
            libraries: hash_libraries(&checkpointed).unwrap(),
            pids: vec![12, 13],
        };
        manifest.verify_libraries(&checkpointed).unwrap();
        let other = bundle("verify-b", b"frontend", b"core v2");
        let error = manifest.verify_libraries(&other).unwrap_err().to_string();
        assert!(error.contains("do not match the checkpoint"), "{error}");
    }

    #[test]
    fn verify_rejects_a_missing_library() {
        let config = bundle("missing", b"frontend", b"core");
        let manifest = Manifest {
            libraries: hash_libraries(&config).unwrap(),
            pids: vec![12],
        };
        std::fs::remove_file(config.bundle_dir.join(CORE_LIBRARY)).unwrap();
        assert!(manifest.verify_libraries(&config).is_err());
    }

    #[test]
    fn manifest_round_trips_with_hex_digests() {
        let config = bundle("manifest", b"frontend", b"core");
        let manifest = Manifest {
            libraries: hash_libraries(&config).unwrap(),
            pids: vec![891, 893],
        };
        manifest.save(&config.bundle_dir).unwrap();
        let text = std::fs::read_to_string(Manifest::path(&config.bundle_dir)).unwrap();
        assert!(
            text.contains(&hex::encode(<sha2::Sha256 as sha2::Digest>::digest(
                b"core"
            )))
        );
        assert_eq!(Manifest::load(&config.bundle_dir).unwrap(), Some(manifest));
        assert_eq!(Manifest::load(Path::new("/nonexistent")).unwrap(), None);
    }

    #[test]
    fn manifest_rejects_a_truncated_digest() {
        let config = bundle("bad-digest", b"frontend", b"core");
        std::fs::write(
            Manifest::path(&config.bundle_dir),
            r#"{"libraries":{"libcuinterpose.so":"abcd"},"pids":[1]}"#,
        )
        .unwrap();
        assert!(Manifest::load(&config.bundle_dir).is_err());
    }

    #[test]
    fn finds_a_mapping_by_exact_path() {
        let maps = "\
7f00-7f01 r--p 00000000 00:2a 1234 /tmp/snapshot-cuda/libcuinterpose.so
7f02-7f03 r-xp 00001000 00:2a 1234 /tmp/snapshot-cuda/libcuinterpose.so
7f04-7f05 r--p 00000000 fd:01 99 /tmp/snapshot-cuda/libcuinterpose_core.so (deleted)
7f06-7f07 rw-p 00000000 00:00 0 [heap]
7f08-7f09 rw-p 00000000 00:00 0
";
        assert_eq!(
            find_mapping(maps, Path::new("/tmp/snapshot-cuda/libcuinterpose.so")).unwrap(),
            Some(MappedFile {
                major: 0,
                minor: 0x2a,
                inode: 1234
            })
        );
        assert_eq!(
            find_mapping(maps, Path::new("/tmp/snapshot-cuda/libcuinterpose_core.so")).unwrap(),
            None
        );
        assert_eq!(
            find_mapping(maps, Path::new("/tmp/other.so")).unwrap(),
            None
        );
    }

    #[test]
    fn finds_this_process_mapping_its_own_executable() {
        let exe = std::fs::read_link("/proc/self/exe").unwrap();
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let mapped = find_mapping(&maps, &exe).unwrap().unwrap();
        let metadata = std::fs::metadata(&exe).unwrap();
        use std::os::unix::fs::MetadataExt;
        assert_eq!(mapped.inode, metadata.ino());
        assert_eq!(mapped.major, libc::major(metadata.dev()));
        assert_eq!(mapped.minor, libc::minor(metadata.dev()));
    }

    #[test]
    fn namespace_pid_is_the_innermost() {
        assert_eq!(
            parse_namespace_pid("Name:\tpython3\nNSpid:\t41872\t893\n"),
            Some(893)
        );
        assert_eq!(
            parse_namespace_pid("Name:\tpython3\nNSpid:\t17\n"),
            Some(17)
        );
        assert_eq!(parse_namespace_pid("Name:\tpython3\n"), None);
        let own = crate::cuinterpose::namespace_pid(std::process::id() as i32).unwrap();
        assert!(own > 0);
    }

    #[test]
    fn sharing_counts_only_cross_process_memory() {
        let records = vec![
            allocation(1, true, 10 << 20),
            allocation(2, true, 98 << 20),
            allocation(3, false, 1 << 30),
            Record::Mapping {
                allocation: AllocationReference {
                    id: [1; 16],
                    creator_pid: 7,
                },
                address: 0x7000_0000,
                size: 10 << 20,
                offset: 0,
                access: Vec::new(),
            },
            Record::Multicast {
                allocation: AllocationReference {
                    id: [4; 16],
                    creator_pid: 7,
                },
                properties: MulticastProperties {
                    devices: 2,
                    size: 2 << 20,
                    handle_types: 1,
                    flags: 0,
                },
                virtual_multicast_handle_count: 1,
            },
        ];
        let sharing = Sharing::of(&records);
        assert_eq!(
            sharing,
            Sharing {
                shared_allocations: 2,
                shared_bytes: 108 << 20,
                multicast_objects: 1,
            }
        );
        assert!(!sharing.is_empty());
        assert!(Sharing::of(&[allocation(5, false, 4096)]).is_empty());
    }
}
