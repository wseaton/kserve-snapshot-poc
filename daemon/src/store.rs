//! A node's checkpoint store: one directory per revision of each workload.
//!
//! ```text
//! <store>/<namespace>/<workload>/<hash>/image/           the checkpoint; it appears whole, by rename
//! <store>/<namespace>/<workload>/<hash>/incoming-<pod>/  a capture or a pull in progress
//! <store>/<namespace>/<workload>/<hash>/failed-<pod>     why this pod no longer tries
//! ```
use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, ensure};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RevisionKey {
    namespace: String,
    workload: String,
    hash: String,
}

impl RevisionKey {
    pub fn new(namespace: &str, workload: &str, hash: &str) -> Result<Self> {
        for part in [namespace, workload, hash] {
            ensure!(
                !part.is_empty()
                    && !part.starts_with('.')
                    && part
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
                "{part:?} cannot name part of a revision"
            );
        }
        Ok(Self {
            namespace: namespace.to_string(),
            workload: workload.to_string(),
            hash: hash.to_string(),
        })
    }

    fn dir(&self, store: &Path) -> PathBuf {
        store
            .join(&self.namespace)
            .join(&self.workload)
            .join(&self.hash)
    }

    #[cfg(any(test, feature = "nixl"))]
    pub fn image_dir(&self, store: &Path) -> PathBuf {
        self.dir(store).join("image")
    }
}

impl FromStr for RevisionKey {
    type Err = anyhow::Error;

    fn from_str(name: &str) -> Result<Self> {
        let parts: Vec<&str> = name.split('/').collect();
        let [namespace, workload, hash] = parts[..] else {
            anyhow::bail!("{name:?} is not <namespace>/<workload>/<hash>");
        };
        Self::new(namespace, workload, hash)
    }
}

impl fmt::Display for RevisionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.namespace, self.workload, self.hash)
    }
}

pub fn ensure_room(dir: &Path, bytes: u64) -> Result<()> {
    let stat = rustix::fs::statvfs(dir).with_context(|| format!("stat {}", dir.display()))?;
    let free = stat.f_bavail.saturating_mul(stat.f_frsize);
    ensure!(
        free >= bytes,
        "{} has {free} bytes free and needs {bytes}",
        dir.display()
    );
    Ok(())
}

/// What cannot be read counts as nothing.
pub fn directory_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => directory_bytes(&entry.path()),
            Ok(kind) if kind.is_file() => entry.metadata().map_or(0, |metadata| metadata.len()),
            _ => 0,
        })
        .sum()
}

pub struct RevisionStore {
    dir: PathBuf,
    staging: PathBuf,
    failure: PathBuf,
}

impl RevisionStore {
    pub fn open(store: &Path, revision: &RevisionKey, pod_uid: &str) -> Result<Self> {
        let dir = revision.dir(store);
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        Ok(Self {
            staging: dir.join(format!("incoming-{pod_uid}")),
            failure: dir.join(format!("failed-{pod_uid}")),
            dir,
        })
    }

    pub fn image_dir(&self) -> PathBuf {
        self.dir.join("image")
    }

    pub fn image_ready(&self) -> bool {
        self.image_dir().is_dir()
    }

    pub fn staging_dir(&self) -> &Path {
        &self.staging
    }

    pub fn ensure_room(&self, bytes: u64) -> Result<()> {
        ensure_room(&self.dir, bytes)
    }

    pub fn clear_staging(&self) -> Result<()> {
        match fs::remove_dir_all(&self.staging) {
            Err(error) if error.kind() != ErrorKind::NotFound => {
                Err(error).context("remove the unfinished image")
            }
            _ => Ok(()),
        }
    }

    /// If another pod published first, that image stays and this one is dropped.
    pub fn publish(&self) -> Result<()> {
        match fs::rename(&self.staging, self.image_dir()) {
            Ok(()) => Ok(()),
            Err(_) if self.image_ready() => self.clear_staging(),
            Err(error) => Err(error).context("publish the image"),
        }
    }

    pub fn failed_before(&self) -> bool {
        self.failure.exists()
    }

    pub fn record_failure(&self, error: &anyhow::Error) -> Result<()> {
        fs::write(&self.failure, format!("{error:#}\n")).context("record the failure")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::anyhow;

    use crate::store::{RevisionKey, RevisionStore, directory_bytes, ensure_room};

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn open(root: &Path, pod: &str) -> RevisionStore {
        let key = RevisionKey::new("serving", "qwen3-8b", "7b9f694bf8").unwrap();
        RevisionStore::open(root, &key, pod).unwrap()
    }

    #[test]
    fn a_revision_name_parses_into_a_path_under_the_store() {
        let key: RevisionKey = "serving/qwen3-8b/7b9f694bf8".parse().unwrap();
        assert_eq!(key.to_string(), "serving/qwen3-8b/7b9f694bf8");
        assert_eq!(
            key.image_dir(Path::new("/store")),
            Path::new("/store/serving/qwen3-8b/7b9f694bf8/image")
        );
        for name in [
            "",
            "serving/qwen3-8b",
            "serving/qwen3-8b/7b9f694bf8/extra",
            "serving//7b9f694bf8",
            "../qwen3-8b/7b9f694bf8",
            "serving/../7b9f694bf8",
            "serving/.hidden/7b9f694bf8",
            "/etc/passwd/x",
            "serving/qwen 3/7b9f694bf8",
            "serving/qwen3-8b/7b9f\\694bf8",
        ] {
            assert!(name.parse::<RevisionKey>().is_err(), "{name:?}");
        }
    }

    #[test]
    fn room_is_checked_against_the_filesystem() {
        let dir = scratch("room");
        fs::create_dir_all(&dir).unwrap();
        ensure_room(&dir, 1).unwrap();
        assert!(ensure_room(&dir, u64::MAX).is_err());
        assert!(ensure_room(&dir.join("missing"), 1).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_directory_is_sized_through_its_subdirectories() {
        let dir = scratch("size");
        fs::create_dir_all(dir.join("img/deep")).unwrap();
        fs::write(dir.join("manifest.json"), [0; 10]).unwrap();
        fs::write(dir.join("img/pages-1.img"), [0; 1000]).unwrap();
        fs::write(dir.join("img/deep/core-1.img"), [0; 100]).unwrap();
        assert_eq!(directory_bytes(&dir), 1110);
        assert_eq!(directory_bytes(&dir.join("missing")), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_published_staging_directory_becomes_the_image() {
        let dir = scratch("publish");
        let store = open(&dir, "a");
        fs::create_dir_all(store.staging_dir().join("img")).unwrap();
        fs::write(store.staging_dir().join("img/pages-1.img"), "ours").unwrap();
        assert!(!store.image_ready());
        store.publish().unwrap();
        assert!(store.image_ready());
        assert!(!store.staging_dir().exists());
        assert_eq!(
            fs::read_to_string(store.image_dir().join("img/pages-1.img")).unwrap(),
            "ours"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_image_published_first_by_another_pod_stays() {
        let dir = scratch("race");
        let (ours, theirs) = (open(&dir, "a"), open(&dir, "b"));
        for (store, content) in [(&theirs, "theirs"), (&ours, "ours")] {
            fs::create_dir_all(store.staging_dir()).unwrap();
            fs::write(store.staging_dir().join("inventory.img"), content).unwrap();
            store.publish().unwrap();
        }
        assert_eq!(
            fs::read_to_string(ours.image_dir().join("inventory.img")).unwrap(),
            "theirs"
        );
        assert!(!ours.staging_dir().exists() && !theirs.staging_dir().exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn publishing_nothing_is_an_error() {
        let dir = scratch("nothing");
        assert!(open(&dir, "a").publish().is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failure_is_remembered_for_its_pod_only() {
        let dir = scratch("failure");
        let (ours, theirs) = (open(&dir, "a"), open(&dir, "b"));
        assert!(!ours.failed_before());
        ours.record_failure(&anyhow!("criu restore failed").context("restore"))
            .unwrap();
        assert!(ours.failed_before() && open(&dir, "a").failed_before());
        assert!(!theirs.failed_before());
        fs::remove_dir_all(dir).unwrap();
    }
}
