//! One revision's directory in a node's checkpoint store.
//!
//! ```text
//! <store>/<namespace>/<workload>/<hash>/image/           the checkpoint; it appears whole, by rename
//! <store>/<namespace>/<workload>/<hash>/incoming-<id>/   a capture or a pull in progress, one per pod
//! ```
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

/// Where the store keeps the revision named `<namespace>/<workload>/<hash>`.
/// The name can come from a peer, so it may not leave the store.
#[cfg(any(test, feature = "nixl"))]
pub fn revision_dir(store: &Path, name: &str) -> Result<PathBuf> {
    let parts: Vec<&str> = name.split('/').collect();
    ensure!(
        parts.len() == 3
            && parts.iter().all(|part| {
                !part.is_empty()
                    && !part.starts_with('.')
                    && part
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            }),
        "{name:?} is not <namespace>/<workload>/<hash>"
    );
    Ok(parts
        .iter()
        .fold(store.to_path_buf(), |dir, part| dir.join(part)))
}

/// Fail unless the filesystem holding `dir` has `bytes` free.
pub fn ensure_room(dir: &Path, bytes: u64) -> Result<()> {
    let stat = rustix::fs::statvfs(dir).with_context(|| format!("stat {}", dir.display()))?;
    let free = stat.f_bavail.saturating_mul(stat.f_frsize);
    ensure!(
        free >= bytes,
        "{} has {free} bytes free, the image needs {bytes}",
        dir.display()
    );
    Ok(())
}

/// The bytes of every file under `dir`; what cannot be read counts as nothing.
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

pub struct State {
    store: PathBuf,
    staging: PathBuf,
}

impl State {
    /// `id` names this pod's unfinished capture or pull among those of other pods.
    pub fn open(store: &Path, id: &str) -> Result<Self> {
        fs::create_dir_all(store).with_context(|| format!("create {}", store.display()))?;
        Ok(Self {
            store: store.to_path_buf(),
            staging: store.join(format!("incoming-{id}")),
        })
    }

    pub fn image_dir(&self) -> PathBuf {
        self.store.join("image")
    }

    pub fn image_ready(&self) -> bool {
        self.image_dir().is_dir()
    }

    pub fn staging_dir(&self) -> &Path {
        &self.staging
    }

    /// Remove what an unfinished capture or pull left behind.
    pub fn clear_staging(&self) -> Result<()> {
        match fs::remove_dir_all(&self.staging) {
            Err(error) if error.kind() != ErrorKind::NotFound => {
                Err(error).context("remove the unfinished image")
            }
            _ => Ok(()),
        }
    }

    /// Make the finished staging directory the node's image of this
    /// revision. If another pod published one first, that one stays and this
    /// one is dropped.
    pub fn publish(&self) -> Result<()> {
        match fs::rename(&self.staging, self.image_dir()) {
            Ok(()) => Ok(()),
            Err(_) if self.image_ready() => self.clear_staging(),
            Err(error) => Err(error).context("publish the image"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use crate::state::{State, directory_bytes, ensure_room, revision_dir};

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-state-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn open(dir: &Path, id: &str) -> State {
        State::open(&dir.join("store/rev"), id).unwrap()
    }

    #[test]
    fn a_revision_name_maps_into_the_store() {
        let store = Path::new("/store");
        assert_eq!(
            revision_dir(store, "serving/qwen3-8b/7b9f694bf8").unwrap(),
            Path::new("/store/serving/qwen3-8b/7b9f694bf8")
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
            assert!(revision_dir(store, name).is_err(), "{name:?}");
        }
    }

    #[test]
    fn room_is_checked_against_the_filesystem() {
        let dir = scratch("room");
        fs::create_dir_all(&dir).unwrap();
        ensure_room(&dir, 1).unwrap();
        let error = ensure_room(&dir, u64::MAX).unwrap_err();
        assert!(format!("{error:#}").contains("bytes free"));
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
    fn a_new_state_has_no_image() {
        let dir = scratch("new");
        let state = open(&dir, "a");
        assert!(!state.image_ready());
        assert_eq!(state.image_dir(), dir.join("store/rev/image"));
        assert_eq!(state.staging_dir(), dir.join("store/rev/incoming-a"));
        assert!(dir.join("store/rev").is_dir());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_published_staging_directory_becomes_the_image() {
        let dir = scratch("publish");
        let state = open(&dir, "a");
        fs::create_dir_all(state.staging_dir().join("img")).unwrap();
        fs::write(state.staging_dir().join("img/pages-1.img"), "ours").unwrap();
        assert!(!state.image_ready());
        state.publish().unwrap();
        assert!(state.image_ready());
        assert!(!state.staging_dir().exists());
        assert_eq!(
            fs::read_to_string(state.image_dir().join("img/pages-1.img")).unwrap(),
            "ours"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_image_published_first_by_another_pod_stays() {
        let dir = scratch("race");
        let (ours, theirs) = (open(&dir, "a"), open(&dir, "b"));
        for (state, content) in [(&theirs, "theirs"), (&ours, "ours")] {
            fs::create_dir_all(state.staging_dir()).unwrap();
            fs::write(state.staging_dir().join("inventory.img"), content).unwrap();
            state.publish().unwrap();
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
    fn clearing_removes_a_partial_image_and_tolerates_none() {
        let dir = scratch("clear");
        let state = open(&dir, "a");
        state.clear_staging().unwrap();
        fs::create_dir_all(state.staging_dir().join("img")).unwrap();
        fs::write(state.staging_dir().join("img/pages-1.img"), "x").unwrap();
        state.clear_staging().unwrap();
        assert!(!state.staging_dir().exists());
        fs::remove_dir_all(dir).unwrap();
    }
}
