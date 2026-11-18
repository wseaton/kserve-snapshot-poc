//! The files an agent shares with the serving container of its pod, and with
//! the other pods of its revision on the node.
//!
//! ```text
//! <state dir>/pull.status    written once by the agent: `local`, `pulled` or `none`
//! <store dir>/image/         the checkpoint of this revision; it appears whole, by rename
//! <store dir>/incoming-<id>/ a pull in progress
//! ```
//!
//! The state directory belongs to the pod. The store directory is on the
//! node's disk and outlives it, so the next pod of the revision on this node
//! restores without a transfer. The serving container publishes an image it
//! took the same way, by renaming a sibling directory to `image`.
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

/// How an agent came by an image, if it did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PullOutcome {
    /// The node already had one.
    Local,
    Pulled,
    None,
}

impl PullOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Pulled => "pulled",
            Self::None => "none",
        }
    }
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

pub struct State {
    dir: PathBuf,
    store: PathBuf,
    staging: PathBuf,
}

impl State {
    /// `id` names this pod's unfinished pull among those of other pods.
    pub fn open(dir: &Path, store: &Path, id: &str) -> Result<Self> {
        for dir in [dir, store] {
            fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        Ok(Self {
            dir: dir.to_path_buf(),
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

    pub fn pull_outcome(&self) -> Result<Option<PullOutcome>> {
        let path = self.dir.join("pull.status");
        let recorded = match fs::read_to_string(&path) {
            Ok(recorded) => recorded,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
        };
        match recorded.trim() {
            "local" => Ok(Some(PullOutcome::Local)),
            "pulled" => Ok(Some(PullOutcome::Pulled)),
            "none" => Ok(Some(PullOutcome::None)),
            other => bail!("{} holds {other:?}", path.display()),
        }
    }

    pub fn record_pull(&self, outcome: PullOutcome) -> Result<()> {
        let staged = self.dir.join("pull.status.tmp");
        fs::write(&staged, format!("{}\n", outcome.as_str()))
            .with_context(|| format!("write {}", staged.display()))?;
        fs::rename(&staged, self.dir.join("pull.status")).context("publish pull.status")
    }

    /// Remove what an unfinished pull left behind.
    pub fn clear_staging(&self) -> Result<()> {
        match fs::remove_dir_all(&self.staging) {
            Err(error) if error.kind() != ErrorKind::NotFound => {
                Err(error).context("remove the unfinished pull")
            }
            _ => Ok(()),
        }
    }

    /// Make the finished pull the node's image of this revision. If another
    /// pod published one first, that one stays and the pull is dropped.
    pub fn publish(&self) -> Result<()> {
        match fs::rename(&self.staging, self.image_dir()) {
            Ok(()) => Ok(()),
            Err(_) if self.image_ready() => self.clear_staging(),
            Err(error) => Err(error).context("publish the pulled image"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use crate::state::{PullOutcome, State, ensure_room};

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-state-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn open(dir: &Path, id: &str) -> State {
        State::open(&dir.join("pod"), &dir.join("store/rev"), id).unwrap()
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
    fn a_new_state_has_no_outcome_and_no_image() {
        let dir = scratch("new");
        let state = open(&dir, "a");
        assert_eq!(state.pull_outcome().unwrap(), None);
        assert!(!state.image_ready());
        assert_eq!(state.image_dir(), dir.join("store/rev/image"));
        assert_eq!(state.staging_dir(), dir.join("store/rev/incoming-a"));
        assert!(dir.join("pod").is_dir() && dir.join("store/rev").is_dir());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_recorded_outcome_is_read_back() {
        let dir = scratch("outcome");
        let state = open(&dir, "a");
        for outcome in [PullOutcome::None, PullOutcome::Local, PullOutcome::Pulled] {
            state.record_pull(outcome).unwrap();
            assert_eq!(state.pull_outcome().unwrap(), Some(outcome));
            assert_eq!(open(&dir, "a").pull_outcome().unwrap(), Some(outcome));
        }
        assert_eq!(
            fs::read_to_string(dir.join("pod/pull.status")).unwrap(),
            "pulled\n"
        );
        assert!(!dir.join("pod/pull.status.tmp").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_unknown_outcome_is_an_error() {
        let dir = scratch("unknown");
        let state = open(&dir, "a");
        fs::write(dir.join("pod/pull.status"), "maybe\n").unwrap();
        assert!(state.pull_outcome().is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_published_pull_becomes_the_image() {
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
    fn publishing_without_a_pull_is_an_error() {
        let dir = scratch("nothing");
        assert!(open(&dir, "a").publish().is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn clearing_removes_a_partial_pull_and_tolerates_none() {
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
