//! A node's checkpoint stores. Each class of store is a directory, with one
//! directory per revision of each workload:
//!
//! ```text
//! <class>/<namespace>/<workload>/<hash>/image/           the checkpoint; it appears whole, by rename
//! <class>/<namespace>/<workload>/<hash>/incoming-<pod>/  a capture or a pull in progress
//! <class>/<namespace>/<workload>/<hash>/failed-<pod>     why this pod no longer tries
//! <class>/<namespace>/<workload>/<hash>/used             touched when a pod of the revision starts
//! ```
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

pub const DEFAULT_CLASS: &str = "default";
const USED: &str = "used";

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

/// One place checkpoints are kept, and what it allows.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Class {
    pub dir: PathBuf,
    /// GiB the directory must have free before a checkpoint is taken.
    #[serde(default)]
    pub min_free_gib: u64,
    /// GiB one namespace may hold here before its checkpoints stop being taken.
    #[serde(default)]
    pub namespace_limit_gib: Option<u64>,
    /// Hours a revision is kept after a pod of it last started.
    #[serde(default)]
    pub retain_hours: Option<u64>,
}

impl Class {
    /// Whether a checkpoint of `revision` may be taken now.
    pub fn room(&self, revision: &RevisionKey) -> Result<()> {
        ensure_room(&self.dir, self.min_free_gib << 30)?;
        if let Some(limit) = self.namespace_limit_gib {
            let held = directory_bytes(&self.dir.join(&revision.namespace));
            ensure!(
                held < limit << 30,
                "namespace {} holds {held} bytes in {} and may hold {limit} GiB",
                revision.namespace,
                self.dir.display()
            );
        }
        Ok(())
    }

    pub fn usage(&self) -> Vec<Usage> {
        let mut held: BTreeMap<(String, String), (u64, u64)> = BTreeMap::new();
        for (revision, dir) in revisions(&self.dir) {
            let (bytes, count) = held
                .entry((revision.namespace, revision.workload))
                .or_default();
            *bytes += directory_bytes(&dir);
            *count += 1;
        }
        held.into_iter()
            .map(|((namespace, workload), (bytes, revisions))| Usage {
                namespace,
                workload,
                bytes,
                revisions,
            })
            .collect()
    }

    pub fn free_bytes(&self) -> Result<u64> {
        free_bytes(&self.dir)
    }

    /// Returns the revisions it removed and the bytes each held.
    pub fn sweep(&self, now: SystemTime) -> Result<Vec<(RevisionKey, u64)>> {
        let Some(hours) = self.retain_hours else {
            return Ok(Vec::new());
        };
        let kept = Duration::from_secs(hours.saturating_mul(3600));
        let mut removed = Vec::new();
        for (revision, dir) in revisions(&self.dir) {
            if now.duration_since(last_used(&dir)?).unwrap_or_default() <= kept {
                continue;
            }
            let bytes = directory_bytes(&dir);
            fs::remove_dir_all(&dir).with_context(|| format!("remove {}", dir.display()))?;
            for emptied in dir.ancestors().skip(1).take(2) {
                let _ = fs::remove_dir(emptied);
            }
            removed.push((revision, bytes));
        }
        Ok(removed)
    }
}

/// What one workload's revisions hold in a class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Usage {
    pub namespace: String,
    pub workload: String,
    pub bytes: u64,
    pub revisions: u64,
}

fn subdirectories(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| Some((entry.file_name().into_string().ok()?, entry.path())))
        .collect()
}

fn revisions(store: &Path) -> Vec<(RevisionKey, PathBuf)> {
    let mut found = Vec::new();
    for (namespace, dir) in subdirectories(store) {
        for (workload, dir) in subdirectories(&dir) {
            for (hash, dir) in subdirectories(&dir) {
                if let Ok(revision) = RevisionKey::new(&namespace, &workload, &hash) {
                    found.push((revision, dir));
                }
            }
        }
    }
    found
}

fn last_used(revision_dir: &Path) -> Result<SystemTime> {
    let marker = revision_dir.join(USED);
    let dated = if marker.exists() {
        marker.as_path()
    } else {
        revision_dir
    };
    fs::metadata(dated)
        .and_then(|metadata| metadata.modified())
        .with_context(|| format!("read when {} was last used", dated.display()))
}

/// The classes of store a daemon offers, by name. One is always `default`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classes(BTreeMap<String, Class>);

impl fmt::Display for Classes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let named: Vec<String> = self
            .iter()
            .map(|(name, class)| format!("{name}={}", class.dir.display()))
            .collect();
        f.write_str(&named.join(","))
    }
}

impl Classes {
    /// `default`, and the classes of the JSON object in `file` if one is given.
    pub fn load(default: Class, file: Option<&Path>) -> Result<Self> {
        let mut classes = match file {
            Some(file) => {
                let text =
                    fs::read_to_string(file).with_context(|| format!("read {}", file.display()))?;
                serde_json::from_str::<BTreeMap<String, Class>>(&text)
                    .with_context(|| format!("parse {}", file.display()))?
            }
            None => BTreeMap::new(),
        };
        ensure!(
            !classes.contains_key(DEFAULT_CLASS),
            "the class {DEFAULT_CLASS:?} is set by the daemon's own options"
        );
        classes.insert(DEFAULT_CLASS.to_string(), default);
        for (name, class) in &classes {
            fs::create_dir_all(&class.dir)
                .with_context(|| format!("create {} for class {name}", class.dir.display()))?;
        }
        Ok(Self(classes))
    }

    /// The class a pod named, or the default when it named none.
    pub fn get(&self, name: Option<&str>) -> Result<&Class> {
        let name = name.unwrap_or(DEFAULT_CLASS);
        match self.0.get(name) {
            Some(class) => Ok(class),
            None => bail!("this node has no store class {name:?}"),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Class)> {
        self.0.iter().map(|(name, class)| (name.as_str(), class))
    }

    /// In whichever class holds it.
    #[cfg(any(test, feature = "nixl"))]
    pub fn image_of(&self, revision: &RevisionKey) -> Option<PathBuf> {
        self.0
            .values()
            .map(|class| revision.image_dir(&class.dir))
            .find(|image| image.is_dir())
    }
}

fn free_bytes(dir: &Path) -> Result<u64> {
    let stat = rustix::fs::statvfs(dir).with_context(|| format!("stat {}", dir.display()))?;
    Ok(stat.f_bavail.saturating_mul(stat.f_frsize))
}

pub fn ensure_room(dir: &Path, bytes: u64) -> Result<()> {
    let free = free_bytes(dir)?;
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
    /// Opening a revision for a pod counts as using it.
    pub fn open(store: &Path, revision: &RevisionKey, pod_uid: &str) -> Result<Self> {
        let dir = revision.dir(store);
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        File::create(dir.join(USED))
            .and_then(|used| used.set_modified(SystemTime::now()))
            .with_context(|| format!("mark {} used", dir.display()))?;
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
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime};

    use anyhow::anyhow;

    use crate::store::{
        Class, Classes, RevisionKey, RevisionStore, USED, Usage, directory_bytes, ensure_room,
    };

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn class(dir: &Path) -> Class {
        Class {
            dir: dir.to_path_buf(),
            min_free_gib: 0,
            namespace_limit_gib: None,
            retain_hours: None,
        }
    }

    fn revision(name: &str) -> RevisionKey {
        name.parse().unwrap()
    }

    /// A revision with an image of `bytes`, last used `idle_hours` ago.
    fn held(store: &Path, name: &str, bytes: usize, idle_hours: u64) -> PathBuf {
        let key = revision(name);
        let opened = RevisionStore::open(store, &key, "pod").unwrap();
        fs::create_dir_all(opened.image_dir()).unwrap();
        fs::write(opened.image_dir().join("pages-1.img"), vec![0; bytes]).unwrap();
        let used = SystemTime::now() - Duration::from_secs(idle_hours * 3600);
        File::options()
            .write(true)
            .open(key.dir(store).join(USED))
            .unwrap()
            .set_modified(used)
            .unwrap();
        key.dir(store)
    }

    #[test]
    fn a_pod_gets_the_class_it_names_and_the_default_when_it_names_none() {
        let dir = scratch("classes");
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("classes.json");
        let shared = dir.join("shared");
        fs::write(
            &file,
            format!(
                r#"{{"shared": {{"dir": "{}", "namespace_limit_gib": 500, "retain_hours": 24}}}}"#,
                shared.display()
            ),
        )
        .unwrap();
        let classes = Classes::load(class(&dir.join("local")), Some(&file)).unwrap();
        assert_eq!(classes.get(None).unwrap().dir, dir.join("local"));
        assert_eq!(
            classes.get(Some("shared")).unwrap(),
            &Class {
                dir: shared.clone(),
                min_free_gib: 0,
                namespace_limit_gib: Some(500),
                retain_hours: Some(24),
            }
        );
        assert!(shared.is_dir() && dir.join("local").is_dir());
        assert!(classes.get(Some("fast")).is_err());
        assert_eq!(
            classes.to_string(),
            format!(
                "default={},shared={}",
                dir.join("local").display(),
                shared.display()
            )
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_classes_file_cannot_redefine_default_or_carry_unknown_fields() {
        let dir = scratch("bad-classes");
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("classes.json");
        for text in [
            r#"{"default": {"dir": "/elsewhere"}}"#,
            r#"{"shared": {"dir": "/shared", "retain_days": 1}}"#,
            r#"{"shared": {"min_free_gib": 1}}"#,
            "not json",
        ] {
            fs::write(&file, text).unwrap();
            assert!(Classes::load(class(&dir), Some(&file)).is_err(), "{text}");
        }
        assert!(Classes::load(class(&dir), Some(&dir.join("missing.json"))).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_image_is_found_in_whichever_class_holds_it() {
        let dir = scratch("image-of");
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("classes.json");
        fs::write(
            &file,
            format!(
                r#"{{"shared": {{"dir": "{}"}}}}"#,
                dir.join("shared").display()
            ),
        )
        .unwrap();
        let classes = Classes::load(class(&dir.join("local")), Some(&file)).unwrap();
        held(&dir.join("shared"), "serving/qwen/aaa", 10, 0);
        let opened = RevisionStore::open(&dir.join("local"), &revision("serving/qwen/bbb"), "pod");
        assert!(opened.is_ok());
        assert_eq!(
            classes.image_of(&revision("serving/qwen/aaa")),
            Some(dir.join("shared/serving/qwen/aaa/image"))
        );
        assert_eq!(classes.image_of(&revision("serving/qwen/bbb")), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_namespace_at_its_limit_gets_no_more_checkpoints() {
        let dir = scratch("limit");
        let limited = Class {
            namespace_limit_gib: Some(1),
            ..class(&dir)
        };
        let full = held(&dir, "serving/qwen/aaa", 16, 0).join("image/pages-2.img");
        limited.room(&revision("serving/glm/bbb")).unwrap();
        File::create(full).unwrap().set_len(1 << 30).unwrap();
        assert!(limited.room(&revision("serving/glm/bbb")).is_err());
        limited.room(&revision("batch/glm/bbb")).unwrap();
        class(&dir).room(&revision("serving/glm/bbb")).unwrap();
        let cramped = Class {
            min_free_gib: u64::MAX >> 30,
            ..class(&dir)
        };
        assert!(cramped.room(&revision("batch/glm/bbb")).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn only_revisions_idle_past_the_retention_are_swept() {
        let dir = scratch("sweep");
        let kept = Class {
            retain_hours: Some(24),
            ..class(&dir)
        };
        let old = held(&dir, "serving/qwen/aaa", 1000, 25);
        let recent = held(&dir, "serving/qwen/bbb", 10, 23);
        let alone = held(&dir, "batch/glm/ccc", 100, 48);
        assert!(class(&dir).sweep(SystemTime::now()).unwrap().is_empty());
        let mut removed = kept.sweep(SystemTime::now()).unwrap();
        removed.sort_by_key(|(_, bytes)| *bytes);
        assert_eq!(
            removed,
            [
                (revision("batch/glm/ccc"), 100),
                (revision("serving/qwen/aaa"), 1000)
            ]
        );
        assert!(!old.exists() && !alone.exists() && recent.join("image").is_dir());
        assert!(!dir.join("batch").exists() && dir.join("serving/qwen").is_dir());
        assert!(kept.sweep(SystemTime::now()).unwrap().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_pod_that_starts_marks_its_revision_used() {
        let dir = scratch("used");
        let kept = Class {
            retain_hours: Some(1),
            ..class(&dir)
        };
        let stale = held(&dir, "serving/qwen/aaa", 10, 5);
        RevisionStore::open(&dir, &revision("serving/qwen/aaa"), "another").unwrap();
        assert!(kept.sweep(SystemTime::now()).unwrap().is_empty());
        assert!(stale.join("image").is_dir());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn usage_adds_up_each_workloads_revisions() {
        let dir = scratch("usage");
        held(&dir, "serving/qwen/aaa", 1000, 0);
        held(&dir, "serving/qwen/bbb", 200, 0);
        held(&dir, "batch/glm/ccc", 30, 0);
        let usage = |namespace: &str, workload: &str, bytes, revisions| Usage {
            namespace: namespace.to_string(),
            workload: workload.to_string(),
            bytes,
            revisions,
        };
        assert_eq!(
            class(&dir).usage(),
            [
                usage("batch", "glm", 30, 1),
                usage("serving", "qwen", 1200, 2)
            ]
        );
        assert!(class(&dir.join("missing")).usage().is_empty());
        assert!(class(&dir).free_bytes().unwrap() > 0);
        fs::remove_dir_all(dir).unwrap();
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
