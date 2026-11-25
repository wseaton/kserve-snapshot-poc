//! A node's checkpoint stores. Each class of store is a directory, with one
//! directory per revision of each workload:
//!
//! ```text
//! <class>/<namespace>/<workload>/<hash>/image/           the checkpoint; it appears whole, by rename
//! <class>/<namespace>/<workload>/<hash>/image/manifest.json  how to restore it, and who captured it
//! <class>/<namespace>/<workload>/<hash>/incoming-<pod>/  a capture or a pull in progress
//! <class>/<namespace>/<workload>/<hash>/failed-<pod>     why this pod no longer tries
//! <class>/<namespace>/<workload>/<hash>/used             touched when a pod of the revision starts
//! ```
//!
//! `<workload>` names a role and a group member too when the pod has them; see
//! [`Workload`].
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use tracing::warn;

pub const DEFAULT_CLASS: &str = "default";
pub const MANIFEST: &str = "manifest.json";
const USED: &str = "used";

/// Separates a workload's name from its role, and the role from the member.
/// Label values contain neither.
const ROLE_SEPARATOR: char = '@';
const MEMBER_SEPARATOR: char = '+';

/// What a checkpoint is of: a workload, the role it plays in a disaggregated
/// deployment, and which pod of a multi-pod group it is. Its path component
/// is `name`, `name@role`, `name+member` or `name@role+member`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Workload {
    name: String,
    role: Option<String>,
    member: Option<u32>,
}

impl Workload {
    pub fn new(name: &str, role: Option<&str>, member: Option<u32>) -> Result<Self> {
        ensure!(valid_part(name), "{name:?} cannot name a workload");
        if let Some(role) = role {
            ensure!(valid_part(role), "{role:?} cannot name a role");
        }
        Ok(Self {
            name: name.to_string(),
            role: role.map(str::to_string),
            member,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Which pod of its group this is, when the group has more than one.
    pub fn member(&self) -> Option<u32> {
        self.member
    }
}

impl FromStr for Workload {
    type Err = anyhow::Error;

    fn from_str(component: &str) -> Result<Self> {
        let (rest, member) = match component.split_once(MEMBER_SEPARATOR) {
            Some((rest, member)) => {
                ensure!(
                    !member.is_empty() && member.chars().all(|c| c.is_ascii_digit()),
                    "{member:?} cannot name a member"
                );
                (rest, Some(member.parse().context("a member index")?))
            }
            None => (component, None),
        };
        let (name, role) = match rest.split_once(ROLE_SEPARATOR) {
            Some((name, role)) => (name, Some(role)),
            None => (rest, None),
        };
        Self::new(name, role, member)
    }
}

impl fmt::Display for Workload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)?;
        if let Some(role) = &self.role {
            write!(f, "{ROLE_SEPARATOR}{role}")?;
        }
        if let Some(member) = self.member {
            write!(f, "{MEMBER_SEPARATOR}{member}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RevisionKey {
    namespace: String,
    workload: Workload,
    hash: String,
}

impl RevisionKey {
    pub fn new(namespace: &str, workload: Workload, hash: &str) -> Result<Self> {
        for part in [namespace, hash] {
            ensure!(valid_part(part), "{part:?} cannot name part of a revision");
        }
        Ok(Self {
            namespace: namespace.to_string(),
            workload,
            hash: hash.to_string(),
        })
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn workload(&self) -> &Workload {
        &self.workload
    }

    fn dir(&self, store: &Path) -> PathBuf {
        store
            .join(&self.namespace)
            .join(self.workload.to_string())
            .join(&self.hash)
    }

    #[cfg(any(test, feature = "p2p"))]
    pub fn image_dir(&self, store: &Path) -> PathBuf {
        self.dir(store).join("image")
    }
}

fn valid_part(part: &str) -> bool {
    !part.is_empty()
        && !part.starts_with('.')
        && part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

impl FromStr for RevisionKey {
    type Err = anyhow::Error;

    fn from_str(name: &str) -> Result<Self> {
        let parts: Vec<&str> = name.split('/').collect();
        let [namespace, workload, hash] = parts[..] else {
            return Err(StoreError::NotARevision {
                name: name.to_string(),
            }
            .into());
        };
        Self::new(namespace, workload.parse()?, hash)
    }
}

impl fmt::Display for RevisionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.namespace, self.workload, self.hash)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("{name:?} is not <namespace>/<workload>/<hash>")]
    NotARevision { name: String },
    #[error("this node has no store class {name:?}")]
    NoClass { name: String },
    #[error("{text:?} is not controller:<uid> or serviceaccount:<name>")]
    NotAPrincipal { text: String },
}

#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("read {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {}", path.display())]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

/// The controller a pod's workload is named after, as the API server
/// reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Owner {
    pub api_version: String,
    pub kind: String,
    pub name: String,
    pub uid: String,
}

impl fmt::Display for Owner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} ({})", self.kind, self.name, self.uid)
    }
}

/// The pod that captured an image, kept in the image's manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attestation {
    pub namespace: String,
    pub pod: String,
    pub uid: String,
    pub service_account: String,
    /// `None` for a pod that no controller owns.
    pub owner: Option<Owner>,
    /// The revision the pod was keyed to.
    pub revision: String,
}

impl Attestation {
    /// Whether pods attested as `self` and `other` may use each other's
    /// image: they have the same controller, or no controller and the same
    /// ServiceAccount.
    pub fn same_principal(&self, other: &Self) -> bool {
        self.namespace == other.namespace && self.principal() == other.principal()
    }

    /// Who a pod must share with the capturing pod to use its image.
    pub fn principal(&self) -> Principal {
        match &self.owner {
            Some(owner) => Principal::Controller {
                uid: owner.uid.clone(),
            },
            None => Principal::ServiceAccount {
                name: self.service_account.clone(),
            },
        }
    }

    /// Whether the image in `image` may be used by pods of `wanted`. An
    /// image captured before images were attested may be used by anyone; one
    /// whose attestation cannot be read by no one.
    #[cfg(any(test, feature = "p2p"))]
    pub fn allows(image: &Path, wanted: &Principal) -> bool {
        match Self::of_image(image) {
            Ok(Some(attestation)) => attestation.principal() == *wanted,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    /// `None` for an image captured before images were attested.
    pub fn of_image(image: &Path) -> Result<Option<Self>, AttestationError> {
        #[derive(Deserialize)]
        struct Attested {
            #[serde(default)]
            attestation: Option<Attestation>,
        }
        let path = image.join(MANIFEST);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(source) => return Err(AttestationError::Read { path, source }),
        };
        match serde_json::from_slice::<Attested>(&bytes) {
            Ok(attested) => Ok(attested.attestation),
            Err(source) => Err(AttestationError::Parse { path, source }),
        }
    }
}

impl fmt::Display for Attestation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pod = format!("pod {}/{} ({})", self.namespace, self.pod, self.uid);
        match &self.owner {
            Some(owner) => write!(f, "{owner} through {pod}"),
            None => write!(
                f,
                "{pod}, owned by no controller, as ServiceAccount {}",
                self.service_account
            ),
        }
    }
}

/// What pods of one namespace share an image by: the UID of their
/// controller, or for pods no controller owns, their ServiceAccount.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Principal {
    Controller { uid: String },
    ServiceAccount { name: String },
}

const CONTROLLER_PRINCIPAL: &str = "controller:";
const SERVICE_ACCOUNT_PRINCIPAL: &str = "serviceaccount:";

impl fmt::Display for Principal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Controller { uid } => write!(f, "{CONTROLLER_PRINCIPAL}{uid}"),
            Self::ServiceAccount { name } => write!(f, "{SERVICE_ACCOUNT_PRINCIPAL}{name}"),
        }
    }
}

impl FromStr for Principal {
    type Err = StoreError;

    fn from_str(text: &str) -> Result<Self, StoreError> {
        let not_one = || StoreError::NotAPrincipal {
            text: text.to_string(),
        };
        let principal = if let Some(uid) = text.strip_prefix(CONTROLLER_PRINCIPAL) {
            Self::Controller {
                uid: uid.to_string(),
            }
        } else if let Some(name) = text.strip_prefix(SERVICE_ACCOUNT_PRINCIPAL) {
            Self::ServiceAccount {
                name: name.to_string(),
            }
        } else {
            return Err(not_one());
        };
        let (Self::Controller { uid: part } | Self::ServiceAccount { name: part }) = &principal;
        if valid_part(part) {
            Ok(principal)
        } else {
            Err(not_one())
        }
    }
}

/// What became of a staging directory on [`RevisionStore::publish`].
#[derive(Debug, PartialEq, Eq)]
pub enum Published {
    /// It is the image now.
    Ours,
    /// An image of the same principal, or one captured before images were
    /// attested, was there first. That image stays.
    Kept,
    /// An image of another principal was there first. That image stays.
    Collided(Box<Collision>),
}

/// Who captured each of two images of one revision. `None` for an image
/// without a readable attestation.
#[derive(Debug, PartialEq, Eq)]
pub struct Collision {
    pub kept: Option<Attestation>,
    pub dropped: Option<Attestation>,
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
                .entry((revision.namespace, revision.workload.to_string()))
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
                if let Ok(revision) =
                    format!("{namespace}/{workload}/{hash}").parse::<RevisionKey>()
                {
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
            None => Err(StoreError::NoClass {
                name: name.to_string(),
            }
            .into()),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Class)> {
        self.0.iter().map(|(name, class)| (name.as_str(), class))
    }

    /// In whichever class holds it.
    #[cfg(any(test, feature = "p2p"))]
    pub fn image_of(&self, revision: &RevisionKey) -> Option<PathBuf> {
        self.0
            .values()
            .map(|class| revision.image_dir(&class.dir))
            .find(|image| image.is_dir())
    }
}

fn described(attestation: Option<&Attestation>) -> String {
    attestation.map_or_else(
        || "an image without a readable attestation".to_string(),
        ToString::to_string,
    )
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

    /// If another pod published first, that image stays and this one is
    /// dropped. A pod of another principal publishing first is logged as a
    /// security event.
    pub fn publish(&self) -> Result<Published> {
        match fs::rename(&self.staging, self.image_dir()) {
            Ok(()) => Ok(Published::Ours),
            Err(_) if self.image_ready() => {
                let kept = Attestation::of_image(&self.image_dir());
                let dropped = Attestation::of_image(&self.staging);
                self.clear_staging()?;
                Ok(match (kept, dropped) {
                    (Ok(None), _) => Published::Kept,
                    (Ok(Some(kept)), Ok(Some(dropped))) if kept.same_principal(&dropped) => {
                        Published::Kept
                    }
                    (kept, dropped) => {
                        let (kept, dropped) = (kept.ok().flatten(), dropped.ok().flatten());
                        warn!(
                            security = "checkpoint-collision",
                            image = %self.image_dir().display(),
                            kept = %described(kept.as_ref()),
                            dropped = %described(dropped.as_ref()),
                            "two principals captured one revision; the image already there stays"
                        );
                        Published::Collided(Box::new(Collision { kept, dropped }))
                    }
                })
            }
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
        Attestation, Class, Classes, Collision, MANIFEST, Owner, Principal, Published, RevisionKey,
        RevisionStore, USED, Usage, Workload, directory_bytes, ensure_room, revisions,
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
        let key: RevisionKey = "serving/qwen3-8b/7b9f694bf8".parse().unwrap();
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

    fn key(role: Option<&str>, member: Option<u32>) -> RevisionKey {
        RevisionKey::new(
            "serving",
            Workload::new("qwen", role, member).unwrap(),
            "5d8c",
        )
        .unwrap()
    }

    #[test]
    fn roles_and_group_members_keep_their_own_revision_of_a_workload() {
        let cases = [
            (None, None, "serving/qwen/5d8c"),
            (Some("prefill"), None, "serving/qwen@prefill/5d8c"),
            (Some("decode"), None, "serving/qwen@decode/5d8c"),
            (None, Some(0), "serving/qwen+0/5d8c"),
            (Some("decode"), Some(0), "serving/qwen@decode+0/5d8c"),
            (Some("decode"), Some(12), "serving/qwen@decode+12/5d8c"),
        ];
        let keys: Vec<RevisionKey> = cases
            .iter()
            .map(|(role, member, name)| {
                let key = key(*role, *member);
                assert_eq!(key.to_string(), *name);
                assert_eq!(revision(name), key, "{name} parses back");
                assert_eq!(key.workload().member(), *member);
                key
            })
            .collect();
        for (i, a) in keys.iter().enumerate() {
            for b in &keys[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert_eq!(
            key(Some("decode"), Some(1)).image_dir(Path::new("/store")),
            Path::new("/store/serving/qwen@decode+1/5d8c/image")
        );

        for (name, role) in [
            ("qwen", Some("")),
            ("qwen", Some("pre@fill")),
            ("qwen", Some("pre+fill")),
            ("qwen", Some(".hidden")),
            ("qwen", Some("pre/fill")),
            ("qw@en", None),
            ("qw+en", None),
            ("", Some("prefill")),
        ] {
            assert!(
                Workload::new(name, role, None).is_err(),
                "{name:?} {role:?}"
            );
        }
        for name in [
            "serving/qwen@/5d8c",
            "serving/@prefill/5d8c",
            "serving/qwen@pre@fill/5d8c",
            "serving/qwen@.prefill/5d8c",
            "serving/qwen+/5d8c",
            "serving/qwen+-1/5d8c",
            "serving/qwen+1x/5d8c",
            "serving/qwen+1+2/5d8c",
            "serving/qwen+99999999999/5d8c",
            "serving/qwen+1@decode/5d8c",
            "serving@x/qwen/5d8c",
            "serving/qwen/5d8c@x",
        ] {
            assert!(name.parse::<RevisionKey>().is_err(), "{name:?}");
        }
    }

    #[test]
    fn a_store_lists_each_role_of_a_workload_as_its_own_revision() {
        let dir = scratch("roles");
        held(&dir, "serving/qwen@prefill/5d8c", 10, 0);
        held(&dir, "serving/qwen@decode/5d8c", 20, 0);
        held(&dir, "serving/qwen@decode+1/5d8c", 20, 0);
        let mut found: Vec<String> = revisions(&dir)
            .into_iter()
            .map(|(revision, _)| revision.to_string())
            .collect();
        found.sort();
        assert_eq!(
            found,
            [
                "serving/qwen@decode+1/5d8c",
                "serving/qwen@decode/5d8c",
                "serving/qwen@prefill/5d8c"
            ]
        );
        fs::remove_dir_all(dir).unwrap();
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
        assert_eq!(store.publish().unwrap(), Published::Ours);
        assert!(store.image_ready());
        assert!(!store.staging_dir().exists());
        assert_eq!(
            fs::read_to_string(store.image_dir().join("img/pages-1.img")).unwrap(),
            "ours"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    fn deployment(name: &str, uid: &str) -> Owner {
        Owner {
            api_version: "apps/v1".to_string(),
            kind: "Deployment".to_string(),
            name: name.to_string(),
            uid: uid.to_string(),
        }
    }

    fn attested(pod: &str, owner: Option<Owner>, service_account: &str) -> Attestation {
        Attestation {
            namespace: "serving".to_string(),
            pod: pod.to_string(),
            uid: format!("{pod}-uid"),
            service_account: service_account.to_string(),
            owner,
            revision: "serving/qwen3-8b/7b9f694bf8".to_string(),
        }
    }

    /// Stage an image holding `content`, its manifest attested as `by`.
    fn stage(store: &RevisionStore, content: &str, by: Option<&Attestation>) {
        fs::create_dir_all(store.staging_dir()).unwrap();
        fs::write(store.staging_dir().join("inventory.img"), content).unwrap();
        let manifest = match by {
            Some(by) => serde_json::json!({ "attestation": by }),
            None => serde_json::json!({}),
        };
        fs::write(store.staging_dir().join(MANIFEST), manifest.to_string()).unwrap();
    }

    fn image_content(store: &RevisionStore) -> String {
        fs::read_to_string(store.image_dir().join("inventory.img")).unwrap()
    }

    #[test]
    fn an_image_published_first_by_another_pod_of_its_controller_stays() {
        let dir = scratch("race");
        let (ours, theirs) = (open(&dir, "a"), open(&dir, "b"));
        let owner = Some(deployment("qwen3-8b", "d-1"));
        stage(
            &theirs,
            "theirs",
            Some(&attested("b", owner.clone(), "default")),
        );
        assert_eq!(theirs.publish().unwrap(), Published::Ours);
        stage(&ours, "ours", Some(&attested("a", owner, "other")));
        assert_eq!(ours.publish().unwrap(), Published::Kept);
        assert_eq!(image_content(&ours), "theirs");
        assert!(!ours.staging_dir().exists() && !theirs.staging_dir().exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn publishing_over_an_image_of_another_principal_is_refused_and_it_survives() {
        let dir = scratch("collision");
        let qwen_pod = attested(
            "qwen-replica",
            Some(deployment("qwen3-8b", "d-1")),
            "default",
        );
        let stranger = attested("stranger", None, "default");
        let lookalike = attested("lookalike", Some(deployment("qwen3-8b", "d-2")), "default");
        let cases = [
            (&stranger, &qwen_pod),
            (&qwen_pod, &stranger),
            (&qwen_pod, &lookalike),
        ];
        for (number, (first, second)) in cases.into_iter().enumerate() {
            let (theirs, ours) = (open(&dir, "first"), open(&dir, "second"));
            stage(&theirs, "first", Some(first));
            assert_eq!(theirs.publish().unwrap(), Published::Ours, "case {number}");
            stage(&ours, "second", Some(second));
            assert_eq!(
                ours.publish().unwrap(),
                Published::Collided(Box::new(Collision {
                    kept: Some(first.clone()),
                    dropped: Some(second.clone()),
                })),
                "case {number}"
            );
            assert_eq!(image_content(&ours), "first", "case {number}");
            assert_eq!(
                Attestation::of_image(&ours.image_dir()).unwrap().as_ref(),
                Some(first),
                "case {number}"
            );
            assert!(!ours.staging_dir().exists(), "case {number}");
            fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn an_image_captured_before_attestation_is_kept_and_counts_as_unowned() {
        let dir = scratch("legacy");
        let (old, new) = (open(&dir, "old"), open(&dir, "new"));
        stage(&old, "old", None);
        assert_eq!(old.publish().unwrap(), Published::Ours);
        assert_eq!(Attestation::of_image(&old.image_dir()).unwrap(), None);
        stage(
            &new,
            "new",
            Some(&attested(
                "new",
                Some(deployment("qwen3-8b", "d-1")),
                "default",
            )),
        );
        assert_eq!(new.publish().unwrap(), Published::Kept);
        assert_eq!(image_content(&new), "old");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_image_without_a_readable_manifest_collides_with_anything() {
        let dir = scratch("unreadable");
        let (theirs, ours) = (open(&dir, "a"), open(&dir, "b"));
        stage(&theirs, "theirs", None);
        fs::write(theirs.staging_dir().join(MANIFEST), "{").unwrap();
        theirs.publish().unwrap();
        assert!(Attestation::of_image(&theirs.image_dir()).is_err());
        let ours_by = attested("b", None, "default");
        stage(&ours, "ours", Some(&ours_by));
        assert_eq!(
            ours.publish().unwrap(),
            Published::Collided(Box::new(Collision {
                kept: None,
                dropped: Some(ours_by),
            }))
        );
        assert_eq!(image_content(&ours), "theirs");
        fs::remove_dir_all(&dir).unwrap();
        assert!(Attestation::of_image(&dir).is_err());
    }

    #[test]
    fn principals_are_a_controller_or_an_unowned_service_account() {
        let qwen = || Some(deployment("qwen3-8b", "d-1"));
        let base = attested("a", qwen(), "default");
        let same = [
            attested("b", qwen(), "default"),
            attested("b", qwen(), "other-sa"),
        ];
        for other in &same {
            assert!(
                base.same_principal(other) && other.same_principal(&base),
                "{other}"
            );
        }
        let different = [
            attested("b", Some(deployment("qwen3-8b", "d-2")), "default"),
            attested("b", None, "default"),
            Attestation {
                namespace: "batch".to_string(),
                ..attested("b", qwen(), "default")
            },
        ];
        for other in &different {
            assert!(
                !base.same_principal(other) && !other.same_principal(&base),
                "{other}"
            );
        }
        let unowned = attested("a", None, "default");
        assert!(unowned.same_principal(&attested("b", None, "default")));
        assert!(!unowned.same_principal(&attested("b", None, "other-sa")));
        assert_eq!(
            unowned.to_string(),
            "pod serving/a (a-uid), owned by no controller, as ServiceAccount default"
        );
        assert_eq!(
            base.to_string(),
            "Deployment qwen3-8b (d-1) through pod serving/a (a-uid)"
        );
    }

    #[test]
    fn a_principal_is_named_in_a_request_and_parsed_back() {
        let controller = attested("a", Some(deployment("qwen3-8b", "d-1")), "default").principal();
        let unowned = attested("a", None, "qwen").principal();
        assert_eq!(
            controller,
            Principal::Controller {
                uid: "d-1".to_string()
            }
        );
        assert_eq!(controller.to_string(), "controller:d-1");
        assert_eq!(unowned.to_string(), "serviceaccount:qwen");
        for principal in [&controller, &unowned] {
            assert_eq!(
                &principal.to_string().parse::<Principal>().unwrap(),
                principal
            );
        }
        for text in [
            "",
            "d-1",
            "controller:",
            "serviceaccount:",
            "controller:../d-1",
            "controller:d 1",
            "serviceaccount:a/b",
            "owner:d-1",
            "controller:.hidden",
        ] {
            assert!(text.parse::<Principal>().is_err(), "{text:?}");
        }
    }

    #[test]
    fn an_image_allows_the_principal_that_captured_it() {
        let dir = scratch("allows");
        let store = open(&dir, "a");
        let qwen = attested("a", Some(deployment("qwen3-8b", "d-1")), "default");
        let controller = qwen.principal();
        let other = Principal::Controller {
            uid: "d-2".to_string(),
        };
        stage(&store, "ours", Some(&qwen));
        store.publish().unwrap();
        assert!(Attestation::allows(&store.image_dir(), &controller));
        assert!(!Attestation::allows(&store.image_dir(), &other));
        assert!(!Attestation::allows(
            &store.image_dir(),
            &Principal::ServiceAccount {
                name: "default".to_string()
            }
        ));

        fs::write(store.image_dir().join(MANIFEST), "{}").unwrap();
        assert!(
            Attestation::allows(&store.image_dir(), &other),
            "an image captured before attestation is anyone's"
        );
        fs::write(store.image_dir().join(MANIFEST), "{").unwrap();
        assert!(!Attestation::allows(&store.image_dir(), &controller));
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
