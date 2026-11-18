//! Where the daemon keeps its things inside a workload pod.

pub const TOOLS: &str = "/run/snapshotd/tools";
pub const IMAGE: &str = "/run/snapshotd/image";
pub const WORK: &str = "/run/snapshotd/work";
/// The cuInterpose bundle, under the tools directory.
pub const CUINTERPOSE: &str = "cuinterpose";
