//! Running CRIU inside a pod, from the node.
use std::fs;
use std::path::Path;

use anyhow::Result;
use tokio::process::Command;
use tracing::warn;

use crate::node::layout::TOOLS;
use crate::node::process::{self, HostPid};

/// `criu <action>` in every namespace of the pod whose init is `init`, and
/// in its cgroup, so what it restores is accounted and confined like the
/// pod's own processes. A process can only be moved into the pod's cgroup
/// from a cgroup namespace that sees both, which is the node's.
///
/// ```text
/// nsenter (node's cgroup namespace) ─► unshare (mounts) ─► mount cgroup2,
///   join the pod's cgroup ─► nsenter (all of the pod's namespaces) ─► criu
/// ```
///
/// CRIU leaves cgroups alone, since its children are already in the right
/// one, and does not lock the network: the connections it keeps are between
/// the processes it has frozen.
pub fn command(init: HostPid, action: &str) -> Command {
    let enter = format!(
        "mount -t cgroup2 none /mnt \
         && echo $$ > \"/mnt$(sed -n 's/^0:://p' /proc/{init}/cgroup)/cgroup.procs\" \
         && exec nsenter -t {init} -a -- \"$@\""
    );
    let mut command = Command::new("nsenter");
    command
        .args(["-t", "1", "-C", "--", "unshare", "-m", "--"])
        .args(["sh", "-c", &enter, "sh"])
        .arg(format!("{TOOLS}/criu"))
        .arg(action)
        .env("LD_LIBRARY_PATH", format!("{TOOLS}/lib"))
        .args(["--libdir", &format!("{TOOLS}/criu-plugins")])
        .args(["--manage-cgroups=ignore", "--network-lock", "skip"])
        .args(["--tcp-established", "--ext-unix-sk"])
        .args(["--file-locks", "--link-remap", "-v4"]);
    command
}

/// Run CRIU; if it fails, log what its own log at `log` says about why.
pub async fn run(command: &mut Command, what: &str, log: &Path) -> Result<()> {
    let outcome = process::run(command, what).await;
    if outcome.is_err() {
        for line in errors(&fs::read_to_string(log).unwrap_or_default()) {
            warn!("criu: {}", line.trim());
        }
    }
    outcome
}

/// The lines of a CRIU log that say why it failed: its errors, or its last
/// lines when it names none.
fn errors(log: &str) -> Vec<&str> {
    let lines: Vec<&str> = log.lines().collect();
    let errors: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| line.contains("Error"))
        .collect();
    if errors.is_empty() {
        lines[lines.len().saturating_sub(5)..].to_vec()
    } else {
        errors
    }
}

#[cfg(test)]
mod tests {
    use crate::node::criu::errors;

    #[test]
    fn a_log_is_cut_down_to_its_errors() {
        let log = "(00.001) Warn  (criu/x.c:1): a\n(00.002) Error (criu/y.c:2): first\n(00.003) fine\n(00.004) Error (criu/z.c:3): second\n";
        assert_eq!(
            errors(log),
            [
                "(00.002) Error (criu/y.c:2): first",
                "(00.004) Error (criu/z.c:3): second"
            ]
        );
        let quiet = "one\ntwo\nthree\nfour\nfive\nsix\n";
        assert_eq!(errors(quiet), ["two", "three", "four", "five", "six"]);
        assert!(errors("").is_empty());
    }
}
