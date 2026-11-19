use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::time::{Instant, sleep};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostPid(pub i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PodPid(pub i32);

impl fmt::Display for HostPid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for PodPid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl HostPid {
    pub fn root(self, path: &str) -> PathBuf {
        PathBuf::from(format!("/proc/{self}/root{path}"))
    }

    pub fn child(self, pid: PodPid) -> Result<HostPid> {
        let tasks = fs::read_dir(format!("/proc/{self}/task"))
            .with_context(|| format!("process {self} is gone"))?;
        for task in tasks.flatten() {
            let Ok(children) = fs::read_to_string(task.path().join("children")) else {
                continue;
            };
            for child in children.split_ascii_whitespace() {
                let Ok(status) = fs::read_to_string(format!("/proc/{child}/status")) else {
                    continue;
                };
                let inner = status
                    .lines()
                    .find_map(|line| line.strip_prefix("NSpid:"))
                    .and_then(|pids| pids.split_ascii_whitespace().last());
                if inner == Some(pid.to_string().as_str()) {
                    return child.parse().map(HostPid).context("child PID");
                }
            }
        }
        bail!("process {self} has no child that is {pid} in its PID namespace")
    }

    /// `None` on a node without SELinux.
    pub fn selinux_context(self) -> Option<String> {
        let attr = fs::read_to_string(format!("/proc/{self}/attr/current")).ok()?;
        selinux_context(&attr).map(str::to_owned)
    }

    pub async fn reaped(self, within: Duration) -> Result<()> {
        let deadline = Instant::now() + within;
        while Path::new(&format!("/proc/{self}")).exists() {
            ensure!(
                Instant::now() < deadline,
                "process {self} is still there {within:?} after its dump"
            );
            sleep(Duration::from_millis(20)).await;
        }
        Ok(())
    }
}

fn selinux_context(attr: &str) -> Option<&str> {
    let context = attr.trim_end_matches(['\0', '\n']);
    (context.matches(':').count() >= 3 && !context.contains(' ')).then_some(context)
}

pub async fn run(command: &mut Command, what: &str) -> Result<()> {
    let output = command
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("start {what}"))?;
    ensure!(
        output.status.success(),
        "{what} failed ({}): {}{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

pub async fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to).with_context(|| format!("create {}", to.display()))?;
    let mut source = from.as_os_str().to_owned();
    source.push("/.");
    run(
        Command::new("cp").arg("-a").arg(source).arg(to),
        &format!("copy {} to {}", from.display(), to.display()),
    )
    .await
}

pub async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || span.in_scope(work)).await?
}

#[cfg(test)]
mod tests {
    use std::process::{Command, Stdio};

    use crate::node::process::{HostPid, PodPid, selinux_context};

    fn sleeper() -> std::process::Child {
        Command::new("sleep")
            .arg("30")
            .stdout(Stdio::null())
            .spawn()
            .unwrap()
    }

    #[test]
    fn a_child_is_found_by_its_pid_in_the_namespace() {
        let mut child = sleeper();
        let own = HostPid(std::process::id() as i32);
        let pid = child.id() as i32;
        assert_eq!(own.child(PodPid(pid)).unwrap(), HostPid(pid));
        assert!(own.child(PodPid(0)).is_err());
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(own.child(PodPid(pid)).is_err());
        assert!(HostPid(i32::MAX).child(PodPid(1)).is_err());
    }

    #[test]
    fn only_an_selinux_context_is_taken_from_a_process_label() {
        assert_eq!(
            selinux_context("system_u:system_r:container_t:s0:c9,c39\0"),
            Some("system_u:system_r:container_t:s0:c9,c39")
        );
        assert_eq!(
            selinux_context("system_u:system_r:spc_t:s0\n"),
            Some("system_u:system_r:spc_t:s0")
        );
        assert_eq!(
            selinux_context("cri-containerd.apparmor.d (enforce)\n"),
            None
        );
        assert_eq!(selinux_context("unconfined\n"), None);
        assert_eq!(selinux_context("kernel"), None);
        assert_eq!(selinux_context(""), None);
    }

    #[tokio::test]
    async fn a_killed_process_is_reaped_only_once_its_parent_waits() {
        let mut child = sleeper();
        let pid = HostPid(child.id() as i32);
        child.kill().unwrap();
        let reaper = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            child.wait().unwrap();
        });
        let started = std::time::Instant::now();
        pid.reaped(std::time::Duration::from_secs(30))
            .await
            .unwrap();
        assert!(started.elapsed() >= std::time::Duration::from_millis(150));
        reaper.join().unwrap();
    }
}
