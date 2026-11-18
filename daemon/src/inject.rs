//! Make a directory of this mount namespace appear inside a container's.
//!
//! ```text
//! open_tree(source, CLONE)  ─►  setns(container's mount namespace)  ─►  move_mount(tree, target)
//! ```
//!
//! `setns` to a mount namespace needs a single-threaded process, so this runs
//! as its own short-lived `snapshotd inject`.
use std::fs::{self, File};
use std::os::fd::AsFd;
use std::path::PathBuf;

use anyhow::{Context, Result};
use rustix::fs::CWD;
use rustix::mount::{MoveMountFlags, OpenTreeFlags, move_mount, open_tree};
use rustix::thread::{LinkNameSpaceType, move_into_link_name_space};

#[derive(clap::Args)]
pub struct Args {
    /// Host PID of a process in the container.
    #[arg(long)]
    pid: i32,
    /// Directory in this mount namespace.
    #[arg(long)]
    source: PathBuf,
    /// Where it appears in the container; created if missing.
    #[arg(long)]
    target: PathBuf,
}

pub fn run(args: &Args) -> Result<()> {
    let tree = open_tree(
        CWD,
        &args.source,
        OpenTreeFlags::OPEN_TREE_CLONE
            | OpenTreeFlags::OPEN_TREE_CLOEXEC
            | OpenTreeFlags::AT_RECURSIVE,
    )
    .with_context(|| format!("clone the mount at {}", args.source.display()))?;
    let namespace = File::open(format!("/proc/{}/ns/mnt", args.pid))
        .with_context(|| format!("open the mount namespace of process {}", args.pid))?;
    move_into_link_name_space(namespace.as_fd(), Some(LinkNameSpaceType::Mount))
        .context("enter the container's mount namespace")?;
    fs::create_dir_all(&args.target)
        .with_context(|| format!("create {} in the container", args.target.display()))?;
    move_mount(
        tree.as_fd(),
        "",
        CWD,
        &args.target,
        MoveMountFlags::MOVE_MOUNT_F_EMPTY_PATH,
    )
    .with_context(|| format!("mount at {} in the container", args.target.display()))
}
