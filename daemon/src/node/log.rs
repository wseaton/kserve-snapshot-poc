//! How the daemon words and times what it reports.
use std::future::Future;

use anyhow::Result;
use tokio::time::Instant;
use tracing::{info, warn};

/// What a pod's log lines are tagged with: the part of its name that tells it
/// from the other pods of its workload.
pub fn short_id(pod: &str) -> &str {
    pod.rsplit('-').next().unwrap_or(pod)
}

pub fn seconds(since: Instant) -> String {
    format!("{:.2}", since.elapsed().as_secs_f64())
}

pub fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1u64 << 30) as f64)
}

/// Run one step of a checkpoint or restore and log how long it took.
pub async fn step<T>(name: &'static str, work: impl Future<Output = Result<T>>) -> Result<T> {
    let started = Instant::now();
    let outcome = work.await;
    match &outcome {
        Ok(_) => info!(seconds = %seconds(started), "{name}"),
        Err(_) => warn!(seconds = %seconds(started), "{name} failed"),
    }
    outcome
}

/// `line` without terminal colour sequences.
pub fn plain(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut rest = line.chars();
    while let Some(c) = rest.next() {
        if c == '\u{1b}' {
            for c in rest.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}

#[cfg(test)]
mod tests {
    use crate::node::log::plain;

    #[test]
    fn colour_sequences_are_removed() {
        let line = "\u{1b}[0;36m(RustFrontend pid=421)\u{1b}[0m \u{1b}[32mINFO\u{1b}[0m ready";
        assert_eq!(plain(line), "(RustFrontend pid=421) INFO ready");
        assert_eq!(plain("no colours"), "no colours");
        assert_eq!(plain("cut off \u{1b}[3"), "cut off ");
    }
}
