use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use tokio::time::Instant;

/// The part of a pod's name that tells it from its siblings. A pod synced
/// from a vCluster is named `<name>-x-<namespace>-x-<vcluster>` on the host.
pub fn short_id(pod: &str) -> &str {
    let name = pod.split("-x-").next().unwrap_or(pod);
    name.rsplit('-').next().unwrap_or(name)
}

pub fn seconds(since: Instant) -> String {
    format!("{:.2}", since.elapsed().as_secs_f64())
}

pub fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1u64 << 30) as f64)
}

/// The last `cap` bytes of a file, or nothing if it cannot be read.
pub fn tail(path: &Path, cap: usize) -> Vec<u8> {
    let cap = u64::try_from(cap).unwrap_or(u64::MAX);
    let read = || -> std::io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let length = file.metadata()?.len();
        file.seek(SeekFrom::Start(length.saturating_sub(cap)))?;
        let mut kept = Vec::new();
        file.take(cap).read_to_end(&mut kept)?;
        Ok(kept)
    };
    read().unwrap_or_default()
}

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
    use std::fs;

    use crate::node::report::{plain, short_id, tail};

    #[test]
    fn colour_sequences_are_removed() {
        let line = "\u{1b}[0;36m(RustFrontend pid=421)\u{1b}[0m \u{1b}[32mINFO\u{1b}[0m ready";
        assert_eq!(plain(line), "(RustFrontend pid=421) INFO ready");
        assert_eq!(plain("no colours"), "no colours");
        assert_eq!(plain("cut off \u{1b}[3"), "cut off ");
    }

    #[test]
    fn a_pod_is_tagged_with_the_end_of_its_own_name() {
        assert_eq!(short_id("snapshot-serve-85f787b8cb-cjv9n"), "cjv9n");
        assert_eq!(
            short_id("qwen3-8b-kserve-6f79c6cd9c-zw7mr-x-kds-smoke-x-kds"),
            "zw7mr"
        );
        assert_eq!(short_id("pod"), "pod");
    }

    #[test]
    fn a_tail_keeps_the_end_of_a_file_up_to_its_cap() {
        let dir = std::env::temp_dir().join(format!("snapshotd-tail-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("restore.log");
        fs::write(&log, "0123456789").unwrap();
        assert_eq!(tail(&log, 4), b"6789");
        assert_eq!(tail(&log, 10), b"0123456789");
        assert_eq!(tail(&log, 4096), b"0123456789");
        assert_eq!(tail(&log, 0), b"");
        assert!(tail(&dir.join("missing.log"), 4096).is_empty());
        fs::remove_dir_all(dir).unwrap();
    }
}
