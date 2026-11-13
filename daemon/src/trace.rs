//! Per-chunk timeline of a transfer, written in the Chrome trace event
//! format so it opens in Perfetto or `chrome://tracing`.
//!
//! Timestamps are wall-clock microseconds, so the sender's and the receiver's
//! files can be laid on one timeline. Each slot of the registered pool is a
//! track; image files that wait for a slot get tracks of their own.
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use serde_json::{Value, json};

/// Tracks for image files start here, above any slot index.
pub const FILE_TRACKS: u64 = 1_000_000;

#[derive(Serialize)]
struct Event {
    name: &'static str,
    ph: &'static str,
    ts: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<u64>,
    pid: u32,
    tid: u64,
    args: Value,
}

/// Collects events when a path is given and does nothing otherwise.
pub struct Trace {
    path: Option<PathBuf>,
    process: &'static str,
    events: Mutex<Vec<Event>>,
}

/// Wall-clock microseconds since the epoch.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_micros() as u64)
}

impl Trace {
    pub fn new(path: Option<PathBuf>, process: &'static str) -> Self {
        Self {
            path,
            process,
            events: Mutex::new(Vec::new()),
        }
    }

    fn push(&self, event: Event) {
        if self.path.is_some()
            && let Ok(mut events) = self.events.lock()
        {
            events.push(event);
        }
    }

    /// A span on `track` that began at `start` and ends now.
    pub fn span(&self, name: &'static str, track: u64, start: u64, args: Value) {
        self.push(Event {
            name,
            ph: "X",
            ts: start,
            dur: Some(now().saturating_sub(start)),
            pid: 0,
            tid: track,
            args,
        });
    }

    /// The value of a counter now.
    pub fn counter(&self, name: &'static str, value: u64) {
        self.push(Event {
            name,
            ph: "C",
            ts: now(),
            dur: None,
            pid: 0,
            tid: 0,
            args: json!({ "value": value }),
        });
    }

    pub fn write(self) -> Result<()> {
        let Some(path) = self.path else { return Ok(()) };
        let events = self
            .events
            .into_inner()
            .map_err(|_| anyhow!("trace lock poisoned"))?;
        let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        serde_json::to_writer(
            BufWriter::new(file),
            &json!({ "process": self.process, "traceEvents": events }),
        )
        .with_context(|| format!("write {}", path.display()))
    }
}
