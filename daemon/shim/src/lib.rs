//! What the node daemon and the shim say to each other, one JSON line each
//! way, on the shim's socket.
use serde::{Deserialize, Serialize};

pub const SOCKET: &str = "/run/snapshotd/shim.sock";
pub const SERVER_LOG: &str = "/run/snapshotd/server.log";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Request {
    Start,
    Release,
    Adopt { pid: i32 },
    Serving,
    State,
    Fail,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reply {
    Started { pid: i32 },
    Done,
    State { started: bool, serving: bool },
    Refused { reason: String },
}
