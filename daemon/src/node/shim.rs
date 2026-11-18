//! The daemon's end of a pod's shim socket.
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufStream};
use tokio::net::UnixStream;
use tokio::time::{Instant, sleep};

use crate::node::process::{HostPid, PodPid};
use crate::shim::{Reply, Request, SOCKET};

const CONNECT_WAIT: Duration = Duration::from_secs(60);

/// How far a shim got before this daemon met it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub started: bool,
    pub serving: bool,
}

pub struct ShimClient {
    stream: BufStream<UnixStream>,
}

impl ShimClient {
    /// Connect through the pod's own filesystem; the shim, `init`, may still
    /// be starting.
    pub async fn connect(init: HostPid) -> Result<Self> {
        let socket = init.root(SOCKET);
        let deadline = Instant::now() + CONNECT_WAIT;
        loop {
            match UnixStream::connect(&socket).await {
                Ok(stream) => {
                    return Ok(Self {
                        stream: BufStream::new(stream),
                    });
                }
                Err(error) if Instant::now() >= deadline => {
                    return Err(error).with_context(|| format!("connect {}", socket.display()));
                }
                Err(_) => sleep(Duration::from_millis(200)).await,
            }
        }
    }

    async fn call(&mut self, request: Request) -> Result<Reply> {
        let mut line = serde_json::to_vec(&request)?;
        line.push(b'\n');
        self.stream.write_all(&line).await?;
        self.stream.flush().await?;
        let mut answer = String::new();
        ensure!(
            self.stream.read_line(&mut answer).await? > 0,
            "the shim closed the connection"
        );
        match serde_json::from_str(&answer).context("parse the shim's reply")? {
            Reply::Refused { reason } => bail!("the shim refused {request:?}: {reason}"),
            reply => Ok(reply),
        }
    }

    async fn done(&mut self, request: Request) -> Result<()> {
        match self.call(request).await? {
            Reply::Done => Ok(()),
            other => bail!("unexpected {other:?}"),
        }
    }

    pub async fn progress(&mut self) -> Result<Progress> {
        match self.call(Request::State).await? {
            Reply::State { started, serving } => Ok(Progress { started, serving }),
            other => bail!("unexpected {other:?}"),
        }
    }

    /// Start the server and return its process.
    pub async fn start(&mut self) -> Result<PodPid> {
        match self.call(Request::Start).await? {
            Reply::Started { pid } => Ok(PodPid(pid)),
            other => bail!("unexpected {other:?}"),
        }
    }

    /// The server is about to be checkpointed; its exit is expected.
    pub async fn release(&mut self) -> Result<()> {
        self.done(Request::Release).await
    }

    /// The server is now the restored process `pid`.
    pub async fn adopt(&mut self, pid: PodPid) -> Result<()> {
        self.done(Request::Adopt { pid: pid.0 }).await
    }

    /// Report the pod ready.
    pub async fn serving(&mut self) -> Result<()> {
        self.done(Request::Serving).await
    }

    /// Make the container exit and start over.
    pub async fn fail(&mut self) -> Result<()> {
        self.done(Request::Fail).await
    }
}
