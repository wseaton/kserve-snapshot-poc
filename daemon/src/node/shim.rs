use std::time::Duration;

use anyhow::{Context, Result, ensure};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufStream};
use tokio::net::UnixStream;
use tokio::time::{Instant, sleep};

use crate::node::process::{HostPid, PodPid};
use snapshotd_shim::{Reply, Request, SOCKET};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub started: bool,
    pub serving: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ShimError {
    #[error("the shim refused {request:?}: {reason}")]
    Refused { request: Request, reason: String },
    #[error("unexpected {reply:?}")]
    Unexpected { reply: Reply },
}

pub struct ShimClient {
    stream: BufStream<UnixStream>,
}

impl ShimClient {
    pub async fn connect(init: HostPid, within: Duration) -> Result<Self> {
        let socket = init.root(SOCKET);
        let deadline = Instant::now() + within;
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
            Reply::Refused { reason } => Err(ShimError::Refused { request, reason }.into()),
            reply => Ok(reply),
        }
    }

    async fn done(&mut self, request: Request) -> Result<()> {
        match self.call(request).await? {
            Reply::Done => Ok(()),
            reply => Err(ShimError::Unexpected { reply }.into()),
        }
    }

    pub async fn progress(&mut self) -> Result<Progress> {
        match self.call(Request::State).await? {
            Reply::State { started, serving } => Ok(Progress { started, serving }),
            reply => Err(ShimError::Unexpected { reply }.into()),
        }
    }

    pub async fn start(&mut self) -> Result<PodPid> {
        match self.call(Request::Start).await? {
            Reply::Started { pid } => Ok(PodPid(pid)),
            reply => Err(ShimError::Unexpected { reply }.into()),
        }
    }

    pub async fn release(&mut self) -> Result<()> {
        self.done(Request::Release).await
    }

    pub async fn adopt(&mut self, pid: PodPid) -> Result<()> {
        self.done(Request::Adopt { pid: pid.0 }).await
    }

    pub async fn serving(&mut self) -> Result<()> {
        self.done(Request::Serving).await
    }

    pub async fn fail(&mut self) -> Result<()> {
        self.done(Request::Fail).await
    }
}
