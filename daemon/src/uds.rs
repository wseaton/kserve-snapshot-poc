//! gRPC over a Unix socket, for the node's own services.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;

pub async fn channel(socket: &Path) -> Result<Channel> {
    let path: PathBuf = socket.to_path_buf();
    Endpoint::from_static("http://unix.socket")
        .connect_with_connector(service_fn(move |_: Uri| {
            let path = path.clone();
            async move { UnixStream::connect(path).await.map(TokioIo::new) }
        }))
        .await
        .with_context(|| format!("connect to {}", socket.display()))
}
