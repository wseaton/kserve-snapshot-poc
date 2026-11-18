//! The container runtime's CRI service: which pods on this node opted in,
//! and the host PID of a pod's container.
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tonic::transport::Channel;

use crate::uds;

#[allow(clippy::all)]
pub mod pb {
    tonic::include_proto!("runtime.v1");
}

use pb::runtime_service_client::RuntimeServiceClient;

const CONTAINER_NAME: &str = "io.kubernetes.container.name";

/// A pod sandbox that is up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sandbox {
    pub id: String,
    pub uid: String,
    pub namespace: String,
    pub name: String,
    pub labels: HashMap<String, String>,
}

pub struct Runtime {
    client: RuntimeServiceClient<Channel>,
}

/// The part of a verbose container status both containerd and CRI-O fill in.
#[derive(Deserialize)]
struct ContainerInfo {
    pid: i32,
}

/// The host PID in a verbose container status.
pub fn container_pid(info: &HashMap<String, String>) -> Result<i32> {
    let info = info
        .get("info")
        .context("the runtime returned no verbose container info")?;
    let info: ContainerInfo =
        serde_json::from_str(info).context("parse the runtime's container info")?;
    if info.pid <= 0 {
        bail!("the container has no process (pid {})", info.pid);
    }
    Ok(info.pid)
}

impl Runtime {
    pub async fn connect(socket: &Path) -> Result<Self> {
        Ok(Self {
            client: RuntimeServiceClient::new(uds::channel(socket).await?),
        })
    }

    /// Ready pod sandboxes carrying `label`, whatever its value.
    pub async fn sandboxes(&mut self, label: &str) -> Result<Vec<Sandbox>> {
        let response = self
            .client
            .list_pod_sandbox(pb::ListPodSandboxRequest {
                filter: Some(pb::PodSandboxFilter {
                    state: Some(pb::PodSandboxStateValue {
                        state: pb::PodSandboxState::SandboxReady.into(),
                    }),
                    ..Default::default()
                }),
            })
            .await
            .context("ListPodSandbox")?;
        Ok(response
            .into_inner()
            .items
            .into_iter()
            .filter(|sandbox| sandbox.labels.contains_key(label))
            .filter_map(|sandbox| {
                let metadata = sandbox.metadata?;
                Some(Sandbox {
                    id: sandbox.id,
                    uid: metadata.uid,
                    namespace: metadata.namespace,
                    name: metadata.name,
                    labels: sandbox.labels,
                })
            })
            .collect())
    }

    pub async fn address(&mut self, sandbox: &Sandbox) -> Result<IpAddr> {
        let response = self
            .client
            .pod_sandbox_status(pb::PodSandboxStatusRequest {
                pod_sandbox_id: sandbox.id.clone(),
                verbose: false,
            })
            .await
            .context("PodSandboxStatus")?;
        let ip = response
            .into_inner()
            .status
            .and_then(|status| status.network)
            .map(|network| network.ip)
            .context("the sandbox has no network status")?;
        ip.parse()
            .with_context(|| format!("pod address {ip:?} of {}", sandbox.name))
    }

    /// The host PID of the running container called `container` in
    /// `sandbox`, once it has one.
    pub async fn pid(&mut self, sandbox: &Sandbox, container: &str) -> Result<Option<i32>> {
        let containers = self
            .client
            .list_containers(pb::ListContainersRequest {
                filter: Some(pb::ContainerFilter {
                    pod_sandbox_id: sandbox.id.clone(),
                    state: Some(pb::ContainerStateValue {
                        state: pb::ContainerState::ContainerRunning.into(),
                    }),
                    ..Default::default()
                }),
            })
            .await
            .context("ListContainers")?
            .into_inner()
            .containers;
        let Some(found) = containers.into_iter().find(|c| {
            c.labels
                .get(CONTAINER_NAME)
                .is_some_and(|name| name == container)
        }) else {
            return Ok(None);
        };
        let status = self
            .client
            .container_status(pb::ContainerStatusRequest {
                container_id: found.id,
                verbose: true,
            })
            .await
            .context("ContainerStatus")?
            .into_inner();
        container_pid(&status.info).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::cri::container_pid;

    fn info(json: &str) -> HashMap<String, String> {
        HashMap::from([("info".to_string(), json.to_string())])
    }

    #[test]
    fn the_pid_is_read_from_containerd_and_crio_info() {
        let containerd = r#"{"sandboxID":"abc","pid":804432,"removing":false,"runtimeSpec":{"process":{"args":["bash"]}}}"#;
        assert_eq!(container_pid(&info(containerd)).unwrap(), 804432);
        let crio = r#"{"sandboxID":"abc","pid":4242,"runtimeSpec":{},"privileged":false}"#;
        assert_eq!(container_pid(&info(crio)).unwrap(), 4242);
    }

    #[test]
    fn a_container_without_a_process_is_an_error() {
        assert!(container_pid(&info(r#"{"pid":0}"#)).is_err());
        assert!(container_pid(&info(r#"{"sandboxID":"abc"}"#)).is_err());
        assert!(container_pid(&info("not json")).is_err());
        assert!(container_pid(&HashMap::new()).is_err());
    }
}
