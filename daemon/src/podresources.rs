//! The kubelet's PodResources service: which GPUs a container was given.
use std::path::Path;

use anyhow::{Context, Result};

use crate::cuda::GpuUuid;
use crate::uds;

#[allow(clippy::all)]
pub mod pb {
    tonic::include_proto!("v1");
}

use pb::pod_resources_lister_client::PodResourcesListerClient;

const GPU: &str = "nvidia.com/gpu";

/// The GPUs of `container` in the pod `namespace/name`, in the order the
/// kubelet lists them.
pub fn gpus_of(
    response: &pb::ListPodResourcesResponse,
    namespace: &str,
    name: &str,
    container: &str,
) -> Result<Vec<GpuUuid>> {
    let pod = response
        .pod_resources
        .iter()
        .find(|pod| pod.namespace == namespace && pod.name == name)
        .with_context(|| format!("the kubelet lists no pod {namespace}/{name}"))?;
    let container = pod
        .containers
        .iter()
        .find(|c| c.name == container)
        .with_context(|| format!("pod {namespace}/{name} has no container {container}"))?;
    container
        .devices
        .iter()
        .filter(|devices| devices.resource_name == GPU)
        .flat_map(|devices| &devices.device_ids)
        .map(|id| id.parse().with_context(|| format!("GPU id {id:?}")))
        .collect()
}

pub async fn gpus(
    socket: &Path,
    namespace: &str,
    name: &str,
    container: &str,
) -> Result<Vec<GpuUuid>> {
    let mut client = PodResourcesListerClient::new(uds::channel(socket).await?);
    let response = client
        .list(pb::ListPodResourcesRequest {})
        .await
        .context("PodResources List")?;
    gpus_of(response.get_ref(), namespace, name, container)
}

#[cfg(test)]
mod tests {
    use crate::podresources::{gpus_of, pb};

    fn devices(resource: &str, ids: &[&str]) -> pb::ContainerDevices {
        pb::ContainerDevices {
            resource_name: resource.to_string(),
            device_ids: ids.iter().map(|id| id.to_string()).collect(),
            ..Default::default()
        }
    }

    fn listing() -> pb::ListPodResourcesResponse {
        let container = |name: &str, devices: Vec<pb::ContainerDevices>| pb::ContainerResources {
            name: name.to_string(),
            devices,
            ..Default::default()
        };
        pb::ListPodResourcesResponse {
            pod_resources: vec![
                pb::PodResources {
                    name: "other".to_string(),
                    namespace: "serving".to_string(),
                    containers: vec![container(
                        "main",
                        vec![devices(
                            "nvidia.com/gpu",
                            &["GPU-74ee314f-4ae6-9540-5d2b-5f2e97d021fa"],
                        )],
                    )],
                },
                pb::PodResources {
                    name: "replica".to_string(),
                    namespace: "serving".to_string(),
                    containers: vec![
                        container("agent", vec![devices("rdma/ib", &["mlx5_0"])]),
                        container(
                            "main",
                            vec![
                                devices("rdma/ib", &["mlx5_1"]),
                                devices(
                                    "nvidia.com/gpu",
                                    &["GPU-03e97d3f-21d4-1c89-f142-657edeaf3566"],
                                ),
                                devices(
                                    "nvidia.com/gpu",
                                    &["GPU-3820b68e-f4c2-8221-ef3c-a9e9476ee6cb"],
                                ),
                            ],
                        ),
                    ],
                },
            ],
        }
    }

    #[test]
    fn a_container_gets_only_its_own_gpus_in_order() {
        let gpus = gpus_of(&listing(), "serving", "replica", "main").unwrap();
        let names: Vec<String> = gpus.iter().map(|gpu| gpu.to_string()).collect();
        assert_eq!(
            names,
            [
                "GPU-03e97d3f-21d4-1c89-f142-657edeaf3566",
                "GPU-3820b68e-f4c2-8221-ef3c-a9e9476ee6cb"
            ]
        );
        assert!(
            gpus_of(&listing(), "serving", "replica", "agent")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn an_unknown_pod_or_container_is_an_error() {
        assert!(gpus_of(&listing(), "serving", "missing", "main").is_err());
        assert!(gpus_of(&listing(), "other-namespace", "replica", "main").is_err());
        assert!(gpus_of(&listing(), "serving", "replica", "missing").is_err());
    }

    #[test]
    fn a_device_id_that_is_not_a_gpu_uuid_is_an_error() {
        let mut listing = listing();
        listing.pod_resources[0].containers[0].devices[0].device_ids = vec!["0".to_string()];
        assert!(gpus_of(&listing, "serving", "other", "main").is_err());
    }
}
