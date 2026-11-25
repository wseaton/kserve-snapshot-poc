//! A pod as the API server knows it, and the controller its workload is
//! named after.
//!
//! ```text
//! Pod ─► ReplicaSet ─► Deployment                              Deployment
//! Pod ─► StatefulSet ─► LeaderWorkerSet                        LeaderWorkerSet
//! Pod ─► StatefulSet ─► LeaderWorkerSet ─► DisaggregatedSet    DisaggregatedSet
//! Pod                                                          none: the opt-in label names it
//! ```
//!
//! Each kind in [`FOLLOWED`] is read and its own controller taken in turn;
//! the first controller of another kind, or the last one read, names the
//! workload.
use std::future::Future;

use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, OwnerReference};
use kube::api::{Api, ApiResource, DynamicObject, GroupVersionKind};

use crate::cri::Sandbox;
use crate::store::{Attestation, Owner, RevisionKey};

/// Controllers that only stand between a pod and the object that owns its
/// template: (group, kind, resource).
const FOLLOWED: [(&str, &str, &str); 4] = [
    ("", "Pod", "pods"),
    ("apps", "ReplicaSet", "replicasets"),
    ("apps", "StatefulSet", "statefulsets"),
    (
        "leaderworkerset.x-k8s.io",
        "LeaderWorkerSet",
        "leaderworkersets",
    ),
];
const MAX_HOPS: usize = 8;
const DEFAULT_SERVICE_ACCOUNT: &str = "default";

#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error("this daemon has no Kubernetes API client")]
    NoClient,
    #[error("read {kind} {namespace}/{name}")]
    Read {
        kind: String,
        namespace: String,
        name: String,
        #[source]
        source: Box<kube::Error>,
    },
    #[error("pod {namespace}/{name} is {in_api:?} in the API server and {on_node:?} on this node")]
    PodReplaced {
        namespace: String,
        name: String,
        in_api: String,
        on_node: String,
    },
    #[error("{object} has two controllers, {first} and {second}")]
    TwoControllers {
        object: String,
        first: String,
        second: String,
    },
    #[error("{kind} {name} is {found:?}, not the {named:?} its dependent names")]
    ControllerReplaced {
        kind: String,
        name: String,
        found: String,
        named: String,
    },
    #[error("more than {MAX_HOPS} controllers stand above the pod")]
    TooDeep,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PodIdentity {
    pub namespace: String,
    pub name: String,
    pub uid: String,
    pub service_account: String,
    /// `None` for a pod that no controller owns.
    pub owner: Option<Owner>,
}

impl PodIdentity {
    pub async fn of(client: &kube::Client, sandbox: &Sandbox) -> Result<Self, IdentityError> {
        let namespace = &sandbox.namespace;
        let pod = Api::<Pod>::namespaced(client.clone(), namespace)
            .get(&sandbox.name)
            .await
            .map_err(|source| IdentityError::Read {
                kind: "Pod".to_string(),
                namespace: namespace.clone(),
                name: sandbox.name.clone(),
                source: Box::new(source),
            })?;
        let service_account = pod
            .spec
            .as_ref()
            .and_then(|spec| spec.service_account_name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| DEFAULT_SERVICE_ACCOUNT.to_string());
        let metadata = pod.metadata;
        let uid = metadata.uid.clone().unwrap_or_default();
        if uid != sandbox.uid {
            return Err(IdentityError::PodReplaced {
                namespace: namespace.clone(),
                name: sandbox.name.clone(),
                in_api: uid,
                on_node: sandbox.uid.clone(),
            });
        }
        let owner = owner_of(namespace, metadata, |reference, resource| {
            let api = Api::<DynamicObject>::namespaced_with(client.clone(), namespace, &resource);
            async move {
                api.get_metadata(&reference.name)
                    .await
                    .map(|partial| partial.metadata)
            }
        })
        .await?;
        Ok(Self {
            namespace: namespace.clone(),
            name: sandbox.name.clone(),
            uid,
            service_account,
            owner,
        })
    }

    pub fn attestation(&self, revision: &RevisionKey) -> Attestation {
        Attestation {
            namespace: self.namespace.clone(),
            pod: self.name.clone(),
            uid: self.uid.clone(),
            service_account: self.service_account.clone(),
            owner: self.owner.clone(),
            revision: revision.to_string(),
        }
    }
}

fn controller(metadata: &ObjectMeta) -> Result<Option<&OwnerReference>, IdentityError> {
    let mut controllers = metadata
        .owner_references
        .iter()
        .flatten()
        .filter(|reference| reference.controller == Some(true));
    let Some(first) = controllers.next() else {
        return Ok(None);
    };
    match controllers.next() {
        None => Ok(Some(first)),
        Some(second) => Err(IdentityError::TwoControllers {
            object: metadata.name.clone().unwrap_or_default(),
            first: format!("{} {}", first.kind, first.name),
            second: format!("{} {}", second.kind, second.name),
        }),
    }
}

/// The resource to read to follow past `reference`, if it is followed.
fn followed(reference: &OwnerReference) -> Option<ApiResource> {
    let (group, version) = reference
        .api_version
        .split_once('/')
        .unwrap_or(("", reference.api_version.as_str()));
    FOLLOWED
        .iter()
        .find(|(followed_group, kind, _)| *followed_group == group && *kind == reference.kind)
        .map(|(_, kind, plural)| {
            ApiResource::from_gvk_with_plural(&GroupVersionKind::gvk(group, version, kind), plural)
        })
}

fn owner(reference: &OwnerReference) -> Owner {
    Owner {
        api_version: reference.api_version.clone(),
        kind: reference.kind.clone(),
        name: reference.name.clone(),
        uid: reference.uid.clone(),
    }
}

/// The controller `pod` in `namespace` belongs to, reading each followed
/// controller with `read` and checking it is the object the reference names.
pub async fn owner_of<Read>(
    namespace: &str,
    pod: ObjectMeta,
    mut read: impl FnMut(OwnerReference, ApiResource) -> Read,
) -> Result<Option<Owner>, IdentityError>
where
    Read: Future<Output = Result<ObjectMeta, kube::Error>>,
{
    let mut found = None;
    let mut metadata = pod;
    for _ in 0..MAX_HOPS {
        let Some(reference) = controller(&metadata)?.cloned() else {
            return Ok(found);
        };
        let Some(resource) = followed(&reference) else {
            return Ok(Some(owner(&reference)));
        };
        let parent =
            read(reference.clone(), resource)
                .await
                .map_err(|source| IdentityError::Read {
                    kind: reference.kind.clone(),
                    namespace: namespace.to_string(),
                    name: reference.name.clone(),
                    source: Box::new(source),
                })?;
        let uid = parent.uid.clone().unwrap_or_default();
        if uid != reference.uid {
            return Err(IdentityError::ControllerReplaced {
                kind: reference.kind,
                name: reference.name,
                found: uid,
                named: reference.uid,
            });
        }
        found = Some(owner(&reference));
        metadata = parent;
    }
    Err(IdentityError::TooDeep)
}

#[cfg(test)]
mod tests {
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, OwnerReference};

    use crate::node::identity::{IdentityError, PodIdentity, owner_of};
    use crate::store::Owner;

    fn reference(api_version: &str, kind: &str, name: &str, controller: bool) -> OwnerReference {
        OwnerReference {
            api_version: api_version.to_string(),
            kind: kind.to_string(),
            name: name.to_string(),
            uid: format!("{name}-uid"),
            controller: Some(controller),
            block_owner_deletion: None,
        }
    }

    fn object(name: &str, owners: Vec<OwnerReference>) -> ObjectMeta {
        ObjectMeta {
            name: Some(name.to_string()),
            uid: Some(format!("{name}-uid")),
            owner_references: (!owners.is_empty()).then_some(owners),
            ..Default::default()
        }
    }

    fn owned(api_version: &str, kind: &str, name: &str) -> Owner {
        Owner {
            api_version: api_version.to_string(),
            kind: kind.to_string(),
            name: name.to_string(),
            uid: format!("{name}-uid"),
        }
    }

    /// The owner of `pod` among the objects `cluster` holds, and the
    /// resources that were read on the way.
    async fn walk(
        pod: ObjectMeta,
        cluster: &[ObjectMeta],
    ) -> (Result<Option<Owner>, IdentityError>, Vec<String>) {
        let mut read = Vec::new();
        let owner = owner_of("serving", pod, |reference, resource| {
            read.push(format!("{}/{}", resource.plural, reference.name));
            let found = cluster
                .iter()
                .find(|object| object.name.as_deref() == Some(reference.name.as_str()))
                .cloned()
                .ok_or_else(|| {
                    kube::Error::Api(Box::new(kube::core::Status {
                        code: 404,
                        message: format!("{} not found", reference.name),
                        reason: "NotFound".to_string(),
                        ..Default::default()
                    }))
                });
            async move { found }
        })
        .await;
        (owner, read)
    }

    #[tokio::test]
    async fn a_deployment_pod_belongs_to_its_deployment() {
        let deployment = reference("apps/v1", "Deployment", "qwen3-8b", true);
        let replicas = reference("apps/v1", "ReplicaSet", "qwen3-8b-7b9f694bf8", true);
        let cluster = [object("qwen3-8b-7b9f694bf8", vec![deployment])];
        let pod = object("qwen3-8b-7b9f694bf8-zw7mr", vec![replicas]);
        let (owner, read) = walk(pod, &cluster).await;
        assert_eq!(
            owner.unwrap(),
            Some(owned("apps/v1", "Deployment", "qwen3-8b"))
        );
        assert_eq!(read, ["replicasets/qwen3-8b-7b9f694bf8"]);
    }

    #[tokio::test]
    async fn a_kserve_pod_belongs_to_the_deployment_kserve_made() {
        let service = reference(
            "serving.kserve.io/v1alpha2",
            "LLMInferenceService",
            "qwen3-8b",
            true,
        );
        let cluster = [
            object(
                "qwen3-8b-kserve-6f79c6cd9c",
                vec![reference("apps/v1", "Deployment", "qwen3-8b-kserve", true)],
            ),
            object("qwen3-8b-kserve", vec![service]),
        ];
        let pod = object(
            "qwen3-8b-kserve-6f79c6cd9c-zw7mr",
            vec![reference(
                "apps/v1",
                "ReplicaSet",
                "qwen3-8b-kserve-6f79c6cd9c",
                true,
            )],
        );
        let (owner, read) = walk(pod, &cluster).await;
        assert_eq!(
            owner.unwrap(),
            Some(owned("apps/v1", "Deployment", "qwen3-8b-kserve")),
            "the template's owner, not the service above it"
        );
        assert_eq!(read, ["replicasets/qwen3-8b-kserve-6f79c6cd9c"]);
    }

    #[tokio::test]
    async fn a_disaggregated_pod_belongs_to_its_set_and_a_worker_to_its_group() {
        const LWS: &str = "leaderworkerset.x-k8s.io/v1";
        let set = reference(
            "disaggregatedset.x-k8s.io/v1",
            "DisaggregatedSet",
            "stand-in-pd",
            true,
        );
        let cluster = [
            object("stand-in-pd-prefill", vec![set]),
            object(
                "stand-in-pd-prefill-0",
                vec![reference(
                    LWS,
                    "LeaderWorkerSet",
                    "stand-in-pd-prefill",
                    true,
                )],
            ),
            object("stand-in-group", Vec::new()),
            object(
                "stand-in-group-lws",
                vec![reference(LWS, "LeaderWorkerSet", "stand-in-group", true)],
            ),
            object(
                "stand-in-group-0",
                vec![reference(
                    "apps/v1",
                    "StatefulSet",
                    "stand-in-group-lws",
                    true,
                )],
            ),
            object(
                "stand-in-group-0-workers",
                vec![reference("v1", "Pod", "stand-in-group-0", true)],
            ),
        ];
        let leader = object(
            "stand-in-pd-prefill-0-0",
            vec![reference(
                "apps/v1",
                "StatefulSet",
                "stand-in-pd-prefill-0",
                true,
            )],
        );
        let (owner, read) = walk(leader, &cluster).await;
        assert_eq!(
            owner.unwrap(),
            Some(owned(
                "disaggregatedset.x-k8s.io/v1",
                "DisaggregatedSet",
                "stand-in-pd"
            ))
        );
        assert_eq!(
            read,
            [
                "statefulsets/stand-in-pd-prefill-0",
                "leaderworkersets/stand-in-pd-prefill"
            ]
        );

        let worker = object(
            "stand-in-group-0-1",
            vec![reference(
                "apps/v1",
                "StatefulSet",
                "stand-in-group-0-workers",
                true,
            )],
        );
        let (owner, _) = walk(worker, &cluster).await;
        assert_eq!(
            owner.unwrap(),
            Some(owned(LWS, "LeaderWorkerSet", "stand-in-group")),
            "a worker's StatefulSet belongs to its leader pod, which is followed too"
        );
    }

    #[tokio::test]
    async fn a_pod_without_a_controller_belongs_to_no_one() {
        let (owner, read) = walk(object("bare", Vec::new()), &[]).await;
        assert_eq!(owner.unwrap(), None);
        assert!(read.is_empty());
        let referenced = object(
            "referenced",
            vec![reference(
                "apps/v1",
                "ReplicaSet",
                "qwen3-8b-7b9f694bf8",
                false,
            )],
        );
        let (owner, read) = walk(referenced, &[]).await;
        assert_eq!(owner.unwrap(), None, "only a controller reference counts");
        assert!(read.is_empty());
        let job = object(
            "job-pod",
            vec![reference("batch/v1", "Job", "warmup", true)],
        );
        let (owner, read) = walk(job, &[]).await;
        assert_eq!(owner.unwrap(), Some(owned("batch/v1", "Job", "warmup")));
        assert!(
            read.is_empty(),
            "a kind that is not followed names the workload unread"
        );
    }

    #[tokio::test]
    async fn a_reference_to_a_missing_or_replaced_controller_is_an_error() {
        let pod = || {
            object(
                "pod",
                vec![reference(
                    "apps/v1",
                    "ReplicaSet",
                    "qwen3-8b-7b9f694bf8",
                    true,
                )],
            )
        };
        let (owner, _) = walk(pod(), &[]).await;
        let error = owner.unwrap_err();
        assert_eq!(
            error.to_string(),
            "read ReplicaSet serving/qwen3-8b-7b9f694bf8"
        );
        assert!(
            matches!(&error, IdentityError::Read { source, .. } if matches!(source.as_ref(), kube::Error::Api(status) if status.code == 404)),
            "{error:?}"
        );
        let replaced = ObjectMeta {
            uid: Some("another-uid".to_string()),
            ..object("qwen3-8b-7b9f694bf8", Vec::new())
        };
        let (owner, _) = walk(pod(), &[replaced]).await;
        let error = owner.unwrap_err();
        assert_eq!(
            error.to_string(),
            "ReplicaSet qwen3-8b-7b9f694bf8 is \"another-uid\", \
             not the \"qwen3-8b-7b9f694bf8-uid\" its dependent names"
        );
        assert!(
            matches!(error, IdentityError::ControllerReplaced { .. }),
            "{error:?}"
        );
    }

    #[tokio::test]
    async fn two_controllers_or_a_loop_are_errors() {
        let twice = object(
            "pod",
            vec![
                reference("apps/v1", "ReplicaSet", "a", true),
                reference("apps/v1", "ReplicaSet", "b", true),
            ],
        );
        let (owner, read) = walk(twice, &[]).await;
        let error = owner.unwrap_err();
        assert_eq!(
            error.to_string(),
            "pod has two controllers, ReplicaSet a and ReplicaSet b"
        );
        assert!(matches!(error, IdentityError::TwoControllers { .. }));
        assert!(read.is_empty());
        let looped = [object(
            "a",
            vec![reference("apps/v1", "ReplicaSet", "a", true)],
        )];
        let (owner, read) = walk(
            object("pod", vec![reference("apps/v1", "ReplicaSet", "a", true)]),
            &looped,
        )
        .await;
        assert!(matches!(owner, Err(IdentityError::TooDeep)), "{owner:?}");
        assert_eq!(read.len(), 8);
    }

    #[test]
    fn a_pod_attests_to_its_identity_and_revision() {
        let identity = PodIdentity {
            namespace: "serving".to_string(),
            name: "qwen3-8b-7b9f694bf8-zw7mr".to_string(),
            uid: "pod-uid".to_string(),
            service_account: "qwen".to_string(),
            owner: Some(owned("apps/v1", "Deployment", "qwen3-8b")),
        };
        let revision = "serving/qwen3-8b/7b9f694bf8".parse().unwrap();
        let attestation = identity.attestation(&revision);
        assert_eq!(attestation.namespace, "serving");
        assert_eq!(attestation.pod, "qwen3-8b-7b9f694bf8-zw7mr");
        assert_eq!(attestation.uid, "pod-uid");
        assert_eq!(attestation.service_account, "qwen");
        assert_eq!(attestation.owner, identity.owner);
        assert_eq!(attestation.revision, "serving/qwen3-8b/7b9f694bf8");
    }
}
