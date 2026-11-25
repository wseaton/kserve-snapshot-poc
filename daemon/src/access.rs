//! Who may read a checkpoint, decided by the API server's authorizer.
//!
//! A revision `<namespace>/<workload>/<hash>` is the namespaced
//! pseudo-resource `checkpoints.snapshot.llm-d.ai`, named after the workload,
//! in its namespace. A Role grants `get` on it like on any other resource:
//!
//! ```yaml
//! rules:
//!   - apiGroups: [snapshot.llm-d.ai]
//!     resources: [checkpoints]
//!     resourceNames: [qwen3-8b]   # optional: one workload only
//!     verbs: [get]
//! ```
//!
//! Restoring a checkpoint into a pod is `get` on `checkpoints`; sending it to
//! another node's daemon is `get` on `checkpoints/transfer`.
use k8s_openapi::api::authorization::v1::{
    ResourceAttributes, SubjectAccessReview, SubjectAccessReviewSpec, SubjectAccessReviewStatus,
};
use kube::api::{Api, PostParams};

use crate::store::RevisionKey;

pub const GROUP: &str = "snapshot.llm-d.ai";
pub const RESOURCE: &str = "checkpoints";
const VERB: &str = "get";
pub const TRANSFER: &str = "transfer";

/// What a checkpoint is read for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    /// Into a pod of this node.
    Restore,
    /// By another node's daemon.
    Transfer,
}

impl Read {
    fn subresource(self) -> Option<&'static str> {
        match self {
            Self::Restore => None,
            Self::Transfer => Some(TRANSFER),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AccessError {
    #[error("this daemon has no Kubernetes API client")]
    NoClient,
    #[error("ask whether {user} may read {revision}")]
    Review {
        user: String,
        revision: String,
        #[source]
        source: Box<kube::Error>,
    },
}

/// Someone who wants a checkpoint, as the API server knows them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subject {
    pub user: String,
    pub groups: Vec<String>,
}

impl Subject {
    /// The user and groups the API server gives a ServiceAccount's token.
    pub fn service_account(namespace: &str, name: &str) -> Self {
        Self {
            user: format!("system:serviceaccount:{namespace}:{name}"),
            groups: vec![
                "system:serviceaccounts".to_string(),
                format!("system:serviceaccounts:{namespace}"),
                "system:authenticated".to_string(),
            ],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Allowed,
    Denied(String),
}

/// The review that asks whether `subject` may read `revision` for `read`.
pub fn review(subject: &Subject, revision: &RevisionKey, read: Read) -> SubjectAccessReview {
    SubjectAccessReview {
        spec: SubjectAccessReviewSpec {
            user: Some(subject.user.clone()),
            groups: Some(subject.groups.clone()),
            resource_attributes: Some(ResourceAttributes {
                group: Some(GROUP.to_string()),
                resource: Some(RESOURCE.to_string()),
                subresource: read.subresource().map(str::to_string),
                verb: Some(VERB.to_string()),
                namespace: Some(revision.namespace().to_string()),
                name: Some(revision.workload().name().to_string()),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// An answer without a status, or one that is not an allow, is a denial.
pub fn verdict(status: Option<SubjectAccessReviewStatus>) -> Verdict {
    let Some(status) = status else {
        return Verdict::Denied("the API server returned no decision".to_string());
    };
    if status.allowed && status.denied != Some(true) {
        return Verdict::Allowed;
    }
    let reason = status.reason.filter(|reason| !reason.is_empty());
    let error = status.evaluation_error.filter(|error| !error.is_empty());
    Verdict::Denied(match (reason, error) {
        (Some(reason), Some(error)) => format!("{reason}; {error}"),
        (Some(reason), None) => reason,
        (None, Some(error)) => error,
        (None, None) => format!("no RBAC rule grants {VERB} on {RESOURCE}.{GROUP}"),
    })
}

/// Whether `subject` may read `revision` for `read`. An error means the API
/// server could not be asked, which callers treat as a denial.
pub async fn may_read(
    client: &kube::Client,
    subject: &Subject,
    revision: &RevisionKey,
    read: Read,
) -> Result<Verdict, AccessError> {
    let answered = Api::<SubjectAccessReview>::all(client.clone())
        .create(&PostParams::default(), &review(subject, revision, read))
        .await
        .map_err(|source| AccessError::Review {
            user: subject.user.clone(),
            revision: revision.to_string(),
            source: Box::new(source),
        })?;
    Ok(verdict(answered.status))
}

#[cfg(test)]
mod tests {
    use k8s_openapi::api::authorization::v1::{ResourceAttributes, SubjectAccessReviewStatus};

    use crate::access::{Read, Subject, Verdict, review, verdict};
    use crate::store::RevisionKey;

    #[test]
    fn a_revision_is_reviewed_as_a_namespaced_checkpoint_named_after_its_workload() {
        let subject = Subject::service_account("serving", "qwen");
        assert_eq!(subject.user, "system:serviceaccount:serving:qwen");
        assert_eq!(
            subject.groups,
            [
                "system:serviceaccounts",
                "system:serviceaccounts:serving",
                "system:authenticated"
            ]
        );
        let revision: RevisionKey = "serving/qwen3-8b@prefill+0/7b9f694bf8".parse().unwrap();
        let spec = review(&subject, &revision, Read::Restore).spec;
        assert_eq!(spec.user.as_deref(), Some(subject.user.as_str()));
        assert_eq!(spec.groups.as_ref(), Some(&subject.groups));
        assert_eq!(spec.non_resource_attributes, None);
        assert_eq!(
            spec.resource_attributes,
            Some(ResourceAttributes {
                group: Some("snapshot.llm-d.ai".to_string()),
                resource: Some("checkpoints".to_string()),
                verb: Some("get".to_string()),
                namespace: Some("serving".to_string()),
                name: Some("qwen3-8b".to_string()),
                ..Default::default()
            })
        );
    }

    #[test]
    fn a_transfer_to_another_node_is_its_own_subresource() {
        let subject = Subject::service_account("snapshotd", "snapshotd");
        let revision: RevisionKey = "serving/qwen3-8b/7b9f694bf8".parse().unwrap();
        let attributes = |read| {
            review(&subject, &revision, read)
                .spec
                .resource_attributes
                .unwrap()
        };
        assert_eq!(attributes(Read::Restore).subresource, None);
        let transfer = attributes(Read::Transfer);
        assert_eq!(transfer.subresource.as_deref(), Some("transfer"));
        assert_eq!(transfer.resource.as_deref(), Some("checkpoints"));
        assert_eq!(transfer.namespace.as_deref(), Some("serving"));
        assert_eq!(transfer.name.as_deref(), Some("qwen3-8b"));
    }

    #[test]
    fn only_an_explicit_allow_is_allowed() {
        let status = |allowed, denied, reason: &str, error: &str| {
            Some(SubjectAccessReviewStatus {
                allowed,
                denied,
                reason: Some(reason.to_string()),
                evaluation_error: Some(error.to_string()),
            })
        };
        let denied = |why: &str| Verdict::Denied(why.to_string());
        let cases = [
            (status(true, None, "", ""), Verdict::Allowed),
            (status(true, Some(false), "bound", ""), Verdict::Allowed),
            (
                status(true, Some(true), "contradiction", ""),
                denied("contradiction"),
            ),
            (
                status(false, None, "", ""),
                denied("no RBAC rule grants get on checkpoints.snapshot.llm-d.ai"),
            ),
            (
                status(false, Some(true), "forbidden by webhook", ""),
                denied("forbidden by webhook"),
            ),
            (
                status(false, None, "", "role not found"),
                denied("role not found"),
            ),
            (
                status(false, None, "no binding", "role not found"),
                denied("no binding; role not found"),
            ),
            (None, denied("the API server returned no decision")),
        ];
        for (number, (status, wanted)) in cases.into_iter().enumerate() {
            assert_eq!(verdict(status), wanted, "case {number}");
        }
    }
}
