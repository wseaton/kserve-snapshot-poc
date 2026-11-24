//! Where a node is, said the way Kueue's `Topology` says it: an ordered list
//! of node labels, widest domain first, and the node's value for each.
//!
//! ```text
//! levels   ib.coreweave.cloud/superpod   ib.coreweave.cloud/leafgroup   kubernetes.io/hostname
//! node A   0                             11                             gf49e9c
//! node B   0                             11                             gf41fb2     shares 2 levels with A
//! node C   0                             391                            gf4334a     shares 1 level with A
//! ```
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;
use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Place(Vec<(String, Option<String>)>);

impl Place {
    /// An empty label value counts as no value.
    pub fn of(levels: &[String], labels: &BTreeMap<String, String>) -> Self {
        Self(
            levels
                .iter()
                .map(|level| {
                    let value = labels.get(level).filter(|value| !value.is_empty());
                    (level.clone(), value.cloned())
                })
                .collect(),
        )
    }

    /// The number of levels, from the widest, at which two nodes are in the
    /// same domain. Nodes described by different levels share none.
    pub fn shared_levels(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(&other.0)
            .take_while(|(ours, theirs)| ours.1.is_some() && ours == theirs)
            .count()
    }

    pub fn levels(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values: Vec<&str> = self
            .0
            .iter()
            .map(|(_, value)| value.as_deref().unwrap_or("?"))
            .collect();
        f.write_str(&values.join("/"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub address: SocketAddr,
    pub place: Place,
    pub free_lanes: usize,
}

/// The nearest to `own` first. Among the equally near, those that can send
/// now come first, and otherwise the order given is kept.
pub fn nearest_first(own: &Place, mut candidates: Vec<Candidate>) -> Vec<Candidate> {
    candidates.sort_by_key(|candidate| {
        (
            Reverse(own.shared_levels(&candidate.place)),
            candidate.free_lanes == 0,
        )
    });
    candidates
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::topology::{Candidate, Place, nearest_first};

    fn levels() -> Vec<String> {
        [
            "ib.coreweave.cloud/superpod",
            "ib.coreweave.cloud/leafgroup",
            "kubernetes.io/hostname",
        ]
        .map(String::from)
        .to_vec()
    }

    fn node(superpod: &str, leafgroup: &str, hostname: &str) -> Place {
        let labels = BTreeMap::from([
            (
                "ib.coreweave.cloud/superpod".to_string(),
                superpod.to_string(),
            ),
            (
                "ib.coreweave.cloud/leafgroup".to_string(),
                leafgroup.to_string(),
            ),
            ("kubernetes.io/hostname".to_string(), hostname.to_string()),
            ("node.coreweave.cloud/rack".to_string(), "237".to_string()),
        ]);
        Place::of(&levels(), &labels)
    }

    #[test]
    fn nodes_share_the_levels_down_to_where_they_part() {
        let a = node("0", "11", "gf49e9c");
        assert_eq!(a.shared_levels(&a), 3);
        assert_eq!(a.shared_levels(&node("0", "11", "gf41fb2")), 2);
        assert_eq!(a.shared_levels(&node("0", "391", "gf4334a")), 1);
        assert_eq!(a.shared_levels(&node("1", "11", "g000000")), 0);
        assert_eq!(a.to_string(), "0/11/gf49e9c");
    }

    #[test]
    fn a_level_below_one_that_differs_does_not_count() {
        let a = node("0", "11", "gf49e9c");
        assert_eq!(a.shared_levels(&node("1", "11", "gf49e9c")), 0);
    }

    #[test]
    fn a_node_without_a_label_shares_nothing_from_there_down() {
        let unlabelled = node("", "", "g80d16c");
        assert_eq!(unlabelled.to_string(), "?/?/g80d16c");
        assert_eq!(unlabelled.shared_levels(&unlabelled), 0);
        assert_eq!(node("0", "11", "gf49e9c").shared_levels(&unlabelled), 0);
        assert_eq!(node("0", "", "a").shared_levels(&node("0", "", "a")), 1);
    }

    #[test]
    fn nodes_described_by_other_levels_share_nothing() {
        let labels = BTreeMap::from([("kubernetes.io/hostname".to_string(), "a".to_string())]);
        let by_host = Place::of(&["kubernetes.io/hostname".to_string()], &labels);
        assert_eq!(node("0", "11", "a").shared_levels(&by_host), 0);
        assert_eq!(Place::default().shared_levels(&Place::default()), 0);
        assert_eq!(Place::default().levels(), 0);
    }

    #[test]
    fn the_nearest_holder_that_can_send_is_asked_first() {
        let own = node("0", "11", "gf49e9c");
        let candidate = |host: u8, place: Place, free_lanes| Candidate {
            address: format!("10.0.0.{host}:8443").parse().unwrap(),
            place,
            free_lanes,
        };
        let far = candidate(1, node("0", "391", "gf4334a"), 2);
        let elsewhere = candidate(2, node("3", "7", "g111111"), 2);
        let near_busy = candidate(3, node("0", "11", "gf41fb2"), 0);
        let near = candidate(4, node("0", "11", "g222222"), 1);
        let unknown = candidate(5, Place::default(), 2);
        let order = nearest_first(
            &own,
            vec![
                far.clone(),
                elsewhere.clone(),
                near_busy.clone(),
                near.clone(),
                unknown.clone(),
            ],
        );
        assert_eq!(order, [near, near_busy, far, elsewhere, unknown]);
    }

    #[test]
    fn without_levels_the_given_order_stands() {
        let candidate = |host: u8, free_lanes| Candidate {
            address: format!("10.0.0.{host}:8443").parse().unwrap(),
            place: Place::default(),
            free_lanes,
        };
        let given = vec![candidate(3, 1), candidate(1, 2), candidate(2, 1)];
        assert_eq!(nearest_first(&Place::default(), given.clone()), given);
    }
}
