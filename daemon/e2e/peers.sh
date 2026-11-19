#!/bin/bash
# End-to-end test of checkpoints moving between nodes, on a three-worker kind cluster with a
# stand-in for vLLM and no GPUs: the daemons with NIXL over TCP, kube-rbac-proxy in front of each,
# and node labels for topology.
#
#   daemon/e2e/peers.sh
#
# It checks that a node without a revision's checkpoint pulls it and restores from it, that a node
# with two holders to choose from takes the nearer, that the proxy turns away callers that are not
# daemons, and that two nodes can pull from each other at once, one of them a server that starts
# prepared.
#
# Four kind nodes on one Linux host run out of inotify instances at the default limit, and CRIU
# needs one: raise fs.inotify.max_user_instances first, as CI does.
#
# Environment:
#   CLUSTER  kind cluster to use, created if missing (default snapshotd-e2e-peers)
#   KEEP     1: leave the cluster and its pods behind (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER=${CLUSTER:-snapshotd-e2e-peers}
IMAGE=localhost/snapshotd-e2e-peers:dev

k() { kubectl --context "kind-$CLUSTER" "$@"; }
node() { k get nodes -l "e2e.snapshotd/order=$1" -o jsonpath='{.items[0].metadata.name}'; }
daemon() { k -n snapshotd get pods -l app=snapshotd --field-selector "spec.nodeName=$1" -o jsonpath='{.items[0].metadata.name}'; }
daemon_ip() { k -n snapshotd get pod "$(daemon "$1")" -o jsonpath='{.status.podIP}'; }
log() { k -n snapshotd logs "$(daemon "$1")" -c snapshotd; }
fail() {
  echo "FAIL: $*" >&2
  for n in "$FAR" "$NEAR" "$HOME_NODE"; do echo "--- $n" >&2; log "$n" 2>&1 | tail -30 >&2 || true; done
  exit 1
}
finish() { [ "${KEEP:-0}" = 1 ] || kind delete cluster --name "$CLUSTER"; }

# Scale a Deployment and wait until it has exactly that many pods, all ready.
settle() {
  k scale "deploy/$1" --replicas="$2"
  k rollout status "deploy/$1" --timeout=180s
  until [ "$(k get pods -l "app=$1" --no-headers | wc -l)" -eq "$2" ]; do sleep 1; done
}
# The node of a Deployment's pod that is not on any of the nodes given.
new_node() {
  local app=$1; shift
  k get pods -l "app=$app" -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | grep -vxF "$(printf '%s\n' "$@")"
}
info_on() { # app node
  local ip
  ip=$(k get pods -l "app=$1" --field-selector "spec.nodeName=$2" -o jsonpath='{.items[0].status.podIP}')
  k -n snapshotd exec "$(daemon "$2")" -c snapshotd -- stand-in info "$ip:50051"
}
pulled_from() { log "$1" | grep "pulled the checkpoint from a peer" | grep -c "peer=$2:8443" || true; }

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --config daemon/e2e/peers/kind.yaml --wait 180s
trap finish EXIT
docker build --load -t "$IMAGE" --target e2e-peers -f images/snapshotd/Dockerfile .
kind load docker-image "$IMAGE" --name "$CLUSTER"

FAR=$(node 1) NEAR=$(node 2) HOME_NODE=$(node 3)

# The certificate cert-manager would issue, made by hand.
certs=$(mktemp -d)
docker run --rm "$IMAGE" sh -c '
  cd /tmp
  key="-newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes"
  openssl req -x509 $key -days 1 -subj /CN=snapshotd-peers-ca -keyout ca.key -out ca.crt 2>/dev/null
  openssl req $key -subj /CN=snapshotd-peers -keyout tls.key -out tls.csr 2>/dev/null
  printf "subjectAltName=DNS:snapshotd-peers\n" > san
  openssl x509 -req -in tls.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 1 -extfile san -out tls.crt 2>/dev/null
  tar c ca.crt tls.crt tls.key' | tar x -C "$certs"
k create namespace snapshotd --dry-run=client -o yaml | k apply -f -
k -n snapshotd create secret generic snapshotd-peers-tls \
  --from-file="$certs/ca.crt" --from-file="$certs/tls.crt" --from-file="$certs/tls.key" \
  --dry-run=client -o yaml | k apply -f -
k apply -k daemon/e2e/peers
k apply -f daemon/e2e/peers/workloads.yaml
k -n snapshotd rollout status ds/snapshotd --timeout=240s
for n in "$FAR" "$NEAR" "$HOME_NODE"; do
  log "$n" | grep -q "answering peers" || fail "the daemon on $n is not answering peers"
done

# One replica at a time: the far rack, then each node of the near rack.
settle near-far 1
[ "$(new_node near-far)" = "$FAR" ] || fail "the first replica is on $(new_node near-far), not $FAR"
first=$(info_on near-far "$FAR")
echo "first replica, on $FAR: $first"
[[ $first == *" Running" ]] || fail "the first replica is not running: $first"

settle near-far 2
[ "$(new_node near-far "$FAR")" = "$NEAR" ] || fail "the second replica is not on $NEAR"
[ "$(info_on near-far "$NEAR")" = "$first" ] || fail "the replica on $NEAR is not a restore of the first"
[ "$(pulled_from "$NEAR" "$(daemon_ip "$FAR")")" = 1 ] || fail "$NEAR did not pull from $FAR"

settle near-far 3
[ "$(info_on near-far "$HOME_NODE")" = "$first" ] || fail "the replica on $HOME_NODE is not a restore of the first"
[ "$(pulled_from "$HOME_NODE" "$(daemon_ip "$NEAR")")" = 1 ] \
  || fail "$HOME_NODE has holders on $NEAR, in its rack, and on $FAR, and did not pull from $NEAR"
log "$HOME_NODE" | grep "asking a peer that has it" | grep -q "shared_levels=1 of=2" \
  || fail "$HOME_NODE did not see $NEAR as sharing its rack"
echo "three replicas of one process, the third pulled from its own rack"

# Only a daemon gets an answer from a daemon's proxy.
hash=$(k get pods -l app=near-far -o jsonpath='{.items[0].metadata.labels.pod-template-hash}')
url="https://$(daemon_ip "$FAR"):8443/v1/checkpoints/default/near-far/$hash"
caller=$(k get pods -l app=near-far -o jsonpath='{.items[0].metadata.name}')
code() { k exec "$caller" -c main -- curl -sk -o /dev/null -w '%{http_code}' "$@" "$url"; }
bearer() { echo "Authorization: Bearer $(k -n "$1" create token "$2" --audience "$3" --duration 10m)"; }
[ "$(code)" = 401 ] || fail "a caller with no token got $(code)"
got=$(code -H "$(bearer default default snapshotd-peers)")
[ "$got" = 403 ] || fail "another ServiceAccount got $got"
got=$(code -H "$(bearer snapshotd snapshotd https://kubernetes.default.svc.cluster.local)")
[ "$got" = 401 ] || fail "a daemon's token for another audience got $got"
got=$(code -H "$(bearer snapshotd snapshotd snapshotd-peers)")
[ "$got" = 200 ] || fail "a daemon's token got $got"
echo "the proxy answers a daemon and nobody else"

# Two revisions that start on different nodes and then each need the other's node: the two
# daemons pull from each other at the same time. One server starts prepared for its checkpoint.
settle left 1
settle right 1
[ "$(new_node left)" = "$HOME_NODE" ] || fail "left started on $(new_node left), not $HOME_NODE"
[ "$(new_node right)" = "$NEAR" ] || fail "right started on $(new_node right), not $NEAR"
left=$(info_on left "$HOME_NODE") right=$(info_on right "$NEAR")
[[ $right == *" Running" ]] || fail "the server that started prepared is not serving: $right"
k scale deploy/left deploy/right --replicas=2
k rollout status deploy/left --timeout=180s
k rollout status deploy/right --timeout=180s
[ "$(info_on left "$NEAR")" = "$left" ] || fail "left on $NEAR is not a restore"
[ "$(info_on right "$HOME_NODE")" = "$right" ] || fail "right on $HOME_NODE is not a restore"
[ "$(pulled_from "$NEAR" "$(daemon_ip "$HOME_NODE")")" = 1 ] || fail "$NEAR did not pull left from $HOME_NODE"
[ "$(pulled_from "$HOME_NODE" "$(daemon_ip "$NEAR")")" = 2 ] || fail "$HOME_NODE did not pull right from $NEAR"
echo "two nodes pulled from each other"
echo PASS
