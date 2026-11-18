#!/bin/bash
# End-to-end test of `snapshotd node` on kind, with a stand-in for vLLM and no GPUs.
#
#   daemon/e2e/run.sh
#
# It checks that the first replica of a revision is checkpointed and comes back as the same
# process, that later replicas are restored from that checkpoint instead of booting, and that a
# new revision boots and is checkpointed again.
#
# Environment:
#   CLUSTER  kind cluster to use, created if missing (default snapshotd-e2e)
#   KEEP     1: leave the cluster and its pods behind (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER=${CLUSTER:-snapshotd-e2e}
IMAGE=localhost/snapshotd-e2e:dev
k() { kubectl --context "kind-$CLUSTER" "$@"; }
fail() { echo "FAIL: $*" >&2; k -n snapshotd logs ds/snapshotd -c snapshotd --tail=40 >&2 || true; exit 1; }
finish() { [ "${KEEP:-0}" = 1 ] || kind delete cluster --name "$CLUSTER"; }

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --wait 180s
trap finish EXIT
docker build --load -t "$IMAGE" -f daemon/e2e/Dockerfile .
kind load docker-image "$IMAGE" --name "$CLUSTER"
k apply -k daemon/e2e
k -n snapshotd rollout status ds/snapshotd --timeout=180s

# Wait for the Deployment to have exactly $1 pods, all ready.
settle() {
  k scale deploy/stand-in --replicas="$1"
  k rollout status deploy/stand-in --timeout=180s
  until [ "$(k get pods -l app=stand-in --no-headers | wc -l)" -eq "$1" ]; do sleep 1; done
}
addresses() { k get pods -l app=stand-in -o jsonpath='{range .items[*]}{.status.podIP}{"\n"}{end}'; }
info() { k -n snapshotd exec ds/snapshotd -c snapshotd -- stand-in info "$1:50051"; }
restores() { k -n snapshotd logs ds/snapshotd -c snapshotd | grep -c "restored from .*/image " || true; }

settle 1
first=$(info "$(addresses)")
echo "first replica: $first"
[[ $first == *" Running" ]] || fail "the first replica is not running: $first"
k -n snapshotd logs ds/snapshotd -c snapshotd | grep -q "restored from .*/incoming-" \
  || fail "the first replica was not checkpointed and restored"

settle 3
for address in $(addresses); do
  got=$(info "$address")
  echo "replica $address: $got"
  [ "$got" = "$first" ] || fail "replica $address is not a restore of the first: $got"
done
[ "$(restores)" = 2 ] || fail "expected 2 restores from the node's checkpoint, saw $(restores)"

# One pod at a time through the new revision: pods of a revision that start together each boot
# and checkpoint, and only one of those checkpoints is kept.
settle 1
k set env deploy/stand-in REVISION=2
settle 1
second=$(info "$(addresses)")
echo "new revision: $second"
[ "$second" != "$first" ] || fail "a new revision came up as the old process"
settle 3
for address in $(addresses); do
  got=$(info "$address")
  [ "$got" = "$second" ] || fail "replica $address of the new revision is not a restore: $got"
done
echo PASS
