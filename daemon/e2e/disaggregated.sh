#!/bin/bash
# End-to-end test of `snapshotd node` under LeaderWorkerSet's DisaggregatedSet on kind, with a
# stand-in for vLLM and no GPUs.
#
#   daemon/e2e/disaggregated.sh
#
# Prefill and decode pods carry the same workload label and the same revision. It checks that
# each role is checkpointed and restored from its own checkpoint, that a template change in one
# role starts a revision both roles capture again, and that the pods of a multi-pod group cold
# boot instead of restoring alone.
#
# Environment:
#   CLUSTER  kind cluster to use, created if missing (default snapshotd-e2e-disaggregated)
#   KEEP     1: leave the cluster and its pods behind (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER=${CLUSTER:-snapshotd-e2e-disaggregated}
IMAGE=localhost/snapshotd-e2e:dev
LWS_VERSION=v1.0.0
SET=disaggregatedset.x-k8s.io

k() { kubectl --context "kind-$CLUSTER" "$@"; }
log() { k -n snapshotd logs ds/snapshotd -c snapshotd; }
fail() { echo "FAIL: $*" >&2; log --tail=40 >&2 || true; exit 1; }
finish() { [ "${KEEP:-0}" = 1 ] || kind delete cluster --name "$CLUSTER"; }

grep -qx "$CLUSTER" <<<"$(kind get clusters)" || kind create cluster --name "$CLUSTER" --wait 180s
trap finish EXIT
docker build --load -t "$IMAGE" --target e2e -f images/snapshotd/Dockerfile .
kind load docker-image "$IMAGE" --name "$CLUSTER"
k apply --server-side \
  -f "https://github.com/kubernetes-sigs/lws/releases/download/$LWS_VERSION/manifests.yaml"
k -n lws-system rollout status deploy/lws-controller-manager --timeout=180s
# The controller is ready before its webhook serves, which turns the workloads away until then.
for attempt in $(seq 60); do
  k apply -k daemon/e2e/disaggregated && break
  [ "$attempt" -lt 60 ] || fail "LWS's webhook never accepted the workloads"
  sleep 2
done
k -n snapshotd rollout status ds/snapshotd --timeout=180s

selector() { echo "$SET/name=stand-in-pd,$SET/role=$1"; }
index() { case $1 in prefill) echo 0 ;; decode) echo 1 ;; esac; }
scale() {
  k patch disaggregatedset stand-in-pd --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/roles/$(index "$1")/spec/replicas\",\"value\":$2}]"
}
# Wait until role $1 has exactly $2 pods, all of revision $3 when given, all ready.
settle() {
  for _ in $(seq 180); do
    local listed ready
    listed=$(k get pods -l "$(selector "$1")" -o jsonpath="{range .items[*]}{.metadata.labels.disaggregatedset\.x-k8s\.io/revision} {.status.conditions[?(@.type=='Ready')].status}{'\n'}{end}")
    ready=$(grep -c " True$" <<<"$listed" || true)
    if [ "$(grep -c . <<<"$listed" || true)" -eq "$2" ] && [ "$ready" -eq "$2" ] \
      && { [ -z "${3:-}" ] || ! grep -qv "^$3 " <<<"$listed"; }; then
      return
    fi
    sleep 1
  done
  fail "role $1 did not settle at $2 ready pods${3:+ of revision $3}"
}
revision() { k get pods -l "$(selector "$1")" -o jsonpath="{.items[0].metadata.labels.disaggregatedset\.x-k8s\.io/revision}"; }
addresses() { k get pods -l "$1" -o jsonpath='{range .items[*]}{.status.podIP}{"\n"}{end}'; }
info() { k -n snapshotd exec ds/snapshotd -c snapshotd -- stand-in info "$1:50051"; }
published() { log | grep -c "checkpoint published on this node image=/store/default/$1/" || true; }

# Every pod of role $1 is a restore of process $2.
all_restores_of() {
  for address in $(addresses "$(selector "$1")"); do
    got=$(info "$address")
    [ "$got" = "$2" ] || fail "$1 pod $address is not a restore of its role's first pod: $got"
  done
}

settle prefill 1
settle decode 1
first=$(revision prefill)
[ "$(revision decode)" = "$first" ] || fail "prefill and decode are on different revisions"
prefill=$(info "$(addresses "$(selector prefill)")")
decode=$(info "$(addresses "$(selector decode)")")
echo "revision $first: prefill $prefill, decode $decode"
[[ $prefill == *" Running" && $decode == *" Running" ]] || fail "a first pod is not running"
[ "$prefill" != "$decode" ] || fail "prefill and decode came up as the same process"
[ "$(published "stand-in-pd@prefill/$first")" = 1 ] || fail "prefill's checkpoint is not under its role"
[ "$(published "stand-in-pd@decode/$first")" = 1 ] || fail "decode's checkpoint is not under its role"

scale prefill 2
scale decode 2
settle prefill 2
settle decode 2
all_restores_of prefill "$prefill"
all_restores_of decode "$decode"
echo "each role's second pod is a restore of that role's first"

# A change to decode alone is a new revision of the whole set, which both roles capture again.
scale prefill 1
scale decode 1
settle prefill 1
settle decode 1
k patch disaggregatedset stand-in-pd --type=json -p '[{"op":"add","path":"/spec/roles/1/spec/leaderWorkerTemplate/workerTemplate/spec/containers/0/env","value":[{"name":"REVISION","value":"2"}]}]'
for _ in $(seq 180); do
  second=$(revision decode)
  [ -n "$second" ] && [ "$second" != "$first" ] && break
  sleep 1
done
[ "$second" != "$first" ] || fail "the template change did not start a new revision"
settle prefill 1 "$second"
settle decode 1 "$second"
[ "$(published "stand-in-pd@prefill/$second")" = 1 ] || fail "prefill was not captured again in revision $second"
[ "$(published "stand-in-pd@decode/$second")" = 1 ] || fail "decode was not captured again in revision $second"
prefill2=$(info "$(addresses "$(selector prefill)")")
[ "$prefill2" != "$prefill" ] || fail "prefill of the new revision came up as the old process"
scale prefill 2
settle prefill 2 "$second"
all_restores_of prefill "$prefill2"
echo "revision $second captured both roles again and restores from its own checkpoints"

# Two pods of one group hold different ranks: each cold boots and none is checkpointed.
group="leaderworkerset.sigs.k8s.io/name=stand-in-group"
k wait pod -l "$group" --for=condition=Ready --timeout=180s
[ "$(k get pods -l "$group" --no-headers | wc -l)" -eq 2 ] || fail "the group does not have two pods"
logged=$(log)
for member in 0 1; do
  grep -q "pod $member of a multi-pod group" <<<"$logged" || fail "group member $member was not cold booted for its group"
done
for address in $(addresses "$group"); do
  [[ $(info "$address") == *" Running" ]] || fail "group pod $address is not serving"
done
! grep -q 'image=/store/default/stand-in-group' <<<"$logged" || fail "a group member was checkpointed"
echo "the pods of a multi-pod group cold boot and are not checkpointed"
echo PASS
