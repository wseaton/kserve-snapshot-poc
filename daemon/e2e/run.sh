#!/bin/bash
# End-to-end test of `snapshotd node` on kind, with a stand-in for vLLM and no GPUs.
#
#   daemon/e2e/run.sh
#
# It checks that the first replica of a revision is checkpointed and comes back as the same
# process, that later replicas are restored from that checkpoint instead of booting and carry
# their pod's security label, that a new revision boots and is checkpointed again, that a pod's
# checkpoint goes to the store class it names, that the daemon's metrics count all of it, that
# a pod naming the HTTP transport goes through the same capture and restore over vLLM's HTTP routes,
# that a checkpoint two pods in a row fail to restore is condemned and captured afresh, and that a
# checkpoint taken on a CPU with a flag the node lacks is recaptured.
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

grep -qx "$CLUSTER" <<<"$(kind get clusters)" || kind create cluster --name "$CLUSTER" --wait 180s
trap finish EXIT
docker build --load -t "$IMAGE" --target e2e -f images/snapshotd/Dockerfile .
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
restores() { k -n snapshotd logs ds/snapshotd -c snapshotd | grep -c "restored .*from=.*/image" || true; }
# The security label of a process in a pod, empty on a node without an LSM.
label() { k exec "$1" -- sh -c "tr -d '\\0' < /proc/$2/attr/current" 2>/dev/null || true; }
server() { k exec "$1" -- cut -d' ' -f1 /proc/1/task/1/children; }

settle 1
first=$(info "$(addresses)")
echo "first replica: $first"
[[ $first == *" Running" ]] || fail "the first replica is not running: $first"
grep -q "restored .*from=.*/incoming-" <<<"$(k -n snapshotd logs ds/snapshotd -c snapshotd)" \
  || fail "the first replica was not checkpointed and restored"

settle 3
for address in $(addresses); do
  got=$(info "$address")
  echo "replica $address: $got"
  [ "$got" = "$first" ] || fail "replica $address is not a restore of the first: $got"
done
[ "$(restores)" = 2 ] || fail "expected 2 restores from the node's checkpoint, saw $(restores)"
for pod in $(k get pods -l app=stand-in -o name); do
  own=$(label "$pod" 1) restored=$(label "$pod" "$(server "$pod")")
  [ "$restored" = "$own" ] || fail "the restored server in $pod is labelled '$restored', its pod '$own'"
done
echo "restored servers carry their pod's label: '${own:-none}'"

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

# A server that prepares itself at startup never serves before its checkpoint: it is captured as
# it comes up, and every replica serves only after a restore.
settle 1
k patch deploy/stand-in --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/command/-","value":"--prepared"}]'
settle 1
third=$(info "$(addresses)")
echo "prepared at startup: $third"
[[ $third == *" Running" ]] || fail "a server that started prepared was not put into service: $third"
[ "$third" != "$second" ] || fail "the prepared revision came up as the old process"
settle 2
for address in $(addresses); do
  got=$(info "$address")
  [ "$got" = "$third" ] || fail "replica $address of the prepared revision is not a restore: $got"
done

# A pod that names a store class has its checkpoint kept there, and one that names a class the
# node does not have boots without one.
class() {
  settle 1
  k patch deploy/stand-in --type=merge \
    -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"snapshot.llm-d.ai/class\":\"$1\"}}}}}"
  settle 1
}
logged() { k -n snapshotd logs ds/snapshotd -c snapshotd | grep -c "$1" || true; }
class scratch
settle 2
[ "$(logged "checkpoint published on this node image=/stores/scratch/default/stand-in/")" = 1 ] \
  || fail "the checkpoint of a pod that named the scratch class is not in /stores/scratch"
[ "$(logged "restored .*from=/stores/scratch/default/stand-in/.*/image")" = 1 ] \
  || fail "the second replica was not restored from the scratch class"
class nowhere
[ "$(logged 'no store class \\"nowhere\\"')" = 1 ] || fail "a pod that named an unknown class was not cold booted for it"
[[ $(info "$(addresses)") == *" Running" ]] || fail "the pod that named an unknown class is not serving"

metrics() {
  k get --raw "/api/v1/namespaces/snapshotd/pods/$(k -n snapshotd get pods -l app=snapshotd -o jsonpath='{.items[0].metadata.name}'):9090/proxy/metrics"
}
for _ in $(seq 20); do
  grep -qF 'snapshotd_store_revisions{class="scratch",namespace="default",workload="stand-in"} 1' <<<"$(metrics)" && break
  sleep 1
done
scraped=$(metrics) || fail "the daemon's metrics could not be read"
for line in \
  'snapshotd_store_revisions{class="scratch",namespace="default",workload="stand-in"} 1' \
  'snapshotd_store_revisions{class="default",namespace="default",workload="stand-in"} 3' \
  'snapshotd_pods_total{namespace="default",workload="stand-in",how="cold_boot"} 1' \
  'snapshotd_pods_total{namespace="default",workload="stand-in",how="capture"} 4' \
  'snapshotd_pods_total{namespace="default",workload="stand-in",how="restore"} 6' \
  'snapshotd_step_seconds_count{step="checkpoint"} 4' \
  'snapshotd_store_free_bytes{class="scratch"}'; do
  grep -qF "$line" <<<"$scraped" || { grep -v '^#' <<<"$scraped" >&2; fail "the daemon's metrics have no: $line"; }
done
echo "the daemon's metrics count what it did and what each class holds"

# A pod that names the HTTP transport is driven through vLLM's HTTP routes, all a server without
# the gRPC control port serves: it is captured as it comes up, and a second replica is restored.
info_http() { k -n snapshotd exec ds/snapshotd -c snapshotd -- stand-in info --http "$1:8000"; }
settle 1
published=$(logged "checkpoint published on this node")
k patch deploy/stand-in --type=json -p '[
  {"op":"remove","path":"/spec/template/metadata/annotations/snapshot.llm-d.ai~1class"},
  {"op":"add","path":"/spec/template/metadata/annotations/snapshot.llm-d.ai~1control","value":"http"},
  {"op":"add","path":"/spec/template/spec/containers/0/command/-","value":"--http-only"}]'
settle 1
fourth=$(info_http "$(addresses)")
echo "over HTTP: $fourth"
[[ $fourth == *" Running" ]] || fail "a pod driven over HTTP is not serving: $fourth"
[ "$fourth" != "$third" ] || fail "the HTTP revision came up as an old process"
[ "$(logged "checkpoint published on this node")" = $((published + 1)) ] \
  || fail "the pod driven over HTTP was not checkpointed"
settle 2
for address in $(addresses); do
  got=$(info_http "$address")
  [ "$got" = "$fourth" ] || fail "replica $address of the HTTP revision is not a restore: $got"
done
[ "$(logged "transport=Http")" -ge 2 ] || fail "the daemon did not drive the HTTP revision's pods over HTTP"

# Each case below runs on a revision of its own, since a revision is recaptured at most once per
# node environment.
on_node() { k -n snapshotd exec ds/snapshotd -c snapshotd -- "$@"; }
revision_dir() {
  echo "/store/default/stand-in/$(k get pods -l app=stand-in -o jsonpath='{.items[0].metadata.labels.pod-template-hash}')"
}
running() {
  for address in $(addresses); do
    [[ $(info_http "$address") == *" Running" ]] || fail "replica $address is not serving after $1"
  done
}

# A checkpoint whose pages are cut short fails CRIU restore in every pod. After two pods fail it,
# the daemon removes it, keeps the evidence, and a later pod captures a new one.
settle 1
k set env deploy/stand-in REVISION=condemned
settle 1
corrupted=$(revision_dir)
on_node sh -c "truncate -s 0 $corrupted/image/img/pages-*.img"
published=$(logged "checkpoint published on this node")
settle 4
for _ in $(seq 60); do
  [ "$(logged "condemned the node's checkpoint")" -ge 1 ] && break
  sleep 1
done
condemns=$(k -n snapshotd logs ds/snapshotd -c snapshotd | grep "condemned the node's checkpoint" || true)
echo "condemned: $condemns"
[ "$(grep -c . <<<"$condemns")" = 1 ] || fail "expected one condemn of the corrupted checkpoint"
[[ $condemns == *"pods=2"* ]] || fail "the checkpoint was not condemned after 2 failed restores"
[ "$(logged "CRIU restore failed")" -ge 2 ] || fail "fewer than 2 pods failed CRIU restore"
on_node sh -c "test -s $corrupted/condemned/*/restores && test -f $corrupted/condemned/*/restore.log" \
  || fail "the condemned checkpoint's evidence was not kept"
[ "$(logged "checkpoint published on this node")" -gt "$published" ] || settle 5
[ "$(logged "checkpoint published on this node")" -gt "$published" ] \
  || fail "no pod captured a new checkpoint after the condemn"
on_node sh -c "test -d $corrupted/image" || fail "the new checkpoint is not the revision's image"
running "the condemn"
restored=$(restores)
settle "$(( $(k get pods -l app=stand-in --no-headers | wc -l) + 1 ))"
[ "$(restores)" -gt "$restored" ] || fail "the new checkpoint was not restored into a later pod"
running "a restore of the new checkpoint"
condemned_metric='snapshotd_condemned_images_total{namespace="default",workload="stand-in"} 1'
grep -qF "$condemned_metric" <<<"$(metrics)" || fail "the daemon's metrics have no: $condemned_metric"
grep -q 'snapshotd_abandoned_revisions_total{' <<<"$(metrics)" && fail "a single condemn abandoned the revision"

# A checkpoint that needs a CPU flag the node lacks is retired, and the next pod captures anew.
settle 1
k set env deploy/stand-in REVISION=cpu-flags
settle 1
flagged=$(revision_dir)
manifest=$(on_node cat "$flagged/image/manifest.json" \
  | jq -e '.environment.cpu_flags += ["snapshotd_e2e_absent"]') \
  || fail "the checkpoint's manifest could not be edited"
k -n snapshotd exec -i ds/snapshotd -c snapshotd -- sh -c "cat > $flagged/image/manifest.json" <<<"$manifest"
published=$(logged "checkpoint published on this node")
settle 2
[ "$(logged "CPU with flags the node lacks: snapshotd_e2e_absent")" -ge 1 ] \
  || fail "a checkpoint that needs a CPU flag the node lacks was not refused"
[ "$(logged "was taken in another environment; replacing it")" -ge 1 ] \
  || fail "a checkpoint that needs a CPU flag the node lacks was not recaptured"
[ "$(logged "checkpoint published on this node")" = $((published + 1)) ] \
  || fail "the recapture was not published"
on_node grep -q snapshotd_e2e_absent "$flagged/image/manifest.json" \
  && fail "the recaptured checkpoint still needs the absent flag"
running "the recapture"
echo PASS
