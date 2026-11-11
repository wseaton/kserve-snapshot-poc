#!/bin/bash
# Cold boot one vLLM server, checkpoint it with CRIU, restore it, and time every phase.
#
#   restore-bench.sh <tag> [vllm args]
#
# Environment:
#   MODEL        model directory (required)
#   TP           tensor parallel size (default 8)
#   MODE         plain: cuda-checkpoint only; cuinterpose: preload cuInterpose and run its
#                prepare and restore phases around the CUDA checkpoint (default plain)
#   REMAP        0: restore onto the same GPUs; rotate: restore rank i onto rank i+1's GPU (default 0)
#   CKPT_DIR     where the CRIU images go (default /work/<tag>/img)
#   SNAPSHOTD    snapshotd binary (default /models/bin/snapshotd)
#   GHOST_LIMIT largest unlinked file CRIU may copy into the image (default 1G)
#   POST_CHECK   command run against the restored server, with NETNS set to the prefix that
#                enters its network namespace; its last output line goes into the result
#   OVERLAY      tarball of patched vLLM files to unpack over the image's, if set
#
# Prints one RESULT line of JSON. Run in a privileged pod whose PID 1 reaps children. Such a pod
# sees every GPU on the node, so the GPUs to use come from the kubelet's device checkpoint.
set -u
TAG=$1; shift
: "${MODEL:?}" "${TP:=8}" "${MODE:=plain}" "${REMAP:=0}"
W=/work/$TAG; rm -rf "$W"; mkdir -p "$W" /work/empty; cd "$W" || exit 1
IMG=${CKPT_DIR:-$W/img}; rm -rf "$IMG"; mkdir -p "$IMG"
SD=${SNAPSHOTD:-/models/bin/snapshotd}
B=/tmp/snapshot-cuda/snapshot-binaries
BUNDLE=/work/cuinterpose
SITE=/usr/local/lib/python3.12/dist-packages

now() { date +%s%N; }
secs() { python3 -c "print(round(($(now) - $1) / 1e9, 3))"; }
declare -A R
fail() { R[result]=$1; report; killall9; exit 1; }
report() {
  local out="{\"tag\": \"$TAG\", \"mode\": \"$MODE\", \"tp\": $TP, \"remap\": \"$REMAP\""
  for k in "${!R[@]}"; do out+=", \"$k\": \"${R[$k]}\""; done
  echo "RESULT $out}"
}
tree() { local p=$1; echo "$p"; for t in /proc/"$p"/task/*; do for c in $(cat "$t"/children 2>/dev/null); do tree "$c"; done; done; }
PIDS=; VP=
killall9() { for p in $( [ -n "$VP" ] && tree "$VP") $PIDS; do kill -9 "$p" 2>/dev/null; done; sleep 5; }
complete() { ${NETNS:-} curl -s -m 600 localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model":"m","prompt":"The capital of France is","max_tokens":32,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'; }

ss -ltn | grep -q ":8000 " && fail PORT_8000_BUSY
pgrep -f "VLLM::" >/dev/null && fail STALE_VLLM
rm -f /dev/shm/link_remap.*
[ -n "${OVERLAY:-}" ] && tar -xzf "$OVERLAY" -C $SITE

mapfile -t ALL < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
mapfile -t MINE < <(python3 - <<'PY'
import json, os
entries = json.load(open("/host-device-plugins/kubelet_internal_checkpoint"))["Data"]["PodDeviceEntries"]
for e in entries:
    if e["PodUID"] == os.environ["POD_UID"] and e["ResourceName"] == "nvidia.com/gpu":
        print("\n".join(d for ids in e["DeviceIDs"].values() for d in ids))
PY
)
[ "${#MINE[@]}" -ge "$TP" ] || fail NOT_ENOUGH_GPUS
SRC=("${MINE[@]:0:$TP}")
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${SRC[*]}") CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_NCCL_SYMM_MEM=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1

LAUNCH=(python3 /exp/noiouring.py)
if [ "$MODE" = cuinterpose ]; then
  mkdir -p $BUNDLE && cp $B/snapshot-cuda/libcuinterpose.so $B/snapshot-cuda/libcuinterpose_core.so $B/cuinterpose-coordinator $BUNDLE/
  rm -f /tmp/cuinterpose-*.sock
  LAUNCH+=(env "LD_PRELOAD=$BUNDLE/libcuinterpose.so")
fi

T=$(now)
setsid "${LAUNCH[@]}" vllm serve "$MODEL" --served-model-name m --tensor-parallel-size "$TP" \
  --enable-sleep-mode --checkpoint-release-weights --host 127.0.0.1 --port 8000 "$@" \
  < /dev/null > "$W/vllm.log" 2>&1 &
VP=$!
for _ in $(seq 3600); do
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 $VP 2>/dev/null || { tail -40 vllm.log; fail VLLM_DIED; }
  sleep 1
done
R[cold_boot_s]=$(secs "$T")
T=$(now); complete > before.txt || fail FIRST_COMPLETION_FAILED; R[cold_first_completion_s]=$(secs "$T")
complete > before.txt
echo "before: $(cat before.txt)"

T=$(now)
curl -s -XPOST localhost:8000/checkpoint/prepare >/dev/null
for _ in $(seq 1200); do
  code=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/checkpoint/status)
  [ "$code" = 200 ] && break; [ "$code" = 500 ] && fail PREPARE_FAILED; sleep 0.5
done
R[prepare_s]=$(secs "$T")
PIDS=$($SD cuda pids $VP); R[cuda_processes]=$(echo "$PIDS" | wc -w)
R[gpu_mib_after_prepare]=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')

if [ "$MODE" = cuinterpose ]; then
  T=$(now)
  $SD cuinterpose prepare --bundle-dir $BUNDLE --checkpoint-dir "$IMG" $PIDS | tee cuinterpose-prepare.log || fail CUINTERPOSE_PREPARE_FAILED
  R[cuinterpose_prepare_s]=$(secs "$T")
fi
T=$(now); $SD cuda checkpoint $PIDS || fail CUDA_CHECKPOINT_FAILED; R[cuda_checkpoint_s]=$(secs "$T")
T=$(now)
LD_LIBRARY_PATH=$B/lib $B/criu dump -t $VP -D "$IMG" -L /work/empty --tcp-established --ext-unix-sk --file-locks \
  --link-remap --ghost-limit "${GHOST_LIMIT:-1G}" -v4 -o dump.log || { grep Error "$IMG/dump.log" | tail -10; fail CRIU_DUMP_FAILED; }
wait $VP 2>/dev/null
R[criu_dump_s]=$(secs "$T"); R[checkpoint_gb]=$(du -s --block-size=1M "$IMG" | awk '{printf "%.1f", $1/1024}')

T_RESTORE=$(now); T=$T_RESTORE
unshare --net bash -c "LD_LIBRARY_PATH=$B/lib $B/ip link set lo up && LD_LIBRARY_PATH=$B/lib exec $B/criu restore \
  -D $IMG -L /work/empty --tcp-established --ext-unix-sk --file-locks --link-remap --restore-detached \
  --pidfile $W/restored.pid -v4 -o restore.log" || { grep Error "$IMG/restore.log" | tail -10; fail CRIU_RESTORE_FAILED; }
R[criu_restore_s]=$(secs "$T")
NETNS="nsenter --net=/proc/$(cat "$W/restored.pid")/ns/net"
GPUS=()
if [ "$REMAP" = rotate ]; then
  TGT=("${SRC[@]:1}" "${SRC[0]}")
  GPUS=(--source "$(IFS=,; echo "${SRC[*]}")" --target "$(IFS=,; echo "${TGT[*]}")" --visible "$(IFS=,; echo "${ALL[*]}")")
fi
T=$(now); $SD cuda restore "${GPUS[@]}" $PIDS || fail CUDA_RESTORE_FAILED; R[cuda_restore_s]=$(secs "$T")
if [ "$MODE" = cuinterpose ]; then
  T=$(now)
  $NETNS $SD cuinterpose restore --bundle-dir $BUNDLE --checkpoint-dir "$IMG" || fail CUINTERPOSE_RESTORE_FAILED
  R[cuinterpose_restore_s]=$(secs "$T")
fi
T=$(now)
$NETNS curl -s -m 3600 -XPOST localhost:8000/checkpoint/restore | tee restore.json; echo
grep -q '"running"' restore.json || { grep -E "CUDA_ERROR|Error" vllm.log | tail -8; fail VLLM_RESTORE_FAILED; }
R[vllm_restore_s]=$(secs "$T"); R[restore_total_s]=$(secs "$T_RESTORE")
T=$(now); complete > after.txt 2>&1; R[restored_first_completion_s]=$(secs "$T")
echo "after:  $(cat after.txt)"
cmp -s before.txt after.txt && R[result]=MATCH || R[result]=DIFFER
[ -n "${POST_CHECK:-}" ] && R[post_check]=$(NETNS=$NETNS "$POST_CHECK" 2> post-check.log | tail -1)
report
killall9
