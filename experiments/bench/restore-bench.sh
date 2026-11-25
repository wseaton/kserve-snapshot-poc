#!/bin/bash
# Cold boot one vLLM server, checkpoint it with CRIU, restore it, and time every phase.
#
#   restore-bench.sh <tag> [vllm args]
#
# Environment:
#   MODEL        model directory (required)
#   TP           tensor parallel size (default 8)
#   DP           data parallel size; the server takes TP*DP GPUs (default 1)
#   MODE         plain: cuda-checkpoint only; cuinterpose: preload cuInterpose and run its
#                prepare and restore phases around the CUDA checkpoint; gpucr: preload GPU-CR's
#                vGPU library, which moves the device memory it tracks into a file before the
#                CUDA checkpoint and back after the CUDA restore, so that memory is in neither
#                the process nor the CRIU image (default plain)
#   GPUCR        directory holding vGPU-NVIDIA.so and cr_client, from one build (default /gpu-cr)
#   GPUCR_STORE  where GPU-CR keeps each process's dump (default /work/<tag>/gpucr)
#   GPUCR_SHM_GB size of that dump for each process; it is allocated in full (default 24)
#   CUDA_TOOL    snapshotd: one process drives the driver API, all PIDs in parallel; helper:
#                Dynamo's cuda-checkpoint-helper, one run for each PID and action, in the order
#                its agent uses (default snapshotd)
#   COMPAT       directory of a newer libcuda to put first on vLLM's library path, if set
#   STREAM       1: CRIU streams the checkpoint into `snapshotd image hold`, which keeps it in
#                huge-page memory and serves the restore from there; nothing is written to
#                CKPT_DIR but logs and sockets; external: CRIU streams to a process that
#                already listens on streamer-capture.sock in CKPT_DIR, such as
#                `snapshotd transfer stream` in the xfer container; with PHASE=restore, CRIU
#                restores from a process serving streamer-serve.sock, such as
#                `snapshotd transfer receive --hold` (default 0)
#   RESERVE_GIB  with STREAM=1, memory to fault in before the dump starts (default 0)
#   CRIU_VERBOSITY  CRIU's -v level (default 4)
#   PHASE        all: capture and restore here; capture: stop after the dump and leave everything
#                a restore on another node needs under /work/<tag>; restore: restore a /work/<tag>
#                captured elsewhere onto this pod's GPUs; serve: cold boot and leave the
#                server running, as a ModelExpress peer for example (default all)
#   GPU_POSITIONS  positions, among the GPUs the container can open, of the ones to run on, such
#                as "4 5 6 7"; they must be allocated to the pod (default: its first TP*DP GPUs).
#                A restore on another node lands on the same positions there
#   RELEASE_WEIGHTS  0: keep the weights in the checkpoint instead of releasing them at prepare
#                and reloading them at restore (default 1)
#   MX           1: load weights through ModelExpress, from a peer over RDMA when one is serving,
#                else from disk; MX_SERVER is its address (default modelexpress-server:8001)
#   NET          pod: restore into the pod's own network instead of a new namespace, so the
#                restored server can reach peers; needs the dumped ports free (default new)
#   KEEP         1: leave the restored server running (default 0)
#   REMAP        0: restore onto the same GPUs; rotate: restore rank i onto rank i+1's GPU (default 0)
#   CKPT_DIR     where the CRIU images go (default /work/<tag>/img)
#   SNAPSHOTD    snapshotd binary (default /models/bin/snapshotd)
#   GHOST_LIMIT largest unlinked file CRIU may copy into the image (default 1G)
#   POST_CHECK   command run against the restored server, with NETNS set to the prefix that
#                enters its network namespace; its last output line goes into the result
#   OVERLAY      tarball of patched vLLM files to unpack over the image's, if set
#
# Prints one RESULT line of JSON. Run in a privileged pod whose PID 1 reaps children. Such a pod
# may see every GPU on the node, so the GPUs to use come from the kubelet's device checkpoint.
set -u
TAG=$1; shift
: "${MODEL:?}" "${TP:=8}" "${DP:=1}" "${MODE:=plain}" "${REMAP:=0}" "${CUDA_TOOL:=snapshotd}" "${STREAM:=0}" "${PHASE:=all}"
NGPU=$((TP * DP))
W=/work/$TAG; IMG=${CKPT_DIR:-$W/img}
if [ "$PHASE" != restore ]; then rm -rf "$W" "$IMG"; fi
mkdir -p "$W" "$IMG" /work/empty; cd "$W" || exit 1
SD=${SNAPSHOTD:-/models/bin/snapshotd}
B=/tmp/snapshot-cuda/snapshot-binaries
BUNDLE=/work/cuinterpose
SITE=/usr/local/lib/python3.12/dist-packages

now() { date +%s%N; }
secs() { python3 -c "print(round(($(now) - $1) / 1e9, 3))"; }
declare -A R
# Record a phase that began at $2: its duration in the result, and its wall-clock span in
# phases.jsonl for trace-merge.py.
lap() {
  R[$1_s]=$(secs "$2")
  echo "{\"name\": \"$1\", \"start_us\": $(($2 / 1000)), \"end_us\": $(($(now) / 1000)), \"node\": \"$(hostname)\"}" >> "$W/phases.jsonl"
}
fail() { R[result]=$1; report; killall9; exit 1; }
report() {
  local out="{\"tag\": \"$TAG\", \"mode\": \"$MODE\", \"cuda_tool\": \"$CUDA_TOOL\", \"stream\": \"$STREAM\", \"tp\": $TP, \"dp\": $DP, \"remap\": \"$REMAP\""
  for k in "${!R[@]}"; do out+=", \"$k\": \"${R[$k]}\""; done
  echo "RESULT $out}"
}
tree() { local p=$1; echo "$p"; for t in /proc/"$p"/task/*; do for c in $(cat "$t"/children 2>/dev/null); do tree "$c"; done; done; }
PIDS=; VP=
HOLD=
killall9() { [ -n "$HOLD" ] && kill "$HOLD" 2>/dev/null; for p in $( [ -n "$VP" ] && tree "$VP") $PIDS; do kill -9 "$p" 2>/dev/null; done
  for _ in $(seq 120); do pgrep -f "VLLM::" >/dev/null || break; sleep 1; done; sleep 5; }
complete() { ${NETNS:-} curl -s -m 600 localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model":"m","prompt":"The capital of France is","max_tokens":32,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'; }

ss -ltn | grep -q ":8000 " && fail PORT_8000_BUSY
pgrep -f "VLLM::" >/dev/null && fail STALE_VLLM
rm -f /dev/shm/link_remap.*
[ -n "${OVERLAY:-}" ] && tar -xzf "$OVERLAY" -C $SITE

# Every GPU this container can open, in device-minor order, the order the driver enumerates them
# in. /proc lists the whole node's; a device node exists only for the ones the pod was given,
# unless the runtime gives a privileged pod all of them.
mapfile -t VISIBLE < <(for f in /proc/driver/nvidia/gpus/*/information; do
  awk '/^GPU UUID/ {u=$3} /^Device Minor/ {m=$3} END {print m, u}' "$f"; done | sort -n |
  while read -r minor uuid; do [ -e "/dev/nvidia$minor" ] && echo "$uuid"; done)
mapfile -t MINE < <(python3 - <<'PY'
import json, os
entries = json.load(open("/host-device-plugins/kubelet_internal_checkpoint"))["Data"]["PodDeviceEntries"]
for e in entries:
    if e["PodUID"] == os.environ["POD_UID"] and e["ResourceName"] == "nvidia.com/gpu":
        print("\n".join(d for ids in e["DeviceIDs"].values() for d in ids))
PY
)
[ "${#MINE[@]}" -ge "$NGPU" ] || fail NOT_ENOUGH_GPUS
mine() { [[ " ${MINE[*]} " == *" $1 "* ]]; }
SRC=()
if [ -n "${GPU_POSITIONS:-}" ]; then
  for p in $GPU_POSITIONS; do mine "${VISIBLE[$p]}" || fail GPU_NOT_ALLOCATED; SRC+=("${VISIBLE[$p]}"); done
else
  for u in "${VISIBLE[@]}"; do mine "$u" && [ "${#SRC[@]}" -lt "$NGPU" ] && SRC+=("$u"); done
fi
[ "${#SRC[@]}" = "$NGPU" ] || fail NOT_ENOUGH_GPUS
if [ "$PHASE" = restore ]; then
  . "$W/capture.env" || fail NO_CAPTURE
  cp "$W/vllm.log.captured" "$W/vllm.log"
  cp -a "$W/shm/." /dev/shm/
  echo 4000000 > /proc/sys/kernel/ns_last_pid
fi
TARGET=("${SRC[@]}")
if [ "$PHASE" = restore ]; then
  # The driver restores each device onto the GPU at the same position among the visible ones as
  # on the source node, so both sides must see the same number. Refuse unless the GPUs it lands
  # on are this pod's.
  [ "${#SRC_VISIBLE[@]}" = "${#VISIBLE[@]}" ] || fail VISIBLE_GPU_COUNT_DIFFERS
  TARGET=()
  for u in "${SRC[@]}"; do
    for p in "${!SRC_VISIBLE[@]}"; do [ "${SRC_VISIBLE[$p]}" = "$u" ] && TARGET+=("${VISIBLE[$p]}"); done
  done
  [ "${#TARGET[@]}" = "${#SRC[@]}" ] || fail SOURCE_GPU_UNKNOWN
  for u in "${TARGET[@]}"; do mine "$u" || fail TARGET_GPU_NOT_ALLOCATED; done
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${TARGET[*]}") CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_NCCL_SYMM_MEM=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1

LAUNCH=(python3 /exp/noiouring.py)
MX_ARGS=()
if [ "${MX:-0}" = 1 ]; then
  LAUNCH+=(env VLLM_PLUGINS=modelexpress "MX_SERVER_ADDRESS=${MX_SERVER:-modelexpress-server:8001}"
    MX_METADATA_PORT=5555 MX_WORKER_GRPC_PORT=6555 MX_RDMA_NIC_PIN=auto
    MX_POD_INFO_DIR=/etc/snapshot-podinfo "UCX_TLS=^rc_gda")
  MX_ARGS=(--load-format modelexpress)
fi
[ -n "${COMPAT:-}" ] && LAUNCH+=(env "LD_LIBRARY_PATH=$COMPAT:$LD_LIBRARY_PATH")
helper() { local action=$1; shift; for p in $PIDS; do $B/cuda-checkpoint-helper --action "$action" --pid "$p" "$@" || return 1; done; }
cuda_checkpoint() {
  if [ "$CUDA_TOOL" = helper ]; then helper lock && helper checkpoint; else $SD cuda checkpoint $PIDS; fi
}
cuda_restore() {
  if [ "$CUDA_TOOL" != helper ]; then $SD cuda restore "${GPUS[@]}" $PIDS; return; fi
  helper restore ${DEVICE_MAP:+--device-map "$DEVICE_MAP"} && helper unlock
}
GC=${GPUCR:-/gpu-cr}; GSTORE=${GPUCR_STORE:-$W/gpucr}
gpucr() { for p in $PIDS; do EXPORT_FILE_PATH=$GSTORE "$GC/cr_client" "$1" -b -p "$p" || return 1; done; }
if [ "$MODE" = gpucr ]; then
  mkdir -p "$GSTORE"
  LAUNCH+=(env "LD_PRELOAD=$GC/vGPU-NVIDIA.so" GPU_VENDOR=NVIDIA "EXPORT_FILE_PATH=$GSTORE" "GPU_CR_SHM_GB=${GPUCR_SHM_GB:-24}")
fi
if [ "$MODE" = cuinterpose ]; then
  mkdir -p $BUNDLE && cp $B/snapshot-cuda/libcuinterpose.so $B/snapshot-cuda/libcuinterpose_core.so $B/cuinterpose-coordinator $BUNDLE/
  rm -f /tmp/cuinterpose-*.sock
  LAUNCH+=(env "LD_PRELOAD=$BUNDLE/libcuinterpose.so")
fi

CRIU_STREAM=(); [ "$STREAM" != 0 ] && CRIU_STREAM=(--stream)
RELEASE=(--checkpoint-release-weights); [ "${RELEASE_WEIGHTS:-1}" = 0 ] && RELEASE=()
if [ "$PHASE" != restore ]; then
if [ ! -L "$HOME/.humming" ]; then
  mkdir -p /compile-cache/humming
  [ -d "$HOME/.humming" ] && cp -a "$HOME/.humming/." /compile-cache/humming/ && rm -r "$HOME/.humming"
  ln -s /compile-cache/humming "$HOME/.humming"
fi
T=$(now)
setsid "${LAUNCH[@]}" vllm serve "$MODEL" --served-model-name m --tensor-parallel-size "$TP" --data-parallel-size "$DP" \
  --enable-sleep-mode "${RELEASE[@]}" "${MX_ARGS[@]}" --host 127.0.0.1 --port 8000 "$@" \
  < /dev/null > "$W/vllm.log" 2>&1 &
VP=$!
for _ in $(seq 3600); do
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 $VP 2>/dev/null || { tail -40 vllm.log; fail VLLM_DIED; }
  sleep 1
done
lap cold_boot "$T"
T=$(now); complete > before.txt || fail FIRST_COMPLETION_FAILED; lap cold_first_completion "$T"
complete > before.txt
echo "before: $(cat before.txt)"
if [ "$PHASE" = serve ]; then R[result]=SERVING; report; exit 0; fi

T=$(now)
curl -s -XPOST localhost:8000/checkpoint/prepare >/dev/null
for _ in $(seq 1200); do
  code=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/checkpoint/status)
  [ "$code" = 200 ] && break; [ "$code" = 500 ] && fail PREPARE_FAILED; sleep 0.5
done
lap prepare "$T"
PIDS=$($SD cuda pids $VP); R[cuda_processes]=$(echo "$PIDS" | wc -w)
R[gpu_mib_after_prepare]=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')

if [ "$MODE" = cuinterpose ]; then
  T=$(now)
  $SD cuinterpose prepare --bundle-dir $BUNDLE --checkpoint-dir "$IMG" $PIDS | tee cuinterpose-prepare.log || fail CUINTERPOSE_PREPARE_FAILED
  lap cuinterpose_prepare "$T"
fi
if [ "$MODE" = gpucr ]; then
  T=$(now); gpucr -c > gpucr-dump.log 2>&1 || { tail -5 gpucr-dump.log; fail GPUCR_DUMP_FAILED; }; lap gpucr_dump "$T"
  R[gpu_mib_after_gpucr]=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
fi
T=$(now); cuda_checkpoint || fail CUDA_CHECKPOINT_FAILED; lap cuda_checkpoint "$T"
if [ "$STREAM" = external ]; then
  for _ in $(seq 600); do [ -S "$IMG/streamer-capture.sock" ] && break; sleep 0.1; done
  [ -S "$IMG/streamer-capture.sock" ] || fail NO_CAPTURE_SOCKET
fi
if [ "$STREAM" = 1 ]; then
  $SD image hold --images-dir "$IMG" --reserve-gib "${RESERVE_GIB:-0}" > "$W/hold.log" 2>&1 &
  HOLD=$!
  for _ in $(seq 1200); do grep -q ^ready "$W/hold.log" && break; sleep 0.1; done
  grep -q ^ready "$W/hold.log" || fail HOLD_NOT_READY
fi
T=$(now)
LD_LIBRARY_PATH=$B/lib $B/criu dump -t $VP -D "$IMG" -L /work/empty --tcp-established --ext-unix-sk --file-locks \
  --link-remap --ghost-limit "${GHOST_LIMIT:-1G}" --libdir $B/criu-plugins "${CRIU_STREAM[@]}" -v"${CRIU_VERBOSITY:-4}" -o dump.log || { grep Error "$IMG/dump.log" | tail -10; fail CRIU_DUMP_FAILED; }
wait $VP 2>/dev/null
for _ in $(seq 600); do ps -eo sid= | awk -v s="$VP" '$1 == s {f = 1} END {exit !f}' || break; sleep 0.1; done
ps -eo sid= | awk -v s="$VP" '$1 == s {f = 1} END {exit !f}' && fail SOURCE_SESSION_ALIVE
lap criu_dump "$T"
if [ "$STREAM" = 1 ]; then
  for _ in $(seq 600); do grep -q ^CAPTURE "$W/hold.log" && break; sleep 0.1; done
  CAPTURE=$(grep ^CAPTURE "$W/hold.log" | cut -d' ' -f2-) || fail CAPTURE_MISSING
  R[checkpoint_gb]=$(echo "$CAPTURE" | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["gb"], 1))')
  R[huge_page_fraction]=$(echo "$CAPTURE" | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["huge_page_fraction"], 3))')
elif [ "$STREAM" = 0 ]; then
  R[checkpoint_gb]=$(du -s --block-size=1M "$IMG" | awk '{printf "%.1f", $1/1024}')
fi
if [ "$PHASE" = capture ]; then
  SRC_VISIBLE=("${VISIBLE[@]}")
  SRC_IP=$(hostname -i)
  declare -p PIDS SRC SRC_VISIBLE SRC_IP > "$W/capture.env"
  cp "$W/vllm.log" "$W/vllm.log.captured"
  mkdir -p "$W/shm" && cp -a /dev/shm/. "$W/shm/"
  R[result]=CAPTURED; report; exit 0
fi
fi

rm -f "$W/restored.pid"
T_RESTORE=$(now); T=$T_RESTORE
RESTORE="LD_LIBRARY_PATH=$B/lib exec $B/criu restore -D $IMG -L /work/empty --ext-unix-sk --file-locks --link-remap \
  --restore-detached ${CRIU_STREAM[*]} --pidfile $W/restored.pid -v${CRIU_VERBOSITY:-4} -o restore.log"
if [ "${NET:-new}" = pod ]; then
  # The source's sockets carry its pod address; Dynamo's CRIU plugin rewrites it to this pod's.
  if [ -n "${SRC_IP:-}" ] && [ "$SRC_IP" != "$(hostname -i)" ]; then
    echo "$SRC_IP $(hostname -i)" > "$W/inet-remap.map"
    export SNAPSHOT_CRIU_INET_REMAP_FILE=$W/inet-remap.map
  fi
  bash -c "$RESTORE --tcp-established --libdir $B/criu-plugins" || { grep Error "$IMG/restore.log" | tail -10; fail CRIU_RESTORE_FAILED; }
else
  unshare --net bash -c "LD_LIBRARY_PATH=$B/lib $B/ip link set lo up && $RESTORE --tcp-established" \
    || { grep Error "$IMG/restore.log" | tail -10; fail CRIU_RESTORE_FAILED; }
fi
lap criu_restore "$T"
NETNS=; [ "${NET:-new}" = pod ] || NETNS="nsenter --net=/proc/$(cat "$W/restored.pid")/ns/net"
GPUS=(); DEVICE_MAP=
if [ "$PHASE" = restore ]; then
  GPUS=(--source "$(IFS=,; echo "${SRC_VISIBLE[*]}")" --target "$(IFS=,; echo "${VISIBLE[*]}")" --visible "$(IFS=,; echo "${VISIBLE[*]}")")
elif [ "$REMAP" = rotate ]; then
  TGT=("${SRC[@]:1}" "${SRC[0]}")
  for u in "${VISIBLE[@]}"; do
    t=$u; for i in "${!SRC[@]}"; do [ "${SRC[$i]}" = "$u" ] && t=${TGT[$i]}; done
    DEVICE_MAP+=${DEVICE_MAP:+,}$u=$t
  done
  GPUS=(--source "$(IFS=,; echo "${SRC[*]}")" --target "$(IFS=,; echo "${TGT[*]}")" --visible "$(IFS=,; echo "${VISIBLE[*]}")")
fi
T=$(now); cuda_restore || fail CUDA_RESTORE_FAILED; lap cuda_restore "$T"
if [ "$MODE" = gpucr ]; then
  T=$(now); gpucr -r > gpucr-restore.log 2>&1 || { tail -5 gpucr-restore.log; fail GPUCR_RESTORE_FAILED; }; lap gpucr_restore "$T"
fi
if [ "$MODE" = cuinterpose ]; then
  T=$(now)
  $NETNS $SD cuinterpose restore --bundle-dir $BUNDLE --checkpoint-dir "$IMG" || fail CUINTERPOSE_RESTORE_FAILED
  lap cuinterpose_restore "$T"
fi
T=$(now)
$NETNS curl -s -m 3600 -XPOST localhost:8000/checkpoint/restore | tee restore.json; echo
grep -q '"running"' restore.json || {
  grep -E "CUDA_ERROR|Error" vllm.log | tail -8
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  fail VLLM_RESTORE_FAILED
}
lap vllm_restore "$T"; lap restore_total "$T_RESTORE"
T=$(now); complete > after.txt 2>&1; lap restored_first_completion "$T"
echo "after:  $(cat after.txt)"
cmp -s before.txt after.txt && R[result]=MATCH || R[result]=DIFFER
[ -n "${POST_CHECK:-}" ] && R[post_check]=$(NETNS=$NETNS "$POST_CHECK" 2> post-check.log | tail -1)
report
[ "${KEEP:-0}" = 1 ] || killall9
