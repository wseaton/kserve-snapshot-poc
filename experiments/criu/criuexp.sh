#!/bin/bash
# usage: criuexp.sh <tag> <remap 0|1> [vllm args]   (run in a privileged pod with 4 GPUs)
set -u
TAG=$1 REMAP=$2; shift 2
W=/work/$TAG; rm -rf $W; mkdir -p $W/img /work/empty; cd $W
ss -ltn | grep -q ":8000 " && { echo "RESULT $TAG: PORT_8000_BUSY"; exit 1; }
pgrep -f "VLLM::" && { echo "RESULT $TAG: STALE_VLLM"; exit 1; }
rm -f /dev/shm/link_remap.*
killall9() { for p in $(tree $VP) $PIDS; do kill -9 $p 2>/dev/null; done; sleep 5; }
B=/tmp/snapshot-cuda/snapshot-binaries
SD=${SNAPSHOTD:-/work/snapshotd}
RESTORE_GPUS=
criu() { LD_LIBRARY_PATH=$B/lib $B/criu "$@"; }
tar -xzf /exp/overlay.tgz -C /usr/local/lib/python3.12/dist-packages
mapfile -t ALL < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
mapfile -t MINE < <(ls /var/run/nvidia-container-devices 2>/dev/null | grep ^GPU- | sort)
[ ${#MINE[@]} -ge 4 ] || MINE=("${ALL[@]:0:4}")
[ ${#MINE[@]} -ge 4 ] || { echo "need 4 allocated GPUs, have: ${MINE[*]}"; exit 1; }
SRC="${MINE[0]},${MINE[1]}"
echo "allocated ${MINE[*]}; source $SRC"
export CUDA_VISIBLE_DEVICES=$SRC CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1} NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  VLLM_USE_NCCL_SYMM_MEM=0 VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm VLLM_WORKER_MULTIPROC_METHOD=spawn \
  UV_USE_IO_URING=0 UV_LOOP_USE_IO_URING=0 USE_LIBUV=0 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
setsid python3 /exp/noiouring.py vllm serve /models/Qwen3-8B --served-model-name m -tp 2 --enable-sleep-mode \
  --checkpoint-release-weights --enable-nccl-comm-suspend --disable-custom-all-reduce \
  --max-model-len 4096 --gpu-memory-utilization 0.6 --host 127.0.0.1 --port 8000 "$@" \
  < /dev/null > $W/vllm.log 2>&1 &
VP=$!
for i in $(seq 600); do curl -sf localhost:8000/health >/dev/null && break; kill -0 $VP 2>/dev/null || { echo "vllm died"; tail -30 vllm.log; exit 1; }; sleep 2; done
complete() { ${NETNS:-} curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model":"m","prompt":"The capital of France is","max_tokens":24,"temperature":0}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])'; }
complete > before.txt; echo "before: $(cat before.txt)"
curl -s -XPOST localhost:8000/checkpoint/prepare >/dev/null
for i in $(seq 120); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/checkpoint/status)" = 200 ] && break; sleep 1; done
curl -s localhost:8000/checkpoint/status; echo
PIDS=
tree() { local p=$1; echo $p; for t in /proc/$p/task/*; do for c in $(cat $t/children 2>/dev/null); do tree $c; done; done; }
PIDS=$($SD cuda pids $VP)
echo "cuda pids: $PIDS"
for p in $(tree $VP); do n=$(ls -l /proc/$p/fd 2>/dev/null | grep -c io_uring); [ "$n" != 0 ] && echo "io_uring: pid $p $(cat /proc/$p/comm) fds=$n"; done
$SD cuda checkpoint $PIDS || { echo "RESULT $TAG: CUDA_CHECKPOINT_FAILED"; killall9; exit 1; }
T0=$(date +%s%N)
criu dump -t $VP -D img -L /work/empty --tcp-established --ext-unix-sk --file-locks --link-remap \
  --ghost-limit 1G -v4 -o dump.log || { echo "RESULT $TAG: CRIU_DUMP_FAILED"; grep -E "Error|err" img/dump.log | tail -15; killall9; exit 1; }
wait $VP 2>/dev/null; echo "criu dump $(( ($(date +%s%N) - T0) / 1000000 )) ms, $(du -sh img | cut -f1)"
sleep 2; ps -eo pid,args | grep -E "vllm|VLLM" | grep -v grep && echo "tree still alive?"
T0=$(date +%s%N)
unshare --net bash -c "LD_LIBRARY_PATH=$B/lib $B/ip link set lo up && LD_LIBRARY_PATH=$B/lib exec $B/criu restore -D img -L /work/empty --tcp-established --ext-unix-sk --file-locks --link-remap --restore-detached --pidfile $W/restored.pid -v4 -o restore.log" || { echo "RESULT $TAG: CRIU_RESTORE_FAILED"; grep -E "Error" img/restore.log | tail -15; killall9; exit 1; }
echo "criu restore $(( ($(date +%s%N) - T0) / 1000000 )) ms, root pid $(cat $W/restored.pid)"
NETNS="nsenter --net=/proc/$(cat $W/restored.pid)/ns/net"
if [ "$REMAP" = 1 ]; then
  RESTORE_GPUS="--source ${MINE[0]},${MINE[1]} --target ${MINE[2]},${MINE[3]} --visible $(IFS=,; echo "${ALL[*]}")"
fi
$SD cuda restore ${RESTORE_GPUS:-} $PIDS || { echo "RESULT $TAG: CUDA_RESTORE_FAILED"; killall9; exit 1; }
$NETNS curl -s -m 300 -XPOST localhost:8000/checkpoint/restore; echo
NETNS=$NETNS complete > after.txt 2>&1; echo "after: $(cat after.txt)"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
cmp -s before.txt after.txt && echo "RESULT $TAG: MATCH" || echo "RESULT $TAG: DIFFER"
grep -E "CUDA_ERROR|Traceback" vllm.log | tail -5
for p in $PIDS; do kill -9 $p 2>/dev/null; done; sleep 5
