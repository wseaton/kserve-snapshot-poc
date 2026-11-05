#!/bin/bash
# usage: exp.sh <tag> <inspect|plain> <patch 0|1> [extra vllm args...]
set -u
TAG=$1 MODE=$2 PATCH=$3; shift 3
W=/work/$TAG; mkdir -p $W; cd $W
SP=/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py
[ -f $SP.orig ] || cp $SP $SP.orig
if [ "$PATCH" = 1 ]; then cp /exp/cuda_communicator.py $SP; else cp $SP.orig $SP; fi
rm -f /tmp/cuinterpose-*.sock
PREFIX=()
[ "$MODE" != plain ] && PREFIX=(env LD_PRELOAD=/tmp/snapshot-cuda/snapshot-binaries/snapshot-cuda/libcuinterpose.so)
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1} NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 \
  VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_NCCL_SYMM_MEM=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ALLOC
setsid "${PREFIX[@]}" vllm serve /models/Qwen3-8B --served-model-name m -tp 2 --enable-sleep-mode \
  --checkpoint-release-weights --enable-nccl-comm-suspend --max-model-len 4096 \
  --gpu-memory-utilization 0.6 --port 8000 "$@" > vllm.log 2>&1 &
VP=$!
for i in $(seq 600); do curl -sf localhost:8000/health >/dev/null && break; kill -0 $VP 2>/dev/null || { echo "vllm died"; tail -40 vllm.log; exit 1; }; sleep 2; done
complete() { curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model":"m","prompt":"The capital of France is","max_tokens":24,"temperature":0}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])'; }
complete > before.txt; echo "before: $(cat before.txt)"
[ "$MODE" = inspect ] && python3 /exp/inspect.py > inspect-running.json
curl -s -XPOST localhost:8000/checkpoint/prepare; echo
for i in $(seq 120); do s=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/checkpoint/status); [ "$s" = 200 ] && break; [ "$s" = 500 ] && break; sleep 1; done
curl -s localhost:8000/checkpoint/status; echo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
if [ "$MODE" = inspect ]; then python3 /exp/inspect.py > inspect-prepared.json; cat inspect-prepared.json; fi
B=/tmp/snapshot-cuda/snapshot-binaries
if [ "$MODE" = cuinterpose ]; then
  python3 /exp/inspect.py > inspect-prepared.json
  PIDS=$(ls /tmp/cuinterpose-*.sock | sed -E 's/.*cuinterpose-([0-9]+).sock/--process \1/' | tr '\n' ' ')
  mkdir -p $W/ck
  $B/cuinterpose-coordinator --prepare --checkpoint-dir $W/ck --socket-dir /tmp $PIDS; echo "coordinator prepare exit $?"
  timeout 240 python3 /exp/cudackpt.py > cudackpt.log 2>&1; echo "cudackpt exit $?"; grep -E "Restore|OK|FAILED" cudackpt.log
  $B/cuinterpose-coordinator --restore --checkpoint-dir $W/ck --socket-dir /tmp $PIDS; echo "coordinator restore exit $?"
fi
if [ "$MODE" = plain ]; then
  WORKERS=$(pgrep -f "VllmWorker|EngineCore" | tr '\n' ' ')
  timeout 240 python3 /exp/cudackpt.py > cudackpt.log 2>&1; echo "cudackpt exit $?"; cat cudackpt.log
  grep -q CUDA_CKPT_OK cudackpt.log || { echo "RESULT $TAG: CUDA_CKPT_FAILED"; kill -9 -- -$VP; sleep 10; exit 1; }
fi
curl -s -m 300 -XPOST localhost:8000/checkpoint/restore; echo
complete > after.txt 2>&1; echo "after: $(cat after.txt)"
cmp -s before.txt after.txt && echo "RESULT $TAG: MATCH" || echo "RESULT $TAG: DIFFER"
grep -E "Error|error|Traceback|CUDA_ERROR" vllm.log | grep -v -i "debug" | tail -15
kill -- -$VP; sleep 15; kill -9 -- -$VP 2>/dev/null; sleep 5
