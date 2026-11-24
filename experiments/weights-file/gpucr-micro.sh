#!/bin/bash
# micro.sh <store dir> <GiB>: time GPU-CR's full dump and restore of <GiB> of device memory.
STORE=$1; GIB=$2; G=/gpu-cr
rm -f /tmp/hold.ready /tmp/hold.check; mkdir -p "$STORE"; rm -f "$STORE"/*
now() { date +%s%N; }; secs() { python3 -c "print(round(($(now) - $1) / 1e9, 2))"; }
export GPU_VENDOR=NVIDIA EXPORT_FILE_PATH=$STORE GPU_CR_SHM_GB=$((GIB + 4)) CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "${GPU_INDEX:-1}p")
used() { nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits; }
LD_PRELOAD=$G/vGPU-NVIDIA.so PYTORCH_ALLOC_CONF=expandable_segments:False python3 /tmp/hold.py "$GIB" > /tmp/hold.log 2>&1 &
for _ in $(seq 600); do [ -f /tmp/hold.ready ] && break; sleep 0.2; done
P=$(cat /tmp/hold.ready); echo "store=$STORE gib=$GIB pid=$P gpu_mib=$(used)"
T=$(now); $G/cr_client -c -b -p "$P" > /tmp/cr-c.log 2>&1 || { tail -3 /tmp/cr-c.log; kill -9 $P; exit 1; }; echo "dump_s=$(secs $T) gpu_mib_after=$(used)"
T=$(now); $G/cr_client -r -b -p "$P" > /tmp/cr-r.log 2>&1 || { tail -3 /tmp/cr-r.log; kill -9 $P; exit 1; }; echo "restore_s=$(secs $T) gpu_mib_after=$(used)"
touch /tmp/hold.check; wait $P
grep -E "CHECK|pinned|ckpt size|restore size|memcpy|Data transfer" /tmp/hold.log | cut -c1-120
