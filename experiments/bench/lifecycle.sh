#!/bin/bash
# Capture a server on one node, copy the checkpoint to another over RDMA, restore it there, and
# leave one Perfetto trace of the whole thing.
#
#   lifecycle.sh <tag> [vllm args]
#
# Environment:
#   SRC_POD, DST_POD   bench pods (pod.yaml) on the two nodes (default restore-bench, restore-bench-b)
#   NS, CONTEXT        namespace and kube context (default weaton-dev, coreweave-waldorf)
#   OVERLAP            0: dump to the source's disk, then copy; 1: CRIU streams into
#                      `snapshotd transfer stream`, which sends while the dump runs and writes
#                      nothing on the source; memory: as 1, and the target keeps the image in
#                      huge-page memory and restores from it, so neither disk is used (default 0)
#   RESERVE_GIB        memory the target makes available before the dump: faulted in with
#                      OVERLAP=memory, allocated and freed once with a /mem image directory (default 0)
#   IMG_DIR            where the CRIU image goes on both nodes (default /work/<tag>/img, the
#                      node's local disk; /mem/<tag> is RAM-backed in pod.yaml)
#   SLOTS, SLOT_MIB    shape of the registered pool (default 64, 64)
#   XFER               snapshotd built with the nixl feature (default /models/bin/snapshotd-nixl)
#   OUT                where the traces and the merged trace go (default ./lifecycle-<tag>)
#   TITLE, SUBTITLE    headings for the rendered page (lifecycle.html in OUT); PNG=1 also
#                      renders it to lifecycle.png with a headless Chrome
#   RESTORE_WAIT       seconds to leave the target pod alone while it restores (default 150)
# Everything restore-bench.sh reads (MODEL, TP, MODE, OVERLAY, SNAPSHOTD, COMPAT, ...) is passed
# through to both pods.
#
# Nothing is exec'ed into the target pod while CRIU restores: a process started there then can
# take a PID the checkpoint needs.
set -euo pipefail
TAG=$1; shift
: "${SRC_POD:=restore-bench}" "${DST_POD:=restore-bench-b}" "${NS:=weaton-dev}" "${CONTEXT:=coreweave-waldorf}"
: "${OVERLAP:=0}" "${SLOTS:=64}" "${SLOT_MIB:=64}" "${XFER:=/models/bin/snapshotd-nixl}" "${RESTORE_WAIT:=150}"
OUT=${OUT:-lifecycle-$TAG}; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
W=/work/$TAG
IMG=${IMG_DIR:-$W/img}
POOL=(--slots "$SLOTS" --slot-mib "$SLOT_MIB")

k() { kubectl --context "$CONTEXT" -n "$NS" "$@"; }
main() { local pod=$1; shift; k exec "$pod" -c main -- "$@"; }
xfer() { local pod=$1; shift; k exec "$pod" -c xfer -- "$@"; }

PASS=(MODEL TP MODE GPU_POSITIONS RELEASE_WEIGHTS CUDA_TOOL SNAPSHOTD COMPAT OVERLAY CRIU_VERBOSITY GHOST_LIMIT POST_CHECK
  NCCL_CUMEM_ENABLE VLLM_ALLREDUCE_USE_FLASHINFER VLLM_USE_SIMPLE_KV_OFFLOAD)
envs() { for name in "${PASS[@]}"; do [ -n "${!name:-}" ] && printf '%s=%q ' "$name" "${!name}"; done; printf 'CKPT_DIR=%q ' "$IMG"; }
ARGS=$(printf '%q ' "$@")

for pod in "$SRC_POD" "$DST_POD"; do k cp "$HERE/restore-bench.sh" "$pod:/work/rb.sh" -c main; done
DST_IP=$(k get pod "$DST_POD" -o jsonpath='{.status.podIP}')

receive() { # <dir> <trace> [more arguments]
  local dir=$1 trace=$2 direct=--direct; shift 2
  # tmpfs has no page cache to bypass; mappings let every slot copy at once.
  case $dir in /mem/*) direct=--mmap ;; esac
  xfer "$DST_POD" sh -c "setsid nohup $XFER transfer receive --dir $dir --listen 0.0.0.0:7700 $direct --trace $trace $* > $trace.log 2>&1 < /dev/null &
    for _ in \$(seq 3000); do grep -q ^listening $trace.log && exit 0; sleep 0.1; done; exit 1"
}

echo "== capture on $SRC_POD"
STREAM=0; [ "$OVERLAP" != 0 ] && STREAM=external
HOLD=(); [ "$OVERLAP" = memory ] && HOLD=(--hold --reserve-gib "${RESERVE_GIB:-0}")
# An image left by an earlier run with this tag would be mixed into the new one, and a RAM-backed
# one keeps holding memory.
case $IMG in /mem/?*|/work/?*)
  main "$DST_POD" python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$IMG" ;;
  *) echo "IMG_DIR must be under /work or /mem" >&2; exit 2 ;;
esac
# A RAM-backed image directory is primed first: allocating and freeing the space once makes the
# kernel reclaim page cache now instead of under every write of the transfer.
case $IMG in /mem/*) if [ "${RESERVE_GIB:-0}" -gt 0 ] && [ "$OVERLAP" != memory ]; then
  xfer "$DST_POD" python3 -c "
import os, threading, time
start = time.time()
def prime(i):
    path = '/mem/.prime-%d' % i
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    os.posix_fallocate(fd, 0, (${RESERVE_GIB} << 30) // 32)
    os.close(fd)
    os.unlink(path)
threads = [threading.Thread(target=prime, args=(i,)) for i in range(32)]
[t.start() for t in threads]; [t.join() for t in threads]
print('PRIME %.1f s for ${RESERVE_GIB} GiB' % (time.time() - start))" | tee "$OUT/prime.txt"
fi ;; esac
# The receiver goes first: with a reservation it faults its memory in before it listens.
[ "$OVERLAP" != 0 ] && receive "$IMG" /work/receive-stream.json ${HOLD[@]+"${HOLD[@]}"}
main "$SRC_POD" sh -c "rm -f /work/capture.log; setsid nohup env $(envs) PHASE=capture STREAM=$STREAM bash /work/rb.sh $TAG $ARGS > /work/capture.log 2>&1 < /dev/null &"
if [ "$OVERLAP" != 0 ]; then
  until main "$SRC_POD" grep -q '"name": "cold_boot"' "$W/phases.jsonl" 2>/dev/null; do sleep 5; done
  xfer "$SRC_POD" sh -c "cd /tmp && setsid nohup $XFER transfer stream --images-dir $IMG --to $DST_IP:7700 ${POOL[*]} --trace /work/send-stream.json > /work/stream.log 2>&1 < /dev/null &"
fi
until main "$SRC_POD" grep -q '^RESULT' /work/capture.log 2>/dev/null; do sleep 10; done
main "$SRC_POD" grep '^RESULT' /work/capture.log | tee "$OUT/capture.json"
grep -q '"CAPTURED"' "$OUT/capture.json"
if [ "$OVERLAP" != 0 ]; then
  until xfer "$SRC_POD" grep -qE '^(STREAM|Error)' /work/stream.log; do sleep 1; done
  xfer "$SRC_POD" grep -E '^(STREAM|Error)' /work/stream.log | tee "$OUT/stream.txt"
fi

echo "== copy to $DST_POD"
receive "$W" /work/receive-files.json
xfer "$SRC_POD" sh -c "cd /tmp && $XFER transfer send --dir $W --to $DST_IP:7700 ${POOL[*]} --trace /work/send-files.json" | grep -E '^(TRANSFER|Error)' | tee "$OUT/transfer.txt"

if [ "$IMG" != "$W/img" ]; then
  # What the source wrote beside the stream: cuInterpose's state, CRIU's logs.
  receive "$IMG" /work/receive-side.json
  xfer "$SRC_POD" sh -c "cd /tmp && $XFER transfer send --dir $IMG --to $DST_IP:7700 ${POOL[*]}" | grep -E '^(TRANSFER|Error)' | tee "$OUT/transfer-side.txt"
fi

echo "== restore on $DST_POD"
RESTORE_STREAM=0; [ "$OVERLAP" = memory ] && RESTORE_STREAM=external
main "$DST_POD" sh -c "rm -f /work/restore.log; setsid nohup env $(envs) PHASE=restore STREAM=$RESTORE_STREAM bash /work/rb.sh $TAG > /work/restore.log 2>&1 < /dev/null &"
end=$(( $(date +%s) + RESTORE_WAIT )); until [ "$(date +%s)" -ge "$end" ]; do sleep 5; done
until main "$DST_POD" grep -q '^RESULT' /work/restore.log 2>/dev/null; do sleep 10; done
main "$DST_POD" grep '^RESULT' /work/restore.log | tee "$OUT/restore.json"

[ "$OVERLAP" = memory ] && xfer "$DST_POD" grep -E '^(HOLD|SERVE)' /work/receive-stream.json.log | tee "$OUT/hold.txt"
main "$DST_POD" cat "$W/phases.jsonl" > "$OUT/phases.jsonl"
TRACES=()
fetch() { # <pod> <remote> <local>
  if xfer "$1" test -s "$2" 2>/dev/null; then xfer "$1" cat "$2" > "$OUT/$3"; TRACES+=("$OUT/$3"); fi
}
fetch "$SRC_POD" /work/send-stream.json send-stream.json
fetch "$DST_POD" /work/receive-stream.json receive-stream.json
fetch "$SRC_POD" /work/send-files.json send-files.json
fetch "$DST_POD" /work/receive-files.json receive-files.json
"$HERE/trace-merge.py" -o "$OUT/lifecycle.json" --phases "$OUT/phases.jsonl" "${TRACES[@]}"
"$HERE/trace-render.py" "$OUT" --title "${TITLE:-$TAG}" --subtitle "${SUBTITLE:-}" ${PNG:+--png "$OUT/lifecycle.png"}
