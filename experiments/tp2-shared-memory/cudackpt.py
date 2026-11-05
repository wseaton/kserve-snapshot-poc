"""Lock all, checkpoint all, restore all, unlock all through the CUDA driver API."""
import ctypes as C, json, os, sys, time

lib = C.CDLL("libcuda.so.1")
assert lib.cuInit(0) == 0
for op in ("Lock", "Checkpoint", "Restore", "Unlock"):
    getattr(lib, "cuCheckpointProcess" + op).argtypes = [C.c_int, C.c_void_p]
lib.cuCheckpointProcessGetState.argtypes = [C.c_int, C.POINTER(C.c_int)]
lib.cuCheckpointProcessGetRestoreThreadId.argtypes = [C.c_int, C.POINTER(C.c_int)]
lib.cuGetErrorName.argtypes = [C.c_int, C.POINTER(C.c_char_p)]

class LockArgs(C.Structure):
    _fields_ = [("timeoutMs", C.c_uint), ("reserved0", C.c_uint), ("reserved1", C.c_uint64 * 7)]

def name(rc):
    n = C.c_char_p(); lib.cuGetErrorName(rc, C.byref(n)); return n.value.decode() if n.value else str(rc)

def cuda_pids():
    me = os.getpid(); pids = []
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == me:
            continue
        tid = C.c_int()
        if lib.cuCheckpointProcessGetRestoreThreadId(int(p), C.byref(tid)) == 0:
            pids.append(int(p))
    return sorted(pids)

pids = [int(x) for x in sys.argv[1:]] or cuda_pids()
print("cuda pids", pids, flush=True)
ok = True
for op in ("Lock", "Checkpoint", "Restore", "Unlock"):
    for pid in pids:
        args = LockArgs(30000) if op == "Lock" else None
        t = time.perf_counter()
        rc = getattr(lib, "cuCheckpointProcess" + op)(pid, C.byref(args) if args else None)
        st = C.c_int(-1); lib.cuCheckpointProcessGetState(pid, C.byref(st))
        print(json.dumps({"op": op, "pid": pid, "rc": name(rc), "state": st.value, "s": round(time.perf_counter() - t, 3)}), flush=True)
        ok &= rc == 0
    if not ok:
        print("FAILED at", op); sys.exit(1)
    if op == "Checkpoint":
        time.sleep(2)
print("CUDA_CKPT_OK")
