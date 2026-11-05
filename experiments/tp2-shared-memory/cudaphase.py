"""Run one CUDA checkpoint phase on every pid: lock|checkpoint|restore|unlock|state.
restore takes --map OLD=NEW,... (GPU UUIDs); the R580 API needs a bijection over every GPU the
processes can see."""
import argparse, ctypes as C, json, sys, time, uuid

class LockArgs(C.Structure):
    _fields_ = [("timeoutMs", C.c_uint), ("reserved0", C.c_uint), ("reserved1", C.c_uint64 * 7)]
class GpuPair(C.Structure):
    _fields_ = [("oldUuid", C.c_ubyte * 16), ("newUuid", C.c_ubyte * 16)]
class RestoreArgs(C.Structure):
    _fields_ = [("gpuPairs", C.POINTER(GpuPair)), ("gpuPairsCount", C.c_uint),
                ("reserved", C.c_char * 44), ("reserved1", C.c_uint64)]

p = argparse.ArgumentParser()
p.add_argument("op"); p.add_argument("pids", nargs="+", type=int); p.add_argument("--map", default="")
a = p.parse_args()
lib = C.CDLL("libcuda.so.1"); assert lib.cuInit(0) == 0
lib.cuGetErrorName.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
lib.cuCheckpointProcessGetState.argtypes = [C.c_int, C.POINTER(C.c_int)]
def name(rc):
    n = C.c_char_p(); lib.cuGetErrorName(rc, C.byref(n)); return n.value.decode() if n.value else str(rc)
def raw(u): return (C.c_ubyte * 16)(*uuid.UUID(u.removeprefix("GPU-")).bytes)
args = None
if a.op == "lock":
    args = LockArgs(30000)
elif a.op == "restore" and a.map:
    pairs = [x.split("=") for x in a.map.split(",")]
    arr = (GpuPair * len(pairs))(*[GpuPair(raw(o), raw(n)) for o, n in pairs])
    args = RestoreArgs(arr, len(pairs))
ok = True
for pid in a.pids:
    st = C.c_int(-1)
    if a.op == "state":
        rc = lib.cuCheckpointProcessGetState(pid, C.byref(st))
        print(json.dumps({"pid": pid, "rc": name(rc), "state": st.value})); continue
    fn = getattr(lib, "cuCheckpointProcess" + a.op.capitalize()); fn.argtypes = [C.c_int, C.c_void_p]
    t = time.perf_counter(); rc = fn(pid, C.byref(args) if args is not None else None)
    lib.cuCheckpointProcessGetState(pid, C.byref(st))
    print(json.dumps({"op": a.op, "pid": pid, "rc": name(rc), "state": st.value, "s": round(time.perf_counter() - t, 3)}), flush=True)
    ok &= rc == 0
sys.exit(0 if ok else 1)
