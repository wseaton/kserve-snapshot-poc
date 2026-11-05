"""Ask every cuInterpose-preloaded process for its tracked CUDA memory records."""
import glob, json, socket, struct, sys
import msgpack

def recvall(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b

def inspect(pid):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(f"/tmp/cuinterpose-{pid}.sock")
    body = msgpack.packb({"version": 1, "body": {"kind": "inspect", "namespace_pid": pid}}, use_bin_type=True)
    s.sendall(struct.pack("<I", len(body)) + body)
    n = struct.unpack("<I", recvall(s, 4))[0]
    return msgpack.unpackb(recvall(s, n), raw=False)["body"]

out = {}
for path in sorted(glob.glob("/tmp/cuinterpose-*.sock")):
    pid = int(path.rsplit("-", 1)[1].split(".")[0])
    try:
        resp = inspect(pid)
    except Exception as e:
        out[pid] = {"error": repr(e)}
        continue
    res = resp["result"]
    if "Err" in res:
        out[pid] = {"error": res["Err"]}
        continue
    records = res["Ok"]["inspection"]["records"]
    allocs = [r["allocation"] for r in records if "allocation" in r]
    shared = [a for a in allocs if a["shared"]]
    kinds = {}
    for r in records:
        k = next(iter(r))
        kinds[k] = kinds.get(k, 0) + 1
    cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")[:80].decode(errors="replace")
    out[pid] = {
        "cmd": cmd,
        "records": kinds,
        "allocations": len(allocs),
        "allocation_bytes": sum(a["size"] for a in allocs),
        "shared": len(shared),
        "shared_bytes": sum(a["size"] for a in shared),
        "shared_detail": sorted({(a["size"], a["handle_types"], a["allocation"]["creator_pid"]) for a in shared}),
    }
print(json.dumps(out, indent=1, default=str))
