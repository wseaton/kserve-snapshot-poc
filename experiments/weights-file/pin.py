import ctypes, mmap, os, sys, time, torch
path, gib = sys.argv[1], int(sys.argv[2])
size = gib << 30
rt = ctypes.CDLL("libcudart.so.13")
rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
rt.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
rt.cudaHostUnregister.argtypes = [ctypes.c_void_p]
H2D, D2H = 1, 2
gpu = torch.empty(size, dtype=torch.uint8, device="cuda")
gpu.random_(0, 255)
want = int(gpu[::1 << 16].sum())
fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
os.ftruncate(fd, size)
host = torch.frombuffer(mmap.mmap(fd, size, flags=mmap.MAP_SHARED), dtype=torch.uint8)
def lap(what, f):
    torch.cuda.synchronize(); t = time.time(); r = f(); torch.cuda.synchronize()
    s = time.time() - t; print(f"{what:34s} {s:6.2f} s  {gib / s:6.1f} GiB/s", flush=True); return r
lap("device to file, pageable", lambda: rt.cudaMemcpy(host.data_ptr(), gpu.data_ptr(), size, D2H))
gpu.zero_()
lap("file to device, pageable", lambda: rt.cudaMemcpy(gpu.data_ptr(), host.data_ptr(), size, H2D))
assert int(gpu[::1 << 16].sum()) == want
e = lap("register the file mapping", lambda: rt.cudaHostRegister(host.data_ptr(), size, 0))
print("register returned", e)
if e == 0:
    gpu.zero_()
    lap("file to device, registered", lambda: rt.cudaMemcpy(gpu.data_ptr(), host.data_ptr(), size, H2D))
    assert int(gpu[::1 << 16].sum()) == want
    lap("device to file, registered", lambda: rt.cudaMemcpy(host.data_ptr(), gpu.data_ptr(), size, D2H))
    lap("unregister", lambda: rt.cudaHostUnregister(host.data_ptr()))
os.unlink(path)
