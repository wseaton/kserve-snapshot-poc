import os, sys, time, torch
gib = int(sys.argv[1])
blocks = [torch.full((1 << 30,), i + 1, dtype=torch.uint8, device="cuda") for i in range(gib)]
torch.cuda.synchronize()
def check():
    return [int(b[::1 << 20].sum()) for b in blocks]
want = check()
print("READY", os.getpid(), flush=True)
open("/tmp/hold.ready", "w").write(str(os.getpid()))
while not os.path.exists("/tmp/hold.check"):
    time.sleep(0.05)
t = time.time()
got = check()
print("CHECK", "MATCH" if got == want else "DIFFER", round(time.time() - t, 3), flush=True)
