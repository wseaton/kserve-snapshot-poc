"""A stand-in for a training job that takes turns on a GPU through the TimeSlice Orchestrator.

Each burst adds to a tensor that fills most of the GPU and checks that every earlier burst's
additions are still there, so a job whose device memory did not survive being parked shows up.
It gives the memory back in a last turn of its own: a job that exits while it is being parked
leaves the GPU full for a moment, and a restore of the next job into a full GPU fails for good.
"""
import os
import time

import torch
from timeslice import TimeSliceOrchestratorClient

JOB, GROUP = os.environ["JOB_ID"], os.environ["GROUP_ID"]
GIB = int(os.environ.get("GIB", "72"))
BURSTS = int(os.environ.get("BURSTS", "4"))
ADDS = int(os.environ.get("ADDS", "10"))
BURST_S = float(os.environ.get("BURST_S", "8"))
IDLE_S = float(os.environ.get("IDLE_S", "4"))


def say(message: str) -> None:
    print(time.strftime("%H:%M:%S"), JOB, message, flush=True)


client = TimeSliceOrchestratorClient(os.environ["ORCHESTRATOR"], JOB, GROUP)
held = None
for burst in range(1, BURSTS + 1):
    asked = time.monotonic()
    with client.on_accelerators(timeout_sec=900.0):
        waited = time.monotonic() - asked
        if held is None:
            held = torch.zeros(GIB << 30, dtype=torch.uint8, device="cuda")
        started = time.monotonic()
        for _ in range(ADDS):
            held.add_(1)
        torch.cuda.synchronize()
        want = burst * ADDS % 256
        intact = bool((held[:: 1 << 20] == want).all())
        say(
            f"burst {burst}: waited {waited:.1f} s for the GPU, "
            f"{torch.cuda.memory_allocated() >> 30} GiB on it, "
            f"counter {int(held[0])} (expected {want}), intact {intact}"
        )
        if not intact:
            raise SystemExit("device memory did not survive being parked")
        time.sleep(max(0.0, BURST_S - (time.monotonic() - started)))
    time.sleep(IDLE_S)
with client.on_accelerators(timeout_sec=900.0):
    held = None
    torch.cuda.empty_cache()
say("done")
