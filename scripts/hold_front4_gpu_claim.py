#!/usr/bin/env python3
"""Hold a tiny CUDA context while a queued model process initializes."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True, choices=range(4))
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES={args.physical_gpu}, got {visible!r}"
        )

    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGTERM) != 0:
        raise OSError("prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != args.parent_pid:
        raise RuntimeError("GPU claim parent exited before initialization")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU claim requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    claim = torch.empty(1, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize(0)

    payload = {
        "claim_pid": os.getpid(),
        "parent_pid": args.parent_pid,
        "physical_gpu": args.physical_gpu,
        "visible_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
    }
    temporary = args.ready_file.with_name(f"{args.ready_file.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.ready_file)

    while os.getppid() == args.parent_pid:
        time.sleep(1)
    del claim


if __name__ == "__main__":
    main()
