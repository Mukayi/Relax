"""Hold ~1.2 GiB on one GPU without computing, so a GPU queue that treats < 1 GiB as free
leaves the card alone during the gaps between experiment runs.

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python gpu_hold.py [MiB]
"""

import signal
import sys
import time

import torch

mib = int(sys.argv[1]) if len(sys.argv) > 1 else 768
held = torch.empty(mib * 2**20, dtype=torch.uint8, device="cuda")
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(3600)
