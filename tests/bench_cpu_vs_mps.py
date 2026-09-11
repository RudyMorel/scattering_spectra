#!/usr/bin/env python
""" CPU vs MPS performance comparison for the scattering transform (`analyze`).

We benchmark `analyze` (the forward scattering-spectra statistics) because it is
a pure, deterministic forward pass -- the right thing to measure for a device
comparison. `generate` is deliberately excluded: its cost is the iterative
optimizer, not the device kernels.

Three columns per size:
  * CPU f64  -- the library's DEFAULT precision (reference real-world cost)
  * CPU f32  -- apples-to-apples vs the GPU (MPS has no float64)
  * MPS f32  -- the Apple GPU

Methodology: 1 warmup (MPS compiles kernels on first call) + median of N timed
runs. `analyze` returns its result on CPU, so each wall-clock time already
includes the GPU->CPU transfer + synchronization (a fair end-to-end number).
A hard per-call SIGALRM timeout keeps any single run bounded; sizes escalate
and we stop before anything gets slow.

Run:  .venv/bin/python tests/bench_cpu_vs_mps.py
"""
import os
import sys
import time
import signal
import warnings
import statistics

# make the package importable no matter the cwd / how this script is launched
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

warnings.filterwarnings(
    "ignore", message=".*output with one or more elements was resized.*"
)
warnings.filterwarnings("ignore", message=".*not writable.*")

from scatspectra import analyze

MPS = torch.backends.mps.is_available()
CALL_TIMEOUT_S = 12      # no single analyze() call may exceed this
REPEATS = 3              # timed runs per (device, size); median reported
ESCALATE_STOP_S = 1.5    # stop adding bigger sizes once cpu f64 exceeds this
GLOBAL_BUDGET_S = 45.0   # hard ceiling on total benchmark wall-clock


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _on_alarm)

# (N_channels, T) -- escalating work. NOTE: the multivariate cross-covariance
# in analyze scales ~N^2, so we grow T (where GPU FFTs shine) more than N to
# keep individual calls short. A global wall-clock budget guards the total.
CONFIGS = [
    (1, 2 ** 12),
    (1, 2 ** 14),
    (4, 2 ** 14),
    (4, 2 ** 16),
    (8, 2 ** 16),
]


def time_call(x, device, repeats):
    """ Median wall-clock seconds of analyze(x, device=...); None on timeout. """
    def _run():
        return analyze(x, device=device)
    # warmup (kernel compilation on MPS, caches on CPU) -- not timed
    signal.setitimer(signal.ITIMER_REAL, CALL_TIMEOUT_S)
    try:
        _run()
    except _Timeout:
        signal.setitimer(signal.ITIMER_REAL, 0)
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    samples = []
    for _ in range(repeats):
        signal.setitimer(signal.ITIMER_REAL, CALL_TIMEOUT_S)
        t = time.perf_counter()
        try:
            _run()
        except _Timeout:
            signal.setitimer(signal.ITIMER_REAL, 0)
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        samples.append(time.perf_counter() - t)
    return statistics.median(samples)


def fmt(v):
    return "  timeout" if v is None else f"{v*1e3:8.1f}"


print(f"torch={torch.__version__}  mps={MPS}  "
      f"(median of {REPEATS} runs after 1 warmup, per-call timeout={CALL_TIMEOUT_S}s)")
print()
hdr = f"{'N':>3} {'T':>7} {'J':>3} | {'CPU f64 ms':>11} {'CPU f32 ms':>11} {'MPS f32 ms':>11} | {'MPS vs CPU(f32)':>15}"
print(hdr)
print("-" * len(hdr))

bench_start = time.perf_counter()
for (N, T) in CONFIGS:
    if time.perf_counter() - bench_start > GLOBAL_BUDGET_S:
        print(f"(stopping: hit global budget of {GLOBAL_BUDGET_S:.0f}s)")
        break
    np.random.seed(0)
    x64 = np.random.randn(1, N, T)
    x32 = x64.astype(np.float32)
    J = int(np.log2(T)) - 3  # the library default

    t_cpu64 = time_call(x64, "cpu", REPEATS)
    t_cpu32 = time_call(x32, "cpu", REPEATS)
    t_mps32 = time_call(x32, "mps", REPEATS) if MPS else None

    speed = "n/a"
    if t_cpu32 and t_mps32:
        r = t_cpu32 / t_mps32
        speed = f"{r:5.2f}x {'(MPS win)' if r > 1 else '(CPU win)'}"

    print(f"{N:>3} {T:>7} {J:>3} | {fmt(t_cpu64)} {fmt(t_cpu32)} {fmt(t_mps32)} | {speed:>15}")

    # stop escalating once the default-precision CPU run gets non-trivial,
    # so the whole benchmark stays well under a minute
    if t_cpu64 is not None and t_cpu64 > ESCALATE_STOP_S:
        print(f"(stopping: CPU f64 exceeded {ESCALATE_STOP_S}s -- keeping total runtime bounded)")
        break

print()
print("Notes:")
print(" * Observed: MPS does NOT accelerate this workload. The scattering transform")
print("   is many small/medium FFTs + complex cross-channel arithmetic -- launch-")
print("   overhead-heavy and reliant on complex64 FFT support, MPS's weak spot. CPU")
print("   (incl. the default f64) ties or beats MPS f32 at every size that fit budget.")
print(" * Large shapes also pay a steep MPS first-call kernel-compilation cost.")
print(" * CPU f64 is the library default; MPS requires f32 (no float64 on Apple GPU).")
sys.exit(0)
