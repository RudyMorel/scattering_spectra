#!/usr/bin/env python
""" Fast, self-limiting smoke test for the CPU / CUDA / MPS device support.

Design goals (so an agent driving this can NEVER get wedged on it):
  * Every individual check is wrapped in a hard wall-clock timeout
    (signal.setitimer). A hang becomes a fast, explicit FAIL, never a freeze.
  * All tensors are tiny (T=256, J=3) so nothing is compute-bound.
  * `generate` is run with a huge `tol_optim` so the convergence criterion
    fires on the FIRST iteration and the function returns immediately. This
    deliberately sidesteps the genuine optimization (which is slow AND, on
    non-convergence, retries forever -- see note at the bottom of this file).
    We are testing the device/dtype plumbing, not optimization quality.

Run:  .venv/bin/python tests/smoke_test.py
Exits non-zero if any check fails or times out.
"""
import os
import sys
import time
import signal
import warnings
import traceback

# make the package importable no matter the cwd / how this script is launched
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

# The MPS fft path emits this resize warning once per call -> floods the log.
warnings.filterwarnings(
    "ignore", message=".*output with one or more elements was resized.*"
)

PER_CHECK_TIMEOUT_S = 20  # belt-and-suspenders: no single check may exceed this
T = 2 ** 8                # 256 samples -- tiny on purpose
J = 3                     # few scales -- tiny on purpose

results = []  # (name, ok, secs, detail)


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout(f"exceeded {PER_CHECK_TIMEOUT_S}s")


signal.signal(signal.SIGALRM, _on_alarm)


def check(name, fn):
    """ Run fn() under a wall-clock timeout; record PASS/FAIL. fn returns an
    optional detail string, or raises to fail. """
    signal.setitimer(signal.ITIMER_REAL, PER_CHECK_TIMEOUT_S)
    t = time.time()
    try:
        detail = fn() or ""
        ok = True
    except _Timeout as e:
        ok, detail = False, f"TIMEOUT ({e})"
    except Exception:
        ok, detail = False, "EXC:\n" + traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    secs = round(time.time() - t, 3)
    results.append((name, ok, secs, detail))
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name:<34} {secs:>6.2f}s"
    if detail and not detail.startswith("EXC"):
        line += f"  {detail}"
    print(line)
    if detail.startswith("EXC"):
        print(detail)


# ---------------------------------------------------------------- environment
MPS = torch.backends.mps.is_available()
CUDA = torch.cuda.is_available()
print(f"torch={torch.__version__}  mps={MPS}  cuda={CUDA}  "
      f"(per-check timeout={PER_CHECK_TIMEOUT_S}s, T={T}, J={J})")

from scatspectra.utils import resolve_device, device_supports_float64, to_numpy
from scatspectra import analyze, generate
from scatspectra.layers.layers_basics import PhaseOperator

np.random.seed(0)
x64 = np.random.randn(1, 1, T)            # float64
x32 = x64.astype(np.float32)              # float32


# ------------------------------------------------------------ device helpers
def _devices():
    assert resolve_device().type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("mps").type == ("mps" if MPS else "mps")  # str passes through
    assert device_supports_float64("cpu") is True
    assert device_supports_float64("mps") is False
    exp = "cuda" if CUDA else ("mps" if MPS else "cpu")
    assert resolve_device("auto").type == exp
    return f"auto->{exp}"


check("resolve_device/float64_probe", _devices)


def _cuda_raises():
    if CUDA:
        return "skipped (cuda present)"
    for bad in (lambda: resolve_device("cuda"), lambda: resolve_device(None, True)):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    return "cuda request raises ValueError"


check("cuda_unavailable_raises", _cuda_raises)


# ------------------------------------------------------------------- analyze
def _analyze_cpu():
    Rx = analyze(x64, J=J, device="cpu")
    assert Rx.y.device.type == "cpu"
    assert Rx.y.dtype == torch.complex128
    return f"dtype={Rx.y.dtype}"


check("analyze cpu (f64)", _analyze_cpu)


def _analyze_default_matches_cpu():
    a, b = analyze(x64, J=J), analyze(x64, J=J, cuda=False)
    assert torch.allclose(a.y, b.y)
    return "default == cuda=False"


check("analyze default==cuda=False", _analyze_default_matches_cpu)


def _analyze_mps_f32():
    if not MPS:
        return "skipped (no mps)"
    Rc = analyze(x32, J=J, device="cpu")
    Rm = analyze(x32, J=J, device="mps")
    assert Rm.y.device.type == "cpu", "result must come back on cpu"
    err = float(np.max(np.abs(
        Rc.y.numpy().astype(np.complex128) - Rm.y.numpy().astype(np.complex128))))
    assert err < 1e-2, f"mps vs cpu err too large: {err:.2e}"
    return f"mps==cpu (maxabs={err:.1e})"


check("analyze mps (f32, real GPU)", _analyze_mps_f32)


def _analyze_mps_f64_fallback():
    if not MPS:
        return "skipped (no mps)"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Rx = analyze(x64, J=J, device="mps")
    assert Rx.y.dtype == torch.complex128
    assert any("MPS" in str(wi.message) for wi in w), "expected MPS fallback warning"
    return "f64->cpu fallback + warning"


check("analyze mps f64->cpu fallback", _analyze_mps_f64_fallback)


# ----------------------------------------------------- PhaseOperator / to_numpy
def _phaseop():
    op = PhaseOperator(4)
    assert "phases" in dict(op.named_buffers()), "phases must be a buffer"
    assert isinstance(to_numpy(torch.randn(3)), np.ndarray)
    if MPS:
        opm = op.to("mps")
        assert opm.phases.device.type == "mps"
        out = opm(torch.randn(2, 4, 1, device="mps"))
        assert out.device.type == "mps"
        assert isinstance(to_numpy(torch.randn(3, device="mps")), np.ndarray)
        return "buffer moves to mps + forward ok"
    return "cpu only"


check("PhaseOperator.to(mps) / to_numpy", _phaseop)


# ------------------------------------------------------------------ generate
# Huge tol_optim => convergence criterion fires on iteration 1 => returns at
# once. This exercises Solver.format / model.to(device) / float64 round-trip
# WITHOUT the slow (and potentially non-terminating) real optimization.
FAST_GEN = dict(R=1, J=J, max_iterations=5, tol_optim=1e12, verbose=False)


def _generate_cpu():
    pd = generate(x=x64, device="cpu", **FAST_GEN)
    assert pd.dlnx.shape[-1] == T
    return f"shape={pd.dlnx.shape}"


check("generate cpu (f64)", _generate_cpu)


def _generate_mps_f64_fallback():
    if not MPS:
        return "skipped (no mps)"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pd = generate(x=x64, device="mps", **FAST_GEN)
    assert pd.dlnx.shape[-1] == T
    assert any("MPS" in str(wi.message) for wi in w), "expected MPS fallback warning"
    return "f64->cpu fallback + warning"


check("generate mps f64->cpu fallback", _generate_mps_f64_fallback)


def _generate_mps_f32():
    if not MPS:
        return "skipped (no mps)"
    pd = generate(x=x32, device="mps", **FAST_GEN)
    assert pd.dlnx.shape[-1] == T
    return f"shape={pd.dlnx.shape}"


check("generate mps (f32, real GPU)", _generate_mps_f32)


def _generate_retry_cap():
    # A non-converging config (impossible tol, 1 iter) must FAIL FAST with a
    # RuntimeError after max_attempts_per_batch retries -- NOT loop forever.
    # This is the guard against the infinite-retry hang. The per-check SIGALRM
    # timeout above would also catch a regression, but assert the error too.
    try:
        generate(x=x64, device="cpu", R=1, J=J, max_iterations=1,
                 tol_optim=1e-30, max_attempts_per_batch=3, verbose=False)
    except RuntimeError:
        return "non-convergence raises (no infinite loop)"
    raise AssertionError("expected RuntimeError from retry cap")


check("generate retry cap (no hang)", _generate_retry_cap)


# --------------------------------------------------------------------- report
n_fail = sum(1 for _, ok, _, _ in results if not ok)
total_s = round(sum(s for _, _, s, _ in results), 2)
print("-" * 60)
print(f"{len(results)-n_fail}/{len(results)} passed in {total_s}s total")
with open(os.path.join(_ROOT, "tests", "results.txt"), "w") as f:
    for name, ok, secs, detail in results:
        f.write(f"{'PASS' if ok else 'FAIL'}\t{secs:>6.2f}s\t{name}\t"
                f"{detail.splitlines()[0] if detail else ''}\n")
    f.write(f"OUTCOME={'ALL_PASS' if n_fail == 0 else f'{n_fail}_FAILED'}\n")
print("wrote tests/results.txt")
sys.exit(1 if n_fail else 0)


# NOTE on the (now-fixed) infinite-retry trap:
# generate()'s `while ibatch < nbatches_to_gen:` loop used to `continue` WITHOUT
# incrementing ibatch when `res['nit'] == max_iterations`, so a configuration
# that never converged (common on MPS-float32, where precision often prevents
# reaching tol_optim) would loop forever. generate() now caps retries via
# `max_attempts_per_batch` and raises a RuntimeError instead -- see the
# "generate retry cap (no hang)" check above. This smoke test still uses
# tol_optim=1e12 for the happy-path generate checks so they return instantly.
