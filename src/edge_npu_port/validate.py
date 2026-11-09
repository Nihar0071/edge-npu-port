"""`estimate --validate local`: compare the estimate with measured per-layer time.

Measured = ONNX Runtime's profiler on this machine's CPU, at ORT_ENABLE_BASIC
optimization. BASIC keeps the graph almost 1:1 with the ONNX file (it does constant
folding and simple fusions, but no layout changes or activation fusion), so each
measured kernel maps back to an estimated layer by node name.

We compare per module (`/model.N/...`) rather than per node: per-node timings of
microsecond ops are noisy, and modules are what you'd discuss with a customer.
We also time the fully optimized session (ORT_ENABLE_ALL) end to end, as the
"what you'd actually ship" number.
"""

import tempfile
import time
from dataclasses import dataclass
from statistics import median

import numpy as np
import onnxruntime as ort

from .calibrate import kernel_times, session_options
from .estimate import Estimate, module_of


@dataclass
class ModuleRow:
    module: str
    estimated_s: float
    measured_s: float

    @property
    def error(self) -> float:
        return (self.estimated_s - self.measured_s) / self.measured_s


@dataclass
class Validation:
    threads: int
    estimated_s: float
    measured_kernels_s: float  # sum of per-node kernel medians at BASIC
    measured_wall_basic_s: float
    measured_wall_all_s: float
    modules: list[ModuleRow]
    unmatched_measured_s: float  # kernel time we couldn't map to an estimated layer

    @property
    def error(self) -> float:
        return (self.estimated_s - self.measured_kernels_s) / self.measured_kernels_s

    @property
    def mape(self) -> float:
        """Mean absolute % error over modules, weighted by measured time (big modules count more)."""
        tot = sum(m.measured_s for m in self.modules)
        return sum(abs(m.error) * m.measured_s for m in self.modules) / tot


def _wall(sess, feeds, runs=20) -> float:
    for _ in range(3):
        sess.run(None, feeds)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feeds)
        ts.append(time.perf_counter() - t)
    return median(ts)


def validate_local(onnx_path: str, est: Estimate, threads: int, runs: int = 20) -> Validation:
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        sess = ort.InferenceSession(onnx_path, session_options(threads, profile_dir=d),
                                    providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        kt = {k: median(v) for k, v in kernel_times(sess, {name: x}, runs).items()}
    basic = ort.InferenceSession(onnx_path, session_options(threads), providers=["CPUExecutionProvider"])
    full = ort.InferenceSession(onnx_path, session_options(threads, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                                providers=["CPUExecutionProvider"])

    est_by_mod = est.by_module()
    meas_by_mod: dict[str, float] = {}
    known = {l.name for l in est.layers}
    unmatched = 0.0
    for node, t in kt.items():
        if node in known or module_of(node) in est_by_mod:
            meas_by_mod[module_of(node)] = meas_by_mod.get(module_of(node), 0.0) + t
        else:
            unmatched += t

    mods = [ModuleRow(m, est_by_mod[m], meas_by_mod[m]) for m in est_by_mod if meas_by_mod.get(m, 0) > 0]
    mods.sort(key=lambda r: -r.measured_s)
    return Validation(threads=threads, estimated_s=est.total_s, measured_kernels_s=sum(kt.values()),
                      measured_wall_basic_s=_wall(basic, {name: x}), measured_wall_all_s=_wall(full, {name: x}),
                      modules=mods, unmatched_measured_s=unmatched)


def render(v: Validation, top: int = 12) -> str:
    ms = lambda s: f"{s * 1e3:.2f}"
    out = [
        "", f"  validation against this machine (ONNX Runtime CPU, {v.threads} threads):",
        f"    estimated            {ms(v.estimated_s):>8} ms",
        f"    measured (kernels)   {ms(v.measured_kernels_s):>8} ms   -> estimate error {v.error:+.0%}",
        f"    measured wall, BASIC {ms(v.measured_wall_basic_s):>8} ms",
        f"    measured wall, ALL   {ms(v.measured_wall_all_s):>8} ms   (fully optimized: what you'd ship)",
        f"    per-module error     {v.mape:.0%} (time-weighted mean |error|)",
        "", f"    {'module':<10}{'est ms':>9}{'meas ms':>9}{'error':>8}",
    ]
    out += [f"    {m.module:<10}{ms(m.estimated_s):>9}{ms(m.measured_s):>9}{m.error:>+8.0%}" for m in v.modules[:top]]
    if v.unmatched_measured_s:
        out.append(f"    (unmatched measured kernel time: {ms(v.unmatched_measured_s)} ms)")
    return "\n".join(out)
