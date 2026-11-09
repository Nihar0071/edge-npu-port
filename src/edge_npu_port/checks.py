"""`edgeport check`: a pass/fail gate for CI.

Nothing counts as ported until the gate says so. Accuracy needs
datasets and minutes, so CI gates on what the graph alone can prove: latency estimate,
memory, and ops that would fall back to the host CPU.
"""

from . import estimate, graph, hardware, inspect_model


def run_checks(onnx_path: str, hw_name: str, max_ms: float | None = None,
               max_peak_mb: float | None = None, allow_fallback: bool = False) -> dict:
    hw = hardware.load(hw_name)
    g = graph.load(onnx_path)
    ins = inspect_model.inspect(g, hw)
    est = estimate.estimate(g, hw)
    checks = []

    if max_ms is not None:
        ms = est.total_s * 1e3
        checks.append({"name": "latency", "pass": ms <= max_ms, "value": round(ms, 3), "limit": max_ms,
                       "detail": f"estimated {ms:.2f} ms vs budget {max_ms} ms (roofline lower bound)"})
    if max_peak_mb is not None:
        mb = ins.peak_live_bytes / 2**20
        checks.append({"name": "peak_activations", "pass": mb <= max_peak_mb, "value": round(mb, 3),
                       "limit": max_peak_mb, "detail": f"peak live activations {mb:.2f} MB at {ins.peak_live_at}"})
    fallback = [o for o in ins.ops if o.level == "fallback"]
    custom = [o for o in ins.ops if o.level == "custom"]
    checks.append({"name": "cpu_fallback", "pass": allow_fallback or not fallback,
                   "value": sum(o.count for o in fallback), "limit": None if allow_fallback else 0,
                   "detail": ", ".join(f"{o.count}x {o.op}" for o in fallback) or "none"})
    checks.append({"name": "custom_kernels_needed", "pass": True, "value": sum(o.count for o in custom),
                   "detail": ", ".join(f"{o.count}x {o.op}" for o in custom) or "none (informational)"})

    return {"model": g.path.name, "target": hw.name, "pass": all(c["pass"] for c in checks),
            "checks": checks, "assumptions": hw.assumptions}
