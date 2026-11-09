"""MCP server: the same checks as the CLI, callable by an agent (Claude Code, etc.).

Each tool returns plain JSON-able dicts. Every non-public hardware number travels
with the result under `assumptions`, so an agent can't present a guess as a spec.

Run:  uv run edgeport-mcp      (stdio transport; see .mcp.json)
"""

from dataclasses import asdict
from pathlib import Path

from mcp.server.mcpserver import MCPServer  # MCP Python SDK 2.x (formerly FastMCP)

from . import estimate as est_mod
from . import graph, hardware, inspect_model

mcp = MCPServer("edge-npu-port", instructions="Edge-NPU feasibility checks for ONNX models. Follow the port-model skill order: twin, inspect, estimate, quant_doctor, check.")


@mcp.tool()
def list_hardware_profiles() -> dict:
    """List bundled hardware profiles (target chips) with their headline numbers and assumptions."""
    out = {}
    for name in hardware.list_profiles():
        p = hardware.load(name)
        out[name] = {"kind": p.kind, "description": p.description,
                     "peak_gmac_per_s": p.peak_macs_per_s / 1e9, "onchip_mb": p.onchip_bytes / 2**20,
                     "ddr_gb_per_s": p.ddr_bytes_per_s / 1e9, "assumptions": p.assumptions}
    return out


@mcp.tool()
def inspect_model(onnx_path: str, hw: str = "nano-gpnpu") -> dict:
    """Step 2 of a port: can this chip run the model at all?

    Reports op support levels (matrix / vector / layout / custom kernel / CPU fallback),
    whether weights and peak live activations fit on-chip, and the layers whose working
    set overflows on-chip memory. `hw` is a bundled profile name or a path to a YAML profile.
    """
    r = inspect_model.inspect(graph.load(onnx_path), hardware.load(hw))
    d = asdict(r)
    d.update(weights_fit=r.weights_fit, activations_fit=r.activations_fit)
    return d


@mcp.tool()
def estimate_latency(onnx_path: str, hw: str = "nano-gpnpu", budget_ms: float | None = None, top: int = 10) -> dict:
    """Step 3 of a port: per-layer roofline latency estimate on a target chip.

    Returns total ms, FPS, MAC utilization, time split by bottleneck (compute / memory /
    vector), the slowest layers, and whether `budget_ms` is met. The roofline is a lower
    bound on latency; on the one chip we validated (M5 CPU) it under-predicted by ~40%.
    """
    p = hardware.load(hw)
    e = est_mod.estimate(graph.load(onnx_path), p)
    slow = sorted(e.layers, key=lambda l: -l.total_s)[:top]
    return {
        "model": e.model, "target": e.profile, "latency_ms": e.total_s * 1e3, "fps": e.fps,
        "mac_utilization": e.mac_utilization, "time_by_bound": {k: v * 1e3 for k, v in e.time_by_bound().items()},
        "meets_budget": None if budget_ms is None else e.total_s * 1e3 <= budget_ms, "budget_ms": budget_ms,
        "slowest_layers": [{"name": l.name, "op": l.op, "ms": l.total_s * 1e3, "bound": l.bound,
                            "macs": l.macs, "ddr_bytes": l.ddr_bytes} for l in slow],
        "caveat": "roofline lower bound; validated on M5 CPU at -42% error; per-shape efficiency not modeled",
        "assumptions": p.assumptions,
    }


@mcp.tool()
def quant_doctor(onnx_path: str, weights: str = "models/yolo11n.pt", hw: str = "nano-gpnpu",
                 eval_images: int = 100, calib_images: int = 32, out_dir: str = "runs/quant-mcp") -> dict:
    """Step 4 of a port: measure int8 accuracy cost and find fixes. SLOW (minutes).

    Scores PyTorch float -> ONNX float -> int8 on COCO val with one evaluator, ranks tensors
    by SQNR, flags Concats whose inputs are crushed by a shared int8 scale (range mixing),
    then keeps blockers + the decode tail (+ low-SQNR layers) in float and prices each fix in
    ms on `hw`. Currently YOLO-style detectors with COCO data in data/. Use small
    eval_images for a first pass; confirm the final choice on more images.
    """
    from . import quant_doctor as qd

    report = qd.run(Path(onnx_path), Path(weights), hw, eval_images, calib_images, [3], Path(out_dir))
    report["chart"] = str(qd.plot(report, Path(out_dir) / "quant_pareto.png"))
    return report


@mcp.tool()
def check(onnx_path: str, hw: str = "nano-gpnpu", max_ms: float | None = None,
          max_peak_mb: float | None = None, allow_fallback: bool = False) -> dict:
    """Gate a model against a target: pass/fail with reasons (same as `edgeport check`, used in CI)."""
    from .checks import run_checks
    return run_checks(onnx_path, hw, max_ms, max_peak_mb, allow_fallback)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
