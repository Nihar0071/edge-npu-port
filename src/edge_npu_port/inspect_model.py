"""Step 2: "Can this chip even run my model?"

Three questions, answered from the graph alone (no hardware needed):
  1. Ops:    which layers run on the fast MAC array, which on slower ALUs,
             which need a custom kernel or fall back to the host CPU?
  2. Memory: do the weights fit on-chip? Does the peak set of live activations?
  3. Spills: which individual layers can't fit their inputs + outputs + weights
             on-chip at once? Those must be tiled through DDR -- the first
             suspects when Step 3 looks for bottlenecks.
"""

from collections import Counter
from dataclasses import dataclass

from .graph import Graph, fuse_silu
from .hardware import Profile


@dataclass
class OpRow:
    op: str
    count: int
    level: str
    macs: int


@dataclass
class LayerRow:
    index: int
    name: str
    op: str
    out_shape: tuple
    working_set_bytes: int


@dataclass
class InspectReport:
    model: str
    profile: str
    opset: int
    num_layers: int
    params: int
    total_macs: int
    weight_bytes: int
    peak_live_bytes: int
    peak_live_at: str
    fused: bool
    unfused_peak_live_bytes: int
    unfused_spilling_layers: int
    onchip_bytes: int
    ops: list[OpRow]
    spilling_layers: int
    top_layers: list[LayerRow]
    warnings: list[str]
    assumptions: list[dict]

    @property
    def weights_fit(self) -> bool:
        return self.weight_bytes <= self.onchip_bytes

    @property
    def activations_fit(self) -> bool:
        return self.peak_live_bytes <= self.onchip_bytes


def inspect(raw: Graph, hw: Profile, top: int = 10, fuse: bool = True) -> InspectReport:
    """Report on the graph as an NPU compiler would see it (SiLU fused) unless fuse=False.

    The as-exported numbers are kept alongside, so the effect of fusion is visible.
    """
    bpe = hw.bytes_per_elem
    raw_peak, _ = raw.peak_live_elems()
    raw_spill = sum(l.working_set(bpe) > hw.onchip_bytes for l in raw.layers)
    g = fuse_silu(raw) if fuse else raw
    counts, macs = Counter(), Counter()
    for l in g.layers:
        counts[l.op] += 1
        macs[l.op] += l.macs
    ops = sorted((OpRow(op, n, hw.level(op), macs[op]) for op, n in counts.items()),
                 key=lambda r: (["matrix", "vector", "layout", "custom", "fallback"].index(r.level), -r.count))

    peak, at = g.peak_live_elems()
    by_ws = sorted(g.layers, key=lambda l: l.working_set(bpe), reverse=True)
    spilling = [l for l in g.layers if l.working_set(bpe) > hw.onchip_bytes]

    warnings = []
    for r in ops:
        if r.level == "fallback":
            warnings.append(f"{r.count}x {r.op} falls back to the host CPU: each one is a DDR round-trip")
        elif r.level == "custom":
            warnings.append(f"{r.count}x {r.op} has no built-in kernel on {hw.name}: needs a custom kernel")
    if g.weight_elems * bpe > hw.onchip_bytes:
        warnings.append("weights don't fit on-chip: every inference re-streams them from DDR")

    return InspectReport(
        model=g.path.name, profile=hw.name, opset=g.opset, num_layers=len(g.layers),
        params=g.weight_elems, total_macs=g.total_macs, weight_bytes=g.weight_elems * bpe,
        peak_live_bytes=peak * bpe,
        peak_live_at=g.layers[at].name if at >= 0 else "input",
        fused=fuse, unfused_peak_live_bytes=raw_peak * bpe, unfused_spilling_layers=raw_spill,
        onchip_bytes=hw.onchip_bytes, ops=ops, spilling_layers=len(spilling),
        top_layers=[LayerRow(l.index, l.name, l.op, l.out_shapes[0], l.working_set(bpe)) for l in by_ws[:top]],
        warnings=warnings, assumptions=hw.assumptions,
    )


def _mb(b: int) -> str:
    return f"{b / 2**20:.2f} MB"


def render(r: InspectReport) -> str:
    ok = lambda b: "fits" if b else "DOES NOT FIT"
    out = [
        f"{r.model} on {r.profile}  (opset {r.opset}, {r.num_layers} layers"
        f"{', Conv+SiLU fused as an NPU compiler would' if r.fused else ', as exported'})",
        f"  params        {r.params / 1e6:.2f} M      compute {r.total_macs / 1e9:.2f} GMAC per inference",
        f"  on-chip       {_mb(r.onchip_bytes)}",
        f"  weights       {_mb(r.weight_bytes):>9}  -> {ok(r.weights_fit)}",
        f"  peak live act {_mb(r.peak_live_bytes):>9}  -> {ok(r.activations_fit)}  (at {r.peak_live_at})",
        f"  layers whose in+out+weights exceed on-chip memory: {r.spilling_layers} / {r.num_layers}",
    ]
    if r.fused:
        out.append(f"  (without fusion: peak live {_mb(r.unfused_peak_live_bytes)}, "
                   f"{r.unfused_spilling_layers} spilling layers)")
    out += [
        "",
        f"  {'op':<14}{'count':>6}  {'runs on':<9}{'GMAC':>8}",
    ]
    out += [f"  {o.op:<14}{o.count:>6}  {o.level:<9}{o.macs / 1e9:>8.3f}" for o in r.ops]
    out += ["", f"  largest working sets:", f"  {'#':>4}  {'op':<10}{'output shape':<22}{'working set':>12}  name"]
    out += [f"  {l.index:>4}  {l.op:<10}{str(l.out_shape):<22}{_mb(l.working_set_bytes):>12}  {l.name}"
            for l in r.top_layers]
    if r.warnings:
        out += ["", "  warnings:"] + [f"    - {w}" for w in r.warnings]
    if r.assumptions:
        out += ["", "  profile assumptions (not vendor specs):"]
        out += [f"    - {a['field']} = {a['value']}: {a['note']}" for a in r.assumptions]
    return "\n".join(out)
