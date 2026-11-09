"""Hardware profiles: the numbers a feasibility check needs about a chip.

A profile is a YAML file. Bundled ones live in `profiles/`; users can pass a path
to their own. Every number that is *not* from a public source must be listed under
`assumptions` -- the report prints them, so nobody mistakes a guess for a spec.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).parent / "profiles"

# How well a chip handles an op:
#   matrix   -> runs on the MAC array (the fast path)
#   vector   -> runs on per-PE ALUs / vector unit (much lower throughput than MACs)
#   layout   -> pure data movement (cost = bytes moved, no math)
#   custom   -> chip can run it, but someone has to write a kernel (programmable NPUs)
#   fallback -> chip can't; it runs on the host CPU (costs a round-trip through DDR)
LEVELS = ("matrix", "vector", "layout", "custom", "fallback")


@dataclass
class Profile:
    name: str
    kind: str
    description: str
    macs_per_cycle: int
    vector_ops_per_cycle: int
    clock_hz: float
    onchip_bytes: int  # largest shared on-chip memory (L2 / SRAM)
    local_bytes_total: int  # sum of per-PE local memories (0 if none)
    ddr_bytes_per_s: float
    bytes_per_elem: int
    op_levels: dict[str, str]
    default_level: str
    assumptions: list[dict]
    sources: list[str]
    onchip_bytes_per_s: float = 64e9  # bandwidth of on-chip memory (L2 / cache)
    layer_overhead_s: float = 0.0  # fixed cost per layer: launch, sync, DMA setup
    fuse_silu: bool = True  # does this target's compiler fuse Conv+SiLU?
    vector_op_cost: dict | None = None  # per-op overrides of estimate.DEFAULT_VECTOR_COST
    conv_macs_per_cycle: int | None = None  # achievable conv rate, if different from the GEMM peak

    def __post_init__(self):
        self.vector_op_cost = self.vector_op_cost or {}

    def macs_per_s(self, op: str) -> float:
        if op in ("Conv", "ConvTranspose") and self.conv_macs_per_cycle:
            return self.conv_macs_per_cycle * self.clock_hz
        return self.peak_macs_per_s

    def level(self, op: str) -> str:
        return self.op_levels.get(op, self.default_level)

    @property
    def peak_macs_per_s(self) -> float:
        return self.macs_per_cycle * self.clock_hz

    @property
    def ridge_point(self) -> float:
        """MACs per DDR byte needed to keep the MAC array busy (roofline 'ridge')."""
        return self.peak_macs_per_s / self.ddr_bytes_per_s


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def load(name_or_path: str) -> Profile:
    p = Path(name_or_path)
    if not p.exists():
        p = PROFILE_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no profile {name_or_path!r}; bundled: {', '.join(list_profiles())}")
    d = yaml.safe_load(p.read_text())

    op_levels = {}
    for level, ops in d["op_support"].items():
        if level == "default":
            continue
        if level not in LEVELS:
            raise ValueError(f"{p.name}: unknown support level {level!r}")
        op_levels.update({op: level for op in ops})

    c, m = d["compute"], d["memory"]
    return Profile(
        name=d["name"], kind=d["kind"], description=d["description"].strip(),
        macs_per_cycle=int(c["macs_per_cycle"]), vector_ops_per_cycle=int(c["vector_ops_per_cycle"]),
        clock_hz=float(c["clock_hz"]), onchip_bytes=int(m["onchip_bytes"]),
        local_bytes_total=int(m.get("local_bytes_per_pe", 0)) * int(m.get("num_pes", 0)),
        ddr_bytes_per_s=float(m["ddr_bytes_per_s"]), bytes_per_elem=int(d["bytes_per_elem"]),
        op_levels=op_levels, default_level=d["op_support"]["default"],
        assumptions=d.get("assumptions", []), sources=d.get("sources", []),
        onchip_bytes_per_s=float(m.get("onchip_bytes_per_s", 64e9)),
        layer_overhead_s=float(c.get("layer_overhead_s", 0.0)),
        fuse_silu=bool(d.get("compiler", {}).get("fuse_silu", True)),
        vector_op_cost=c.get("vector_op_cost", {}),
        conv_macs_per_cycle=c.get("conv_macs_per_cycle"),
    )
