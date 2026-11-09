"""`edgeport` command line. Each subcommand is one step of a port."""

import argparse
import json
import sys
from pathlib import Path


def _cmd_twin(args: argparse.Namespace) -> int:
    from . import twin

    onnx_path = Path(args.onnx) if args.onnx else twin.export_onnx(args.weights, Path(args.out_dir), args.imgsz)
    images = [Path(p) for p in args.images]
    if not images:
        import ultralytics
        images = sorted((Path(ultralytics.__file__).parent / "assets").glob("*.jpg"))

    reports = twin.twin_check(args.weights, onnx_path, images, args.imgsz)
    ok = all(twin.passes(r) for r in reports)
    if args.json:
        print(json.dumps({"onnx": str(onnx_path), "pass": ok, "images": twin.as_dict(reports)}, indent=2))
    else:
        print(f"ONNX: {onnx_path}")
        print(f"{'image':<14}{'box max|d|':>12}{'box rel':>10}{'score max|d|':>14}{'score rel':>11}{'dets pt/onnx/match':>21}")
        for r in reports:
            print(f"{r.image:<14}{r.boxes.max_abs:>12.2e}{r.boxes.max_rel:>10.1e}"
                  f"{r.scores.max_abs:>14.2e}{r.scores.max_rel:>11.1e}"
                  f"{f'{r.detections_pt}/{r.detections_onnx}/{r.detections_matched}':>21}")
        print("PASS: ONNX is a faithful twin" if ok else "FAIL: outputs differ beyond float rounding")
    return 0 if ok else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from . import graph, hardware, inspect_model

    report = inspect_model.inspect(graph.load(args.onnx), hardware.load(args.hw), args.top, fuse=not args.no_fuse)
    if args.json:
        d = asdict(report)
        d.update(weights_fit=report.weights_fit, activations_fit=report.activations_fit)
        print(json.dumps(d, indent=2, default=str))
    else:
        print(inspect_model.render(report))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from . import calibrate

    prof = calibrate.calibrate(args.threads, args.onchip_mb * 2**20, args.name)
    path = calibrate.write(prof, Path(args.out or f"profiles/{args.name}.yaml"))
    m = prof["measured"]
    print(f"wrote {path}")
    print(f"  peak MAC rate   {prof['compute']['macs_per_cycle']} GMAC/s  (2048^3 MatMul in {m['matmul_2048_s'] * 1e3:.1f} ms)")
    print(f"  DDR bandwidth   {prof['memory']['ddr_bytes_per_s'] / 1e9:.0f} GB/s")
    print(f"  on-chip bw      {prof['memory']['onchip_bytes_per_s'] / 1e9:.0f} GB/s")
    print(f"  vector rate     {prof['compute']['vector_ops_per_cycle']} Gop/s")
    print(f"  layer overhead  {prof['compute']['layer_overhead_s'] * 1e6:.1f} us")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    from . import estimate, graph, hardware

    hw = hardware.load(args.hw)
    est = estimate.estimate(graph.load(args.onnx), hw)
    print(estimate.render(est, args.top, args.budget_ms))
    if args.validate == "local":
        from . import validate
        threads = int(hw.description.split(" threads")[0].split()[-1]) if hw.kind == "cpu" else args.threads
        print(validate.render(validate.validate_local(args.onnx, est, threads)))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="edgeport", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("twin", help="Step 1: export to ONNX and prove it matches PyTorch")
    t.add_argument("--weights", default="models/yolo11n.pt")
    t.add_argument("--onnx", help="skip export and check this ONNX file")
    t.add_argument("--out-dir", default="models")
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--json", action="store_true")
    t.add_argument("images", nargs="*", help="images to compare on (default: Ultralytics sample images)")
    t.set_defaults(fn=_cmd_twin)

    i = sub.add_parser("inspect", help="Step 2: op support and memory fit on a target chip")
    i.add_argument("onnx")
    i.add_argument("--hw", default="nano-gpnpu", help="bundled profile name or path to a YAML profile")
    i.add_argument("--top", type=int, default=10, help="how many largest-working-set layers to list")
    i.add_argument("--no-fuse", action="store_true", help="report the graph as exported, without Conv+SiLU fusion")
    i.add_argument("--json", action="store_true")
    i.set_defaults(fn=_cmd_inspect)

    c = sub.add_parser("calibrate", help="measure this machine's CPU roofline into a hardware profile")
    c.add_argument("--name", default="local-cpu")
    c.add_argument("--threads", type=int, default=6, help="ORT intra-op threads (default: 6 = M5 Pro P-cores)")
    c.add_argument("--onchip-mb", type=int, default=16, help="last-level cache per cluster, MB")
    c.add_argument("--out", help="output YAML (default: profiles/<name>.yaml)")
    c.set_defaults(fn=_cmd_calibrate)

    e = sub.add_parser("estimate", help="Step 3: per-layer roofline latency on a target chip")
    e.add_argument("onnx")
    e.add_argument("--hw", default="nano-gpnpu")
    e.add_argument("--budget-ms", type=float, help="latency budget to check against, e.g. 16.7 for 60 FPS")
    e.add_argument("--top", type=int, default=10)
    e.add_argument("--validate", choices=["local"], help="compare with measured per-layer time on this machine")
    e.add_argument("--threads", type=int, default=6)
    e.set_defaults(fn=_cmd_estimate)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
