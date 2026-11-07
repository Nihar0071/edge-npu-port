"""inspect / graph sanity checks.

Two kinds:
  * synthetic graphs where we can compute the right answer by hand
  * YOLO11n cross-checked against an independent counter (Ultralytics/thop)
"""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from edge_npu_port import graph, hardware, inspect_model

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / "models" / "yolo11n.onnx"


def _conv_silu_model(tmp_path: Path) -> Path:
    """x[1,8,16,16] -> Conv 3x3 (8->16) -> SiLU (Sigmoid+Mul) -> y."""
    w = numpy_helper.from_array(np.zeros((16, 8, 3, 3), np.float32), "w")
    b = numpy_helper.from_array(np.zeros(16, np.float32), "b")
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Sigmoid", ["c"], ["s"]),
        helper.make_node("Mul", ["c", "s"], ["y"]),
    ]
    g = helper.make_graph(nodes, "t",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 16, 16])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 16, 16])],
                          initializer=[w, b])
    p = tmp_path / "conv_silu.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), p)
    return p


def test_conv_macs_by_hand(tmp_path):
    g = graph.load(_conv_silu_model(tmp_path))
    # every output element (16*16*16) needs C_in*kH*kW = 8*3*3 MACs
    assert g.layers[0].macs == 16 * 16 * 16 * 8 * 3 * 3
    assert g.weight_elems == 16 * 8 * 3 * 3 + 16


def test_fusion_and_peak_by_hand(tmp_path):
    g = graph.load(_conv_silu_model(tmp_path))
    x, y = 8 * 16 * 16, 16 * 16 * 16
    # unfused: at the Mul, c, s and y are all alive (x died after the Conv... but it's alive *during* it)
    assert g.peak_live_elems()[0] == 3 * y
    f = graph.fuse_silu(g)
    assert [l.op for l in f.layers] == ["Conv"] and f.layers[0].attrs["fused"] == "SiLU"
    assert f.peak_live_elems()[0] == x + y  # input and output of the fused conv


@pytest.mark.skipif(not YOLO.exists(), reason="run `edgeport twin` first")
def test_yolo_matches_independent_counter():
    g = graph.load(YOLO)
    # Ultralytics reports 6.673 GFLOPs at 640 (= 3.337 GMAC; thop also counts BN/activations)
    assert abs(g.total_macs / 3.337e9 - 1) < 0.03
    r = inspect_model.inspect(g, hardware.load("nano-gpnpu"))
    assert r.unfused_peak_live_bytes > r.peak_live_bytes  # fusion must help
    assert sum(o.count for o in r.ops if o.op == "Sigmoid") == 1  # 77 SiLUs fused; the 1 left is the Detect head's class-score sigmoid
