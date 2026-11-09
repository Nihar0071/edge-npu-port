"""The C++ kernel must match the numpy integer reference bit-for-bit, in every mode,
on random data -- including awkward shapes where the last tile is partial."""

import json
import subprocess

import numpy as np
import pytest

from edge_npu_port import kernel_harness as kh

pytestmark = pytest.mark.skipif(not kh.SIM.exists(), reason="build sim/ first")


def _write_random(d, pixels, cin, cout, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (pixels, cin), dtype=np.int8)
    w = rng.integers(-127, 128, (cout, cin), dtype=np.int8)
    bias = rng.integers(-5000, 5000, cout).astype(np.int32)
    m0, shift = kh.quantize_multiplier(rng.uniform(1e-4, 5e-3, cout))
    lut = rng.integers(-128, 128, 256, dtype=np.int8)
    xz, pz = int(rng.integers(-128, 128)), int(rng.integers(-20, 20))
    for name, a in [("x", x), ("w", w), ("bias", bias), ("m0", m0), ("shift", shift), ("lut", lut)]:
        a.tofile(d / f"{name}.bin")
    (d / "meta.txt").write_text(f"pixels {pixels}\ncin {cin}\ncout {cout}\nx_zero {xz}\npre_zero {pz}\n")
    acc = (x.astype(np.int64) - xz) @ w.astype(np.int64).T + bias
    pre = kh.requantize_np(acc, m0[None], shift[None], pz)
    return lut[(pre + 128).astype(np.int64)]


@pytest.mark.parametrize("pixels,cin,cout", [(6400, 64, 32), (1000, 48, 16), (77, 16, 8)])
@pytest.mark.parametrize("mode", ["v0", "v1", "v2"])
def test_bit_exact_vs_numpy(tmp_path, pixels, cin, cout, mode):
    y_np = _write_random(tmp_path, pixels, cin, cout, seed=pixels + cin)
    p = subprocess.run([str(kh.SIM), str(tmp_path), "--mode", mode], capture_output=True, text=True)
    r = json.loads(p.stdout)
    assert r["bit_exact"], "tiled kernel differs from C++ scalar reference"
    y_sim = np.fromfile(tmp_path / "y_sim.bin", dtype=np.int8).reshape(pixels, cout)
    assert (y_sim == y_np).all(), "C++ differs from the numpy integer reference"


def test_double_buffering_helps(tmp_path):
    _write_random(tmp_path, 6400, 256, 64, seed=1)
    cyc = {m: json.loads(subprocess.run([str(kh.SIM), str(tmp_path), "--mode", m], capture_output=True,
                                        text=True).stdout)["cycles"] for m in ("v0", "v1", "v2")}
    assert cyc["v0"] > cyc["v1"] > cyc["v2"]
