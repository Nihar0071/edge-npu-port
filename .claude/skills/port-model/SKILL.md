---
name: port-model
description: Port an ONNX model to an edge NPU target with the edge-npu-port tools. Use when asked whether a model will run on a chip, how fast, what int8 will cost in accuracy, or to port/quantize a new model for an edge target.
---

# Port a model to an edge NPU

A port is done when every **gate** below is green, in this order. Each step measures against the previous one, so an error is always attributable to the step that introduced it. The order is standard porting practice: prove the twin, choose precision by measured cost, then kernels each checked against a CPU reference.

Tools: the `edge-npu-port` MCP server (`list_hardware_profiles`, `inspect_model`, `estimate_latency`, `quant_doctor`, `check`), or the same commands via `uv run edgeport <cmd> --help`.

## 0. Pin the spec

Write down the target profile (`list_hardware_profiles`, or a user-supplied YAML), the latency budget in ms, and the allowed accuracy drop in metric points.
**Gate:** all three are numbers the user agreed to.

## 1. Twin

`uv run edgeport twin --weights <model.pt>` exports with static shapes and opset 17, then compares PyTorch and ONNX per output group.
**Gate:** `PASS` (relative error < 1e-4 per group, every detection matched). A twin that fails stops the port: every later number would be unattributable.

## 2. Inspect

`inspect_model(onnx, hw)`.
**Gate:** every op is `matrix`, `vector` or `layout` on the target, or each `custom`/`fallback` op is listed in the report to the user with its count. Weight and activation overflow are recorded, not yet judged.

## 3. Estimate

`estimate_latency(onnx, hw, budget_ms)`.
**Gate:** the estimate is under budget *with margin*. The roofline is a lower bound (validated −42% on an M5 CPU), so treat < 60% of budget as likely, 60–100% as "pending measurement", > 100% as a miss. Report the slowest layers and their bound (compute / memory / vector).

## 4. Precision by measured cost

`quant_doctor(onnx, weights, hw, eval_images=100)` for a first pass, then rerun the chosen configuration with ≥ 500 images.
Read the result in this order:
1. Stage-by-stage accuracy: ONNX float must equal PyTorch float (else go back to step 1).
2. `range_mixing` flags: these are blockers; they keep the int8 model near zero whatever else you do.
3. The fix ladder: pick the cheapest row within the accuracy budget, judged by the task metric, never by SQNR alone.
**Gate:** chosen configuration within the accuracy budget on ≥ 500 images, with its added ms from the ladder.

## 5. Kernels (only for layers the estimate flags and the compiler can't serve)

Each custom kernel has a scalar CPU reference and is **bit-exact** against it on real layer data and random shapes, before any cycle count counts. Compare against the framework's int8 output too, and explain every LSB of difference (see `edgeport kernel`).
**Gate:** bit-exact vs reference in every variant; speedup reported against the MAC roofline.

## 6. Gate in CI

`uv run edgeport check <onnx> --hw <profile> --max-ms <budget>` (wired in `.github/workflows/edge-check.yml`).
**Gate:** `PASS`.

## Report

One table per step with its gate result, then: total estimated latency including the step 4 fix cost, accuracy drop, every assumption printed by the profile, and what remains unmeasured. Present assumed hardware numbers as assumptions.
