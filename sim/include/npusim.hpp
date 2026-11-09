// npusim: a tile-level model of a small GPNPU-style core.
//
// NOT an instruction-set simulator. It models the three things that decide a
// kernel's speed on this kind of chip, and nothing else:
//   1. the MAC array   : 64 PEs x 8 int8 MACs, all PEs run in lockstep
//   2. per-PE ALUs     : 1 op/cycle per PE (requantize, lookup tables)
//   3. one DMA engine  : moves tiles DDR <-> PE local memory, one transfer at a time
// Time advances per *tile* (a chunk of work that fits in local memory), not per
// instruction. Every number below is printed in the report; the starred ones are
// assumptions, not published specs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npusim {

struct Config {
  int num_pes = 64;           // 8 x 8 grid
  int macs_per_pe = 8;        // int8 MACs per PE per cycle
  int alu_ops_per_pe = 1;     // * ALU ops per PE per cycle
  int local_bytes = 4096;     // * local memory per PE
  int ddr_bytes_per_cycle = 16;    // 16 GB/s at 1 GHz
  int ddr_latency_cycles = 500;    // * fixed cost to start a DDR transfer
  int l2_bytes_per_cycle = 64;     // * L2 -> PE broadcast bandwidth
  int l2_latency_cycles = 50;      // * fixed cost to start an L2 broadcast
  int epilogue_ops_per_elem = 3;   // requant multiply, round/shift/clamp, SiLU lookup
  double clock_hz = 1e9;
};

// One int8 1x1 convolution with a fused requantize + SiLU epilogue.
// Layout is HWC ("channels last"): each pixel's channels are contiguous, so a
// PE's pixels arrive as one contiguous DMA block.
struct Conv1x1 {
  int pixels = 0, cin = 0, cout = 0;
  int32_t x_zero = 0;            // input zero point (asymmetric int8 activations)
  int32_t pre_zero = 0;          // zero point of the requantized pre-activation
  std::vector<int8_t> x;         // [pixels][cin]
  std::vector<int8_t> w;         // [cout][cin]   symmetric int8 weights (zero point 0)
  std::vector<int32_t> bias;     // [cout]        int32, scale = s_x * s_w[c]
  std::vector<int32_t> m0;       // [cout]        fixed-point multiplier (Q31)
  std::vector<int32_t> shift;    // [cout]        right shift, >= 0
  std::vector<int8_t> lut;       // [256]         SiLU in the int8 domain, indexed by (q + 128)
};

enum class Mode { V0_SERIAL, V1_DOUBLE_BUFFER, V2_OVERLAP_WEIGHTS };

struct Event {
  std::string resource;  // "dma" or "compute"
  std::string kind;      // "load", "store", "compute"
  int tile;
  int64_t start, end;    // cycles
};

struct Result {
  std::vector<int8_t> y;  // [pixels][cout]
  int64_t cycles = 0;
  int64_t ideal_mac_cycles = 0;
  int64_t dma_busy = 0, compute_busy = 0;
  int pixels_per_pe = 0, cout_block = 0, tiles = 0, local_bytes_used = 0;
  std::vector<Event> timeline;
};

// Bytes of PE local memory a tiling needs. Double buffering keeps two input and two
// output tiles, so the next tile can load while the current one computes.
int local_footprint(const Conv1x1& k, int pixels_per_pe, int cout_block, bool double_buffer);

// Largest pixels-per-PE that fits in local memory (0 if none fits).
int auto_pixels_per_pe(const Config& c, const Conv1x1& k, int cout_block, bool double_buffer);

// Straightforward scalar reference: the "CPU reference" every kernel must match bit-exactly.
std::vector<int8_t> reference(const Conv1x1& k);

// The tiled kernel, run on the timing model.
Result run(const Config& c, const Conv1x1& k, Mode mode, int pixels_per_pe, int cout_block);

// gemmlowp / TFLite fixed-point requantization (Jacob et al. 2018, section 2.2)
int32_t requantize(int32_t acc, int32_t m0, int32_t shift, int32_t zero);

}  // namespace npusim
