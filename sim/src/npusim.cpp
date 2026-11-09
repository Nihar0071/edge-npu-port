#include "npusim.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace npusim {

// ---- fixed-point requantization ------------------------------------------------
// The int32 accumulator has scale s_x*s_w. The output wants scale s_y. So
//   y = zero + acc * M,   M = s_x*s_w/s_y  (a real number, usually < 1)
// Integer-only hardware can't multiply by a float, so M is stored as
//   M = m0 * 2^-31 * 2^-shift,  m0 in [2^30, 2^31)
// and applied as a rounding high-multiply followed by a rounding right shift.

static int32_t sat_round_doubling_high_mul(int32_t a, int32_t b) {
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
  int64_t ab = static_cast<int64_t>(a) * b;
  int64_t nudge = ab >= 0 ? (1LL << 30) : (1 - (1LL << 30));
  return static_cast<int32_t>((ab + nudge) / (1LL << 31));  // C++ '/' truncates toward zero
}

static int32_t rounding_divide_by_pot(int32_t x, int exponent) {
  if (exponent == 0) return x;
  int32_t mask = static_cast<int32_t>((1LL << exponent) - 1);
  int32_t remainder = x & mask;
  int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

int32_t requantize(int32_t acc, int32_t m0, int32_t shift, int32_t zero) {
  int32_t v = rounding_divide_by_pot(sat_round_doubling_high_mul(acc, m0), shift) + zero;
  return std::clamp(v, -128, 127);
}

static int8_t silu_lut(const Conv1x1& k, int32_t q) { return k.lut[q + 128]; }

// ---- reference ------------------------------------------------------------------
std::vector<int8_t> reference(const Conv1x1& k) {
  std::vector<int8_t> y(static_cast<size_t>(k.pixels) * k.cout);
  for (int p = 0; p < k.pixels; ++p)
    for (int o = 0; o < k.cout; ++o) {
      int32_t acc = k.bias[o];
      for (int i = 0; i < k.cin; ++i)
        acc += (static_cast<int32_t>(k.x[p * k.cin + i]) - k.x_zero) * k.w[o * k.cin + i];
      y[p * k.cout + o] = silu_lut(k, requantize(acc, k.m0[o], k.shift[o], k.pre_zero));
    }
  return y;
}

// ---- tiling ---------------------------------------------------------------------
int local_footprint(const Conv1x1& k, int P, int cb, bool dbl) {
  int bufs = dbl ? 2 : 1;
  return bufs * P * k.cin       // input tile (int8)
       + bufs * P * k.cout      // output tile (int8)
       + k.cin * cb             // current weight block, broadcast from L2
       + P * cb * 4;            // int32 accumulators for the current block
}

int auto_pixels_per_pe(const Config& c, const Conv1x1& k, int cb, bool dbl) {
  int best = 0;
  for (int P = 1; P <= k.pixels; ++P)
    if (local_footprint(k, P, cb, dbl) <= c.local_bytes) best = P;
    else break;
  return best;
}

Result run(const Config& c, const Conv1x1& k, Mode mode, int P, int cb) {
  if (k.cout % cb) throw std::invalid_argument("cout must be a multiple of cout_block");
  const bool dbl = mode != Mode::V0_SERIAL;
  Result r;
  r.pixels_per_pe = P;
  r.cout_block = cb;
  r.local_bytes_used = local_footprint(k, P, cb, dbl);
  if (r.local_bytes_used > c.local_bytes) throw std::invalid_argument("tile does not fit in PE local memory");

  // Zero-point trick: sum_i (x_i - zx) * w_i = sum_i x_i*w_i - zx * sum_i w_i.
  // Fold the second term into the bias once, so the inner loop is a plain int8 dot product.
  std::vector<int32_t> bias_folded(k.cout);
  for (int o = 0; o < k.cout; ++o) {
    int32_t wsum = 0;
    for (int i = 0; i < k.cin; ++i) wsum += k.w[o * k.cin + i];
    bias_folded[o] = k.bias[o] - k.x_zero * wsum;
  }

  const int px_per_tile = c.num_pes * P;
  r.tiles = (k.pixels + px_per_tile - 1) / px_per_tile;
  r.y.assign(static_cast<size_t>(k.pixels) * k.cout, 0);
  r.ideal_mac_cycles = static_cast<int64_t>(k.pixels) * k.cin * k.cout / (c.num_pes * c.macs_per_pe);

  auto dma_cycles = [&](int64_t bytes) { return c.ddr_latency_cycles + (bytes + c.ddr_bytes_per_cycle - 1) / c.ddr_bytes_per_cycle; };
  const int blocks = k.cout / cb;
  const int64_t bcast = c.l2_latency_cycles + (k.cin * cb + c.l2_bytes_per_cycle - 1) / c.l2_bytes_per_cycle;

  std::vector<int64_t> load_end(r.tiles + 1, 0), comp_end(r.tiles, 0), store_end(r.tiles, 0);
  int64_t dma_free = 0, comp_free = 0;
  auto dma = [&](const char* kind, int t, int64_t ready, int64_t bytes) {
    int64_t s = std::max(dma_free, ready), e = s + dma_cycles(bytes);
    dma_free = e;
    r.dma_busy += e - s;
    r.timeline.push_back({"dma", kind, t, s, e});
    return e;
  };

  auto tile_pixels = [&](int t) { return std::min(px_per_tile, k.pixels - t * px_per_tile); };

  if (dbl) load_end[0] = dma("load", 0, 0, static_cast<int64_t>(tile_pixels(0)) * k.cin);
  for (int t = 0; t < r.tiles; ++t) {
    const int npx = tile_pixels(t);
    const int p0 = t * px_per_tile;
    // lockstep: the tile takes as long as its busiest PE (the last tile may leave PEs idle)
    const int P_busy = std::min(P, npx);

    int64_t ready;
    if (!dbl) {
      load_end[t] = dma("load", t, std::max(dma_free, comp_free), static_cast<int64_t>(npx) * k.cin);
      ready = load_end[t];
    } else {
      // prefetch the next tile into the other input buffer (freed when compute(t-1) ended)
      if (t + 1 < r.tiles)
        load_end[t + 1] = dma("load", t + 1, t > 0 ? comp_end[t - 1] : 0,
                              static_cast<int64_t>(tile_pixels(t + 1)) * k.cin);
      // the output buffer we're about to write was used by tile t-2: wait for its store
      ready = std::max(load_end[t], t >= 2 ? store_end[t - 2] : 0);
    }

    // ---- compute: the actual int8 math, in the tiled loop order ----
    for (int pe = 0; pe < c.num_pes; ++pe)
      for (int j = 0; j < P; ++j) {
        const int p = p0 + pe * P + j;
        if (p >= p0 + npx) break;
        for (int b = 0; b < blocks; ++b)
          for (int oo = 0; oo < cb; ++oo) {
            const int o = b * cb + oo;
            int32_t acc = bias_folded[o];
            const int8_t* xp = &k.x[static_cast<size_t>(p) * k.cin];
            const int8_t* wp = &k.w[static_cast<size_t>(o) * k.cin];
            for (int i = 0; i < k.cin; ++i) acc += static_cast<int32_t>(xp[i]) * wp[i];
            r.y[static_cast<size_t>(p) * k.cout + o] = silu_lut(k, requantize(acc, k.m0[o], k.shift[o], k.pre_zero));
          }
      }

    // ---- compute: timing ----
    const int64_t mac = static_cast<int64_t>(P_busy) * k.cin * cb / c.macs_per_pe;  // per block, per PE
    int64_t body = (mode == Mode::V2_OVERLAP_WEIGHTS)
        ? bcast + blocks * std::max(mac, bcast)  // next block's weights broadcast during this block's MACs
        : blocks * (bcast + mac);                // broadcast, then compute, block by block
    const int64_t epilogue = static_cast<int64_t>(P_busy) * k.cout * c.epilogue_ops_per_elem / c.alu_ops_per_pe;
    const int64_t s = std::max(comp_free, ready), e = s + body + epilogue;
    comp_free = comp_end[t] = e;
    r.compute_busy += e - s;
    r.timeline.push_back({"compute", "compute", t, s, e});

    store_end[t] = dma("store", t, e, static_cast<int64_t>(npx) * k.cout);
  }
  r.cycles = std::max(dma_free, comp_free);
  return r;
}

}  // namespace npusim
