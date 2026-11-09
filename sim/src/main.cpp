// npusim CLI.
//   npusim <data_dir> [--mode v0|v1|v2] [--pixels-per-pe N|auto] [--cout-block N] [--sweep]
// Reads the kernel's tensors from <data_dir> (written by the Python harness),
// checks the tiled kernel against the scalar reference bit-for-bit, and prints a
// JSON report (cycles, utilization, timeline) to stdout.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "npusim.hpp"

using namespace npusim;

template <typename T>
static std::vector<T> read_bin(const std::string& path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<T> v(n);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
  if (static_cast<size_t>(f.gcount()) != n * sizeof(T)) throw std::runtime_error("short read: " + path);
  return v;
}

static std::map<std::string, long> read_meta(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::map<std::string, long> m;
  std::string k;
  long v;
  while (f >> k >> v) m[k] = v;
  return m;
}

static Conv1x1 load(const std::string& dir) {
  auto m = read_meta(dir + "/meta.txt");
  Conv1x1 k;
  k.pixels = static_cast<int>(m.at("pixels"));
  k.cin = static_cast<int>(m.at("cin"));
  k.cout = static_cast<int>(m.at("cout"));
  k.x_zero = static_cast<int32_t>(m.at("x_zero"));
  k.pre_zero = static_cast<int32_t>(m.at("pre_zero"));
  k.x = read_bin<int8_t>(dir + "/x.bin", static_cast<size_t>(k.pixels) * k.cin);
  k.w = read_bin<int8_t>(dir + "/w.bin", static_cast<size_t>(k.cout) * k.cin);
  k.bias = read_bin<int32_t>(dir + "/bias.bin", k.cout);
  k.m0 = read_bin<int32_t>(dir + "/m0.bin", k.cout);
  k.shift = read_bin<int32_t>(dir + "/shift.bin", k.cout);
  k.lut = read_bin<int8_t>(dir + "/lut.bin", 256);
  return k;
}

static const char* name(Mode m) {
  return m == Mode::V0_SERIAL ? "v0" : m == Mode::V1_DOUBLE_BUFFER ? "v1" : "v2";
}

static std::string json(const Config& c, const Conv1x1& k, Mode mode, const Result& r, bool bitexact, bool timeline) {
  std::ostringstream o;
  o << "{\"mode\":\"" << name(mode) << "\",\"bit_exact\":" << (bitexact ? "true" : "false")
    << ",\"cycles\":" << r.cycles << ",\"ideal_mac_cycles\":" << r.ideal_mac_cycles
    << ",\"mac_utilization\":" << static_cast<double>(r.ideal_mac_cycles) / r.cycles
    << ",\"time_us\":" << r.cycles / c.clock_hz * 1e6
    << ",\"dma_busy\":" << r.dma_busy << ",\"compute_busy\":" << r.compute_busy
    << ",\"pixels_per_pe\":" << r.pixels_per_pe << ",\"cout_block\":" << r.cout_block
    << ",\"tiles\":" << r.tiles << ",\"local_bytes_used\":" << r.local_bytes_used
    << ",\"local_bytes\":" << c.local_bytes
    << ",\"shape\":{\"pixels\":" << k.pixels << ",\"cin\":" << k.cin << ",\"cout\":" << k.cout << "}";
  if (timeline) {
    o << ",\"timeline\":[";
    for (size_t i = 0; i < r.timeline.size(); ++i) {
      const auto& e = r.timeline[i];
      o << (i ? "," : "") << "[\"" << e.resource << "\",\"" << e.kind << "\"," << e.tile << "," << e.start << "," << e.end << "]";
    }
    o << "]";
  }
  o << "}";
  return o.str();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: npusim <data_dir> [--mode v0|v1|v2] [--pixels-per-pe N|auto] [--cout-block N] [--sweep]\n";
    return 2;
  }
  std::string dir = argv[1], mode_s = "v1", ppe = "auto";
  int cb = 4;
  bool sweep = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode") mode_s = argv[++i];
    else if (a == "--pixels-per-pe") ppe = argv[++i];
    else if (a == "--cout-block") cb = std::stoi(argv[++i]);
    else if (a == "--sweep") sweep = true;
  }
  try {
    Config c;
    Conv1x1 k = load(dir);
    const auto ref = reference(k);
    const std::map<std::string, Mode> modes = {{"v0", Mode::V0_SERIAL}, {"v1", Mode::V1_DOUBLE_BUFFER},
                                               {"v2", Mode::V2_OVERLAP_WEIGHTS}};
    if (sweep) {  // every mode x every tile size that fits
      std::cout << "[";
      bool first = true;
      for (auto& [s, m] : modes) {
        int maxp = auto_pixels_per_pe(c, k, cb, m != Mode::V0_SERIAL);
        for (int P = 1; P <= maxp; ++P) {
          Result r = run(c, k, m, P, cb);
          std::cout << (first ? "" : ",") << json(c, k, m, r, r.y == ref, false);
          first = false;
        }
      }
      std::cout << "]\n";
      return 0;
    }
    Mode m = modes.at(mode_s);
    int P = ppe == "auto" ? auto_pixels_per_pe(c, k, cb, m != Mode::V0_SERIAL) : std::stoi(ppe);
    Result r = run(c, k, m, P, cb);
    bool ok = r.y == ref;
    std::ofstream(dir + "/y_sim.bin", std::ios::binary).write(reinterpret_cast<const char*>(r.y.data()), r.y.size());
    std::ofstream(dir + "/y_ref.bin", std::ios::binary).write(reinterpret_cast<const char*>(ref.data()), ref.size());
    std::cout << json(c, k, m, r, ok, true) << "\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
}
