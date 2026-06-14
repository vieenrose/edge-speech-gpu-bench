// Benchmark a model through the (cuDNN-free) CUDA EP: peak RSS + warm inference time.
// All dynamic dims -> 1 (batch=1, one chunk). RSS is dominated by weights + CUDA/cuBLAS
// context, so dynamic=1 is representative for RAM; latency is per-chunk (indicative).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include "onnxruntime_cxx_api.h"

static long peak_rss_mb() {
  FILE* f = fopen("/proc/self/status", "r"); if (!f) return -1;
  char line[256]; long kb = 0;
  while (fgets(line, sizeof line, f)) if (!strncmp(line, "VmHWM:", 6)) { sscanf(line + 6, "%ld", &kb); break; }
  fclose(f); return kb / 1024;
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: ort_bench <model.onnx> <name>\n"); return 1; }
  const char* model = argv[1]; const char* name = argv[2];
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "bench");
  Ort::SessionOptions so; so.SetIntraOpNumThreads(4);
  OrtCUDAProviderOptions cu; memset(&cu, 0, sizeof cu); cu.device_id = 0;
  cu.gpu_mem_limit = (size_t)2 * 1024 * 1024 * 1024;  // cap arena (GB10's 128GB UMA overflows ORT-1.11 math)
  cu.arena_extend_strategy = 1;                       // kSameAsRequested (avoid next-pow2 of huge size)
  so.AppendExecutionProvider_CUDA(cu);
  Ort::Session sess(env, model, so);
  Ort::AllocatorWithDefaultOptions alloc;
  size_t nin = sess.GetInputCount(), nout = sess.GetOutputCount();
  std::vector<std::string> in_names, out_names;
  for (size_t i = 0; i < nin; i++) { char* n = sess.GetInputName(i, alloc); in_names.push_back(n); alloc.Free(n); }
  for (size_t i = 0; i < nout; i++) { char* n = sess.GetOutputName(i, alloc); out_names.push_back(n); alloc.Free(n); }
  std::vector<const char*> inp, outp;
  for (auto& s : in_names) inp.push_back(s.c_str());
  for (auto& s : out_names) outp.push_back(s.c_str());

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::mt19937 rng(7); std::normal_distribution<float> nd(0, 0.1f);
  const int64_t SEQ = (argc > 3) ? atoll(argv[3]) : 100;
  std::vector<Ort::Value> ins;
  std::vector<std::vector<float>> fbuf(nin);
  std::vector<std::vector<int64_t>> i64buf(nin);
  std::vector<std::vector<int32_t>> i32buf(nin);
  for (size_t i = 0; i < nin; i++) {
    auto ti = sess.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    auto shape = ti.GetShape();
    for (size_t k = 0; k < shape.size(); k++)
      if (shape[k] < 0) shape[k] = (k == 0) ? 1 : SEQ;  // batch->1, other dynamic (seq)->SEQ
    size_t cnt = 1; for (auto d : shape) cnt *= (size_t)d;
    auto et = ti.GetElementType();
    bool is_len = in_names[i].find("len") != std::string::npos;  // length-type -> sequence (=1)
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      i64buf[i].assign(cnt, is_len ? SEQ : 0);
      ins.push_back(Ort::Value::CreateTensor<int64_t>(mem, i64buf[i].data(), cnt, shape.data(), shape.size()));
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      i32buf[i].assign(cnt, is_len ? (int32_t)SEQ : 0);
      ins.push_back(Ort::Value::CreateTensor<int32_t>(mem, i32buf[i].data(), cnt, shape.data(), shape.size()));
    } else {  // float
      fbuf[i].resize(cnt); for (auto& v : fbuf[i]) v = nd(rng);
      ins.push_back(Ort::Value::CreateTensor<float>(mem, fbuf[i].data(), cnt, shape.data(), shape.size()));
    }
  }
  auto run = [&]() { sess.Run(Ort::RunOptions{}, inp.data(), ins.data(), nin, outp.data(), nout); };
  for (int i = 0; i < 3; i++) run();  // warmup (JIT + arena grow)
  std::vector<double> t;
  for (int i = 0; i < 10; i++) {
    auto a = std::chrono::high_resolution_clock::now();
    run();
    auto b = std::chrono::high_resolution_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
  }
  std::sort(t.begin(), t.end());
  printf("%-12s  warm_ms=%7.2f  peak_rss=%ld MB\n", name, t[t.size() / 2], peak_rss_mb());
  return 0;
}
