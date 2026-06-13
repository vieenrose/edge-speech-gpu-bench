// Validate the cuDNN-free CUDA Conv against ORT's CPU Conv reference.
#include <cstdio>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include "onnxruntime_cxx_api.h"

static std::pair<std::vector<std::string>, std::vector<std::vector<float>>>
run(Ort::Env& env, const char* model, bool cuda) {
  Ort::SessionOptions so;
  if (cuda) { OrtCUDAProviderOptions cu; cu.device_id = 0; so.AppendExecutionProvider_CUDA(cu); }
  Ort::Session sess(env, model, so);
  Ort::AllocatorWithDefaultOptions alloc;
  size_t nin = sess.GetInputCount(), nout = sess.GetOutputCount();
  std::vector<std::string> innames, outnames;
  for (size_t i = 0; i < nin; i++) innames.push_back(sess.GetInputNameAllocated(i, alloc).get());
  for (size_t i = 0; i < nout; i++) outnames.push_back(sess.GetOutputNameAllocated(i, alloc).get());
  std::vector<const char*> inp, outp;
  for (auto& s : innames) inp.push_back(s.c_str());
  for (auto& s : outnames) outp.push_back(s.c_str());
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::mt19937 rng(123); std::normal_distribution<float> nd(0, 1);  // same seed both runs
  std::vector<Ort::Value> ins; std::vector<std::vector<float>> bufs(nin);
  for (size_t i = 0; i < nin; i++) {
    auto shape = sess.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    size_t cnt = 1; for (auto d : shape) cnt *= d;
    bufs[i].resize(cnt); for (auto& v : bufs[i]) v = nd(rng);
    ins.push_back(Ort::Value::CreateTensor<float>(mem, bufs[i].data(), cnt, shape.data(), shape.size()));
  }
  auto res = sess.Run(Ort::RunOptions{}, inp.data(), ins.data(), nin, outp.data(), nout);
  std::vector<std::vector<float>> outs(nout);
  for (size_t i = 0; i < nout; i++) {
    float* d = res[i].GetTensorMutableData<float>();
    auto sh = res[i].GetTensorTypeAndShapeInfo().GetShape();
    size_t cnt = 1; for (auto x : sh) cnt *= x;
    outs[i].assign(d, d + cnt);
  }
  return {outnames, outs};
}

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "convtest");
  const char* model = "/tmp/conv_test.onnx";
  auto cpu = run(env, model, false);
  auto cu = run(env, model, true);
  double maxdiff = 0;
  for (size_t i = 0; i < cpu.second.size(); i++) {
    double md = 0;
    for (size_t j = 0; j < cpu.second[i].size(); j++)
      md = std::max(md, (double)std::fabs(cpu.second[i][j] - cu.second[i][j]));
    printf("  %-10s max|CPU-CUDA| = %.3e  (n=%zu)\n", cpu.first[i].c_str(), md, cpu.second[i].size());
    maxdiff = std::max(maxdiff, md);
  }
  printf("OVERALL max|CPU-CUDA| = %.3e  ->  %s\n", maxdiff,
         maxdiff < 1e-3 ? "PASS (cuDNN-free conv matches CPU reference)" : "FAIL");
  return maxdiff < 1e-3 ? 0 : 1;
}
