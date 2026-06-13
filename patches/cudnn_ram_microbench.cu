// Measure how much libcudnn faults into RSS when you actually use a cuDNN conv,
// vs the cuBLAS-only path. This is the per-process RAM the cuDNN-free EP avoids.
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static long lib_rss_kb(const char* needle) {
  FILE* f = fopen("/proc/self/smaps", "r");
  if (!f) return -1;
  char line[1024]; long rss = 0; int in = 0;
  while (fgets(line, sizeof line, f)) {
    if (((line[0]>=0x30&&line[0]<=0x39)||(line[0]>=0x61&&line[0]<=0x66)) && strstr(line," r")) in = (strstr(line, needle) != NULL);
    if (in && strncmp(line, "Rss:", 4) == 0) { long v = 0; sscanf(line + 4, "%ld", &v); rss += v; }
  }
  fclose(f); return rss;
}

int main() {
  printf("start:               libcudnn RSS = %ld KB\n", lib_rss_kb("libcudnn"));
  cublasHandle_t cb; cublasCreate(&cb);
  // a cuBLAS GEMM (the cuDNN-free conv uses this path) — should NOT fault in cuDNN
  int n = 256; float *A, *B, *C; cudaMalloc(&A, n*n*4); cudaMalloc(&B, n*n*4); cudaMalloc(&C, n*n*4);
  float one = 1, zero = 0;
  cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, B, n, &zero, C, n);
  cudaDeviceSynchronize();
  printf("after cuBLAS GEMM:    libcudnn RSS = %ld KB   (cuDNN-free conv path)\n", lib_rss_kb("libcudnn"));

  cudnnHandle_t cd; cudnnCreate(&cd);
  printf("after cudnnCreate:    libcudnn RSS = %ld KB\n", lib_rss_kb("libcudnn"));

  // a real cuDNN convolution forward (what stock ORT Conv does)
  cudnnTensorDescriptor_t xd, yd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cvd;
  cudnnCreateTensorDescriptor(&xd); cudnnCreateTensorDescriptor(&yd);
  cudnnCreateFilterDescriptor(&wd); cudnnCreateConvolutionDescriptor(&cvd);
  int N=1,Cc=64,H=32,W=32, M=64,kh=3,kw=3;
  cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,Cc,H,W);
  cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M,Cc,kh,kw);
  cudnnSetConvolution2dDescriptor(cvd, 1,1, 1,1, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  int on,oc,oh,ow; cudnnGetConvolution2dForwardOutputDim(cvd, xd, wd, &on,&oc,&oh,&ow);
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on,oc,oh,ow);
  float *x,*w,*y; cudaMalloc(&x,(size_t)N*Cc*H*W*4); cudaMalloc(&w,(size_t)M*Cc*kh*kw*4); cudaMalloc(&y,(size_t)on*oc*oh*ow*4);
  size_t ws=0; cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  cudnnGetConvolutionForwardWorkspaceSize(cd, xd, wd, cvd, yd, algo, &ws);
  void* wsp=nullptr; if (ws) cudaMalloc(&wsp, ws);
  cudnnConvolutionForward(cd, &one, xd, x, wd, w, cvd, algo, wsp, ws, &zero, yd, y);
  cudaDeviceSynchronize();
  printf("after cuDNN conv fwd: libcudnn RSS = %ld KB   (stock ORT Conv path)\n", lib_rss_kb("libcudnn"));
  return 0;
}
