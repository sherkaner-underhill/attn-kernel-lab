// SPDX-License-Identifier: Apache-2.0
// Instruction-family microbenchmark: legacy E4M3xE4M3->F32 mma.sync vs
// block-scaled MXFP8 (kind::mxf8f6f4.block_scale, UE8M0 unity scales) on
// SM120a, using the exact m16n8k32 fragment layout the production PV path
// uses.  Answers the instruction-family question in isolation: correctness
// (bit-compare under unity scales) and pure register-resident issue-rate
// throughput for each family.
//
// Build:  /usr/local/cuda/bin/nvcc -O3 -arch=sm_120a -o pv_mma_bench pv_mma_bench.cu
// Run:    ./pv_mma_bench [iters]        (default 10000)
//
// No torch, tiny footprint (<50 MB) — safe next to the resident server.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                        \
  do {                                                               \
    cudaError_t e = (x);                                             \
    if (e != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
              cudaGetErrorString(e), __FILE__, __LINE__);            \
      exit(1);                                                       \
    }                                                                \
  } while (0)

#define LEGACY_MMA(d, a, b)                                          \
  asm volatile(                                                      \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "         \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"      \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])               \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),       \
        "r"(b[1]))

#define MX_MMA(d, a, b, sa, sb)                                      \
  asm volatile(                                                      \
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale" \
      ".scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "                      \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "       \
      "{%10}, {0,0}, {%11}, {0,0};\n"                                \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])               \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),       \
        "r"(b[1]), "r"(sa), "r"(sb))

// ---------------- correctness: one warp, same inputs, both families -------
__global__ void correctness_kernel(const unsigned* A, const unsigned* B,
                                   float* d_legacy, float* d_mx) {
  const int t = threadIdx.x & 31;
  unsigned a[4] = {A[t], A[t + 32], A[t + 64], A[t + 96]};
  unsigned b[2] = {B[t], B[t + 32]};
  const unsigned unity = 0x7f7f7f7fu;  // UE8M0 127 -> 2^0 in every byte
  float dl[4] = {0.f, 0.f, 0.f, 0.f};
  float dm[4] = {0.f, 0.f, 0.f, 0.f};
  LEGACY_MMA(dl, a, b);
  MX_MMA(dm, a, b, unity, unity);
  for (int i = 0; i < 4; ++i) {
    d_legacy[t * 4 + i] = dl[i];
    d_mx[t * 4 + i] = dm[i];
  }
}

#define S8_MMA(d, a, b)                                              \
  asm volatile(                                                      \
      "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "   \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"      \
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])               \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),       \
        "r"(b[1]))

// e2m1 FP4: m16n8k64, same 4+2 register fragment shape, 2x K depth.
#define FP4_MMA(d, a, b, sa, sb)                                     \
  asm volatile(                                                      \
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale"     \
      ".scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "                      \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "       \
      "{%10}, {0,0}, {%11}, {0,0};\n"                                \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])               \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),       \
        "r"(b[1]), "r"(sa), "r"(sb))

// ---------------- throughput: NACC independent accumulators ---------------
constexpr int NACC = 8;
constexpr double FLOP_PER_MMA = 2.0 * 16 * 8 * 32;      // 8192 (k32 forms)
constexpr double FLOP_PER_MMA_K64 = 2.0 * 16 * 8 * 64;  // 16384 (fp4)

__global__ void bench_legacy(const unsigned* A, const unsigned* B, float* out,
                             int iters) {
  const int t = threadIdx.x & 31;
  unsigned a[4] = {A[t], A[t + 32], A[t + 64], A[t + 96]};
  unsigned b[2] = {B[t], B[t + 32]};
  float acc[NACC][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < NACC; ++u) LEGACY_MMA(acc[u], a, b);
  }
  float s = 0.f;
  for (int u = 0; u < NACC; ++u)
    for (int i = 0; i < 4; ++i) s += acc[u][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void bench_mx(const unsigned* A, const unsigned* B, float* out,
                         int iters) {
  const int t = threadIdx.x & 31;
  unsigned a[4] = {A[t], A[t + 32], A[t + 64], A[t + 96]};
  unsigned b[2] = {B[t], B[t + 32]};
  const unsigned unity = 0x7f7f7f7fu;
  float acc[NACC][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < NACC; ++u) MX_MMA(acc[u], a, b, unity, unity);
  }
  float s = 0.f;
  for (int u = 0; u < NACC; ++u)
    for (int i = 0; i < 4; ++i) s += acc[u][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void bench_s8(const unsigned* A, const unsigned* B, float* out,
                         int iters) {
  const int t = threadIdx.x & 31;
  unsigned a[4] = {A[t], A[t + 32], A[t + 64], A[t + 96]};
  unsigned b[2] = {B[t], B[t + 32]};
  int acc[NACC][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < NACC; ++u) S8_MMA(acc[u], a, b);
  }
  int s = 0;
  for (int u = 0; u < NACC; ++u)
    for (int i = 0; i < 4; ++i) s += acc[u][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
}

__global__ void bench_fp4(const unsigned* A, const unsigned* B, float* out,
                          int iters) {
  const int t = threadIdx.x & 31;
  unsigned a[4] = {A[t], A[t + 32], A[t + 64], A[t + 96]};
  unsigned b[2] = {B[t], B[t + 32]};
  const unsigned unity = 0x7f7f7f7fu;
  float acc[NACC][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < NACC; ++u) FP4_MMA(acc[u], a, b, unity, unity);
  }
  float s = 0.f;
  for (int u = 0; u < NACC; ++u)
    for (int i = 0; i < 4; ++i) s += acc[u][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

static double time_kernel(void (*k)(const unsigned*, const unsigned*, float*,
                                    int),
                          const unsigned* A, const unsigned* B, float* out,
                          int iters, int blocks, int threads) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  k<<<blocks, threads>>>(A, B, out, iters);  // warmup + JIT-free check
  CK(cudaDeviceSynchronize());
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep) {
    CK(cudaEventRecord(e0));
    k<<<blocks, threads>>>(A, B, out, iters);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best_ms) best_ms = ms;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return best_ms;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 10000;

  // Random e4m3 bytes with NaN encodings (0x7f/0xff) remapped: the two
  // instruction families are not required to agree bit-for-bit on NaNs.
  const int ABYTES = 128 * 4, BBYTES = 64 * 4;
  unsigned char ha[ABYTES], hb[BBYTES];
  srand(20260828);
  for (int i = 0; i < ABYTES; ++i) {
    unsigned char v = (unsigned char)(rand() & 0xff);
    if ((v & 0x7f) == 0x7f) v ^= 0x40;
    ha[i] = v;
  }
  for (int i = 0; i < BBYTES; ++i) {
    unsigned char v = (unsigned char)(rand() & 0xff);
    if ((v & 0x7f) == 0x7f) v ^= 0x40;
    hb[i] = v;
  }
  unsigned *dA, *dB;
  float *dL, *dM, *dOut;
  CK(cudaMalloc(&dA, ABYTES));
  CK(cudaMalloc(&dB, BBYTES));
  CK(cudaMalloc(&dL, 128 * sizeof(float)));
  CK(cudaMalloc(&dM, 128 * sizeof(float)));
  CK(cudaMemcpy(dA, ha, ABYTES, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hb, BBYTES, cudaMemcpyHostToDevice));

  // ---- correctness ----
  correctness_kernel<<<1, 32>>>(dA, dB, dL, dM);
  CK(cudaDeviceSynchronize());
  float hl[128], hm[128];
  CK(cudaMemcpy(hl, dL, sizeof(hl), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hm, dM, sizeof(hm), cudaMemcpyDeviceToHost));
  int mismatch = 0;
  float max_abs = 0.f;
  for (int i = 0; i < 128; ++i) {
    if (memcmp(&hl[i], &hm[i], 4) != 0) ++mismatch;
    float d = fabsf(hl[i] - hm[i]);
    if (d > max_abs) max_abs = d;
  }
  printf("correctness: %d/128 outputs bit-mismatch, max_abs_diff=%g\n",
         mismatch, max_abs);

  // ---- throughput ----
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  const int threads = 256;
  const int blocks = prop.multiProcessorCount * 8;
  CK(cudaMalloc(&dOut, (size_t)blocks * threads * sizeof(float)));
  const double warps = (double)blocks * (threads / 32);
  const double flop = warps * iters * NACC * FLOP_PER_MMA;

  const double flop4 = warps * iters * NACC * FLOP_PER_MMA_K64;

  double ms_l = time_kernel(bench_legacy, dA, dB, dOut, iters, blocks, threads);
  double ms_m = time_kernel(bench_mx, dA, dB, dOut, iters, blocks, threads);
  double ms_s = time_kernel(bench_s8, dA, dB, dOut, iters, blocks, threads);
  double ms_4 = time_kernel(bench_fp4, dA, dB, dOut, iters, blocks, threads);
  printf("SMs=%d blocks=%d iters=%d  flop/kernel(k32)=%.3e\n",
         prop.multiProcessorCount, blocks, iters, flop);
  printf("legacy  E4M3xE4M3->F32   : %8.3f ms  %8.1f TF/s\n", ms_l,
         flop / (ms_l * 1e-3) / 1e12);
  printf("mxfp8   block-scale 1X   : %8.3f ms  %8.1f TF/s\n", ms_m,
         flop / (ms_m * 1e-3) / 1e12);
  printf("int8    S8xS8->S32       : %8.3f ms  %8.1f TOP/s\n", ms_s,
         flop / (ms_s * 1e-3) / 1e12);
  printf("fp4     mxf4 e2m1 k64 2X : %8.3f ms  %8.1f TF/s\n", ms_4,
         flop4 / (ms_4 * 1e-3) / 1e12);
  printf("ratios vs legacy fp8: mx=%.3f int8=%.3f fp4=%.3f\n", ms_l / ms_m,
         ms_l / ms_s, (flop4 / ms_4) / (flop / ms_l));
  return 0;
}
