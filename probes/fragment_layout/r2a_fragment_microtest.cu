// SPDX-License-Identifier: Apache-2.0
// Stage-1 micro-test: the two fragment mappings the fused kernel depends on.
//
// (A) C->A identity: the fp32 accumulator of mma.m16n8k32 (S tiles) can be
//     packed DIRECTLY into the bf16 A-fragment of mma.m16n8k16 (PV) with no
//     cross-lane traffic:  for P's k16 chunk j (= S n-tiles 2j, 2j+1):
//        a0 = pack(c[2j][0],   c[2j][1])     a1 = pack(c[2j][2],   c[2j][3])
//        a2 = pack(c[2j+1][0], c[2j+1][1])   a3 = pack(c[2j+1][2], c[2j+1][3])
// (B) V via ldmatrix.x2.trans: row-major V[k16 x n8] in smem -> col-major
//     B-fragment of mma.m16n8k16.
//
// One warp computes O = (Q x K^T) x V at m16, k32(QK), BN=16 (two S tiles),
// n8 output, with small-integer inputs; CPU reference must match EXACTLY.
//
// Build: nvcc -O3 -arch=sm_120a -o r2a r2a_fragment_microtest.cu
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
  printf("CUDA error %s at %d\n", cudaGetErrorString(err_), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_trans(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_fp8(float& c0, float& c1, float& c2, float& c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// Q[16x32] fp8, K[16x32] fp8 (two n8 tiles), V[16x8] bf16 -> O[16x8] fp32
__global__ void micro(const __nv_fp8_e4m3* Q, const __nv_fp8_e4m3* K,
                      const __nv_bfloat16* V, float* O) {
  __shared__ __align__(16) unsigned char q_sm[16 * 32];
  __shared__ __align__(16) unsigned char k_sm[16 * 32];
  __shared__ __align__(16) unsigned char v_sm[16 * 8 * 2];
  const int lane = threadIdx.x;
  for (int i = lane; i < 16 * 32; i += 32) { q_sm[i] = ((const unsigned char*)Q)[i]; }
  for (int i = lane; i < 16 * 32; i += 32) { k_sm[i] = ((const unsigned char*)K)[i]; }
  for (int i = lane; i < 16 * 8; i += 32)
    ((__nv_bfloat16*)v_sm)[i] = V[i];
  __syncwarp();

  // A fragment: Q 16x32 fp8 == 16x16 b16 -> ldmatrix x4
  uint32_t a0, a1, a2, a3;
  { int r = lane % 16, half = lane / 16;
    ldsm_x4(a0, a1, a2, a3, q_sm + r * 32 + half * 16); }

  // S tiles n=0,1: B fragments from K rows [0..8) and [8..16)
  float c[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  for (int n = 0; n < 2; ++n) {
    uint32_t b0, b1;
    int r = lane % 8;
    ldsm_x2(b0, b1, k_sm + (n * 8 + r) * 32 + ((lane / 8) % 2) * 16);
    mma_fp8(c[n][0], c[n][1], c[n][2], c[n][3], a0, a1, a2, a3, b0, b1);
  }

  // (A) C->A identity packing, k16 chunk j=0 covers S tiles 0,1
  uint32_t pa0 = pack_bf16(c[0][0], c[0][1]);
  uint32_t pa1 = pack_bf16(c[0][2], c[0][3]);
  uint32_t pa2 = pack_bf16(c[1][0], c[1][1]);
  uint32_t pa3 = pack_bf16(c[1][2], c[1][3]);

  // (B) V[16x8] bf16 row-major in smem -> col-major B-frag via ldmatrix trans
  uint32_t vb0, vb1;
  { int r = lane % 16;
    ldsm_x2_trans(vb0, vb1, v_sm + r * 8 * 2); }

  float o[4] = {0, 0, 0, 0};
  mma_bf16(o[0], o[1], o[2], o[3], pa0, pa1, pa2, pa3, vb0, vb1);

  // write O per the m16n8 C layout
  int row = lane / 4, colp = (lane % 4) * 2;
  O[row * 8 + colp] = o[0];
  O[row * 8 + colp + 1] = o[1];
  O[(row + 8) * 8 + colp] = o[2];
  O[(row + 8) * 8 + colp + 1] = o[3];
}

// ---------------------------------------------------------------------------
// Probe: mma.m16n8k32 e4m3 A-fragment byte order. For test lane L (0..3),
// register r, byte b: set that single byte to fp8 1.0 (0x38) in lane L only,
// with B[k][n] = bit n of k (n=0..4, exact 0/1 values). The C row of lane L
// then reads out k* in binary. Everything else zero.
__global__ void probe_a_layout(float* out /* [4][4][4][8] lane,reg,byte -> C row */) {
  __shared__ __align__(16) unsigned char b_sm[32 * 32];
  const int lane = threadIdx.x;
  // B: 8 rows (n) x 32 cols (k) in the QK/K layout (row-per-n), value bit n of k
  for (int i = lane; i < 8 * 32; i += 32) {
    int n = i / 32, k = i % 32;
    b_sm[n * 32 + k] = ((k >> n) & 1) ? 0x38 : 0x00;  // e4m3 1.0 / 0.0
  }
  __syncwarp();
  uint32_t bb0, bb1;
  { int r = lane % 8;
    ldsm_x2(bb0, bb1, b_sm + r * 32 + ((lane / 8) % 2) * 16); }

  for (int L = 0; L < 4; ++L)
    for (int reg = 0; reg < 4; ++reg)
      for (int byte = 0; byte < 4; ++byte) {
        uint32_t a[4] = {0, 0, 0, 0};
        if ((lane % 4) == L && lane < 4) a[reg] = 0x38u << (8 * byte);
        float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        mma_fp8(c0, c1, c2, c3, a[0], a[1], a[2], a[3], bb0, bb1);
        // write full C tile; host decodes
        int row = lane / 4, colp = (lane % 4) * 2;
        float* dst = out + ((L * 4 + reg) * 4 + byte) * 16 * 8;
        dst[row * 8 + colp] = c0; dst[row * 8 + colp + 1] = c1;
        dst[(row + 8) * 8 + colp] = c2; dst[(row + 8) * 8 + colp + 1] = c3;
        __syncwarp();
      }
}

int main() {
  __nv_fp8_e4m3 hq[16 * 32], hk[16 * 32];
  __nv_bfloat16 hv[16 * 8];
  srand(11);
  for (int i = 0; i < 16 * 32; ++i) hq[i] = __nv_fp8_e4m3((float)(rand() % 5 - 2));
  for (int i = 0; i < 16 * 32; ++i) hk[i] = __nv_fp8_e4m3((float)(rand() % 5 - 2));
  for (int i = 0; i < 16 * 8; ++i) hv[i] = __float2bfloat16((float)(rand() % 5 - 2));

  // CPU reference: O = (Q K^T) V   (S in fp32, P == S here, no softmax)
  float S[16][16], Oref[16][8] = {};
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      float d = 0;
      for (int k = 0; k < 32; ++k) d += (float)hq[m * 32 + k] * (float)hk[n * 32 + k];
      S[m][n] = d;
    }
  for (int m = 0; m < 16; ++m)
    for (int j = 0; j < 8; ++j)
      for (int n = 0; n < 16; ++n)
        Oref[m][j] += (float)__float2bfloat16(S[m][n]) * (float)hv[n * 8 + j];

  __nv_fp8_e4m3 *dq, *dk; __nv_bfloat16* dv; float* dO;
  CHECK(cudaMalloc(&dq, sizeof(hq))); CHECK(cudaMalloc(&dk, sizeof(hk)));
  CHECK(cudaMalloc(&dv, sizeof(hv))); CHECK(cudaMalloc(&dO, 16 * 8 * 4));
  CHECK(cudaMemcpy(dq, hq, sizeof(hq), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dk, hk, sizeof(hk), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dv, hv, sizeof(hv), cudaMemcpyHostToDevice));
  micro<<<1, 32>>>(dq, dk, dv, dO);
  CHECK(cudaDeviceSynchronize());
  float hO[16 * 8];
  CHECK(cudaMemcpy(hO, dO, sizeof(hO), cudaMemcpyDeviceToHost));
  double maxerr = 0;
  for (int m = 0; m < 16; ++m)
    for (int j = 0; j < 8; ++j)
      maxerr = fmax(maxerr, fabs(hO[m * 8 + j] - Oref[m][j]));
  printf("micro-test C->A + V-trans: max_abs_err=%.4f (Oref[0][0]=%.1f got=%.1f)\n",
         maxerr, Oref[0][0], hO[0]);
  printf(maxerr < 1.0 ? "FRAGMENT MAPPINGS OK\n" : "FRAGMENT MAPPINGS WRONG\n");

  // ---- fp8 A-layout probe ----
  float* dP; CHECK(cudaMalloc(&dP, 4*4*4*16*8*4));
  CHECK(cudaMemset(dP, 0, 4*4*4*16*8*4));
  probe_a_layout<<<1, 32>>>(dP);
  CHECK(cudaDeviceSynchronize());
  float* hP = (float*)malloc(4*4*4*16*8*4);
  CHECK(cudaMemcpy(hP, dP, 4*4*4*16*8*4, cudaMemcpyDeviceToHost));
  printf("fp8 A-fragment map (test lanes 0..3): entries (lane%%4, reg, byte) -> (row, k)\n");
  for (int L = 0; L < 4; ++L)
    for (int reg = 0; reg < 4; ++reg)
      for (int byte = 0; byte < 4; ++byte) {
        const float* C = hP + ((L*4+reg)*4+byte)*16*8;
        int found_row = -1, kk = -1;
        for (int m = 0; m < 16 && found_row < 0; ++m) {
          int k = 0; bool any = false;
          for (int n = 0; n < 5; ++n) { if (C[m*8+n] > 0.5f) { k |= (1 << n); any = true; } }
          bool zerobit_hit = false;
          // detect k=0 case: row contributes only if some bit set; check bit pattern
          for (int n = 0; n < 8; ++n) if (C[m*8+n] != 0.f) zerobit_hit = true;
          if (any || (zerobit_hit)) { found_row = m; kk = k; }
        }
        // k==0 gives all-zero row (undetectable); mark unknown as k might be 0
        printf("L%d r%d b%d -> row=%d k=%d%s\n", L, reg, byte, found_row, kk,
               found_row < 0 ? " (all-zero: k==0 for this slot?)" : "");
      }
  return maxerr < 1.0 ? 0 : 1;
}
