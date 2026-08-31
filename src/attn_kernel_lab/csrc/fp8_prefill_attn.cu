// Low-precision fused prefill attention -- mma.sync implementation family
// (SM89 Ada / SM120 Blackwell-consumer; developed and qualified on the
// RTX PRO 6000 Blackwell Server Edition).
//
// SPDX-License-Identifier: Apache-2.0
// Adapted from the `origin-private` implementation defined in
// THIRD_PARTY_NOTICES.md. The operator's normative definition lives in
// docs/OPERATOR_CONTRACT.md -- where this comment and that document disagree,
// the document wins.
//
// Fused online-softmax attention with INT8 QK by default (fp8 e4m3 QK
// selectable for A/B), and per-head selectable PV precision: fp8 PV
// (default) or bf16 PV (conservative fallback).  One kernel launch serves
// all query heads of one request (grid.y = q head); GQA group size is a
// runtime parameter.
//
// Design notes (proven by the public probe/micro-test suite in
// probes/fragment_layout/r2a_fragment_microtest.cu):
//
//  * S = QK^T uses mma.m16n8k32 (int8 default / e4m3 selectable) with
//    per-ROW Q scales (sm_scale folded in; see quant.py quantize_q -- an
//    earlier per-64-row-block Q scheme was superseded, measured as the
//    dominant term of the conservative mode's excess error) and per-64-row
//    K tile scales.
//  * fp8 PV exploits the empirically probed A-fragment byte order of
//    mma.m16n8k32.e4m3: thread (r=lane/4, c=lane%4) holds A[r][4c+b] in
//    reg0 byte b, A[r+8][4c+b] in reg1, and +16 columns in regs 2/3.
//    Consequently each thread's own S accumulators pack DIRECTLY into its
//    PV A-fragment registers -- zero cross-lane traffic -- provided V's kv
//    rows are pre-permuted by the fixed 32-permutation SIGMA and stored
//    transposed d-major, tile-major fp8 (built by the Python side).
//  * P's fp8 scale is the constant 448: online softmax bounds p <= 1, so
//    448 is the exact per-row-half amax scale.  It folds into the exp:
//    p448 = ex2(fma(s, log2e, log2(448) - m*log2e)) -- zero extra work.
//    One global-per-(kv-head) V scale folds into the epilogue.
//  * bf16 PV (fallback heads) uses the zero-shuffle C->A packing into
//    mma.m16n8k16.bf16 with ldmatrix.x2.trans V loads.
//
// Numerics (measured, see README): vs an fp32 reference on real-workload
// tensors, row-normalized output error median ~2% (fp8 PV) vs the bf16
// baseline's own ~0.35%; depth-stable 64k..446k.  Synthetic full-depth
// verify at N=446k: max_abs_err 1.0e-4.
//
// Throughput (measured on RTX PRO 6000, CUDA 12.9): 446 TF/s average over
// a 24-head GQA 14x32k chunked-prefill schedule at 446k context, vs 333
// TF/s for the bf16 SDPA baseline and 680 TF/s cuBLASLt fp8 GEMM roof --
// that was the BM=64/NW=4 shape.  At the BM=128/NW=8 default below, the
// same 24-head GQA deployment shape (M=32768, N=446464) measures 530 TF/s
// vs 415 TF/s for BM=64 on this kernel (1.28x), matching the 445 -> 568
// TF/s the standalone research kernel gets from the same shape change.
//
// Limitations (PoC): head_dim 256 only; page_size 1 pools; no sliding
// window / logit cap / cross attention (a consuming framework falls back
// to its stock backend for those); K/V gathered+quantized per forward by
// the Python side.  NOTE the total gather/quantize cost over a 446k
// 14-chunk prefill is DISPUTED (~0.2-0.3 s claimed vs ~7-8 s implied by
// the checked per-call baseline JSON) and is a Phase 3 measurement item;
// do not cite either figure as settled.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

// ---- CTA tile shape ---------------------------------------------------
// BM Q rows per CTA spread over NW warps, 16 Q rows each (the per-warp
// fragment layout is fixed at 16 rows, so BM == NW*16 always).  BM=128/
// NW=8 doubles the warps resident per CTA -- occupancy is 1 CTA/SM either
// way at this smem footprint, so more warps per CTA is the only lever for
// latency hiding -- and halves the K/V gmem traffic per Q row.  Measured
// on the standalone research kernel at the deployment GQA shape (24 q
// heads / 4 kv heads, 14x32k chunked prefill at 446k context):
// 446 -> 570 TF/s.  Override at build time with -DBM_D=64 -DNW_D=4.
#ifndef BM_D
#define BM_D 128
#endif
#ifndef NW_D
#define NW_D 8
#endif
constexpr int BM = BM_D;               // NW*16 (16 Q rows per warp)
constexpr int NW = NW_D;               // warps (16 Q rows each)
static_assert(BM == NW * 16, "BM must equal NW*16");
// Fallback shape for launches the wide shape cannot serve (see the smem
// budget in the launcher).  BM must be a multiple of it so one Q-row
// padding granularity (quant.py MPAD) satisfies both.
constexpr int BM_NARROW = 64;
static_assert(BM % BM_NARROW == 0, "BM must be a multiple of BM_NARROW");

constexpr int BN = 64;
constexpr int HD = 256;
constexpr int KROWB = HD;              // fp8 K/Q smem row bytes (XOR-swizzled)
constexpr int VROWB8 = BN;             // fp8 V^T smem row bytes (XOR-swizzled)
constexpr int VROWB16 = HD * 2 + 16;   // bf16 V smem row bytes (BN rows, padded)
constexpr int VSMEM8 = HD * VROWB8;    // 16384
constexpr int VSMEM16 = BN * VROWB16;  // 33792
constexpr int VSMEM = (VSMEM8 > VSMEM16) ? VSMEM8 : VSMEM16;
constexpr int KSTAGES = 2;             // K double-buffered (both shapes)
// V stages.  The wide shape is fp8-PV only by construction (it traps on a
// bf16-PV head), so its V region is exactly VSTAGES_WIDE x VSMEM8, and the
// XOR swizzle's 8704 B saving is what makes VSTAGES_WIDE = 2 fit under the
// 101376 B opt-in.  The narrow shape keeps ONE stage: it may carry bf16-PV
// heads, whose V layout is 33792 B on its own.  VST below is compile-time
// per instantiation, so each kernel gets exactly one tile schedule and no
// runtime branch survives.
constexpr int VSTAGES_WIDE = 2;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LOG2_448 = 8.807354922057604f;

// ---- XOR shared-memory swizzle ----------------------------------------
// The fp8 Q/K/V^T rows used to carry 16 B of padding purely to break bank
// conflicts (a 256 B or 64 B row stride puts 8 consecutive rows on the same
// banks).  Padding costs 8704 B of the ~99 KB budget at BM=128, which is
// exactly what stands between this kernel and a second V stage, so the rows
// are power-of-two again and the conflicts are broken by permuting the
// 16-byte atoms WITHIN each row instead.
//
// Shared memory is 32 banks x 4 B = 128 B wide.  A region whose row is R
// bytes packs 128/R rows into one bank window (R <= 128), so a 16 B access
// lands on window column  (row % (128/R)) * (R/16) + atom, taken mod 8.  An
// ldmatrix group is 8 CONSECUTIVE rows read at ONE atom index, so
// conflict-free means those 8 accesses must cover the window's 8 columns
// once each.  XOR-ing the atom index with the row bits that do not already
// move the access inside its window does exactly that:
//
//   R = 256 (Q, K): 16 atoms/row, each row spans 2 windows, so the column
//                   is atom % 8 and the whole row index is free -> ^ (row & 7).
//   R =  64 (V^T8): 4 atoms/row, 2 rows per window, so the column is
//                   (row & 1) * 4 + atom -- bit 0 of row already splits the
//                   window in half, and the permutation must come from the
//                   NEXT two bits -> ^ ((row >> 1) & 3).  The naive
//                   ^ (row & 3) repeats every 4 rows while the group spans
//                   8, so rows 0 and 4 collide and a 2-way conflict remains.
//
// Both maps are bijections on one row's atoms, so every cp.async loop still
// writes each row exactly once, and every byte reaches the same mma register
// it did before: this is address-only, hence bit-exact.  Every cp16 write
// and every ldmatrix read of these three regions goes through them.
static_assert(KROWB == 256, "swz_qk's (row & 7) mask assumes 16 atoms/row");
static_assert(VROWB8 == 64, "swz_v8's ((row>>1) & 3) mask assumes 4 atoms/row");
__device__ __forceinline__ int swz_qk(int row, int col) {
  return row * KROWB + (((col >> 4) ^ (row & 7)) << 4);
}
__device__ __forceinline__ int swz_v8(int row, int col) {
  return row * VROWB8 + (((col >> 4) ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
               "r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
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
__device__ __forceinline__ void mma_s8(int32_t& c0, int32_t& c1, int32_t& c2,
                                       int32_t& c3, uint32_t a0, uint32_t a1,
                                       uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
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
// bytes b0..b3 = e4m3(f0..f3); cvt packs operand a into the HIGH byte
__device__ __forceinline__ uint32_t pack_fp8x4(float f0, float f1, float f2, float f3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(f1), "f"(f0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(f3), "f"(f2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
__device__ __forceinline__ float fast_exp2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y;
}

// Q8 [H, M, HD] fp8, K8 [KVH, ntmax*BN, HD] fp8,
// VT8 [KVH, ntmax, HD, BN] fp8 (SIGMA-permuted V^T; may be null),
// VB16 [KVH, ntmax*BN, HD] bf16 (fallback-head V; may be null),
// O [H, M, HD] bf16, qscale [H, M] (PER ROW), kscale [KVH, ntmax], vscale [KVH],
// pv8_mask [H] (1 = fp8 PV, 0 = bf16 PV).
// Bottom-right causal: q row r attends kv cols <= prefix + r.
template <bool QKI8, int BMT>
__global__ void __launch_bounds__(BMT / 16 * 32)
fp8_prefill_attn_kernel(const uint8_t* __restrict__ Q8,
                        const uint8_t* __restrict__ K8,
                        const uint8_t* __restrict__ VT8,
                        const __nv_bfloat16* __restrict__ VB16,
                        __nv_bfloat16* __restrict__ O,
                        const float* __restrict__ qscale,
                        const float* __restrict__ kscale,
                        const float* __restrict__ vscale,
                        const float* __restrict__ vlog2r,
                        const float* __restrict__ vinvr,
                        const float* __restrict__ vmean,
                        const uint8_t* __restrict__ pv8_mask,
                        int M, int N, int prefix, int ntmax, int grp,
                        float* __restrict__ LSE) {
  constexpr int NWT = BMT / 16;        // warps in this CTA (16 Q rows each)
  static_assert(BMT == NWT * 16, "BMT must be a multiple of 16");
  // V stages for THIS instantiation (see VSTAGES_WIDE): 2 on the wide,
  // fp8-PV-only shape; 1 on the narrow one, which may carry bf16-PV heads.
  constexpr int VST = (BMT == BM_NARROW) ? 1 : VSTAGES_WIDE;
  extern __shared__ unsigned char smem[];
  unsigned char* q_sm = smem;                                  // BMT x KROWB
  unsigned char* k_sm = q_sm + BMT * KROWB;                    // KSTAGES x BN x KROWB
  unsigned char* v_sm = k_sm + KSTAGES * BN * KROWB;           // VST x VSMEM8 / VSMEM

  const int qh = blockIdx.y, kvh = qh / grp;
  const bool pv8 = pv8_mask[qh] != 0;
  // The wide tile is sized for the fp8 V staging only; a bf16-PV CTA here
  // means the host's all_pv8 flag disagreed with the mask (smem overrun
  // territory).  Trap loudly instead of corrupting shared memory.
  if (BMT != BM_NARROW && !pv8) { __trap(); }
  const uint8_t* Qh = Q8 + (size_t)qh * M * HD;
  __nv_bfloat16* Oh = O + (size_t)qh * M * HD;
  const uint8_t* Kh = K8 + (size_t)kvh * ntmax * BN * HD;
  const uint8_t* VTh = pv8 ? VT8 + (size_t)kvh * ntmax * HD * BN : nullptr;
  const __nv_bfloat16* VBh = pv8 ? nullptr : VB16 + (size_t)kvh * ntmax * BN * HD;

  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int m0 = blockIdx.x * BMT;
  // per-ROW Q scales: each thread's two fixed rows
  const int qrow = m0 + warp * 16 + lane / 4;
  const float q_s0 = qscale[(size_t)qh * M + qrow];
  const float q_s1 = qscale[(size_t)qh * M + qrow + 8];

  // load Q tile
  for (int i = tid; i < BMT * HD / 16; i += NWT * 32) {
    int row = i / (HD / 16), col = (i % (HD / 16)) * 16;
    cp16(q_sm + swz_qk(row, col), Qh + (size_t)(m0 + row) * HD + col);
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();

  // causal extent for this CTA
  int hi = prefix + m0 + BMT;
  if (hi > N) hi = N;
  const int ntiles = (hi + BN - 1) / BN;

  auto prefetch_k = [&](int t) {
    if (t < ntiles) {
      const int n0 = t * BN;
      unsigned char* kd = k_sm + (t % KSTAGES) * BN * KROWB;
      for (int i = tid; i < BN * HD / 16; i += NWT * 32) {
        int row = i / (HD / 16), col = (i % (HD / 16)) * 16;
        int gr = n0 + row;
        if (gr < N) cp16(kd + swz_qk(row, col), Kh + (size_t)gr * HD + col);
      }
    }
    cp_commit();
  };
  // Mirrors prefetch_k's tail behaviour: past the last tile it copies
  // nothing but STILL commits an (empty) group.  The VST == 2 schedule
  // prefetches V one tile ahead, so it calls this with t == ntiles, and the
  // wait depth below is only uniform if the commit COUNT is.
  auto prefetch_v = [&](int t) {
    if (t < ntiles) {
      if (pv8) {
        // VT tile t: HD rows x BN bytes, contiguous (builder zero-pads past N)
        const uint8_t* src = VTh + (size_t)t * HD * BN;
        unsigned char* vd = v_sm + (t % VST) * VSMEM8;
        for (int i = tid; i < HD * BN / 16; i += NWT * 32) {
          int row = i / (BN / 16), col = (i % (BN / 16)) * 16;
          cp16(vd + swz_v8(row, col), src + row * BN + col);
        }
      } else {
        // bf16-PV reaches here only on the narrow shape, where VST == 1
        const int n0 = t * BN;
        for (int i = tid; i < BN * HD * 2 / 16; i += NWT * 32) {
          int row = i / (HD * 2 / 16), col = (i % (HD * 2 / 16)) * 16;
          int gr = n0 + row;
          if (gr < N) {
            cp16(v_sm + row * VROWB16 + col,
                 (const unsigned char*)(VBh + (size_t)gr * HD) + col);
          } else {
            // beyond-N rows: stale smem could decode as bf16 NaN and poison
            // the masked-P (0 x NaN) accumulation -- zero-fill explicitly
            *reinterpret_cast<uint4*>(v_sm + row * VROWB16 + col) =
                make_uint4(0, 0, 0, 0);
          }
        }
      }
    }
    cp_commit();
  };

  // ---- tile pipeline ----------------------------------------------------
  // Two schedules, selected at COMPILE time by VST.
  //
  // VST == 2 (wide shape, fp8-PV only).  Both of tile t+1's copies are
  // issued at the TOP of iteration t, immediately after the barrier that
  // ends tile t-1's reads, and are waited on at the top of iteration t+1 --
  // a full iteration of latency hiding for BOTH K and V, at one cp.async
  // wait and ONE __syncthreads() per tile:
  //
  //     pre-loop         : V(0), K(0)
  //     iter t, in order : cp_wait<0>; sync; issue V(t+1), K(t+1);
  //                        S(t) on k_sm[t%2]; softmax; PV(t) on v_sm[t%2]
  //
  //   point in the schedule             pending groups, oldest -> newest
  //   --------------------------------  --------------------------------
  //   entry of iter 0 (after pre-loop)  V(0), K(0)
  //   entry of iter t, t >= 1           V(t), K(t)    [issued in iter t-1]
  //   after cp_wait<0>() in iter t      (none)
  //   after the two issues in iter t    V(t+1), K(t+1)  == "entry of t+1"
  //
  //   Exactly two groups are outstanding at every iteration entry, and they
  //   are exactly the two this iteration consumes, so the wait is a full
  //   drain and its depth never changes -- including on the last iteration,
  //   because prefetch_v(ntiles) / prefetch_k(ntiles) copy nothing but still
  //   commit.  The single barrier covers both hazard directions:
  //     RAW  it publishes every warp's cp.async arrivals for K(t) and V(t)
  //          (cp_wait retires only the calling thread's own copies).
  //     WAR  it sits after iteration t-1's S and PV, the last readers of
  //          k_sm[(t+1)%2] and v_sm[(t+1)%2] -- the two buffers the issues
  //          just below overwrite.
  //   K cannot run one tile further ahead than this: k_sm has two stages, so
  //   K(t+2) would target k_sm[t%2], which S(t) is about to read.
  //
  // VST == 1 (narrow shape, may carry bf16-PV heads).  V is single-buffered,
  // so V(t) cannot be issued until PV(t-1) has released the buffer; the
  // original two-wait schedule is kept verbatim:
  //
  //     pre-loop : K(0)
  //     iter t   : issue V(t), K(t+1); cp_wait<2>; sync; S(t) + softmax;
  //                cp_wait<1>; sync; PV(t); sync
  //
  //   i.e. [V(t), K(t+1)] in flight after the issues, with K(t) -- older
  //   than both -- retired by the cp_wait<2>.
  if (VST >= 2) {
    prefetch_v(0);
    prefetch_k(0);
  } else {
    prefetch_k(0);
  }

  // per-thread state: two row-halves (r = warp*16 + lane/4, and +8)
  float m_i[2] = {-INFINITY, -INFINITY};
  float l_i[2] = {0.f, 0.f};                        // fp8-PV: sums are x448
  float o_acc[HD / 8][4];
  #pragma unroll
  for (int d = 0; d < HD / 8; ++d) { o_acc[d][0]=o_acc[d][1]=o_acc[d][2]=o_acc[d][3]=0.f; }

  const int row_a = m0 + warp * 16 + lane / 4;      // q row (half A)
  const int colp = (lane % 4) * 2;                  // fragment col pair base
  // K2 (fruit report): (col <= prefix+row) && (col < N) is exactly
  // col <= min(prefix+row, N-1), and prefix/row_a/N never change inside the
  // tile loop -- so the whole causal test collapses to one per-kernel bound
  // per half, re-based per tile below, leaving a single compare against a
  // compile-time literal per score. Integer-exact; the predicate takes the
  // same value for every (n, j), so the masked scores are bit-identical.
  const int causal_lim[2] = {min(prefix + row_a, N - 1),
                             min(prefix + row_a + 8, N - 1)};
  const float pc_base = pv8 ? LOG2_448 : 0.f;
  // Q A-fragment address, hoisted: the swizzle needs the row index inside
  // the WHOLE q_sm tile (not inside this warp's 16-row slice), so the warp
  // offset is folded into the row rather than into a base pointer.
  const int qsm_row = warp * 16 + (lane % 16);
  const int qsm_half = (lane / 16) * 16;

  for (int t = 0; t < ntiles; ++t) {
    if (VST >= 2) {
      // retire V(t), K(t); then run tile t+1's copies under this tile
      cp_wait<0>();
      __syncthreads();
      prefetch_v(t + 1);
      prefetch_k(t + 1);
    } else {
      // groups in flight after these issues: [V(t), K(t+1)]; anything older
      // (i.e. K(t)) must complete before the S matmuls read it.
      prefetch_v(t);
      prefetch_k(t + 1);
      cp_wait<2>();
      __syncthreads();
    }
    const unsigned char* kb = k_sm + (t % KSTAGES) * BN * KROWB;
    const unsigned char* vb = v_sm + (t % VST) * VSMEM8;
    const float k_s = kscale[kvh * ntmax + t];
    // per-tile V scale fold: fp8-PV packs P as p*448*r_t (r_t = vs_t/vs_max)
    // via the exp constant; the l-sum is corrected back by 1/r_t below.
    const float log2rt = pv8 ? vlog2r[kvh * ntmax + t] : 0.f;
    const float invrt = pv8 ? vinvr[kvh * ntmax + t] : 1.f;
    const int n0 = t * BN;

    // ---- S = Q K^T for BN=64: 8 n8 tiles x 8 k32 steps ------------------
    // QKI8: int8 x int8 -> exact int32 dots (the ONLY QK error is input
    // rounding, ~0.4% rms vs e4m3's ~3% -- the SageAttention design);
    // converted to fp32 once after the k-loop. Fragment byte layout is
    // shared by all 1-byte mma operand types (s8 == e4m3 plumbing).
    float s[8][4];
    if (QKI8) {
      int32_t si[8][4];
      #pragma unroll
      for (int n = 0; n < 8; ++n) { si[n][0]=si[n][1]=si[n][2]=si[n][3]=0; }
      #pragma unroll
      for (int k32 = 0; k32 < HD / 32; ++k32) {
        uint32_t a0,a1,a2,a3;
        ldsm_x4(a0,a1,a2,a3, q_sm + swz_qk(qsm_row, k32*32 + qsm_half));
        #pragma unroll
        for (int n = 0; n < 8; n += 2) {
          uint32_t b0,b1,b2,b3;
          { int g = lane / 8, r = lane % 8;
            ldsm_x4(b0,b1,b2,b3,
                    kb + swz_qk((n + (g >> 1)) * 8 + r, k32*32 + (g & 1) * 16)); }
          mma_s8(si[n][0],si[n][1],si[n][2],si[n][3],     a0,a1,a2,a3, b0,b1);
          mma_s8(si[n+1][0],si[n+1][1],si[n+1][2],si[n+1][3], a0,a1,a2,a3, b2,b3);
        }
      }
      #pragma unroll
      for (int n = 0; n < 8; ++n)
        #pragma unroll
        for (int j = 0; j < 4; ++j) s[n][j] = (float)si[n][j];
    } else {
      #pragma unroll
      for (int n = 0; n < 8; ++n) { s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f; }
      #pragma unroll
      for (int k32 = 0; k32 < HD / 32; ++k32) {
        uint32_t a0,a1,a2,a3;
        ldsm_x4(a0,a1,a2,a3, q_sm + swz_qk(qsm_row, k32*32 + qsm_half));
        #pragma unroll
        for (int n = 0; n < 8; n += 2) {
          // x4 = B-fragments for n8 tiles n and n+1 at this k32
          uint32_t b0,b1,b2,b3;
          { int g = lane / 8, r = lane % 8;
            ldsm_x4(b0,b1,b2,b3,
                    kb + swz_qk((n + (g >> 1)) * 8 + r, k32*32 + (g & 1) * 16)); }
          mma_fp8(s[n][0],s[n][1],s[n][2],s[n][3],     a0,a1,a2,a3, b0,b1);
          mma_fp8(s[n+1][0],s[n+1][1],s[n+1][2],s[n+1][3], a0,a1,a2,a3, b2,b3);
        }
      }
    }
    const float sc_h[2] = {q_s0 * k_s, q_s1 * k_s};
    // scale + causal mask (bottom-right): kv col n0+8n+colp{,+1} <= prefix+row
    //
    // Fully-visible fast path (exact): if the tile's LAST column is visible
    // to this warp's FIRST row (min row = m0 + warp*16) and inside N, then
    // every (row, col) predicate in the warp passes and the per-score bounds
    // are dead work -- >99.9% of tile iterations at flagship geometry cross
    // no frontier.  Boundary/ragged tiles keep the original path verbatim.
    // Warp-uniform by construction (no divergence).
    //
    // DEFAULT OFF (-DFP8PA_ENABLE_FULLVIS to opt in): measured 2026-08-28
    // (interleaved single_head_fp8_32kx446k, 3 rounds), ALONE it is -3.4%,
    // but STACKED on the rescale skip below it is +1.2% (36.25 -> 36.69 ms)
    // -- once the rescale body is gone, the duplicated score loop's code
    // size/branch cost outweighs the predicate removal.  Kept because a
    // later pipeline restructure (N32 reuse, TMA) can shift that balance.
#ifdef FP8PA_ENABLE_FULLVIS
    constexpr bool kFullVis = true;
#else
    constexpr bool kFullVis = false;
#endif
    const int nlast = n0 + BN - 1;
    const bool tile_fullvis =
        kFullVis && (nlast <= prefix + m0 + warp * 16) && (nlast < N);
    float rowmax[2] = {-INFINITY, -INFINITY};
    if (tile_fullvis) {
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          int half = j / 2;                    // 0: row_a, 1: row_a+8
          float val = s[n][j] * sc_h[half];
          s[n][j] = val;
          rowmax[half] = fmaxf(rowmax[half], val);
        }
      }
    } else {
      const int rem0 = causal_lim[0] - n0 - colp;   // 2 IADD per tile,
      const int rem1 = causal_lim[1] - n0 - colp;   // then ISETP vs literal
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          int half = j / 2;                    // 0: row_a, 1: row_a+8
          float val = s[n][j] * sc_h[half];
          bool ok = (n * 8 + (j % 2)) <= (half ? rem1 : rem0);
          s[n][j] = ok ? val : -INFINITY;
          rowmax[half] = fmaxf(rowmax[half], s[n][j]);
        }
      }
    }
    // quad reduce (lanes sharing a row differ in lane%4)
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      rowmax[0] = fmaxf(rowmax[0], __shfl_xor_sync(0xffffffff, rowmax[0], off));
      rowmax[1] = fmaxf(rowmax[1], __shfl_xor_sync(0xffffffff, rowmax[1], off));
    }
    float m_new[2] = {fmaxf(m_i[0], rowmax[0]), fmaxf(m_i[1], rowmax[1])};
    float alpha[2], pc[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      alpha[h] = (m_i[h] == -INFINITY) ? 0.f
                                       : fast_exp2((m_i[h] - m_new[h]) * LOG2E);
      m_i[h] = m_new[h];
      // per-half exp constant; fp8 PV folds the 448 P scale AND the
      // per-tile V-scale ratio in here (both free: pc is per-tile anyway)
      pc[h] = (m_new[h] == -INFINITY) ? -INFINITY
                                      : pc_base + log2rt - m_new[h] * LOG2E;
    }
    // p = exp(s - m_new)  (x448 under fp8 PV); -inf s -> ex2(-inf) = 0
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        int half = j / 2;
        // K1 (fruit report): the former (pc == -inf ? 0 : ...) guard was
        // provably dead -- when pc[half] is -inf every s[n][j] for that half
        // is -inf too (the quad reduction covers all 64 columns), and
        // ex2.approx.ftz(fma(-inf, LOG2E, -inf)) = +0.0f, the same bits the
        // guard returned; and the state never occurs anyway (column 0 of
        // tile 0 is always visible, so every row max is finite from the
        // first tile). 32 FSEL/thread/tile removed.
        s[n][j] = fast_exp2(fmaf(s[n][j], LOG2E, pc[half]));
      }
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      float part = 0.f;
      #pragma unroll
      for (int n = 0; n < 8; ++n) part += s[n][h*2] + s[n][h*2+1];
      // part sums p*448*r_t (fp32, pre-fp8-rounding); undo the tile ratio
      l_i[h] = l_i[h] * alpha[h] + part * invrt;
    }
    // rescale O -- skipped (exactly) when no row owned by this warp set a
    // new running max this tile: alpha == 1.0f then, and x * 1.0f is the
    // identity, so the 128-FMUL body is dead work on the (predicted ~97%
    // of) record-free tiles.  ex2.approx(0) == 1.0f exactly, so a stable
    // max yields alpha == 1 bitwise.  A NaN alpha compares != 1 and still
    // runs the rescale, preserving NaN propagation.  Warp-uniform vote (no
    // divergence).  -DFP8PA_DISABLE_SKIP_RESCALE restores the original
    // unconditional loop for A/B.
#ifdef FP8PA_DISABLE_SKIP_RESCALE
    constexpr bool kSkipRescale = false;
#else
    constexpr bool kSkipRescale = true;
#endif
    if (!kSkipRescale ||
        __any_sync(0xffffffffu, (alpha[0] != 1.f) | (alpha[1] != 1.f))) {
      #pragma unroll
      for (int d = 0; d < HD / 8; ++d) {
        o_acc[d][0] *= alpha[0]; o_acc[d][1] *= alpha[0];
        o_acc[d][2] *= alpha[1]; o_acc[d][3] *= alpha[1];
      }
    }
    if (VST < 2) {
      // single V buffer: V(t) went out at the top of THIS iteration and is
      // only guaranteed here (K(t+1) may still be in flight)
      cp_wait<1>();
      __syncthreads();
    }
    if (pv8) {
      // ---- PV fp8: 2 k32 chunks x 32 d-tiles; A-fragments straight from s
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t pa0 = pack_fp8x4(s[4*j][0],   s[4*j][1],   s[4*j+1][0], s[4*j+1][1]);
        uint32_t pa1 = pack_fp8x4(s[4*j][2],   s[4*j][3],   s[4*j+1][2], s[4*j+1][3]);
        uint32_t pa2 = pack_fp8x4(s[4*j+2][0], s[4*j+2][1], s[4*j+3][0], s[4*j+3][1]);
        uint32_t pa3 = pack_fp8x4(s[4*j+2][2], s[4*j+2][3], s[4*j+3][2], s[4*j+3][3]);
        #pragma unroll
        for (int d = 0; d < HD / 8; d += 2) {
          // x4 = B-fragments for d-tiles d and d+1 of this chunk
          uint32_t vb0,vb1,vb2,vb3;
          { int g = lane / 8, r = lane % 8;
            ldsm_x4(vb0,vb1,vb2,vb3,
                    vb + swz_v8((d + (g >> 1)) * 8 + r, j*32 + (g & 1) * 16)); }
          mma_fp8(o_acc[d][0],o_acc[d][1],o_acc[d][2],o_acc[d][3],
                  pa0,pa1,pa2,pa3, vb0,vb1);
          mma_fp8(o_acc[d+1][0],o_acc[d+1][1],o_acc[d+1][2],o_acc[d+1][3],
                  pa0,pa1,pa2,pa3, vb2,vb3);
        }
      }
    } else {
      // ---- PV bf16: P (4 k16 chunks) x V (zero-shuffle C->A packing) ----
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t pa0 = pack_bf16(s[2*j][0],   s[2*j][1]);
        uint32_t pa1 = pack_bf16(s[2*j][2],   s[2*j][3]);
        uint32_t pa2 = pack_bf16(s[2*j+1][0], s[2*j+1][1]);
        uint32_t pa3 = pack_bf16(s[2*j+1][2], s[2*j+1][3]);
        #pragma unroll
        for (int d = 0; d < HD / 8; ++d) {
          uint32_t vb0, vb1;
          { int r = lane % 16;
            ldsm_x2_trans(vb0, vb1, vb + (j*16 + r) * VROWB16 + d * 16); }
          mma_bf16(o_acc[d][0], o_acc[d][1], o_acc[d][2], o_acc[d][3],
                   pa0, pa1, pa2, pa3, vb0, vb1);
        }
      }
    }
    // WAR for next iteration's prefetch_v(t+1) into the one V buffer; the
    // VST == 2 schedule needs no barrier here (its top-of-loop one covers
    // both buffers).
    if (VST < 2) __syncthreads();
  }

  // epilogue: l across quad, divide (fp8 PV: x448 cancels; times vscale)
  #pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    l_i[0] += __shfl_xor_sync(0xffffffff, l_i[0], off);
    l_i[1] += __shfl_xor_sync(0xffffffff, l_i[1], off);
  }
  const float onum = pv8 ? vscale[kvh] : 1.f;
  float inv_l[2] = {l_i[0] > 0.f ? onum / l_i[0] : 0.f,
                    l_i[1] > 0.f ? onum / l_i[1] : 0.f};
  if (LSE != nullptr && lane % 4 == 0) {
    // Base-2 log-sum-exp of the masked scores (the FA2/CUTLASS wrapper
    // convention): lse2 = m*log2(e) + log2(l_true).  Under fp8 PV, l_i
    // carries 448 * l_true -- the P-scale fold, accumulated PRE-rounding --
    // so subtract log2(448); bf16 PV carries l_true directly.  All four
    // lanes of a quad hold identical l/m after the xor-reduce above; one
    // writes.  A row with no unmasked column cannot occur for real rows
    // (row r always sees >= prefix+1 >= 1 columns); the l==0 guard keeps
    // padded rows finite-or--inf rather than NaN.
    const float lse_sub = pv8 ? LOG2_448 : 0.f;
    const int lr0 = m0 + warp * 16 + lane / 4;
    if (lr0 < M)
      LSE[(size_t)qh * M + lr0] = l_i[0] > 0.f
          ? fmaf(m_i[0], LOG2E, log2f(l_i[0]) - lse_sub) : -INFINITY;
    if (lr0 + 8 < M)
      LSE[(size_t)qh * M + lr0 + 8] = l_i[1] > 0.f
          ? fmaf(m_i[1], LOG2E, log2f(l_i[1]) - lse_sub) : -INFINITY;
  }
  #pragma unroll
  for (int d = 0; d < HD / 8; ++d) {
    int col = d * 8 + colp;
    int r0 = m0 + warp * 16 + lane / 4;
    // V mean add-back (fp8 PV only): O = P*(V - mean) + mean, exact since
    // the softmax weights sum to 1. Neutralizes massive-activation V
    // channels that would otherwise set the fp8 range.
    const float vm0 = pv8 ? vmean[kvh * HD + col] : 0.f;
    const float vm1 = pv8 ? vmean[kvh * HD + col + 1] : 0.f;
    if (r0 < M) {
      Oh[(size_t)r0 * HD + col]     = __float2bfloat16(fmaf(o_acc[d][0], inv_l[0], vm0));
      Oh[(size_t)r0 * HD + col + 1] = __float2bfloat16(fmaf(o_acc[d][1], inv_l[0], vm1));
    }
    if (r0 + 8 < M) {
      Oh[(size_t)(r0+8) * HD + col]     = __float2bfloat16(fmaf(o_acc[d][2], inv_l[1], vm0));
      Oh[(size_t)(r0+8) * HD + col + 1] = __float2bfloat16(fmaf(o_acc[d][3], inv_l[1], vm1));
    }
  }
}

}  // namespace

// O [H, Mpad, 256] bf16 <- attention(Q8, K8, VT8/VB16) with bottom-right
// causal masking (q row r sees kv cols <= prefix + r).  See kernel comment
// for tensor layouts.  Rows of Q8 beyond the real token count are compute
// padding: garbage in, never stored beyond Mpad, sliced off by the caller.
void fp8_prefill_attn(torch::Tensor q8, torch::Tensor k8, torch::Tensor vt8,
                      torch::Tensor vb16, torch::Tensor o, torch::Tensor qscale,
                      torch::Tensor kscale, torch::Tensor vscale,
                      torch::Tensor vlog2r, torch::Tensor vinvr,
                      torch::Tensor vmean,
                      torch::Tensor pv8_mask, int64_t n, int64_t prefix,
                      bool any_pv8, bool all_pv8, bool qk_i8,
                      c10::optional<torch::Tensor> lse_opt) {
  const bool want_lse =
      lse_opt.has_value() && lse_opt->defined() && lse_opt->numel() > 0;
  torch::Tensor lse;
  if (want_lse) {
    lse = *lse_opt;
    TORCH_CHECK(lse.is_cuda() && lse.dtype() == torch::kFloat32 &&
                    lse.is_contiguous(),
                "lse must be a contiguous fp32 CUDA tensor");
    TORCH_CHECK(lse.numel() >= (int64_t)q8.size(0) * q8.size(1),
                "lse must hold [H, M] rows");
  }
  TORCH_CHECK(q8.is_cuda() && q8.dtype() == torch::kUInt8 && q8.is_contiguous());
  TORCH_CHECK(k8.is_cuda() && k8.dtype() == torch::kUInt8 && k8.is_contiguous());
  TORCH_CHECK(o.is_cuda() && o.dtype() == torch::kBFloat16 && o.is_contiguous());
  TORCH_CHECK(qscale.dtype() == torch::kFloat32 && kscale.dtype() == torch::kFloat32);
  TORCH_CHECK(qscale.numel() >= (int64_t)q8.size(0) * q8.size(1),
              "qscale must be per-row [H, M]");
  TORCH_CHECK(vscale.dtype() == torch::kFloat32);
  TORCH_CHECK(pv8_mask.dtype() == torch::kUInt8 && pv8_mask.is_cuda());
  const int H = q8.size(0), M = q8.size(1);
  const int KVH = k8.size(0), ntmax = kscale.size(1);
  TORCH_CHECK(q8.size(2) == HD && k8.size(2) == HD, "head_dim must be 256");
  TORCH_CHECK(M % BM == 0, "M must be padded to a multiple of BM (see "
                           "quant.py MPAD)");
  TORCH_CHECK(H % KVH == 0, "q heads must be a multiple of kv heads");
  TORCH_CHECK(k8.size(1) == (int64_t)ntmax * BN, "K rows must be ntmax*64");
  // The kernel itself only needs n to fit the padded buffers; kv beyond
  // the last row's causal reach is simply never visible.
  TORCH_CHECK(n <= (int64_t)ntmax * BN, "n exceeds padded kv buffers");
  // any/all are computed host-side (the mask is static per config) so the
  // launch stays sync-free; the mask tensor itself is read per-CTA on device.
  if (any_pv8) {
    TORCH_CHECK(vt8.numel() >= (int64_t)KVH * ntmax * HD * BN,
                "VT8 workspace too small for fp8-PV heads");
    TORCH_CHECK(vlog2r.numel() >= (int64_t)KVH * ntmax &&
                vinvr.numel() >= (int64_t)KVH * ntmax &&
                vmean.numel() >= (int64_t)KVH * HD,
                "per-tile V fold arrays / vmean too small");
    TORCH_CHECK(vlog2r.dtype() == torch::kFloat32 &&
                vinvr.dtype() == torch::kFloat32 &&
                vmean.dtype() == torch::kFloat32);
  }
  if (!all_pv8) {
    TORCH_CHECK(vb16.numel() >= (int64_t)KVH * ntmax * BN * HD,
                "VB16 workspace too small for bf16-PV heads");
  }

  // ---- tile-shape selection + shared-memory budget --------------------
  // Q tile + K double-buffer are BM-proportional; the V staging buffer is
  // the union of the two PV layouts, and the bf16 one (still padded, it is
  // narrow-shape only) is 17 KB larger than the XOR-swizzled fp8 one:
  //
  //   BM=128, fp8-PV x2   : 32768 + 32768 + 2*16384 = 98304 B  fits
  //   BM=128, any bf16-PV : 32768 + 32768 +   33792 = 99328 B  would fit,
  //                         but the wide shape traps on bf16-PV by design
  //   BM= 64, any bf16-PV : 16384 + 32768 +   33792 = 82944 B  fits
  //
  // With +16 B padded rows and a single V stage those were 90112 / 103424 /
  // 86016 B; the swizzle is what buys the second V stage at BM=128.
  //
  // SM120's per-block opt-in ceiling is 101376 B.  The wide shape serves
  // only launches whose heads are ALL fp8-PV -- its V region is sized for
  // the fp8 layout alone and the kernel traps on a bf16-PV head there -- so
  // a launch carrying one runs the BM_NARROW instantiation.  Tile shape is
  // numerics-neutral (per-row softmax state, per-64-row K/Q scales, same
  // tile visit order; the extra tiles a wider CTA sweeps for its early
  // rows are fully causal-masked, contributing p == 0 and alpha == 1
  // exactly), verified bit-exact BM=64 vs BM=128 -- so this only ever
  // changes speed.
  constexpr size_t SMEM_KSTAGE = (size_t)KSTAGES * BN * KROWB;
  constexpr size_t SMEM_WIDE =
      (size_t)BM * KROWB + SMEM_KSTAGE + (size_t)VSTAGES_WIDE * VSMEM8;
  constexpr size_t SMEM_NARROW =
      (size_t)BM_NARROW * KROWB + SMEM_KSTAGE + VSMEM;
  const size_t optin =
      at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  const bool wide = all_pv8 && BM != BM_NARROW && SMEM_WIDE <= optin;
  const size_t smem = wide ? SMEM_WIDE : SMEM_NARROW;
  TORCH_CHECK(smem <= optin, "fp8 prefill attn needs ", smem,
              " B of shared memory, device opt-in limit is ", optin, " B");

  const int bm_rt = wide ? BM : BM_NARROW;
  auto kfn = qk_i8 ? (wide ? fp8_prefill_attn_kernel<true, BM>
                           : fp8_prefill_attn_kernel<true, BM_NARROW>)
                   : (wide ? fp8_prefill_attn_kernel<false, BM>
                           : fp8_prefill_attn_kernel<false, BM_NARROW>);
  // one opt-in per (qk_i8, shape); each pair has a fixed smem size
  static bool smem_configured[4] = {false, false, false, false};
  bool& configured = smem_configured[(qk_i8 ? 2 : 0) + (wide ? 1 : 0)];
  if (!configured) {
    cudaError_t st = cudaFuncSetAttribute(
        (const void*)kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(st == cudaSuccess, "cudaFuncSetAttribute(dynamic smem = ",
                smem, " B) failed: ", cudaGetErrorString(st));
    configured = true;
  }
  dim3 grid(M / bm_rt, H);
  auto stream = at::cuda::getCurrentCUDAStream();
  kfn<<<grid, bm_rt / 16 * 32, smem, stream>>>(
      q8.data_ptr<uint8_t>(), k8.data_ptr<uint8_t>(),
      any_pv8 ? vt8.data_ptr<uint8_t>() : nullptr,
      all_pv8 ? nullptr
              : reinterpret_cast<__nv_bfloat16*>(vb16.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
      qscale.data_ptr<float>(), kscale.data_ptr<float>(),
      vscale.data_ptr<float>(),
      any_pv8 ? vlog2r.data_ptr<float>() : nullptr,
      any_pv8 ? vinvr.data_ptr<float>() : nullptr,
      any_pv8 ? vmean.data_ptr<float>() : nullptr,
      pv8_mask.data_ptr<uint8_t>(),
      M, (int)n, (int)prefix, ntmax, H / KVH,
      want_lse ? lse.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_prefill_attn", &fp8_prefill_attn,
        "Fused low-precision prefill attention (mma.sync family), per-head PV "
        "precision. Optional trailing `lse` (contiguous fp32, >= [H, M]) "
        "receives the BASE-2 log-sum-exp per (head, row); omit for the "
        "output-only v1 behaviour.",
        py::arg("q8"), py::arg("k8"), py::arg("vt8"), py::arg("vb16"),
        py::arg("o"), py::arg("qscale"), py::arg("kscale"), py::arg("vscale"),
        py::arg("vlog2r"), py::arg("vinvr"), py::arg("vmean"),
        py::arg("pv8_mask"), py::arg("n"), py::arg("prefix"),
        py::arg("any_pv8"), py::arg("all_pv8"), py::arg("qk_i8"),
        py::arg("lse") = py::none());
}
