// SPDX-License-Identifier: Apache-2.0
// Toolchain gate for the unity-scale MXFP8 PV experiment.
//
// Question: does the DEPLOYED CUDA 12.9 ptxas accept the SM120a block-scaled
// MXFP8 mma (kind::mxf8f6f4.block_scale) and emit the intended SASS form?
// The external ~2x issue-rate evidence was collected on CUDA 13.3; nothing
// proceeds until 12.9 passes this gate on the RTX PRO 6000.
//
// Compile-only probe, three candidate PTX spellings (-DV=1|2|3):
//   V1: shape.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
//   V2: same without the explicit .scale_vec::1X (1X is the documented default)
//   V3: kind/block_scale qualifiers before the shape
//
// nvcc -arch=sm_120a -DV=1 -cubin -o probe.cubin mx_syntax_probe.cu
// cuobjdump -sass probe.cubin | grep -i mma

#include <cstdio>

__global__ void probe(const unsigned* A, const unsigned* B, unsigned sa,
                      unsigned sb, float* D) {
  const int t = threadIdx.x & 31;
  unsigned a0 = A[t], a1 = A[t + 32], a2 = A[t + 64], a3 = A[t + 96];
  unsigned b0 = B[t], b1 = B[t + 32];
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#if V == 1
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
      "scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {0,0}, {%11}, {0,0};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa),
        "r"(sb));
#elif V == 2
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
      "f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {0,0}, {%11}, {0,0};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa),
        "r"(sb));
#elif V == 3
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {0,0}, {%11}, {0,0};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa),
        "r"(sb));
#else
#error "define -DV=1|2|3"
#endif
  D[t * 4 + 0] = d0;
  D[t * 4 + 1] = d1;
  D[t * 4 + 2] = d2;
  D[t * 4 + 3] = d3;
}

int main() {
  printf("mx_syntax_probe V%d: compiled\n", (int)V);
  return 0;
}
