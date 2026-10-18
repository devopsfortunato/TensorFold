// Flash Next's routed NVFP4 experts for prompt chunks (ours; work/research/R2-prefill.md W-B). nvfp4_expert_kernel
// (fn_nvfp4_experts.cu, the copy of TensorFold's src/tensorfold/cuda/nvfp4/experts.cu) gives a warp one plan item of
// at most 16 pairs and one 32-column block: per 16-row pass it streams that block's weights and decodes every weight
// fragment and scale again. Here a warp's pass takes T row tiles of 16 (an item of any size, the plan's tile chosen
// by the caller): each 32-input group's weight words are loaded, decoded and their e4m3 scales converted once and
// applied to all T tiles, so weight reads and the decode's ALU work drop by T.
//
// The bits are nvfp4_expert_kernel's: a pair row's sums never depend on the other rows of its mma (each row of an
// m16n8k16 is its own dot), and every output value goes through the same steps in the same order: for each 32-input
// group g, for each 16-input half h: p = mma(a, b, 0) (a fresh zero accumulator), acc = fmaf(p, e4m3 scale, acc);
// then acc *= the (expert, matrix) fp32 scale and the same epilogue (experts.cuh). The weight words, scale bytes and
// decode are the same (the block layout of experts.py `pack`). Built with the same flags (-O3).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "experts.cuh"

namespace tf_fn_experts_prompt {

constexpr int BLOCK4 = 36;  // uint4 a (32 columns, 32 inputs) block: 32 lanes' code words, then 4 of scales

// nvfp4_expert_kernel's decode (fn_nvfp4_experts.cu): two E2M1 codes of `w` at bits s and s + 16 -> a bf16 pair.
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
  const uint32_t v = w >> s;
  const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// experts.cuh's mma with a fresh zero accumulator, its result in registers of its own: the same instruction (C = +0.0),
// without the copy of the A fragment into the result registers the "+f" form costs when A is reused for 8 mmas
__device__ __forceinline__ void mma0(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

// A warp's stage: the lane's weight words and scales for one 32-input group (as nvfp4_expert_kernel's Stage), and the
// inputs of T row tiles (rows gq and gq + 8 of each).
template <int M, int T>
struct Stage {
  uint4 w[M];
  uint4 s[M];
  uint4 xq[T][2];  // the A fragment of each row tile and half: {a0, a1, a2, a3} (rows gq, gq + 8)
};

template <int M, int T, int MS = M>
__device__ __forceinline__ void load_stage(Stage<M, T>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* const (&x0)[T], const __nv_bfloat16* const (&x1)[T],
                                           const bool (&v0)[T], const bool (&v1)[T]) {
  const uint4* b = blk + (size_t)g * (MS * BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc(b + m * BLOCK4 + lane);
    st.s[m] = ld_nc(b + m * BLOCK4 + 32 + t);
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int r = 0; r < T; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint2 xa = v0[r] ? __ldg(reinterpret_cast<const uint2*>(x0[r] + k0 + 16 * h)) : zero;
      const uint2 xb = v1[r] ? __ldg(reinterpret_cast<const uint2*>(x1[r] + k0 + 16 * h)) : zero;
      st.xq[r][h] = make_uint4(xa.x, xb.x, xa.y, xb.y);
    }
}

// nvfp4_expert_kernel's compute_stage with each weight fragment decoded and each scale converted once for T row
// tiles; every row tile's values take the same mma, fmaf and order as there.
template <int M, int T>
__device__ __forceinline__ void compute_stage(float (&acc)[T][M][1][NTW][4], const Stage<M, T>& st,
                                              const bool (&hi)[T]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const int sh = (j & 1) * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
        for (int r = 0; r < T; ++r) {
          const uint32_t a0 = st.xq[r][h].x, a1 = st.xq[r][h].y, a2 = st.xq[r][h].z, a3 = st.xq[r][h].w;
          float p[4];
          mma0(p, a0, a1, a2, a3, b0, b1);
          float(&a)[4] = acc[r][m][0][j];
          a[0] = fmaf(p[0], s0, a[0]);
          a[1] = fmaf(p[1], s1, a[1]);
          if (hi[r]) {
            a[2] = fmaf(p[2], s0, a[2]);
            a[3] = fmaf(p[3], s1, a[3]);
          }
        }
      }
    }
}

// X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N, limit, skip: nvfp4_expert_kernel's arguments
// and loop; a warp's pass takes 16 * T pairs of its item instead of 16 (items of any size).
// EPI 5: gate/up's SwiGLU from a gate pass: `gate` [pairs, N] holds bf16(gate sums x scale) (EPI 3 of the gate
// matrix), this pass the up matrix: swiglu(gate, up) as epilogue<2> computes it (its first step is bf16(g), which the
// stored gate already is), the same bf16 store.
__device__ __forceinline__ void epilogue_up(const float (&acc)[1][1][NTW][4], void* out, const __nv_bfloat16* gate,
                                            int N, int col0, int pr0, int pr1, bool v0, bool v1, float limit) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const int col = col0 + 8 * j;
      const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(gate + row + col);
      const float o0 = swiglu(__low2float(g2), acc[0][0][j][2 * h], limit);
      const float o1 = swiglu(__high2float(g2), acc[0][0][j][2 * h + 1], limit);
      *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) = __floats2bfloat162_rn(o0, o1);
    }
  }
}

// MS: matrices stored a block group (gate/up: 2), MOFF: the first one this pass reads (a one-matrix pass of gate/up)
template <int M, int EPI, int T, int WARPS, int MS = M, int MOFF = 0>
__device__ __forceinline__ void prompt_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                            int NB, const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N,
                                            float limit, int skip, const __nv_bfloat16* __restrict__ gate = nullptr) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (MS * BLOCK4) + MOFF * BLOCK4;
    for (int p0 = 0; p0 < cnt; p0 += 16 * T) {
      bool v0[T], v1[T];
      int pr0[T], pr1[T];
      const __nv_bfloat16* x0[T];
      const __nv_bfloat16* x1[T];
#pragma unroll
      for (int r = 0; r < T; ++r) {
        const int r0 = p0 + 16 * r;
        v0[r] = r0 + gq < cnt;
        v1[r] = r0 + gq + 8 < cnt;
        pr0[r] = v0[r] ? __ldg(members + first + r0 + gq) : 0;
        pr1[r] = v1[r] ? __ldg(members + first + r0 + gq + 8) : 0;
        const int x0r = slots ? pr0[r] / slots : pr0[r], x1r = slots ? pr1[r] / slots : pr1[r];
        x0[r] = X + (size_t)x0r * x_stride;
        x1[r] = X + (size_t)x1r * x_stride;
      }
      float acc[T][M][1][NTW][4];
#pragma unroll
      for (int r = 0; r < T; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[r][m][0][j][0] = acc[r][m][0][j][1] = acc[r][m][0][j][2] = acc[r][m][0][j][3] = 0.f;
      constexpr int D = 2;
      Stage<M, T> st[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < KG) load_stage<M, T, MS>(st[d], blk, d, lane, t, x0, x1, v0, v1);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            compute_stage<M, T>(acc, st[d], v1);
            if (g + D < KG) load_stage<M, T, MS>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < T; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float gs = __ldg(scale + e * MS + MOFF + m);
#pragma unroll
          for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[r][m][0][j][q] *= gs;
        }
        if constexpr (EPI == 5) {
          epilogue_up(acc[r], out, gate, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
        } else {
          epilogue<EPI, M, 1>(acc[r], 0, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
        }
      }
    }
  }
}

}  // namespace tf_fn_experts_prompt

// gate/up SwiGLU (M 2, epilogue 2), down fp32 (M 1, epilogue 0) and bf16 (3); 4 warps a CTA, each warp T row tiles
// of 16 a pass. extern "C" names: fn_prompt4_<gu|f32|b16>_t<T>.
#define TF_PROMPT4(NAME, M, EPI, T) TF_PROMPT4B(NAME, M, EPI, T, )
#define TF_PROMPT4B(NAME, M, EPI, T, MINB)                                                                         \
  extern "C" __global__ void __launch_bounds__(128 MINB)                                                         \
      NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,             \
           const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,                        \
           const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,         \
           float limit, int skip) {                                                                              \
    tf_fn_experts_prompt::prompt_body<M, EPI, T, 4>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, \
                                                     out, N, limit, skip);                                       \
  }
TF_PROMPT4(fn_prompt4_gu_t2, 2, 2, 2)
TF_PROMPT4(fn_prompt4_gu_t3, 2, 2, 3)
TF_PROMPT4(fn_prompt4_gu_t4, 2, 2, 4)
TF_PROMPT4(fn_prompt4_f32_t2, 1, 0, 2)
TF_PROMPT4(fn_prompt4_f32_t4, 1, 0, 4)
TF_PROMPT4(fn_prompt4_b16_t2, 1, 3, 2)
TF_PROMPT4(fn_prompt4_b16_t4, 1, 3, 4)
#define COMMA4 , 4
// the same with at least 4 CTAs an SM (<= 128 registers): more warps in flight, the same arithmetic
TF_PROMPT4B(fn_prompt4_gu_t2o, 2, 2, 2, COMMA4)
TF_PROMPT4B(fn_prompt4_f32_t2o, 1, 0, 2, COMMA4)
TF_PROMPT4B(fn_prompt4_b16_t2o, 1, 3, 2, COMMA4)

// gate/up in two one-matrix passes at 4 CTAs an SM (half the accumulators): the gate's bf16(sums x scale) into `out`
// of the first, then the up pass's SwiGLU reading them (`gate`), so the act bytes are epilogue<2>'s
extern "C" __global__ void __launch_bounds__(128, 4)
    fn_prompt4_gate_t2o(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  tf_fn_experts_prompt::prompt_body<1, 3, 2, 4, 2, 0>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out,
                                                      N, limit, skip);
}
extern "C" __global__ void __launch_bounds__(128, 4)
    fn_prompt4_up_t2o(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                      const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                      float limit, int skip, const __nv_bfloat16* __restrict__ gate) {
  tf_fn_experts_prompt::prompt_body<1, 5, 2, 4, 2, 1>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out,
                                                      N, limit, skip, gate);
}
