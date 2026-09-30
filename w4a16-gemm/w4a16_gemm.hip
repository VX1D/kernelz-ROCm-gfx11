// out[M,N] (bf16) = x[M,K] (bf16) @ dequant(w)[N,K]^T on RDNA3 (gfx11) WMMA.
// w: symmetric int4 stored as q+8 nibbles (low nibble = even k), fp16 scale per 128-wide group.
// Weights are dequantized in LDS as bf16(q * float(bf16(scale))) with round-to-nearest-even.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

namespace {

constexpr int BN = 128, WM = 4, WN = 2, BK = 32, PAD = 8, GROUP = 128;

__device__ __forceinline__ uint32_t rne_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    return u + 0x7FFFu + ((u >> 16) & 1u);
}

__device__ __forceinline__ float bf16_round(float f) {
    return __uint_as_float(rne_bits(f) & 0xFFFF0000u);
}

template <int BM>
__global__ void __launch_bounds__(WM * WN * 32)
w4a16_gemm_kernel(const uint16_t * __restrict__ x, const uint8_t * __restrict__ w,
                  const _Float16 * __restrict__ scales, uint16_t * __restrict__ out,
                  int M, int K, int ldx, int ldo) {
    constexpr int NT = WM * WN * 32, LDK = BK + PAD;
    constexpr int FI = BM / WM / 16, FJ = BN / WN / 16;
    // x: 16-byte chunks of 8 bf16; w: 8-byte chunks of 16 int4.
    constexpr int XCH = BM * BK / 8 / NT, WCH = BN * BK / 16 / NT;
    static_assert(XCH >= 1 && WCH >= 1 && BM * BK / 8 % NT == 0 && BN * BK / 16 % NT == 0, "tile split");

    __shared__ __attribute__((aligned(16))) uint16_t xs[2][BM * LDK];
    __shared__ __attribute__((aligned(16))) uint16_t ws[2][BN * LDK];

    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int wm = wave / WN, wn = wave % WN;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int groups = K / GROUP;

    const uint16_t * xp[XCH];
    bool xok[XCH];
    int xso[XCH];
#pragma unroll
    for (int c = 0; c < XCH; ++c) {
        const int id = tid + c * NT, r = id / 4, cc = id % 4;
        xok[c] = m0 + r < M;
        xp[c] = x + (size_t) (m0 + (xok[c] ? r : 0)) * ldx + cc * 8;
        xso[c] = r * LDK + cc * 8;
    }
    const uint8_t * wp[WCH];
    const _Float16 * sp[WCH];
    int wso[WCH];
#pragma unroll
    for (int c = 0; c < WCH; ++c) {
        const int id = tid + c * NT, r = id / 2, cc = id % 2;
        wp[c] = w + (size_t) (n0 + r) * (K / 2) + cc * 8;
        sp[c] = scales + (size_t) (n0 + r) * groups;
        wso[c] = r * LDK + cc * 16;
    }

    uint4 xreg[XCH];
    uint2 wreg[WCH];
    float sreg[WCH];

    auto load = [&](int kt) {
        const int k = kt * BK;
#pragma unroll
        for (int c = 0; c < XCH; ++c) {
            xreg[c] = xok[c] ? *(const uint4 *) (xp[c] + k) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int c = 0; c < WCH; ++c) {
            wreg[c] = *(const uint2 *) (wp[c] + k / 2);
            sreg[c] = (float) sp[c][k / GROUP];
        }
    };

    auto store = [&](int buf) {
#pragma unroll
        for (int c = 0; c < XCH; ++c) {
            *(uint4 *) &xs[buf][xso[c]] = xreg[c];
        }
#pragma unroll
        for (int c = 0; c < WCH; ++c) {
            const float sc = bf16_round(sreg[c]), bias = -8.0f * sc;
            const uint32_t words[2] = {wreg[c].x, wreg[c].y};
            uint32_t packed[8];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t lo = words[i] & 0x0F0F0F0Fu, hi = (words[i] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float fl = __builtin_fmaf((float) ((lo >> (8 * j)) & 0xFFu), sc, bias);
                    const float fh = __builtin_fmaf((float) ((hi >> (8 * j)) & 0xFFu), sc, bias);
                    packed[i * 4 + j] = __builtin_amdgcn_perm(rne_bits(fh), rne_bits(fl), 0x07060302u);
                }
            }
            uint4 * wd = (uint4 *) &ws[buf][wso[c]];
            wd[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            wd[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
    };

    f32x8 acc[FI][FJ];
#pragma unroll
    for (int i = 0; i < FI; ++i)
#pragma unroll
        for (int j = 0; j < FJ; ++j)
            acc[i][j] = (f32x8) {0, 0, 0, 0, 0, 0, 0, 0};

    const int fr = lane & 15;
    auto compute = [&](int buf, int ks) {
        bf16x16 a[FI], b[FJ];
#pragma unroll
        for (int i = 0; i < FI; ++i)
            a[i] = *(const bf16x16 *) &xs[buf][(wm * FI * 16 + i * 16 + fr) * LDK + ks];
#pragma unroll
        for (int j = 0; j < FJ; ++j)
            b[j] = *(const bf16x16 *) &ws[buf][(wn * FJ * 16 + j * 16 + fr) * LDK + ks];
#pragma unroll
        for (int i = 0; i < FI; ++i)
#pragma unroll
            for (int j = 0; j < FJ; ++j)
                acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b[j], acc[i][j]);
    };

    const int KT = K / BK;
    load(0);
    store(0);
    __syncthreads();

    for (int kt = 0; kt < KT - 1; ++kt) {
        const int buf = kt & 1;
        load(kt + 1);
        compute(buf, 0);
        // Dequantize the next stage between the two WMMA halves so VALU work interleaves with WMMA.
        store(buf ^ 1);
        compute(buf, 16);
#pragma unroll
        for (int g = 0; g < FI * FJ; ++g) {
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x002, 1, 0);
        }
        __syncthreads();
    }
    compute((KT - 1) & 1, 0);
    compute((KT - 1) & 1, 16);

    const int hi = lane >> 4;
#pragma unroll
    for (int i = 0; i < FI; ++i) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int m = m0 + wm * FI * 16 + i * 16 + 2 * e + hi;
            if (m < M) {
#pragma unroll
                for (int j = 0; j < FJ; ++j) {
                    out[(size_t) m * ldo + n0 + wn * FJ * 16 + j * 16 + fr] = (uint16_t) (rne_bits(acc[i][j][e]) >> 16);
                }
            }
        }
    }
}

} // namespace

// Returns 0 on success, 1 for unsupported shapes/strides, 2 for a launch error.
extern "C" int w4a16_gemm(const void * x, const void * w, const void * scales, void * out,
                          int M, int N, int K, int ldx, int ldo, void * stream) {
    if (M <= 0 || N % BN != 0 || K % GROUP != 0 || ldx % 8 != 0) {
        return 1;
    }
    constexpr int SMALL_BM = 128, LARGE_BM = 256;
    const int bm = M < 1024 ? SMALL_BM : LARGE_BM;
    dim3 grid(N / BN, (M + bm - 1) / bm);
    if (bm == SMALL_BM) {
        hipLaunchKernelGGL(w4a16_gemm_kernel<SMALL_BM>, grid, dim3(WM * WN * 32), 0, (hipStream_t) stream,
                           (const uint16_t *) x, (const uint8_t *) w, (const _Float16 *) scales,
                           (uint16_t *) out, M, K, ldx, ldo);
    } else {
        hipLaunchKernelGGL(w4a16_gemm_kernel<LARGE_BM>, grid, dim3(WM * WN * 32), 0, (hipStream_t) stream,
                           (const uint16_t *) x, (const uint8_t *) w, (const _Float16 *) scales,
                           (uint16_t *) out, M, K, ldx, ldo);
    }
    return hipGetLastError() == hipSuccess ? 0 : 2;
}
