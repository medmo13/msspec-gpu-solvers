// FP16-storage complex LU for the V2 solver (step 1d). Blocked half-precision-storage LU, ported from the earlier GPU version.
//
// The n x n complex matrix lives in FP16 (one __half2 = 4 bytes per element, column major), i.e. half
// the memory of the FP32 matrix. The factorisation is a blocked right-looking LU with partial pivoting:
//   * each panel (rows k..n, w = min(nb, n-k) columns) is converted to FP32 and factorised by
//     cusolverDnCgetrf (exact partial pivoting over all remaining rows), then rounded back to FP16;
//   * row interchanges are applied to the trailing columns only (left columns keep "delayed"
//     pivots; the solve applies them to the right-hand side panel by panel);
//   * U12 = L11^-1 A12 (cublasCtrsm, FP32) on column tiles of width tc, converted FP16 -> FP32 -> FP16;
//   * the trailing update A22 -= L21 U12 is ONE real FP16 tensor-core GEMM (cublasGemmEx, CUDA_R_16F
//     inputs, FP32 accumulate) on the interleaved real/imag storage (see k_build_abig, k_build_bbig).
// cuBLAS has no complex FP16 GEMM, so FP16 is the *storage* precision and FP32 the arithmetic.
// The caller refines the solution with an FP64 residual (msspec_v2_gpu.cu, shared with the other methods).
//
// Kept from the earlier GPU version, with its defaults and no environment switches: nb = min(1024, n), tc = default_tc(n),
// TF32 math mode at the start of factor, real FP16 trailing update.
// Dropped from the earlier GPU version: the look-ahead path (second stream and panel buffers), the complex TF32 trailing
// update (MSSPEC_GPU_HALF_HGEMM=0 path of the earlier GPU version), the MSSPEC_GPU_* knobs, the profiling timers, the debug
// printing and the max-element probe.
// The stream and the cuBLAS / cuSOLVER handles are owned by the caller and are not created or destroyed here.
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstddef>

namespace half_lu {

// FP32 complex tile (src, leading dimension lds) -> FP16 matrix (dst, leading dimension ldd); m rows, w columns
__global__ void k_f2h(const cuComplex *src, size_t lds, __half2 *dst, size_t ldd, int m, int w) {
  const size_t total = static_cast<size_t>(m) * w;
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; t < total;
       t += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t i = t % m, j = t / m;
    const cuComplex v = src[i + j * lds];
    dst[i + j * ldd] = __floats2half2_rn(v.x, v.y);
  }
}

// FP16 matrix -> FP32 complex tile
__global__ void k_h2f(const __half2 *src, size_t lds, cuComplex *dst, size_t ldd, int m, int w) {
  const size_t total = static_cast<size_t>(m) * w;
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; t < total;
       t += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t i = t % m, j = t / m;
    const __half2 v = src[i + j * lds];
    dst[i + j * ldd] = make_cuComplex(__low2float(v), __high2float(v));
  }
}

// one thread per column: apply the cnt row interchanges (r0+i <-> piv[r0+i]) in order
__global__ void k_swap_cols(__half2 *A, size_t ld, int col0, int ncols, const int *piv, int r0, int cnt) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ncols) return;
  __half2 *col = A + static_cast<size_t>(col0 + j) * ld;
  for (int i = 0; i < cnt; ++i) {
    const int a = r0 + i, b = piv[a];
    if (a != b) {
      const __half2 t = col[a];
      col[a] = col[b];
      col[b] = t;
    }
  }
}

__global__ void k_swap_rhs(cuComplex *B, size_t ldb, int nrhs, const int *piv, int r0, int cnt) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nrhs) return;
  cuComplex *col = B + static_cast<size_t>(j) * ldb;
  for (int i = 0; i < cnt; ++i) {
    const int a = r0 + i, b = piv[a];
    if (a != b) {
      const cuComplex t = col[a];
      col[a] = col[b];
      col[b] = t;
    }
  }
}

// Real-FP16 trailing update: a complex product C -= L U is ONE real GEMM on the interleaved storage:
// C' (2mr x wc, FP16 view of the half2 matrix) -= [A1 A2] (2mr x 2w) * [Br; Bi] (2w x wc),
// with A1 = L (re,im interleaved), A2 = (-im,re) interleaved, Br/Bi the real/imag planes of U.
__global__ void k_build_abig(const cuComplex *panel, size_t ldp, int row0, int mr, int w, __half2 *abig) {
  const size_t total = static_cast<size_t>(mr) * w;
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; t < total;
       t += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t i = t % mr, j = t / mr;
    const cuComplex v = panel[row0 + i + j * ldp];
    abig[i + j * mr] = __floats2half2_rn(v.x, v.y);
    abig[i + (static_cast<size_t>(w) + j) * mr] = __floats2half2_rn(-v.y, v.x);
  }
}

__global__ void k_build_bbig(const cuComplex *u, int w, int wc, __half *bbig) {
  const size_t total = static_cast<size_t>(w) * wc;
  for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; t < total;
       t += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t r = t % w, j = t / w;
    const cuComplex v = u[r + j * w];
    bbig[r + j * 2 * static_cast<size_t>(w)] = __float2half_rn(v.x);
    bbig[w + r + j * 2 * static_cast<size_t>(w)] = __float2half_rn(v.y);
  }
}

// cuSOLVER panel pivots are 1-based and local to the panel; store them as absolute 0-based rows
__global__ void k_abs_piv(const int *ppiv, int *piv, int k, int cnt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) piv[k + i] = k + ppiv[i] - 1;
}

struct HalfLU {
  int n = 0, nb = 0, tc = 0, lwork = 0;
  cudaStream_t stream = nullptr;     // all work is queued on this stream (not owned)
  cusolverDnHandle_t solver = nullptr;   // not owned
  cublasHandle_t blas = nullptr;         // not owned
  __half2 *A = nullptr;              // n x n FP16 storage, column major, 4 n^2 bytes
  cuComplex *panel = nullptr, *wu = nullptr, *wc = nullptr, *cwork = nullptr;
  __half *bbig = nullptr;
  int *piv = nullptr, *ppiv = nullptr, *info = nullptr;
};

// wider column tiles are faster up to n ~ 21 000; above that the extra workspace would not fit in 4 GiB next to the FP16 matrix
inline int default_tc(int n) { return n <= 21000 ? 2048 : 1024; }

inline void release(HalfLU &h) {
  if (h.A) cudaFree(h.A);
  if (h.panel) cudaFree(h.panel);
  if (h.wu) cudaFree(h.wu);
  if (h.wc) cudaFree(h.wc);
  if (h.bbig) cudaFree(h.bbig);
  if (h.cwork) cudaFree(h.cwork);
  if (h.piv) cudaFree(h.piv);
  if (h.ppiv) cudaFree(h.ppiv);
  if (h.info) cudaFree(h.info);
  h = HalfLU{};
}

// Allocates the FP16 matrix and the workspace on the given stream and handles (sizes and defaults of the earlier GPU version).
inline bool alloc(HalfLU &h, int n, cudaStream_t stream, cusolverDnHandle_t solver, cublasHandle_t blas) {
  h = HalfLU{};
  h.n = n;
  h.nb = std::min(1024, n);
  h.tc = std::max(h.nb, default_tc(n));
  h.stream = stream;
  h.solver = solver;
  h.blas = blas;
  const size_t N = static_cast<size_t>(n);
  bool ok = cudaMalloc(reinterpret_cast<void **>(&h.A), 4 * N * N) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&h.panel), 8 * N * h.nb) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&h.wu), 8 * static_cast<size_t>(h.nb) * h.tc) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&h.wc), 8 * N * h.tc) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&h.bbig), 4 * static_cast<size_t>(h.nb) * h.tc) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&h.piv), 4 * N) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&h.ppiv), 4 * static_cast<size_t>(h.nb)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void **>(&h.info), 4) == cudaSuccess;
  if (ok) ok = cusolverDnSetStream(h.solver, h.stream) == CUSOLVER_STATUS_SUCCESS &&
               cublasSetStream(h.blas, h.stream) == CUBLAS_STATUS_SUCCESS;
  if (ok) ok = cusolverDnCgetrf_bufferSize(h.solver, n, h.nb, h.panel, n, &h.lwork) == CUSOLVER_STATUS_SUCCESS && h.lwork >= 0;
  if (ok) ok = cudaMalloc(reinterpret_cast<void **>(&h.cwork), 8 * static_cast<size_t>(std::max(h.lwork, 1))) == cudaSuccess;
  if (!ok) {
    cudaGetLastError();
    release(h);
    return false;
  }
  return true;
}

// FP32 device tile (n rows, cols columns, leading dimension n) -> FP16 matrix columns [first_col, first_col+cols)
inline void upload_tile(HalfLU &h, const cuComplex *d_tile, int first_col, int cols) {
  const size_t total = static_cast<size_t>(h.n) * cols;
  const int blocks = static_cast<int>(std::min<size_t>(4096, (total + 255) / 256));
  k_f2h<<<blocks, 256, 0, h.stream>>>(d_tile, h.n, h.A + static_cast<size_t>(first_col) * h.n, h.n, h.n, cols);
}

inline int blocks_for(size_t count) { return static_cast<int>(std::min<size_t>(4096, (count + 255) / 256)); }

// Blocked LU of the FP16 matrix in place. Returns false on any cuSOLVER/cuBLAS status, panel info != 0, or CUDA error.
inline bool factor(HalfLU &h) {
  const int n = h.n;
  const cuComplex one = make_cuComplex(1.f, 0.f);
  const float fm1 = -1.f, fone = 1.f;
  cublasSetMathMode(h.blas, CUBLAS_TF32_TENSOR_OP_MATH);
  for (int k = 0; k < n; k += h.nb) {
    const int m = n - k, w = std::min(h.nb, m);
    __half2 *Akk = h.A + static_cast<size_t>(k) + static_cast<size_t>(k) * n;
    k_h2f<<<blocks_for(static_cast<size_t>(m) * w), 256, 0, h.stream>>>(Akk, n, h.panel, m, m, w);
    if (cusolverDnCgetrf(h.solver, m, w, h.panel, m, h.cwork, h.ppiv, h.info) != CUSOLVER_STATUS_SUCCESS) return false;
    int info = -1;
    if (cudaMemcpyAsync(&info, h.info, sizeof(int), cudaMemcpyDeviceToHost, h.stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(h.stream) != cudaSuccess || info != 0) return false;
    k_abs_piv<<<(w + 255) / 256, 256, 0, h.stream>>>(h.ppiv, h.piv, k, w);
    k_f2h<<<blocks_for(static_cast<size_t>(m) * w), 256, 0, h.stream>>>(h.panel, m, Akk, n, m, w);
    const int mr = m - w;
    if (mr > 0)    // wc doubles as the [A1 A2] buffer: (m-w) x 2w half2 <= 8 n nb bytes <= 8 n tc
      k_build_abig<<<blocks_for(static_cast<size_t>(mr) * w), 256, 0, h.stream>>>(h.panel, m, w, mr, w, reinterpret_cast<__half2 *>(h.wc));
    for (int c = k + w; c < n; c += h.tc) {
      const int wc = std::min(h.tc, n - c);
      __half2 *A12 = h.A + static_cast<size_t>(k) + static_cast<size_t>(c) * n;
      k_swap_cols<<<(wc + 127) / 128, 128, 0, h.stream>>>(h.A, n, c, wc, h.piv, k, w);
      k_h2f<<<blocks_for(static_cast<size_t>(w) * wc), 256, 0, h.stream>>>(A12, n, h.wu, w, w, wc);
      if (cublasCtrsm(h.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, w, wc, &one,
                      h.panel, m, h.wu, w) != CUBLAS_STATUS_SUCCESS) return false;
      k_f2h<<<blocks_for(static_cast<size_t>(w) * wc), 256, 0, h.stream>>>(h.wu, w, A12, n, w, wc);
      if (mr > 0) {
        __half2 *A22 = h.A + static_cast<size_t>(k + w) + static_cast<size_t>(c) * n;
        k_build_bbig<<<blocks_for(static_cast<size_t>(w) * wc), 256, 0, h.stream>>>(h.wu, w, wc, h.bbig);
        cublasSetMathMode(h.blas, CUBLAS_DEFAULT_MATH);
        if (cublasGemmEx(h.blas, CUBLAS_OP_N, CUBLAS_OP_N, 2 * mr, wc, 2 * w, &fm1, h.wc, CUDA_R_16F, 2 * mr, h.bbig,
                         CUDA_R_16F, 2 * w, &fone, A22, CUDA_R_16F, 2 * n, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
      }
    }
  }
  const cudaError_t e1 = cudaGetLastError();
  const cudaError_t e2 = cudaStreamSynchronize(h.stream);
  return e1 == cudaSuccess && e2 == cudaSuccess;
}

// B (n x nrhs, leading dimension n, FP32 complex, device) <- A^-1 B using the FP16 factors. Plain FP32 solves.
inline bool solve(HalfLU &h, cuComplex *B, int nrhs) {
  const int n = h.n;
  const cuComplex one = make_cuComplex(1.f, 0.f), minus_one = make_cuComplex(-1.f, 0.f);
  cublasSetMathMode(h.blas, CUBLAS_DEFAULT_MATH);
  for (int k = 0; k < n; k += h.nb) {                      // forward: b <- L_k^-1 P_k b
    const int m = n - k, w = std::min(h.nb, m);
    k_swap_rhs<<<(nrhs + 63) / 64, 64, 0, h.stream>>>(B, n, nrhs, h.piv, k, w);
    const __half2 *Akk = h.A + static_cast<size_t>(k) + static_cast<size_t>(k) * n;
    k_h2f<<<blocks_for(static_cast<size_t>(m) * w), 256, 0, h.stream>>>(Akk, n, h.panel, m, m, w);
    if (cublasCtrsm(h.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, w, nrhs, &one,
                    h.panel, m, B + k, n) != CUBLAS_STATUS_SUCCESS) return false;
    if (m > w &&
        cublasCgemm(h.blas, CUBLAS_OP_N, CUBLAS_OP_N, m - w, nrhs, w, &minus_one, h.panel + w, m, B + k, n, &one,
                    B + k + w, n) != CUBLAS_STATUS_SUCCESS) return false;
  }
  const int last = ((n - 1) / h.nb) * h.nb;
  for (int k = last; k >= 0; k -= h.nb) {                  // backward: U
    const int w = std::min(h.nb, n - k), rest = n - k - w;
    if (rest > 0) {
      // U12 is w x rest; wc holds n*tc complex >= nb*rest as long as tc >= nb*rest/n, so chunk by tc*n/nb columns
      const int chunk = static_cast<int>(std::min<size_t>(rest, (static_cast<size_t>(h.tc) * n) / h.nb));
      for (int c = 0; c < rest; c += chunk) {
        const int cw = std::min(chunk, rest - c);
        const __half2 *U12 = h.A + static_cast<size_t>(k) + static_cast<size_t>(k + w + c) * n;
        k_h2f<<<blocks_for(static_cast<size_t>(w) * cw), 256, 0, h.stream>>>(U12, n, h.wc, w, w, cw);
        if (cublasCgemm(h.blas, CUBLAS_OP_N, CUBLAS_OP_N, w, nrhs, cw, &minus_one, h.wc, w, B + k + w + c, n, &one,
                        B + k, n) != CUBLAS_STATUS_SUCCESS) return false;
      }
    }
    const __half2 *Akk = h.A + static_cast<size_t>(k) + static_cast<size_t>(k) * n;
    k_h2f<<<blocks_for(static_cast<size_t>(w) * w), 256, 0, h.stream>>>(Akk, n, h.panel, w, w, w);
    if (cublasCtrsm(h.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, w, nrhs, &one,
                    h.panel, w, B + k, n) != CUBLAS_STATUS_SUCCESS) return false;
  }
  const cudaError_t e1 = cudaGetLastError();
  const cudaError_t e2 = cudaStreamSynchronize(h.stream);
  return e1 == cudaSuccess && e2 == cudaSuccess;
}

}  // namespace half_lu
