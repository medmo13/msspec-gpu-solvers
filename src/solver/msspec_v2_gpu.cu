/* MSSPEC V2 GPU solver, step 1d: FP32 matrix on the device with FP32 LU by cuSOLVER
 * or by MAGMA 2.9, FP16-stored matrix with the blocked LU (half_lu.cuh), and
 * (the *_refined methods only) resident FP64 refinement.
 *
 * Methods (value of MSSPEC_V2_GPU_METHOD):
 *   cusolver_refined    FP32 LU, then FP64 residual refinement on the GPU
 *   cusolver_unrefined  FP32 LU only, result cast to FP64
 *   magma_refined       as cusolver_refined, LU by magma_cgetrf_gpu / magma_cgetrs_gpu
 *   magma_unrefined     as cusolver_unrefined, LU by MAGMA
 *   own_refined         FP16-stored matrix (4 n^2 bytes), half_lu LU, then the same refinement
 *   own_unrefined       as own_refined, without refinement
 *   magma_host_refined  FP32 matrix in pinned host memory (8 n^2 bytes), LU by MAGMA magma_cgetrf
 *                       (CPU interface, hybrid), solve by LAPACK cgetrs_ on the host, then the same refinement
 *   magma_host_unrefined as magma_host_refined, without refinement
 * Any other value is refused by msspec_v2_gpu_begin_ (exit code 3).
 *
 * Entry points, called from V2_GPU_INV_MAT (v2_gpu_inv_mat.f). Arguments are
 * passed by reference. The method string has a trailing hidden size_t length.
 *   msspec_v2_gpu_begin_    row descriptors, Gaunt cache; allocates the n x n device matrix
 *   msspec_v2_gpu_column_   pair functions of one column atom; assembles its columns on the device
 *   msspec_v2_gpu_solve_    Cgetrf, Cgetrs, optional refinement; FP64 solution out
 *
 * The n x n matrix is never built on the host and never copied host to device.
 * The small per-pair data (YPAIR, HPAIR, ATTPAIR), the descriptors and the
 * Gaunt cache are prepared on the host in Fortran, as in the earlier GPU version, and uploaded.
 * The refinement residual regenerates each column tile on the device from the
 * pair data kept from column time, as in the earlier GPU version (ped_assembly.cu).
 *
 * Early start: a constructor (run when the library is loaded) starts a pthread
 * that sets the device and creates the stream and the cuBLAS and cuSOLVER
 * handles, when MSSPEC_V2_GPU_METHOD names a known method. begin joins it and
 * reports the wait as init_wait_s.
 *
 * Switches for the ladder (own_* only where stated):
 *   MSSPEC_V2_EARLY_START=0  no early start thread; GPU init happens in begin (init_wait_s).
 *   MSSPEC_V2_GPU_BUILD=0    MS is built on the CPU by the Fortran stock loop (FP64) and
 *                            uploaded by msspec_v2_gpu_upload_ (FP32 copy, FP16 storage).
 *                            own_unrefined only; the *_refined methods exit 3 with
 *                            cpu_build_refined_unsupported, the other methods with cpu_build_own_only.
 *   The proof line ends with gpu_build=, early_start= and copy_s= (host to device copy time).
 *
 * No CPU fallback. Any failure prints one line
 *   MSSPEC_V2_GPU_FAILED <reason> method=<m> n=<n>
 * and exits with code 3. Each solve prints one proof line
 *   MSSPEC_V2_GPU method=<m> n=<n> nrhs=<k> init_wait_s=<> build_s=<> lu_s=<> refine_iters=<> refine_s=<>
 */
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <magma_v2.h>

#include "half_lu.cuh"

#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define V2_TOLERANCE 1.0e-14
#define V2_MAX_CORRECTIONS 5
#define V2_CHECK(call, what)                                  \
  do {                                                        \
    cudaError_t v2_status_ = (call);                          \
    if (v2_status_ != cudaSuccess)                            \
      v2_fail_cuda((what), v2_status_, v2_n);                 \
  } while (0)

enum { V2_NONE = 0, V2_REFINED = 1, V2_UNREFINED = 2,
       V2_MAGMA_REFINED = 3, V2_MAGMA_UNREFINED = 4,
       V2_OWN_REFINED = 5, V2_OWN_UNREFINED = 6,
       V2_MAGMA_HOST_REFINED = 7, V2_MAGMA_HOST_UNREFINED = 8 };

/* LAPACK cgetrs (OpenBLAS/reference LAPACK, Fortran linkage). trans_len is the hidden length of trans. */
extern "C" void cgetrs_(const char *trans, const int *n, const int *nrhs,
                        const cuComplex *a, const int *lda, const int *ipiv,
                        cuComplex *b, const int *ldb, int *info,
                        size_t trans_len);

struct V2Tile {
  int col0, ncols, absorber;
  cuComplex *ypair, *hpair, *attpair;   /* device; own_*: the shared v2_ypair/v2_hpair/v2_attpair */
  cuComplex *hy, *hh, *ha;              /* own_* only: host FP32 pair data of this tile */
  int *col_l, *col_m, *col_ind;
};

static int v2_method = V2_NONE;
static char v2_method_name[64] = "";
static int v2_thread_started = 0;
static int v2_init_done = 0;
static int v2_init_status = 0;
static char v2_init_what[120] = "";
static pthread_t v2_init_thread;
static cudaStream_t v2_stream = NULL;
static cublasHandle_t v2_blas = NULL;
static cusolverDnHandle_t v2_solver = NULL;
static int v2_use_magma = 0;
static magma_queue_t v2_magma_queue = NULL;
static int *v2_ipiv_host = NULL;
static cuComplex *v2_host_matrix = NULL;  /* magma_host_*: FP32 matrix in pinned host memory */
static half_lu::HalfLU v2_hl;          /* own_* only: FP16 matrix and blocked LU workspace */
static int v2_ready = 0;               /* 1 after a successful begin (replaces the v2_matrix test) */

static int v2_parse_method(const char *name) {
  if (strcmp(name, "cusolver_refined") == 0) return V2_REFINED;
  if (strcmp(name, "cusolver_unrefined") == 0) return V2_UNREFINED;
  if (strcmp(name, "magma_refined") == 0) return V2_MAGMA_REFINED;
  if (strcmp(name, "magma_unrefined") == 0) return V2_MAGMA_UNREFINED;
  if (strcmp(name, "own_refined") == 0) return V2_OWN_REFINED;
  if (strcmp(name, "own_unrefined") == 0) return V2_OWN_UNREFINED;
  if (strcmp(name, "magma_host_refined") == 0) return V2_MAGMA_HOST_REFINED;
  if (strcmp(name, "magma_host_unrefined") == 0) return V2_MAGMA_HOST_UNREFINED;
  return V2_NONE;
}

static int v2_is_magma_host(int method) {
  return method == V2_MAGMA_HOST_REFINED || method == V2_MAGMA_HOST_UNREFINED;
}

static int v2_is_magma(int method) {
  return method == V2_MAGMA_REFINED || method == V2_MAGMA_UNREFINED ||
         v2_is_magma_host(method);
}

static int v2_is_own(int method) {
  return method == V2_OWN_REFINED || method == V2_OWN_UNREFINED;
}

/* Methods whose tiles share one device pair set and keep their pair data on the host (plan of the earlier GPU version):
 * own_* and magma_host_*. v2_is_own stays the half_lu test. */
static int v2_shared_pair_set(int method) {
  return v2_is_own(method) || v2_is_magma_host(method);
}

static int v2_refines(void) {
  return v2_method == V2_REFINED || v2_method == V2_MAGMA_REFINED ||
         v2_method == V2_OWN_REFINED || v2_method == V2_MAGMA_HOST_REFINED;
}

static int v2_n = 0, v2_natoms = 0, v2_nl = 0, v2_ngaunt = 0, v2_maxlocal = 0;
static cuComplex *v2_matrix = NULL, *v2_work = NULL, *v2_tile = NULL;
static int *v2_ipiv = NULL, *v2_info = NULL;
static int *v2_row_atom = NULL, *v2_row_l = NULL, *v2_row_m = NULL;
static int *v2_row_gntcol = NULL;
static cuComplex *v2_row_tlk = NULL;
static float *v2_gnt = NULL;
static struct V2Tile *v2_tiles = NULL;
static int v2_ntiles = 0, v2_tiles_capacity = 0;
static cuComplex *v2_ypair = NULL, *v2_hpair = NULL, *v2_attpair = NULL;  /* own_*: one device pair set */
static size_t v2_ny = 0, v2_nh = 0, v2_na = 0;
static double v2_init_wait = 0.0, v2_build_s = 0.0;
static int v2_cpu_build_requested = 0;  /* MSSPEC_V2_GPU_BUILD=0 (read in begin) */
static int v2_cpu_build = 0;            /* 1 after msspec_v2_gpu_upload_ (the path that ran) */
static double v2_copy_s = 0.0;          /* host to device copy time in the CPU build */
static double v2_begin_end = 0.0;       /* monotonic time at the end of begin */

/* 1 when the environment variable is set and equal to value. */
static int v2_env_is(const char *name, const char *value) {
  const char *setting = getenv(name);
  return setting != NULL && strcmp(setting, value) == 0;
}

static double v2_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static void v2_fail(const char *reason, int n) {
  fprintf(stderr, "MSSPEC_V2_GPU_FAILED %s method=%s n=%d\n", reason,
          v2_method_name, n);
  fflush(stderr);
  exit(3);
}

/* Device memory budget (stderr lines starting MSSPEC_V2_GPU_BUDGET). A line with
 * free_drop is the free device memory lost since the previous budget line. */
static size_t v2_budget_last_free = 0;

static void v2_budget_req(const char *what, size_t bytes) {
  fprintf(stderr, "MSSPEC_V2_GPU_BUDGET alloc %s bytes=%zu\n", what, bytes);
  fflush(stderr);
}

static void v2_budget(const char *what, size_t bytes) {
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
    cudaGetLastError();
    fprintf(stderr, "MSSPEC_V2_GPU_BUDGET %s bytes=%zu meminfo=unavailable\n",
            what, bytes);
    fflush(stderr);
    return;
  }
  long long drop = 0;
  if (v2_budget_last_free != 0)
    drop = (long long)v2_budget_last_free - (long long)free_b;
  fprintf(stderr,
          "MSSPEC_V2_GPU_BUDGET %s bytes=%zu free=%zu total=%zu free_drop=%lld ntiles=%d\n",
          what, bytes, free_b, total_b, drop, v2_ntiles);
  fflush(stderr);
  v2_budget_last_free = free_b;
}

static void v2_fail_cuda(const char *what, cudaError_t status, int n) {
  char reason[200];
  snprintf(reason, sizeof reason, "%s:%s", what, cudaGetErrorName(status));
  v2_budget(what, 0);
  v2_fail(reason, n);
}

static void v2_fail_code(const char *what, int code, int n) {
  char reason[160];
  snprintf(reason, sizeof reason, "%s:%d", what, code);
  v2_fail(reason, n);
}

/* A half_lu call returned false: name the pending CUDA error if there is one. */
static void v2_fail_half(const char *what, int n) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) v2_fail_cuda(what, e, n);
  v2_fail(what, n);
}

static int v2_init_gpu(void) {
  cudaError_t e = cudaSetDevice(0);
  if (e != cudaSuccess) {
    snprintf(v2_init_what, sizeof v2_init_what, "init:cudaSetDevice:%s",
             cudaGetErrorName(e));
    return 1;
  }
  e = cudaFree(0);
  if (e != cudaSuccess) {
    snprintf(v2_init_what, sizeof v2_init_what, "init:cudaFree0:%s",
             cudaGetErrorName(e));
    return 1;
  }
  e = cudaStreamCreate(&v2_stream);
  if (e != cudaSuccess) {
    snprintf(v2_init_what, sizeof v2_init_what, "init:cudaStreamCreate:%s",
             cudaGetErrorName(e));
    return 1;
  }
  if (cublasCreate(&v2_blas) != CUBLAS_STATUS_SUCCESS ||
      cublasSetStream(v2_blas, v2_stream) != CUBLAS_STATUS_SUCCESS) {
    snprintf(v2_init_what, sizeof v2_init_what, "init:cublas");
    return 1;
  }
  if (cusolverDnCreate(&v2_solver) != CUSOLVER_STATUS_SUCCESS ||
      cusolverDnSetStream(v2_solver, v2_stream) != CUSOLVER_STATUS_SUCCESS) {
    snprintf(v2_init_what, sizeof v2_init_what, "init:cusolver");
    return 1;
  }
  if (v2_use_magma) {
    if (magma_init() != MAGMA_SUCCESS) {
      snprintf(v2_init_what, sizeof v2_init_what, "init:magma_init");
      return 1;
    }
    magma_queue_create_from_cuda(0, v2_stream, v2_blas, NULL,
                                 &v2_magma_queue);
    if (v2_magma_queue == NULL) {
      snprintf(v2_init_what, sizeof v2_init_what, "init:magma_queue");
      return 1;
    }
  }
  return 0;
}

static void *v2_init_main(void *unused) {
  (void)unused;
  v2_init_status = v2_init_gpu();
  return NULL;
}

/* Early start. Runs when the library is loaded. Does nothing when the
 * variable is unset, empty or names an unknown method (begin refuses those). */
__attribute__((constructor)) static void msspec_v2_gpu_constructor(void) {
  const char *setting = getenv("MSSPEC_V2_GPU_METHOD");
  if (setting == NULL || setting[0] == '\0') return;
  const int method = v2_parse_method(setting);
  if (method == V2_NONE) return;
  if (v2_env_is("MSSPEC_V2_EARLY_START", "0")) return;
  v2_use_magma = v2_is_magma(method);
  if (pthread_create(&v2_init_thread, NULL, v2_init_main, NULL) == 0)
    v2_thread_started = 1;
}

static void v2_release(void) {
  int t;
  v2_ready = 0;
  half_lu::release(v2_hl);
  if (v2_host_matrix) cudaFreeHost(v2_host_matrix);
  v2_host_matrix = NULL;
  if (v2_matrix) cudaFree(v2_matrix);
  if (v2_work) cudaFree(v2_work);
  if (v2_tile) cudaFree(v2_tile);
  if (v2_ipiv) cudaFree(v2_ipiv);
  if (v2_info) cudaFree(v2_info);
  free(v2_ipiv_host);
  if (v2_row_atom) cudaFree(v2_row_atom);
  if (v2_row_l) cudaFree(v2_row_l);
  if (v2_row_m) cudaFree(v2_row_m);
  if (v2_row_gntcol) cudaFree(v2_row_gntcol);
  if (v2_row_tlk) cudaFree(v2_row_tlk);
  if (v2_gnt) cudaFree(v2_gnt);
  if (v2_ypair) cudaFree(v2_ypair);
  if (v2_hpair) cudaFree(v2_hpair);
  if (v2_attpair) cudaFree(v2_attpair);
  v2_ypair = NULL; v2_hpair = NULL; v2_attpair = NULL;
  v2_ny = 0; v2_nh = 0; v2_na = 0;
  for (t = 0; t < v2_ntiles; ++t) {
    if (v2_tiles[t].hy != NULL) {          /* own_*: device pair set is shared, freed above */
      free(v2_tiles[t].hy);
      free(v2_tiles[t].hh);
      free(v2_tiles[t].ha);
    } else {
      cudaFree(v2_tiles[t].ypair);
      cudaFree(v2_tiles[t].hpair);
      cudaFree(v2_tiles[t].attpair);
    }
    cudaFree(v2_tiles[t].col_l);
    cudaFree(v2_tiles[t].col_m);
    cudaFree(v2_tiles[t].col_ind);
  }
  v2_matrix = NULL; v2_work = NULL; v2_tile = NULL;
  v2_ipiv = NULL; v2_info = NULL; v2_ipiv_host = NULL;
  v2_row_atom = NULL; v2_row_l = NULL; v2_row_m = NULL;
  v2_row_gntcol = NULL; v2_row_tlk = NULL; v2_gnt = NULL;
  v2_ntiles = 0;
}

/* Assemble columns col0 .. col0+ncols-1 of the matrix into out (leading
 * dimension n). Same formula as the CPU branch of INV_MAT_MS, in FP32, as in
 * earlier GPU version, ped_assembly.cu:assemble_atom_columns. */
__global__ void v2_assemble_kernel(
    int n, int ncols, int col0, int absorber, int nl, int ngaunt, int maxlocal,
    const int *row_atom, const int *row_l, const int *row_m,
    const int *row_gntcol, const cuComplex *row_tlk,
    const int *col_l, const int *col_m, const int *col_ind,
    const cuComplex *ypair, const cuComplex *hpair,
    const cuComplex *attpair, const float *gnt, cuComplex *out) {
  const size_t element = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= (size_t)n * (size_t)ncols) return;
  const int row = (int)(element % (size_t)n);
  const int column = (int)(element / (size_t)n);
  const int atom = row_atom[row];
  if (atom == absorber) {
    out[element] = make_cuComplex(row == col0 + column ? 1.0f : 0.0f, 0.0f);
    return;
  }
  const int lcol = col_l[column];
  const int mcol = col_m[column];
  const int ind = col_ind[column];
  const int lrow = row_l[row];
  const int mrow = row_m[row];
  const int gntcol = row_gntcol[row];
  const int l_stride = 2 * nl + 1;
  const int m_stride = 4 * nl + 1;
  const size_t y_atom_offset = (size_t)(atom - 1) * l_stride * m_stride;
  const size_t h_atom_offset = (size_t)(atom - 1) * l_stride;
  const int m = mcol - mrow;
  const int abs_m = abs(m);
  float sum_real = 0.0f;
  float sum_imag = 0.0f;
  for (int l = abs(lrow - lcol); l <= lrow + lcol; l += 2) {
    if (abs_m > l) continue;
    const cuComplex h = hpair[h_atom_offset + l];
    const cuComplex y = ypair[y_atom_offset +
                              (size_t)(m + 2 * nl) * l_stride + l];
    const size_t g_index = (size_t)l +
        (size_t)(ngaunt + 1) * ((size_t)(ind - 1) +
                                (size_t)maxlocal * (size_t)(gntcol - 1));
    const float coefficient = gnt[g_index];
    const float product_real = h.x * y.x - h.y * y.y;
    const float product_imag = h.x * y.y + h.y * y.x;
    float phase_real = product_real;
    float phase_imag = product_imag;
    if (l % 4 == 1) {
      phase_real = -product_imag;
      phase_imag = product_real;
    } else if (l % 4 == 2) {
      phase_real = -product_real;
      phase_imag = -product_imag;
    } else if (l % 4 == 3) {
      phase_real = product_imag;
      phase_imag = -product_real;
    }
    sum_real += phase_real * coefficient;
    sum_imag += phase_imag * coefficient;
  }
  const float scale = attpair[atom - 1].x * 12.566370614359172f;
  const float scattering_real = -sum_imag * scale;
  const float scattering_imag = sum_real * scale;
  const cuComplex tlk = row_tlk[row];
  float result_real = -(tlk.x * scattering_real - tlk.y * scattering_imag);
  float result_imag = -(tlk.x * scattering_imag + tlk.y * scattering_real);
  if (row == col0 + column) result_real += 1.0f;
  out[element] = make_cuComplex(result_real, result_imag);
}

/* r -= A_tile x_tile in FP64 (subtract_operator_tile of the earlier GPU version). row_norms collects
 * the FP32 tile row sums |re|+|im| (the matrix norm used by the stop test). */
__global__ void v2_residual_kernel(int n, int nrhs, int col0, int ncols,
                                   const cuComplex *tile,
                                   const cuDoubleComplex *x,
                                   cuDoubleComplex *res, double *row_norms) {
  const size_t element = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= (size_t)n * (size_t)nrhs) return;
  const int row = (int)(element % (size_t)n);
  const int rhs = (int)(element / (size_t)n);
  double real = 0.0, imag = 0.0, norm = 0.0;
  for (int c = 0; c < ncols; ++c) {
    const cuComplex a = tile[row + (size_t)c * n];
    const cuDoubleComplex xv = x[(size_t)(col0 + c) + (size_t)rhs * n];
    real += (double)a.x * xv.x - (double)a.y * xv.y;
    imag += (double)a.x * xv.y + (double)a.y * xv.x;
    if (rhs == 0) norm += fabs((double)a.x) + fabs((double)a.y);
  }
  res[element].x -= real;
  res[element].y -= imag;
  if (rhs == 0) row_norms[row] += norm;
}

/* FP32 LU of v2_matrix in place. cuSOLVER takes and returns pivots and info on
 * the device; MAGMA takes host pivots and host info, and its legacy routines
 * wait for the device before they return (cudaDeviceSynchronize below). */
static void v2_factor(int N) {
  if (v2_is_own(v2_method)) {
    if (!half_lu::factor(v2_hl)) v2_fail_half("half_lu_factor", N);
    return;
  }
  if (v2_is_magma_host(v2_method)) {
    /* magma_cgetrf: CPU interface, hybrid; MAGMA moves blocks to the GPU itself. Pivots are 1-based (LAPACK). */
    magma_int_t info = 0;
    if (magma_cgetrf(N, N, v2_host_matrix, N, v2_ipiv_host, &info) !=
        MAGMA_SUCCESS)
      v2_fail_code("magma_host_getrf_status", 0, N);
    V2_CHECK(cudaDeviceSynchronize(), "magma_host_getrf_sync");
    if (info != 0) v2_fail_code("magma_host_getrf_info", info, N);
    return;
  }
  if (v2_use_magma) {
    magma_int_t info = 0;
    if (magma_cgetrf_gpu(N, N, v2_matrix, N, v2_ipiv_host, &info) !=
        MAGMA_SUCCESS)
      v2_fail_code("magma_getrf_status", 0, N);
    V2_CHECK(cudaDeviceSynchronize(), "magma_getrf_sync");
    if (info != 0) v2_fail_code("magma_getrf_info", info, N);
    return;
  }
  if (cusolverDnCgetrf(v2_solver, N, N, v2_matrix, N, v2_work, v2_ipiv,
                       v2_info) != CUSOLVER_STATUS_SUCCESS)
    v2_fail_code("cusolver_getrf_status", 0, N);
  V2_CHECK(cudaStreamSynchronize(v2_stream), "getrf_sync");
  int info = 0;
  V2_CHECK(cudaMemcpy(&info, v2_info, sizeof(int), cudaMemcpyDeviceToHost),
           "memcpy_getrf_info");
  if (info != 0) v2_fail_code("cusolver_getrf_info", info, N);
}

/* Solve v2_matrix X = rhs in place (FP32, rhs is N x R on the device).
 * pass is 0 for the first solve and 1 for a refinement correction; it only
 * names the failure. */
static void v2_solve32(int N, int R, cuComplex *rhs, int pass) {
  if (v2_is_own(v2_method)) {
    if (!half_lu::solve(v2_hl, rhs, R))
      v2_fail_half(pass ? "half_lu_solve_correction" : "half_lu_solve", N);
    return;
  }
  if (v2_is_magma_host(v2_method)) {
    /* rhs is host memory. The factors are the host matrix from magma_cgetrf; LAPACK cgetrs_ (host) solves
     * with them. Host solve for every n: the factors (8 n^2 bytes) do not fit the device at large n. */
    char trans = 'N';
    int n32 = N, r32 = R, lda = N, ldb = N, info32 = 0;
    cgetrs_(&trans, &n32, &r32, v2_host_matrix, &lda, v2_ipiv_host, rhs, &ldb,
            &info32, (size_t)1);
    if (info32 != 0) v2_fail_code("host_getrs_info", info32, N);
    return;
  }
  if (v2_use_magma) {
    magma_int_t info = 0;
    if (magma_cgetrs_gpu(MagmaNoTrans, N, R, v2_matrix, N, v2_ipiv_host, rhs,
                         N, &info) != MAGMA_SUCCESS)
      v2_fail_code("magma_getrs_status", pass, N);
    V2_CHECK(cudaDeviceSynchronize(), "magma_getrs_sync");
    if (info != 0) v2_fail_code("magma_getrs_info", info, N);
    return;
  }
  if (cusolverDnCgetrs(v2_solver, CUBLAS_OP_N, N, R, v2_matrix, N, v2_ipiv,
                       rhs, N, v2_info) != CUSOLVER_STATUS_SUCCESS)
    v2_fail_code("cusolver_getrs_status", pass, N);
  V2_CHECK(cudaStreamSynchronize(v2_stream), "getrs_sync");
  int info = 0;
  V2_CHECK(cudaMemcpy(&info, v2_info, sizeof(int), cudaMemcpyDeviceToHost),
           "memcpy_getrs_info");
  if (info != 0) v2_fail_code("cusolver_getrs_info", info, N);
}

extern "C" void msspec_v2_gpu_begin_(
    const char *method, int *n, int *natoms, int *nl, int *ngaunt,
    int *maxlocal, const double *gnt_cache, const int *row_atom,
    const int *row_l, const int *row_m, const int *row_gntcol,
    const cuDoubleComplex *row_tlk, size_t method_len) {
  const size_t copy_len = method_len < sizeof(v2_method_name) - 1
                              ? method_len : sizeof(v2_method_name) - 1;
  memcpy(v2_method_name, method, copy_len);
  v2_method_name[copy_len] = '\0';
  v2_method = v2_parse_method(v2_method_name);
  if (v2_method == V2_NONE) v2_fail("unknown_method", *n);
  v2_use_magma = v2_is_magma(v2_method);
  v2_cpu_build_requested = v2_env_is("MSSPEC_V2_GPU_BUILD", "0");
  v2_cpu_build = 0;
  if (v2_cpu_build_requested && v2_refines())
    v2_fail("cpu_build_refined_unsupported", *n);
  if (v2_cpu_build_requested && !v2_is_own(v2_method))
    v2_fail("cpu_build_own_only", *n);

  double t0 = v2_now();
  if (v2_thread_started && !v2_init_done) {
    pthread_join(v2_init_thread, NULL);
    v2_init_done = 1;
  } else if (!v2_init_done) {
    v2_init_status = v2_init_gpu();
    v2_init_done = 1;
  }
  v2_init_wait = v2_now() - t0;
  if (v2_init_status != 0) v2_fail(v2_init_what, *n);

  v2_budget_last_free = 0;
  v2_budget("begin_start", 0);
  v2_release();
  v2_budget_last_free = 0;
  v2_budget("begin_after_release", 0);
  v2_build_s = 0.0;
  v2_copy_s = 0.0;
  v2_n = *n;
  v2_natoms = *natoms;
  v2_nl = *nl;
  v2_ngaunt = *ngaunt;
  v2_maxlocal = *maxlocal;
  if (v2_n <= 0 || v2_natoms <= 0 || v2_nl < 0 || v2_ngaunt < 0 ||
      v2_maxlocal <= 0)
    v2_fail("bad_dimensions", v2_n);

  t0 = v2_now();
  if (v2_is_own(v2_method)) {
    /* FP16 matrix (4 n^2 bytes) and blocked LU workspace; no FP32 n x n matrix */
    if (!half_lu::alloc(v2_hl, v2_n, v2_stream, v2_solver, v2_blas))
      v2_fail("half_lu_alloc", v2_n);
    {
      const size_t N = (size_t)v2_n;
      v2_budget_req("half_lu_A_fp16", 4 * N * N);
      v2_budget_req("half_lu_panel", 8 * N * (size_t)v2_hl.nb);
      v2_budget_req("half_lu_wu", 8 * (size_t)v2_hl.nb * (size_t)v2_hl.tc);
      v2_budget_req("half_lu_wc", 8 * N * (size_t)v2_hl.tc);
      v2_budget_req("half_lu_bbig", 4 * (size_t)v2_hl.nb * (size_t)v2_hl.tc);
      v2_budget_req("half_lu_piv_ppiv_info", 4 * N + 4 * (size_t)v2_hl.nb + 4);
      v2_budget_req("half_lu_cwork", 8 * (size_t)(v2_hl.lwork > 1 ? v2_hl.lwork : 1));
      v2_budget("half_lu_all", 0);
    }
  } else if (v2_is_magma_host(v2_method)) {
    /* magma_host_*: FP32 matrix in pinned host memory (8 n^2 bytes). No device matrix, no cuSOLVER work area. */
    const size_t nn = (size_t)v2_n * (size_t)v2_n;
    v2_budget_req("host_matrix_pinned", nn * sizeof(cuComplex));
    V2_CHECK(cudaMallocHost((void **)&v2_host_matrix, nn * sizeof(cuComplex)),
             "cudaMallocHost_matrix");
    v2_budget("host_matrix_pinned", nn * sizeof(cuComplex));
    v2_ipiv_host = (int *)malloc((size_t)v2_n * sizeof(int));
    if (v2_ipiv_host == NULL) v2_fail("malloc_ipiv_host", v2_n);
  } else {
    const size_t nn = (size_t)v2_n * (size_t)v2_n;
    V2_CHECK(cudaMalloc((void **)&v2_matrix, nn * sizeof(cuComplex)),
             "cudaMalloc_matrix");
    V2_CHECK(cudaMalloc((void **)&v2_ipiv, (size_t)v2_n * sizeof(int)),
             "cudaMalloc_ipiv");
    V2_CHECK(cudaMalloc((void **)&v2_info, sizeof(int)), "cudaMalloc_info");
    if (v2_use_magma) {
      v2_ipiv_host = (int *)malloc((size_t)v2_n * sizeof(int));
      if (v2_ipiv_host == NULL) v2_fail("malloc_ipiv_host", v2_n);
    }
    int lwork = 0;
    if (cusolverDnCgetrf_bufferSize(v2_solver, v2_n, v2_n, v2_matrix, v2_n,
                                    &lwork) != CUSOLVER_STATUS_SUCCESS)
      v2_fail("cusolver_getrf_bufferSize", v2_n);
    V2_CHECK(cudaMalloc((void **)&v2_work,
                        (size_t)(lwork > 0 ? lwork : 1) * sizeof(cuComplex)),
             "cudaMalloc_work");
  }
  V2_CHECK(cudaMalloc((void **)&v2_tile,
                      (size_t)v2_n * (size_t)v2_maxlocal * sizeof(cuComplex)),
           "cudaMalloc_tile");
  v2_budget("tile_v2_tile", (size_t)v2_n * (size_t)v2_maxlocal * sizeof(cuComplex));
  if (v2_shared_pair_set(v2_method)) {
    /* own_*, magma_host_*: the pair arrays of one column are the same size for every column (ypair_count, hpair_count, natoms of the earlier GPU version) */
    v2_ny = (size_t)(2 * v2_nl + 1) * (size_t)(4 * v2_nl + 1) * (size_t)v2_natoms;
    v2_nh = (size_t)(2 * v2_nl + 1) * (size_t)v2_natoms;
    v2_na = (size_t)v2_natoms;
    V2_CHECK(cudaMalloc((void **)&v2_ypair, v2_ny * sizeof(cuComplex)),
             "cudaMalloc_pair_ypair");
    V2_CHECK(cudaMalloc((void **)&v2_hpair, v2_nh * sizeof(cuComplex)),
             "cudaMalloc_pair_hpair");
    V2_CHECK(cudaMalloc((void **)&v2_attpair, v2_na * sizeof(cuComplex)),
             "cudaMalloc_pair_attpair");
    v2_budget("own_pair_set", (v2_ny + v2_nh + v2_na) * sizeof(cuComplex));
  }

  V2_CHECK(cudaMalloc((void **)&v2_row_atom, (size_t)v2_n * sizeof(int)),
           "cudaMalloc_row_atom");
  V2_CHECK(cudaMalloc((void **)&v2_row_l, (size_t)v2_n * sizeof(int)),
           "cudaMalloc_row_l");
  V2_CHECK(cudaMalloc((void **)&v2_row_m, (size_t)v2_n * sizeof(int)),
           "cudaMalloc_row_m");
  V2_CHECK(cudaMalloc((void **)&v2_row_gntcol, (size_t)v2_n * sizeof(int)),
           "cudaMalloc_row_gntcol");
  V2_CHECK(cudaMemcpy(v2_row_atom, row_atom, (size_t)v2_n * sizeof(int),
                      cudaMemcpyHostToDevice), "memcpy_row_atom");
  V2_CHECK(cudaMemcpy(v2_row_l, row_l, (size_t)v2_n * sizeof(int),
                      cudaMemcpyHostToDevice), "memcpy_row_l");
  V2_CHECK(cudaMemcpy(v2_row_m, row_m, (size_t)v2_n * sizeof(int),
                      cudaMemcpyHostToDevice), "memcpy_row_m");
  V2_CHECK(cudaMemcpy(v2_row_gntcol, row_gntcol,
                      (size_t)v2_n * sizeof(int), cudaMemcpyHostToDevice),
           "memcpy_row_gntcol");

  cuComplex *host_tlk = (cuComplex *)malloc((size_t)v2_n * sizeof(cuComplex));
  if (host_tlk == NULL) v2_fail("malloc_row_tlk", v2_n);
  for (int i = 0; i < v2_n; ++i)
    host_tlk[i] = make_cuComplex((float)row_tlk[i].x, (float)row_tlk[i].y);
  V2_CHECK(cudaMalloc((void **)&v2_row_tlk,
                      (size_t)v2_n * sizeof(cuComplex)), "cudaMalloc_row_tlk");
  V2_CHECK(cudaMemcpy(v2_row_tlk, host_tlk,
                      (size_t)v2_n * sizeof(cuComplex),
                      cudaMemcpyHostToDevice), "memcpy_row_tlk");
  free(host_tlk);

  const size_t gnt_count = (size_t)(v2_ngaunt + 1) * (size_t)v2_maxlocal *
                           (size_t)v2_maxlocal;
  float *host_gnt = (float *)malloc(gnt_count * sizeof(float));
  if (host_gnt == NULL) v2_fail("malloc_gnt", v2_n);
  for (size_t i = 0; i < gnt_count; ++i)
    host_gnt[i] = (float)gnt_cache[i];
  V2_CHECK(cudaMalloc((void **)&v2_gnt, gnt_count * sizeof(float)),
           "cudaMalloc_gnt");
  V2_CHECK(cudaMemcpy(v2_gnt, host_gnt, gnt_count * sizeof(float),
                      cudaMemcpyHostToDevice), "memcpy_gnt");
  free(host_gnt);
  v2_budget("rows_gnt", (size_t)v2_n * (4 * sizeof(int) + sizeof(cuComplex)) +
            gnt_count * sizeof(float));
  v2_build_s += v2_now() - t0;
  v2_begin_end = v2_now();
  v2_ready = 1;
}

extern "C" void msspec_v2_gpu_column_(
    int *firstcol, int *ncols, int *absorber, int *natoms, int *nl,
    int *ngaunt, int *maxlocal, const int *col_l, const int *col_m,
    const int *col_ind, const cuDoubleComplex *ypair,
    const cuDoubleComplex *hpair, const double *attpair) {
  const double t0 = v2_now();
  const int col0 = *firstcol - 1;
  const int nc = *ncols;
  if (!v2_ready || *natoms != v2_natoms || *nl != v2_nl ||
      *ngaunt != v2_ngaunt || *maxlocal != v2_maxlocal)
    v2_fail("column_before_begin", v2_n);
  if (nc <= 0 || nc > v2_maxlocal || col0 < 0 || col0 + nc > v2_n ||
      *absorber < 1 || *absorber > v2_natoms)
    v2_fail("bad_column_block", v2_n);

  const size_t l_count = (size_t)(2 * v2_nl + 1);
  const size_t m_count = (size_t)(4 * v2_nl + 1);
  const size_t ny = l_count * m_count * (size_t)v2_natoms;
  const size_t nh = l_count * (size_t)v2_natoms;
  const size_t na = (size_t)v2_natoms;

  /* Pair data: FP64 from Fortran, converted to FP32 on the host (as in the earlier GPU version), then uploaded once. */
  cuComplex *host_y = (cuComplex *)malloc(ny * sizeof(cuComplex));
  cuComplex *host_h = (cuComplex *)malloc(nh * sizeof(cuComplex));
  cuComplex *host_a = (cuComplex *)malloc(na * sizeof(cuComplex));
  if (host_y == NULL || host_h == NULL || host_a == NULL)
    v2_fail("malloc_pairs", v2_n);
  for (size_t i = 0; i < ny; ++i)
    host_y[i] = make_cuComplex((float)ypair[i].x, (float)ypair[i].y);
  for (size_t i = 0; i < nh; ++i)
    host_h[i] = make_cuComplex((float)hpair[i].x, (float)hpair[i].y);
  for (size_t i = 0; i < na; ++i)
    host_a[i] = make_cuComplex((float)attpair[i], 0.0f);

  if (v2_ntiles == v2_tiles_capacity) {
    const int capacity = v2_tiles_capacity == 0 ? 16 : 2 * v2_tiles_capacity;
    struct V2Tile *grown = (struct V2Tile *)realloc(
        v2_tiles, (size_t)capacity * sizeof(struct V2Tile));
    if (grown == NULL) v2_fail("realloc_tiles", v2_n);
    v2_tiles = grown;
    v2_tiles_capacity = capacity;
  }
  struct V2Tile *tile = &v2_tiles[v2_ntiles];
  tile->col0 = col0;
  tile->ncols = nc;
  tile->absorber = *absorber;
  const int own_pairs = v2_shared_pair_set(v2_method);
  tile->hy = NULL;
  tile->hh = NULL;
  tile->ha = NULL;
  if (own_pairs) {
    /* own_*: one device pair set for all tiles (plan of the earlier GPU version). This tile's FP32 pair data stays on the
     * host (tile->hy/hh/ha) and is uploaded again per tile in the refinement residual. */
    tile->ypair = v2_ypair;
    tile->hpair = v2_hpair;
    tile->attpair = v2_attpair;
    tile->hy = host_y;
    tile->hh = host_h;
    tile->ha = host_a;
    V2_CHECK(cudaMemcpy(v2_ypair, host_y, ny * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "memcpy_ypair");
    V2_CHECK(cudaMemcpy(v2_hpair, host_h, nh * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "memcpy_hpair");
    V2_CHECK(cudaMemcpy(v2_attpair, host_a, na * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "memcpy_attpair");
  } else {
    V2_CHECK(cudaMalloc((void **)&tile->ypair, ny * sizeof(cuComplex)),
             "cudaMalloc_ypair");
    V2_CHECK(cudaMalloc((void **)&tile->hpair, nh * sizeof(cuComplex)),
             "cudaMalloc_hpair");
    V2_CHECK(cudaMalloc((void **)&tile->attpair, na * sizeof(cuComplex)),
             "cudaMalloc_attpair");
    V2_CHECK(cudaMemcpy(tile->ypair, host_y, ny * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "memcpy_ypair");
    V2_CHECK(cudaMemcpy(tile->hpair, host_h, nh * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "memcpy_hpair");
    V2_CHECK(cudaMemcpy(tile->attpair, host_a, na * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "memcpy_attpair");
    free(host_y);
    free(host_h);
    free(host_a);
  }
  V2_CHECK(cudaMalloc((void **)&tile->col_l, (size_t)nc * sizeof(int)),
           "cudaMalloc_col_l");
  V2_CHECK(cudaMalloc((void **)&tile->col_m, (size_t)nc * sizeof(int)),
           "cudaMalloc_col_m");
  V2_CHECK(cudaMalloc((void **)&tile->col_ind, (size_t)nc * sizeof(int)),
           "cudaMalloc_col_ind");
  V2_CHECK(cudaMemcpy(tile->col_l, col_l, (size_t)nc * sizeof(int),
                      cudaMemcpyHostToDevice), "memcpy_col_l");
  V2_CHECK(cudaMemcpy(tile->col_m, col_m, (size_t)nc * sizeof(int),
                      cudaMemcpyHostToDevice), "memcpy_col_m");
  V2_CHECK(cudaMemcpy(tile->col_ind, col_ind, (size_t)nc * sizeof(int),
                      cudaMemcpyHostToDevice), "memcpy_col_ind");
  v2_ntiles += 1;
  v2_budget("column_tile", own_pairs ? 3 * (size_t)nc * sizeof(int)
                                     : (ny + nh + na) * sizeof(cuComplex));

  const size_t count = (size_t)v2_n * (size_t)nc;
  const int blocks = (int)((count + 255) / 256);
  const int own = v2_is_own(v2_method);
  const int host_lu = v2_is_magma_host(v2_method);
  /* own_*: assemble this block in FP32 in v2_tile, then store it in FP16 (stage tile -> half matrix, as in the earlier GPU version).
   * magma_host_*: assemble in v2_tile, then copy it to the pinned host matrix (FP32). */
  cuComplex *out = (own || host_lu) ? v2_tile : v2_matrix + (size_t)col0 * (size_t)v2_n;
  v2_assemble_kernel<<<blocks, 256, 0, v2_stream>>>(
      v2_n, nc, col0, *absorber, v2_nl, v2_ngaunt, v2_maxlocal,
      v2_row_atom, v2_row_l, v2_row_m, v2_row_gntcol, v2_row_tlk,
      tile->col_l, tile->col_m, tile->col_ind,
      tile->ypair, tile->hpair, tile->attpair, v2_gnt, out);
  V2_CHECK(cudaGetLastError(), "assemble_kernel");
  if (own) {
    half_lu::upload_tile(v2_hl, v2_tile, col0, nc);
    V2_CHECK(cudaGetLastError(), "upload_tile");
  }
  if (host_lu) {
    V2_CHECK(cudaMemcpyAsync(v2_host_matrix + (size_t)col0 * (size_t)v2_n, v2_tile,
                             count * sizeof(cuComplex), cudaMemcpyDeviceToHost,
                             v2_stream), "download_tile");
  }
  V2_CHECK(cudaStreamSynchronize(v2_stream), "assemble_sync");
  v2_build_s += v2_now() - t0;
}

/* CPU build (MSSPEC_V2_GPU_BUILD=0, own_unrefined only). The host has built MS in FP64 (column-major
 * n x n, leading dimension n) and calls this once after begin. Blocks of maxlocal columns are converted
 * to FP32 on the host, copied to v2_tile, and stored in FP16 by half_lu::upload_tile (stage tile of the earlier GPU version). */
extern "C" void msspec_v2_gpu_upload_(int *n, const cuDoubleComplex *sm) {
  const int N = *n;
  if (!v2_cpu_build_requested || !v2_ready || N != v2_n || !v2_is_own(v2_method) || v2_cpu_build)
    v2_fail("upload_before_begin", N);
  cuComplex *host32 = (cuComplex *)malloc((size_t)N * (size_t)v2_maxlocal * sizeof(cuComplex));
  if (host32 == NULL) v2_fail("malloc_upload", N);
  for (int col0 = 0; col0 < N; col0 += v2_maxlocal) {
    const int nc = (N - col0 < v2_maxlocal) ? N - col0 : v2_maxlocal;
    const size_t total = (size_t)N * (size_t)nc;
    for (size_t i = 0; i < total; ++i)
      host32[i] = make_cuComplex((float)sm[(size_t)col0 * (size_t)N + i].x,
                                 (float)sm[(size_t)col0 * (size_t)N + i].y);
    V2_CHECK(cudaStreamSynchronize(v2_stream), "upload_presync");
    const double tc = v2_now();
    V2_CHECK(cudaMemcpyAsync(v2_tile, host32, total * sizeof(cuComplex),
                             cudaMemcpyHostToDevice, v2_stream), "upload_copy");
    V2_CHECK(cudaStreamSynchronize(v2_stream), "upload_copy_sync");
    v2_copy_s += v2_now() - tc;
    half_lu::upload_tile(v2_hl, v2_tile, col0, nc);
    V2_CHECK(cudaGetLastError(), "upload_tile");
  }
  V2_CHECK(cudaStreamSynchronize(v2_stream), "upload_sync");
  free(host32);
  v2_cpu_build = 1;
  v2_build_s += v2_now() - v2_begin_end;
}

/* Residual r = b - A x in FP64 over all tiles, regenerated on the device.
 * Returns eta = max|r| / (max_row sum|A| * max|x| + max|b|), the stop test of the earlier GPU version. */
static double v2_residual_eta(int n, int nrhs, const cuDoubleComplex *b_dev,
                              cuDoubleComplex *x_host, cuDoubleComplex *x_dev,
                              cuDoubleComplex *r_dev, double *rn_dev,
                              cuDoubleComplex *r_host, double *rn_host,
                              double rhs_norm) {
  const size_t count = (size_t)n * (size_t)nrhs;
  V2_CHECK(cudaMemcpy(x_dev, x_host, count * sizeof(cuDoubleComplex),
                      cudaMemcpyHostToDevice), "memcpy_x_residual");
  V2_CHECK(cudaMemcpy(r_dev, b_dev, count * sizeof(cuDoubleComplex),
                      cudaMemcpyDeviceToDevice), "memcpy_r_residual");
  V2_CHECK(cudaMemset(rn_dev, 0, (size_t)n * sizeof(double)), "memset_rn");
  for (int t = 0; t < v2_ntiles; ++t) {
    const struct V2Tile *tile = &v2_tiles[t];
    const size_t tile_count = (size_t)n * (size_t)tile->ncols;
    const int blocks = (int)((tile_count + 255) / 256);
    if (tile->hy != NULL) {
      /* own_*: this tile's pair data is host-resident; copy it into the shared device set. Async on
       * v2_stream, so the copy is ordered after the previous tile's kernels (no overwrite race). */
      V2_CHECK(cudaMemcpyAsync(tile->ypair, tile->hy, v2_ny * sizeof(cuComplex),
                               cudaMemcpyHostToDevice, v2_stream), "residual_upload_ypair");
      V2_CHECK(cudaMemcpyAsync(tile->hpair, tile->hh, v2_nh * sizeof(cuComplex),
                               cudaMemcpyHostToDevice, v2_stream), "residual_upload_hpair");
      V2_CHECK(cudaMemcpyAsync(tile->attpair, tile->ha, v2_na * sizeof(cuComplex),
                               cudaMemcpyHostToDevice, v2_stream), "residual_upload_attpair");
    }
    v2_assemble_kernel<<<blocks, 256, 0, v2_stream>>>(
        n, tile->ncols, tile->col0, tile->absorber, v2_nl, v2_ngaunt,
        v2_maxlocal, v2_row_atom, v2_row_l, v2_row_m, v2_row_gntcol,
        v2_row_tlk, tile->col_l, tile->col_m, tile->col_ind,
        tile->ypair, tile->hpair, tile->attpair, v2_gnt, v2_tile);
    V2_CHECK(cudaGetLastError(), "residual_assemble_kernel");
    const int rblocks = (int)((count + 255) / 256);
    v2_residual_kernel<<<rblocks, 256, 0, v2_stream>>>(
        n, nrhs, tile->col0, tile->ncols, v2_tile, x_dev, r_dev, rn_dev);
    V2_CHECK(cudaGetLastError(), "residual_kernel");
  }
  V2_CHECK(cudaStreamSynchronize(v2_stream), "residual_sync");
  V2_CHECK(cudaMemcpy(r_host, r_dev, count * sizeof(cuDoubleComplex),
                      cudaMemcpyDeviceToHost), "memcpy_r_host");
  V2_CHECK(cudaMemcpy(rn_host, rn_dev, (size_t)n * sizeof(double),
                      cudaMemcpyDeviceToHost), "memcpy_rn_host");

  double matrix_norm = 0.0;
  for (int row = 0; row < n; ++row)
    if (rn_host[row] > matrix_norm) matrix_norm = rn_host[row];
  double residual_norm = 0.0, solution_norm = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const cuDoubleComplex r = r_host[i];
    const cuDoubleComplex xv = x_host[i];
    if (!isfinite(r.x) || !isfinite(r.y) || !isfinite(xv.x) ||
        !isfinite(xv.y))
      return HUGE_VAL;
    const double rm = fabs(r.x) + fabs(r.y);
    const double xm = fabs(xv.x) + fabs(xv.y);
    if (rm > residual_norm) residual_norm = rm;
    if (xm > solution_norm) solution_norm = xm;
  }
  return residual_norm / (matrix_norm * solution_norm + rhs_norm);
}

extern "C" void msspec_v2_gpu_solve_(int *n, int *nrhs,
                                     const cuDoubleComplex *b,
                                     cuDoubleComplex *x) {
  const int N = *n;
  const int R = *nrhs;
  if (N != v2_n || R <= 0 || !v2_ready)
    v2_fail("bad_solve_arguments", N);
  if (v2_cpu_build_requested != v2_cpu_build)
    v2_fail("build_mode_mismatch", N);
  const size_t count = (size_t)N * (size_t)R;
  const size_t count_bytes = count * sizeof(cuComplex);
  v2_budget("solve_start", 0);

  /* Right-hand side in FP64 (kept for the residual) and FP32 (for the solves). */
  double rhs_norm = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double m = fabs(b[i].x) + fabs(b[i].y);
    if (m > rhs_norm) rhs_norm = m;
  }

  const double t_lu0 = v2_now();
  cuComplex *host32 = (cuComplex *)malloc(count_bytes);
  if (host32 == NULL) v2_fail("malloc_rhs32", N);
  for (size_t i = 0; i < count; ++i)
    host32[i] = make_cuComplex((float)b[i].x, (float)b[i].y);
  /* magma_host_*: the solve runs on host32 (host LAPACK); the other methods solve in dev32 on the device. */
  const int host_lu = v2_is_magma_host(v2_method);
  cuComplex *dev32 = NULL;
  if (!host_lu) {
    V2_CHECK(cudaMalloc((void **)&dev32, count_bytes), "cudaMalloc_rhs32");
    V2_CHECK(cudaMemcpy(dev32, host32, count_bytes, cudaMemcpyHostToDevice),
             "memcpy_rhs32");
  }

  v2_factor(N);
  v2_solve32(N, R, host_lu ? host32 : dev32, 0);
  if (!host_lu)
    V2_CHECK(cudaMemcpy(host32, dev32, count_bytes, cudaMemcpyDeviceToHost),
             "memcpy_x32");
  for (size_t i = 0; i < count; ++i) {
    x[i].x = (double)host32[i].x;
    x[i].y = (double)host32[i].y;
  }
  const double lu_s = v2_now() - t_lu0;

  int iterations = 0;
  double refine_s = 0.0;
  if (v2_refines()) {
    const double t_ref0 = v2_now();
    cuDoubleComplex *dev_b = NULL, *dev_x = NULL, *dev_r = NULL;
    double *dev_rn = NULL;
    cuDoubleComplex *host_r = (cuDoubleComplex *)malloc(
        count * sizeof(cuDoubleComplex));
    double *host_rn = (double *)malloc((size_t)N * sizeof(double));
    if (host_r == NULL || host_rn == NULL) v2_fail("malloc_residual", N);
    V2_CHECK(cudaMalloc((void **)&dev_b, count * sizeof(cuDoubleComplex)),
             "cudaMalloc_b64");
    V2_CHECK(cudaMalloc((void **)&dev_x, count * sizeof(cuDoubleComplex)),
             "cudaMalloc_x64");
    V2_CHECK(cudaMalloc((void **)&dev_r, count * sizeof(cuDoubleComplex)),
             "cudaMalloc_r64");
    V2_CHECK(cudaMalloc((void **)&dev_rn, (size_t)N * sizeof(double)),
             "cudaMalloc_rn");
    V2_CHECK(cudaMemcpy(dev_b, b, count * sizeof(cuDoubleComplex),
                        cudaMemcpyHostToDevice), "memcpy_b64");

    double eta = v2_residual_eta(N, R, dev_b, x, dev_x, dev_r, dev_rn,
                                 host_r, host_rn, rhs_norm);
    if (!isfinite(eta)) v2_fail("refine_nonfinite", N);
    if (v2_is_own(v2_method))
      fprintf(stderr, "MSSPEC_V2_GPU_REFINE method=%s correction=0 eta=%.6e\n",
              v2_method_name, eta);
    while (eta > V2_TOLERANCE && iterations < V2_MAX_CORRECTIONS) {
      for (size_t i = 0; i < count; ++i)
        host32[i] = make_cuComplex((float)host_r[i].x, (float)host_r[i].y);
      if (!host_lu)
        V2_CHECK(cudaMemcpy(dev32, host32, count_bytes,
                            cudaMemcpyHostToDevice), "memcpy_corr_rhs");
      v2_solve32(N, R, host_lu ? host32 : dev32, 1);
      if (!host_lu)
        V2_CHECK(cudaMemcpy(host32, dev32, count_bytes,
                            cudaMemcpyDeviceToHost), "memcpy_corr");
      for (size_t i = 0; i < count; ++i) {
        x[i].x += (double)host32[i].x;
        x[i].y += (double)host32[i].y;
      }
      ++iterations;
      eta = v2_residual_eta(N, R, dev_b, x, dev_x, dev_r, dev_rn,
                            host_r, host_rn, rhs_norm);
      if (!isfinite(eta)) v2_fail("refine_nonfinite", N);
      if (v2_is_own(v2_method))
        fprintf(stderr, "MSSPEC_V2_GPU_REFINE method=%s correction=%d eta=%.6e\n",
                v2_method_name, iterations, eta);
    }
    if (!(eta <= V2_TOLERANCE)) v2_fail("refine_not_converged", N);
    cudaFree(dev_b);
    cudaFree(dev_x);
    cudaFree(dev_r);
    cudaFree(dev_rn);
    free(host_r);
    free(host_rn);
    refine_s = v2_now() - t_ref0;
  }

  for (size_t i = 0; i < count; ++i)
    if (!isfinite(x[i].x) || !isfinite(x[i].y))
      v2_fail("nonfinite_solution", N);
  cudaFree(dev32);
  free(host32);

  fprintf(stderr,
          "MSSPEC_V2_GPU method=%s n=%d nrhs=%d init_wait_s=%.6f "
          "build_s=%.6f lu_s=%.6f refine_iters=%d refine_s=%.6f "
          "gpu_build=%d early_start=%d copy_s=%.6f\n",
          v2_method_name, N, R, v2_init_wait, v2_build_s, lu_s,
          iterations, refine_s, v2_cpu_build ? 0 : 1,
          v2_thread_started ? 1 : 0, v2_copy_s);
  fflush(stderr);
}
