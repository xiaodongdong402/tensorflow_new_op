#include <iostream>

#if GPU
#define EIGEN_USE_GPU

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"



//#include "cuda_runtime.h"
//#include "curand.h"
//#include "cublas_v2.h"

namespace dzhang{

#define BLOCK 512

dim3 cuda_gridsize(size_t n);

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// The number of cuda threads to use. Since work is assigned to SMs at the
// granularity of a block, 128 is chosen to allow utilizing more SMs for
// smaller input sizes.
constexpr int CAFFE_CUDA_NUM_THREADS = 128;
// The maximum number of blocks to use in the default kernel call. We set it to
// 4096 which would work for compute capability 2.x (where 65536 is the limit).
// This number is very carelessly chosen. Ideally, one would like to look at
// the hardware at runtime, and pick the number of blocks that makes most
// sense for the specific runtime environment. This is a todo item.
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int CAFFE_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
          CAFFE_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}


void check_error(cudaError_t status);

void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void cuda_pull_array(float *x_gpu, float *x, size_t n);

void cuda_push_array(float *x_gpu, float *x, size_t n);

void fill_gpu(int N, float ALPHA, float * X, int INCX);

void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);

float *cuda_make_array(float *x, size_t n);



}


#endif
