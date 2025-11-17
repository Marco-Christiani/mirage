#pragma once
#include "tasks/common/common_header.cuh"
namespace kernel {

template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void rms_norm_backward_impl(
    void const *input_ptr,
    void const *grad_output_ptr,
    void const *weight_ptr,
    void *grad_input_ptr,
    void *grad_weight_ptr,
    float eps) {
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("BACKWARD RMSNorm executing! BATCH=%d, HIDDEN=%d\n", BATCH_SIZE, HIDDEN_DIM);
    printf("  input_ptr=%p, grad_output_ptr=%p, weight_ptr=%p\n", input_ptr, grad_output_ptr, weight_ptr);
    printf("  grad_input_ptr=%p, grad_weight_ptr=%p\n", grad_input_ptr, grad_weight_ptr);
    printf("BACKWARD RMSNorm executing! BATCH=%d, HIDDEN=%d\n", BATCH_SIZE, HIDDEN_DIM);
    T *grad_input = static_cast<T *>(grad_input_ptr);
    grad_input[0] = T(999.0f);  // Magic value
  }
  __syncthreads();

  static_assert(BATCH_SIZE == 1);
  extern __shared__ char smem[];
  static_assert(HIDDEN_DIM % NUM_THREADS == 0);
  
  constexpr int ELTS_PER_THREAD = HIDDEN_DIM / NUM_THREADS;
  constexpr int BYTES_PER_THREAD = ELTS_PER_THREAD * sizeof(T);
  constexpr int BYTES_PER_CP = []() {
    if constexpr (BYTES_PER_THREAD % 16 == 0) {
      return 16;
    } else if constexpr (BYTES_PER_THREAD % 8 == 0) {
      return 8;
    } else {
      static_assert(BYTES_PER_THREAD % 4 == 0);
      return 4;
    }
  }();
  constexpr int CHUNK_SIZE = BYTES_PER_CP / sizeof(T);
  constexpr int TILE_SIZE = NUM_THREADS * CHUNK_SIZE;
  static_assert(HIDDEN_DIM % TILE_SIZE == 0);
  constexpr int NUM_TILES = HIDDEN_DIM / TILE_SIZE;
  constexpr int NUM_CHUNKS = BATCH_SIZE * HIDDEN_DIM / CHUNK_SIZE;

  T const *__restrict__ d_grad_output = static_cast<T const *>(grad_output_ptr);
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_grad_input = static_cast<T *>(grad_input_ptr);
  T *__restrict__ d_grad_weight = static_cast<T *>(grad_weight_ptr);

  // Shared memory layout
  constexpr size_t SHARED_GRAD_OUTPUT_OFFSET = 0;
  constexpr size_t SHARED_INPUT_OFFSET = SHARED_GRAD_OUTPUT_OFFSET + sizeof(T) * HIDDEN_DIM;
  constexpr size_t SHARED_WEIGHT_OFFSET = SHARED_INPUT_OFFSET + sizeof(T) * HIDDEN_DIM;
  constexpr size_t SHARED_GRAD_INPUT_OFFSET = SHARED_WEIGHT_OFFSET + sizeof(T) * HIDDEN_DIM;
  constexpr size_t SHARED_GRAD_WEIGHT_OFFSET = SHARED_GRAD_INPUT_OFFSET + sizeof(T) * HIDDEN_DIM;
  constexpr size_t REDUCE_BUFFER_OFFSET = SHARED_GRAD_WEIGHT_OFFSET + sizeof(T) * HIDDEN_DIM;
  
  T *shared_grad_output = (T *)(smem + SHARED_GRAD_OUTPUT_OFFSET);
  T *shared_input = (T *)(smem + SHARED_INPUT_OFFSET);
  T *shared_weight = (T *)(smem + SHARED_WEIGHT_OFFSET);
  T *shared_grad_input = (T *)(smem + SHARED_GRAD_INPUT_OFFSET);
  T *shared_grad_weight = (T *)(smem + SHARED_GRAD_WEIGHT_OFFSET);
  float *reduce_smem = reinterpret_cast<float *>(smem + REDUCE_BUFFER_OFFSET);

  // Load data into shared memory
  {
    load_smem<T, BYTES_PER_CP>(shared_grad_output + threadIdx.x * CHUNK_SIZE,
                               d_grad_output + threadIdx.x * CHUNK_SIZE);
    load_smem<T, BYTES_PER_CP>(shared_input + threadIdx.x * CHUNK_SIZE,
                               d_input + threadIdx.x * CHUNK_SIZE);
    load_smem<T, BYTES_PER_CP>(shared_weight + threadIdx.x * CHUNK_SIZE,
                               d_weight + threadIdx.x * CHUNK_SIZE);
    cp_async_fence();
  }

  // Step 1: Compute RMS
  float sum_sq = 0.0f;
#pragma unroll
  for (int for_idx = 0; for_idx < NUM_TILES; for_idx++) {
    if (for_idx + 1 < NUM_TILES) {
      load_smem<T, BYTES_PER_CP>(
          shared_grad_output + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE,
          d_grad_output + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE);
      load_smem<T, BYTES_PER_CP>(
          shared_input + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE,
          d_input + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE);
      load_smem<T, BYTES_PER_CP>(
          shared_weight + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE,
          d_weight + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE);
      cp_async_fence();
      cp_async_wait<1>();
    } else if (for_idx + 1 == NUM_TILES) {
      cp_async_wait<0>();
    }
    __syncthreads();
#pragma unroll
    for (int i = threadIdx.x; i < TILE_SIZE; i += NUM_THREADS) {
      float val = (float)shared_input[for_idx * TILE_SIZE + i];
      sum_sq += val * val;
    }
  }

  // Reduce sum_sq across threads
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }
  if (threadIdx.x % 32 == 0) {
    reduce_smem[threadIdx.x / 32] = sum_sq;
  }
  __syncthreads();
  sum_sq = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;
#pragma unroll
  for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }
  if (threadIdx.x == 0) {
    reduce_smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrt(reduce_smem[0] / float(HIDDEN_DIM) + eps);

  // Step 2: Compute sum(dy * w * x) for the correction term
  float sum_dy_w_x = 0.0f;
#pragma unroll
  for (int i = threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
    float dy = (float)shared_grad_output[i];
    float w = (float)shared_weight[i];
    float x = (float)shared_input[i];
    sum_dy_w_x += dy * w * x;
  }

  // Reduce sum_dy_w_x across threads
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    sum_dy_w_x += shfl_xor_sync(sum_dy_w_x, offset);
  }
  if (threadIdx.x % 32 == 0) {
    reduce_smem[threadIdx.x / 32] = sum_dy_w_x;
  }
  __syncthreads();
  sum_dy_w_x = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;
#pragma unroll
  for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
    sum_dy_w_x += shfl_xor_sync(sum_dy_w_x, offset);
  }
  if (threadIdx.x == 0) {
    reduce_smem[0] = sum_dy_w_x;
  }
  __syncthreads();

  float correction = reduce_smem[0] / float(HIDDEN_DIM) * rms_rcp * rms_rcp;

  // Step 3: Compute gradients
#pragma unroll
  for (int i = threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
    float dy = (float)shared_grad_output[i];
    float w = (float)shared_weight[i];
    float x = (float)shared_input[i];
    
    // grad_weight = dy * x * rms_rcp
    float dw = dy * x * rms_rcp;
    shared_grad_weight[i] = (T)dw;
    
    // grad_input = dy * w * rms_rcp - x * correction
    float dx = dy * w * rms_rcp - x * correction;
    shared_grad_input[i] = (T)dx;
  }
  __syncthreads();

  // Step 4: Write outputs to global memory
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS; i += NUM_THREADS) {
    if constexpr (BYTES_PER_CP == 16) {
      *((__uint128_t *)((void *)&d_grad_input[i * CHUNK_SIZE])) =
          *((__uint128_t *)((void *)&shared_grad_input[i * CHUNK_SIZE]));
      *((__uint128_t *)((void *)&d_grad_weight[i * CHUNK_SIZE])) =
          *((__uint128_t *)((void *)&shared_grad_weight[i * CHUNK_SIZE]));
    } else if constexpr (BYTES_PER_CP == 8) {
      *((uint64_t *)((void *)&d_grad_input[i * CHUNK_SIZE])) =
          *((uint64_t *)((void *)&shared_grad_input[i * CHUNK_SIZE]));
      *((uint64_t *)((void *)&d_grad_weight[i * CHUNK_SIZE])) =
          *((uint64_t *)((void *)&shared_grad_weight[i * CHUNK_SIZE]));
    } else {
      *((uint32_t *)((void *)&d_grad_input[i * CHUNK_SIZE])) =
          *((uint32_t *)((void *)&shared_grad_input[i * CHUNK_SIZE]));
      *((uint32_t *)((void *)&d_grad_weight[i * CHUNK_SIZE])) =
          *((uint32_t *)((void *)&shared_grad_weight[i * CHUNK_SIZE]));
    }
  }
}

} // namespace kernel