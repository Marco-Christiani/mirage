#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

// Linear backward: compute grad_input and grad_weight
// grad_input = grad_output @ weight          [batch_size, in_features]
// grad_weight = grad_output.T @ input        [out_features, in_features]
template <typename T, int BATCH_SIZE, int IN_FEATURES, int OUT_FEATURES>
__device__ __forceinline__ void linear_backward_impl(
    void const *input_ptr,       // x: [BATCH_SIZE, IN_FEATURES]
    void const *weight_ptr,      // W: [OUT_FEATURES, IN_FEATURES]
    void const *grad_output_ptr, // dL/dy: [BATCH_SIZE, OUT_FEATURES]
    void *grad_input_ptr,        // dL/dx: [BATCH_SIZE, IN_FEATURES]
    void *grad_weight_ptr        // dL/dW: [OUT_FEATURES, IN_FEATURES]
) {
  static_assert(BATCH_SIZE == 1, "Currently only support BATCH_SIZE=1");

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_grad_output = static_cast<T const *>(grad_output_ptr);
  T *__restrict__ d_grad_input = static_cast<T *>(grad_input_ptr);
  T *__restrict__ d_grad_weight = static_cast<T *>(grad_weight_ptr);

  // Compute grad_input = grad_output @ weight
  // grad_input[i] = sum_j(grad_output[j] * weight[j, i])
  // Each thread computes one or more elements of grad_input
  for (int i = threadIdx.x; i < IN_FEATURES; i += blockDim.x) {
    float acc = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < OUT_FEATURES; j++) {
      float dout = static_cast<float>(d_grad_output[j]);
      float w = static_cast<float>(d_weight[j * IN_FEATURES + i]);
      acc += dout * w;
    }
    d_grad_input[i] = static_cast<T>(acc);
  }

  // Compute grad_weight = grad_output.T @ input
  // grad_weight[j, i] = grad_output[j] * input[i]
  // Since batch_size=1, this is just an outer product
  __syncthreads();

  int total_elements = OUT_FEATURES * IN_FEATURES;
  for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
    int j = idx / IN_FEATURES;  // output feature index
    int i = idx % IN_FEATURES;  // input feature index

    float dout = static_cast<float>(d_grad_output[j]);
    float x = static_cast<float>(d_input[i]);
    d_grad_weight[j * IN_FEATURES + i] = static_cast<T>(dout * x);
  }
}

} // namespace kernel
