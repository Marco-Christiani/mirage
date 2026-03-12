/// Layer 1: Graph building + search.
///
/// Graph construction is stateless (no GPU, no context, no side effects).
/// Search requires a device handle for GPU-based fingerprint checking.
#pragma once

#include "mirage/c/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Graph building
// ---------------------------------------------------------------------------

/// Create an empty kernel graph.
mirage_status_t mirage_graph_create(mirage_graph_t **out);
void mirage_graph_destroy(mirage_graph_t *graph);

/// Add a graph input tensor.
mirage_status_t mirage_graph_new_input(mirage_graph_t *graph,
                                       mirage_tensor_spec_t const *spec,
                                       mirage_tensor_t *out);

/// Kernel-level operations.
mirage_status_t mirage_graph_matmul(mirage_graph_t *graph,
                                    mirage_tensor_t lhs,
                                    mirage_tensor_t rhs,
                                    mirage_tensor_t *out);

mirage_status_t mirage_graph_unary(mirage_graph_t *graph,
                                   mirage_unary_op_t op,
                                   mirage_tensor_t input,
                                   mirage_tensor_t *out);

mirage_status_t mirage_graph_binary(mirage_graph_t *graph,
                                    mirage_binary_op_t op,
                                    mirage_tensor_t lhs,
                                    mirage_tensor_t rhs,
                                    mirage_tensor_t *out);

mirage_status_t mirage_graph_reduction(mirage_graph_t *graph,
                                       mirage_tensor_t input,
                                       int32_t dim,
                                       int32_t factor,
                                       mirage_tensor_t *out);

mirage_status_t mirage_graph_rms_norm(mirage_graph_t *graph,
                                      mirage_tensor_t input,
                                      int32_t normalized_size,
                                      mirage_tensor_t *out);

mirage_status_t mirage_graph_mark_output(mirage_graph_t *graph,
                                         mirage_tensor_t tensor);

// ---------------------------------------------------------------------------
// Device handle
// ---------------------------------------------------------------------------

/// Create a device handle for GPU-bound operations (search).
/// device_ordinal: CUDA device index (0-based).
mirage_status_t mirage_device_create(int32_t device_ordinal,
                                     mirage_device_t **out);
void mirage_device_destroy(mirage_device_t *device);

/// Query device memory.
mirage_status_t mirage_device_mem_info(mirage_device_t const *device,
                                       size_t *out_free,
                                       size_t *out_total);

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

typedef struct mirage_search_options {
  uint32_t max_candidates; // 0 = use default (1024)
  uint8_t verbose;
  uint8_t formal_verify;
  // Search space configuration (all nullable = use defaults):
  int32_t const *grid_dims;
  size_t num_grid_dims;
  int32_t const *block_dims;
  size_t num_block_dims;
  int32_t const *imaps;
  size_t num_imaps;
  int32_t const *omaps;
  size_t num_omaps;
  int32_t const *fmaps;
  size_t num_fmaps;
  int32_t const *franges;
  size_t num_franges;
  // Memory limits (0 = use compile-time default):
  size_t max_dmem_size;
  size_t max_smem_size;
  size_t max_dmem_fp_size;
  size_t max_smem_fp_size;
  size_t max_num_threadblocks;
} mirage_search_options_t;

/// Run superoptimization. Does not modify `graph`.
/// Returns candidates ranked best-first.
mirage_status_t mirage_search(mirage_device_t *device,
                              mirage_graph_t const *graph,
                              mirage_search_options_t const *options,
                              mirage_search_result_t **out);

void mirage_search_result_destroy(mirage_search_result_t *result);

/// Number of candidates found (0 = no improvement over input graph).
size_t mirage_search_result_count(mirage_search_result_t const *result);

/// Get candidate graph by index (0 = best).
/// Returned pointer is valid until the search result is destroyed.
mirage_graph_t const *mirage_search_result_get(
    mirage_search_result_t const *result,
    size_t index);

#ifdef __cplusplus
}
#endif
