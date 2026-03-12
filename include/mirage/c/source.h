/// Layer 2: Transpilation to CUDA source.
///
/// Transpiles a kernel graph (original or search candidate) into CUDA source
/// code. Pure CPU operation, no GPU required.
#pragma once

#include "mirage/c/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Source traits
// ---------------------------------------------------------------------------

/// Bitfield describing properties of transpiled source.
typedef uint32_t mirage_source_traits_t;
enum {
  /// Source calls host-side libraries (cuBLAS, etc.). Cannot be compiled
  /// without those libraries and cannot run without a host-side launcher.
  MIRAGE_SOURCE_USES_HOST_LIBS = 1 << 0,
  /// Source contains __global__ CUDA kernels. These can be extracted,
  /// compiled separately, and launched by the consumer.
  MIRAGE_SOURCE_HAS_KERNELS = 1 << 1,
  /// All compute in the source is device-callable (__device__ functions).
  /// Future capability -- not produced by the current transpiler.
  MIRAGE_SOURCE_DEVICE_CALLABLE = 1 << 2,
};

// ---------------------------------------------------------------------------
// Transpile options
// ---------------------------------------------------------------------------

typedef struct mirage_transpile_options {
  int32_t target_cc;    // Compute capability (e.g. 80, 90). 0 = native.
  uint8_t profiling;    // Emit profiling instrumentation.
  int32_t pipeline_stages; // 0 = default.
  /// Traits the consumer requires in the output. If the transpiler cannot
  /// satisfy them, transpilation returns MIRAGE_STATUS_UNSUPPORTED.
  /// 0 = no constraints.
  mirage_source_traits_t required_traits;
} mirage_transpile_options_t;

// ---------------------------------------------------------------------------
// Transpile + source inspection
// ---------------------------------------------------------------------------

/// Transpile a kernel graph to CUDA source.
/// `graph` may be the original input graph or a search candidate.
mirage_status_t mirage_transpile(mirage_graph_t const *graph,
                                 mirage_transpile_options_t const *options,
                                 mirage_source_t **out);

void mirage_source_destroy(mirage_source_t *source);

/// Query the traits of the transpiled source.
mirage_source_traits_t mirage_source_traits(mirage_source_t const *source);

/// Full CUDA source code (null-terminated).
/// Valid until the source handle is destroyed.
char const *mirage_source_code(mirage_source_t const *source);
size_t mirage_source_code_len(mirage_source_t const *source);

/// Intermediate buffer size in bytes (the `buf` parameter in the generated
/// _execute_mugraph function). The caller must allocate this on the device.
size_t mirage_source_buf_size(mirage_source_t const *source);

/// Maximum shared memory used by any kernel in the source.
size_t mirage_source_max_smem(mirage_source_t const *source);

/// Output tensor directives: shape and stride information for each graph
/// output. The caller uses these to allocate output buffers.
size_t mirage_source_num_outputs(mirage_source_t const *source);
mirage_status_t mirage_source_output_spec(mirage_source_t const *source,
                                          size_t index,
                                          mirage_tensor_spec_t *out);

/// Number of distinct __global__ kernels in the source (custom ops).
/// This is 0 when the graph only contains kernel-level library ops (e.g.
/// standalone matmul). It is >= 1 when the superoptimizer found a fused
/// threadblock implementation.
size_t mirage_source_num_kernels(mirage_source_t const *source);

/// Per-kernel launch metadata.
typedef struct mirage_kernel_meta {
  char const *func_name;
  size_t func_name_len;
  size_t smem_bytes;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
} mirage_kernel_meta_t;

mirage_status_t mirage_source_kernel_meta(mirage_source_t const *source,
                                          size_t index,
                                          mirage_kernel_meta_t *out);

// ---------------------------------------------------------------------------
// Kernel argument mapping
// ---------------------------------------------------------------------------

/// Describes where a single kernel argument pointer comes from.
typedef enum mirage_arg_source {
  /// Pointer is input_tensors[index] from the dispatch call.
  MIRAGE_ARG_INPUT = 0,
  /// Pointer is output_tensors[index] from the dispatch call.
  MIRAGE_ARG_OUTPUT = 1,
  /// Pointer is (char*)buf + buf_offset in the workspace buffer.
  MIRAGE_ARG_BUF = 2,
} mirage_arg_source_t;

typedef struct mirage_kernel_arg {
  mirage_arg_source_t source;
  /// For INPUT/OUTPUT: index into the input/output tensor array.
  /// For BUF: byte offset into the workspace buffer.
  size_t index_or_offset;
} mirage_kernel_arg_t;

/// Number of arguments for kernel at `kernel_index`.
size_t mirage_source_kernel_num_args(mirage_source_t const *source,
                                     size_t kernel_index);

/// Get argument mapping for a kernel.
/// `arg_index` is 0-based within the kernel's argument list.
mirage_status_t mirage_source_kernel_arg(mirage_source_t const *source,
                                         size_t kernel_index,
                                         size_t arg_index,
                                         mirage_kernel_arg_t *out);

#ifdef __cplusplus
}
#endif
