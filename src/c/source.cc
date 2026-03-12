#include "internal.h"

#include "mirage/c/source.h"
#include "mirage/c/types.h"

#include "mirage/kernel/customized.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpile.h"
#include "mirage/type.h"

#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

bool from_mirage_dtype(mirage::type::DataType dtype,
                       mirage_dtype_t *out_dtype) {
  switch (dtype) {
  case mirage::type::DT_FLOAT16:
    *out_dtype = MIRAGE_DTYPE_F16;
    return true;
  case mirage::type::DT_BFLOAT16:
    *out_dtype = MIRAGE_DTYPE_BF16;
    return true;
  case mirage::type::DT_FLOAT32:
    *out_dtype = MIRAGE_DTYPE_F32;
    return true;
  case mirage::type::DT_DOUBLE:
    *out_dtype = MIRAGE_DTYPE_F64;
    return true;
  default:
    return false;
  }
}

/// Per-argument mapping for a kernel launch.
struct ArgInfo {
  mirage_arg_source_t source;
  size_t index_or_offset;
};

/// Per-kernel metadata stored alongside the transpiled source.
struct KernelInfo {
  std::string func_name;
  size_t smem_bytes;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  std::vector<ArgInfo> args;
};

/// Determine whether the graph has any KN_CUSTOMIZED_OP operators.
bool graph_has_customized_ops(mirage::kernel::Graph const *g) {
  for (auto const *op : g->operators) {
    if (op->op_type == mirage::type::KN_CUSTOMIZED_OP) {
      return true;
    }
  }
  return false;
}

/// Determine whether the graph uses library ops that require host libraries
/// (e.g. standalone matmul maps to cuBLAS via kn::gemm).
bool graph_uses_host_libs(mirage::kernel::Graph const *g) {
  for (auto const *op : g->operators) {
    if (op->op_type == mirage::type::KN_MATMUL_OP) {
      return true;
    }
  }
  return false;
}

} // namespace

// ---------------------------------------------------------------------------
// mirage_source: owns transpile result and derived metadata
// ---------------------------------------------------------------------------

struct mirage_source {
  std::string code;
  size_t buf_size;
  size_t max_smem;
  mirage_source_traits_t traits;
  std::vector<mirage::transpiler::OutputTensorDirective> output_directives;
  std::vector<KernelInfo> kernels;
};

// ---------------------------------------------------------------------------
// Transpile
// ---------------------------------------------------------------------------

mirage_status_t mirage_transpile(mirage_graph_t const *graph,
                                 mirage_transpile_options_t const *options,
                                 mirage_source_t **out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }

  mirage::transpiler::TranspilerConfig cfg;
  cfg.target_cc = 80; // default
  cfg.profiling = false;
  cfg.num_consumer_wgs = 2;
  cfg.num_producer_wgs = 1;
  cfg.pipeline_stages = 2;
  cfg.enable_online_softmax = false;

  if (options != nullptr) {
    if (options->target_cc > 0) {
      cfg.target_cc = options->target_cc;
    }
    cfg.profiling = options->profiling != 0;
    if (options->pipeline_stages > 0) {
      cfg.pipeline_stages = options->pipeline_stages;
    }
  }

  // Check if the graph can satisfy required traits before transpiling.
  mirage_source_traits_t required = 0;
  if (options != nullptr) {
    required = options->required_traits;
  }

  bool has_custom = graph_has_customized_ops(graph->graph);
  bool uses_host = graph_uses_host_libs(graph->graph);

  // If the consumer requires no host libs but the graph needs them, fail early.
  if ((required & MIRAGE_SOURCE_HAS_KERNELS) && !has_custom) {
    return MIRAGE_STATUS_UNSUPPORTED;
  }

  auto result = mirage::transpiler::transpile(graph->graph, cfg,
                                              graph->input_strides);
  if (result.error_type != mirage::transpiler::CUDA_T_SUCCESS) {
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }

  auto *source = new (std::nothrow) mirage_source_t{};
  if (source == nullptr) {
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }

  source->code = std::move(result.code);
  source->buf_size = result.buf_size;
  source->max_smem = result.max_smem_size;
  source->output_directives = std::move(result.output_directives);

  // Compute traits.
  source->traits = 0;
  if (uses_host) {
    source->traits |= MIRAGE_SOURCE_USES_HOST_LIBS;
  }
  if (has_custom) {
    source->traits |= MIRAGE_SOURCE_HAS_KERNELS;
  }

  // Build a map from tensor GUID to its role in the graph:
  // - Graph inputs: identified by KN_INPUT_OP output tensors
  // - Graph outputs: identified by KN_OUTPUT_OP input tensors
  // - Intermediates: everything else (lives in the workspace buffer)
  //
  // For inputs/outputs, store the index (0-based order of appearance).
  // For intermediates, store the byte offset in the workspace buffer,
  // computed from the tensor's data_offset field.

  struct TensorRole {
    mirage_arg_source_t source;
    size_t index_or_offset;
  };
  std::unordered_map<int64_t, TensorRole> tensor_roles; // keyed by GUID

  {
    int input_idx = 0;
    int output_idx = 0;
    for (auto const *op : graph->graph->operators) {
      if (op->op_type == mirage::type::KN_INPUT_OP) {
        for (auto const &t : op->output_tensors) {
          tensor_roles[t.guid] = {MIRAGE_ARG_INPUT,
                                  static_cast<size_t>(input_idx++)};
        }
      } else if (op->op_type == mirage::type::KN_OUTPUT_OP) {
        for (auto const &t : op->input_tensors) {
          // Mark the tensor feeding into the output op. If it's already
          // known (e.g. as intermediate), upgrade it to output.
          tensor_roles[t.guid] = {MIRAGE_ARG_OUTPUT,
                                  static_cast<size_t>(output_idx++)};
        }
      }
    }
    // Everything not yet in the map is an intermediate.
    for (auto const *op : graph->graph->operators) {
      for (auto const &t : op->output_tensors) {
        if (tensor_roles.find(t.guid) == tensor_roles.end()) {
          tensor_roles[t.guid] = {
              MIRAGE_ARG_BUF,
              static_cast<size_t>(t.data_offset >= 0 ? t.data_offset : 0)};
        }
      }
    }
  }

  // Extract per-kernel metadata from KN_CUSTOMIZED_OP operators.
  for (auto const *op : graph->graph->operators) {
    if (op->op_type != mirage::type::KN_CUSTOMIZED_OP) {
      continue;
    }
    auto const *custom_op =
        static_cast<mirage::kernel::KNCustomizedOp const *>(op);
    auto const &bg = custom_op->bgraph;

    KernelInfo ki{};
    ki.func_name = "custom_kernel_" + std::to_string(source->kernels.size());
    ki.smem_bytes = 0;
    ki.grid_dim[0] = bg.grid_dim.x;
    ki.grid_dim[1] = bg.grid_dim.y;
    ki.grid_dim[2] = bg.grid_dim.z;
    ki.block_dim[0] = bg.block_dim.x;
    ki.block_dim[1] = bg.block_dim.y;
    ki.block_dim[2] = bg.block_dim.z;

    // Build argument list: outputs first, then inputs (matches transpiler
    // convention in transpiler_kn.cc line 626).
    for (auto const &t : custom_op->output_tensors) {
      auto it = tensor_roles.find(t.guid);
      if (it != tensor_roles.end()) {
        ki.args.push_back({it->second.source, it->second.index_or_offset});
      } else {
        // Fallback: treat as intermediate at offset 0.
        ki.args.push_back({MIRAGE_ARG_BUF, 0});
      }
    }
    for (auto const &t : custom_op->input_tensors) {
      auto it = tensor_roles.find(t.guid);
      if (it != tensor_roles.end()) {
        ki.args.push_back({it->second.source, it->second.index_or_offset});
      } else {
        ki.args.push_back({MIRAGE_ARG_BUF, 0});
      }
    }

    source->kernels.push_back(std::move(ki));
  }

  // If we have kernel metadata and max_smem, assign it. The transpiler gives
  // a single max_smem_size across all kernels.
  for (auto &ki : source->kernels) {
    ki.smem_bytes = source->max_smem;
  }

  // Verify required traits if specified.
  if (required != 0 && (source->traits & required) != required) {
    delete source;
    return MIRAGE_STATUS_UNSUPPORTED;
  }

  *out = source;
  return MIRAGE_STATUS_OK;
}

void mirage_source_destroy(mirage_source_t *source) { delete source; }

mirage_source_traits_t mirage_source_traits(mirage_source_t const *source) {
  if (source == nullptr) {
    return 0;
  }
  return source->traits;
}

char const *mirage_source_code(mirage_source_t const *source) {
  if (source == nullptr) {
    return "";
  }
  return source->code.c_str();
}

size_t mirage_source_code_len(mirage_source_t const *source) {
  if (source == nullptr) {
    return 0;
  }
  return source->code.size();
}

size_t mirage_source_buf_size(mirage_source_t const *source) {
  if (source == nullptr) {
    return 0;
  }
  return source->buf_size;
}

size_t mirage_source_max_smem(mirage_source_t const *source) {
  if (source == nullptr) {
    return 0;
  }
  return source->max_smem;
}

size_t mirage_source_num_outputs(mirage_source_t const *source) {
  if (source == nullptr) {
    return 0;
  }
  return source->output_directives.size();
}

mirage_status_t mirage_source_output_spec(mirage_source_t const *source,
                                          size_t index,
                                          mirage_tensor_spec_t *out) {
  if (source == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (index >= source->output_directives.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }

  auto const &dir = source->output_directives[index];

  // Zero-initialize.
  *out = {};

  // The output directives don't carry dtype directly; default to F16.
  out->dtype = MIRAGE_DTYPE_F16;
  out->rank = static_cast<uint32_t>(dir.shape.size());
  if (out->rank > MIRAGE_MAX_RANK) {
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }

  for (uint32_t i = 0; i < out->rank; ++i) {
    out->dims[i] = dir.shape[i];
  }
  for (uint32_t i = 0; i < out->rank && i < dir.strides.size(); ++i) {
    out->strides[i] = static_cast<int64_t>(dir.strides[i]);
  }

  return MIRAGE_STATUS_OK;
}

size_t mirage_source_num_kernels(mirage_source_t const *source) {
  if (source == nullptr) {
    return 0;
  }
  return source->kernels.size();
}

mirage_status_t mirage_source_kernel_meta(mirage_source_t const *source,
                                          size_t index,
                                          mirage_kernel_meta_t *out) {
  if (source == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (index >= source->kernels.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }

  auto const &ki = source->kernels[index];
  out->func_name = ki.func_name.c_str();
  out->func_name_len = ki.func_name.size();
  out->smem_bytes = ki.smem_bytes;
  out->grid_dim[0] = ki.grid_dim[0];
  out->grid_dim[1] = ki.grid_dim[1];
  out->grid_dim[2] = ki.grid_dim[2];
  out->block_dim[0] = ki.block_dim[0];
  out->block_dim[1] = ki.block_dim[1];
  out->block_dim[2] = ki.block_dim[2];

  return MIRAGE_STATUS_OK;
}

size_t mirage_source_kernel_num_args(mirage_source_t const *source,
                                     size_t kernel_index) {
  if (source == nullptr || kernel_index >= source->kernels.size()) {
    return 0;
  }
  return source->kernels[kernel_index].args.size();
}

mirage_status_t mirage_source_kernel_arg(mirage_source_t const *source,
                                         size_t kernel_index,
                                         size_t arg_index,
                                         mirage_kernel_arg_t *out) {
  if (source == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (kernel_index >= source->kernels.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto const &ki = source->kernels[kernel_index];
  if (arg_index >= ki.args.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }

  out->source = ki.args[arg_index].source;
  out->index_or_offset = ki.args[arg_index].index_or_offset;
  return MIRAGE_STATUS_OK;
}
