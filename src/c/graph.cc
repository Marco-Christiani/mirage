#include "internal.h"

#include "mirage/c/graph.h"
#include "mirage/c/types.h"

#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/search/search_c.h"
#include "mirage/type.h"

#include <new>
#include <vector>

namespace {

mirage::type::DataType to_mirage_dtype(mirage_dtype_t dtype) {
  switch (dtype) {
  case MIRAGE_DTYPE_F16:
    return mirage::type::DT_FLOAT16;
  case MIRAGE_DTYPE_BF16:
    return mirage::type::DT_BFLOAT16;
  case MIRAGE_DTYPE_F32:
    return mirage::type::DT_FLOAT32;
  case MIRAGE_DTYPE_F64:
    return mirage::type::DT_DOUBLE;
  default:
    return mirage::type::DT_UNKNOWN;
  }
}

bool to_unary_operator_type(mirage_unary_op_t op,
                            mirage::type::KNOperatorType *out_type) {
  switch (op) {
  case MIRAGE_UNARY_EXP:
    *out_type = mirage::type::KN_EXP_OP;
    return true;
  case MIRAGE_UNARY_SQRT:
    *out_type = mirage::type::KN_SQRT_OP;
    return true;
  case MIRAGE_UNARY_SILU:
    *out_type = mirage::type::KN_SILU_OP;
    return true;
  case MIRAGE_UNARY_GELU:
    *out_type = mirage::type::KN_GELU_OP;
    return true;
  case MIRAGE_UNARY_RELU:
    *out_type = mirage::type::KN_RELU_OP;
    return true;
  case MIRAGE_UNARY_LOG:
    *out_type = mirage::type::KN_LOG_OP;
    return true;
  default:
    return false;
  }
}

bool to_binary_operator_type(mirage_binary_op_t op,
                             mirage::type::KNOperatorType *out_type) {
  switch (op) {
  case MIRAGE_BINARY_ADD:
    *out_type = mirage::type::KN_ADD_OP;
    return true;
  case MIRAGE_BINARY_MUL:
    *out_type = mirage::type::KN_MUL_OP;
    return true;
  case MIRAGE_BINARY_DIV:
    *out_type = mirage::type::KN_DIV_OP;
    return true;
  case MIRAGE_BINARY_POW:
    *out_type = mirage::type::KN_POW_OP;
    return true;
  default:
    return false;
  }
}

constexpr uint32_t k_default_search_capacity = 1024;
constexpr uint32_t k_max_search_capacity = 4096;

} // namespace

// mirage_graph is defined in internal.h.

static mirage_status_t append_single_output(mirage_graph_t *g,
                                            mirage::kernel::KNOperator *op,
                                            mirage_tensor_t *out) {
  if (g == nullptr || out == nullptr || op == nullptr) {
    delete op;
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (op->output_tensors.size() != 1) {
    delete op;
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }
  g->graph->operators.push_back(op);
  g->tensors.push_back(op->output_tensors[0]);
  *out = static_cast<mirage_tensor_t>(g->tensors.size() - 1);
  return MIRAGE_STATUS_OK;
}

// ---------------------------------------------------------------------------
// mirage_search_result: owns candidate graphs
// ---------------------------------------------------------------------------

struct mirage_search_result {
  std::vector<mirage_graph_t *> candidates;

  ~mirage_search_result() {
    for (auto *c : candidates) {
      delete c;
    }
  }
};

// ---------------------------------------------------------------------------
// mirage_device: wraps device ordinal for search
// ---------------------------------------------------------------------------

struct mirage_device {
  int32_t ordinal;
};

// ---------------------------------------------------------------------------
// Public API: status string
// ---------------------------------------------------------------------------

char const *mirage_status_string(mirage_status_t status) {
  switch (status) {
  case MIRAGE_STATUS_OK:
    return "ok";
  case MIRAGE_STATUS_INVALID_ARGUMENT:
    return "invalid_argument";
  case MIRAGE_STATUS_INTERNAL_ERROR:
    return "internal_error";
  case MIRAGE_STATUS_UNSUPPORTED:
    return "unsupported";
  case MIRAGE_STATUS_NOT_FOUND:
    return "not_found";
  default:
    return "unknown_status";
  }
}

// ---------------------------------------------------------------------------
// Graph building
// ---------------------------------------------------------------------------

mirage_status_t mirage_graph_create(mirage_graph_t **out) {
  if (out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  try {
    *out = new mirage_graph_t{};
  } catch (std::bad_alloc const &) {
    *out = nullptr;
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }
  return MIRAGE_STATUS_OK;
}

void mirage_graph_destroy(mirage_graph_t *graph) { delete graph; }

mirage_status_t mirage_graph_new_input(mirage_graph_t *graph,
                                       mirage_tensor_spec_t const *spec,
                                       mirage_tensor_t *out) {
  if (graph == nullptr || spec == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (spec->rank == 0 || spec->rank > MIRAGE_MAX_RANK) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }

  auto m_dtype = to_mirage_dtype(spec->dtype);
  if (m_dtype == mirage::type::DT_UNKNOWN) {
    return MIRAGE_STATUS_UNSUPPORTED;
  }

  std::vector<int> shape(spec->rank);
  for (uint32_t i = 0; i < spec->rank; ++i) {
    if (spec->dims[i] <= 0 || spec->dims[i] > INT32_MAX) {
      return MIRAGE_STATUS_INVALID_ARGUMENT;
    }
    shape[i] = static_cast<int>(spec->dims[i]);
  }

  // Compute strides: use caller-provided strides if non-zero, else row-major.
  std::vector<size_t> strides(spec->rank);
  bool has_strides = false;
  for (uint32_t i = 0; i < spec->rank; ++i) {
    if (spec->strides[i] != 0) {
      has_strides = true;
      break;
    }
  }
  if (has_strides) {
    for (uint32_t i = 0; i < spec->rank; ++i) {
      strides[i] = static_cast<size_t>(spec->strides[i]);
    }
  } else {
    size_t s = 1;
    for (uint32_t i = spec->rank; i-- > 0;) {
      strides[i] = s;
      s *= static_cast<size_t>(shape[i]);
    }
  }

  auto t = graph->graph->new_input(shape, strides, m_dtype,
                                   mirage::layout::DmemRowMajor);
  graph->tensors.push_back(t);
  graph->input_strides.push_back(strides);
  *out = static_cast<mirage_tensor_t>(graph->tensors.size() - 1);
  return MIRAGE_STATUS_OK;
}

mirage_status_t mirage_graph_matmul(mirage_graph_t *graph, mirage_tensor_t lhs,
                                    mirage_tensor_t rhs,
                                    mirage_tensor_t *out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (lhs >= graph->tensors.size() || rhs >= graph->tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto *op =
      graph->graph->create_matmul_op(graph->tensors[lhs], graph->tensors[rhs]);
  return append_single_output(graph, op, out);
}

mirage_status_t mirage_graph_unary(mirage_graph_t *graph, mirage_unary_op_t op,
                                   mirage_tensor_t input,
                                   mirage_tensor_t *out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (input >= graph->tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  mirage::type::KNOperatorType op_type{};
  if (!to_unary_operator_type(op, &op_type)) {
    return MIRAGE_STATUS_UNSUPPORTED;
  }
  auto *created =
      graph->graph->create_elementunary_op(graph->tensors[input], op_type);
  return append_single_output(graph, created, out);
}

mirage_status_t mirage_graph_binary(mirage_graph_t *graph,
                                    mirage_binary_op_t op, mirage_tensor_t lhs,
                                    mirage_tensor_t rhs,
                                    mirage_tensor_t *out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (lhs >= graph->tensors.size() || rhs >= graph->tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  mirage::type::KNOperatorType op_type{};
  if (!to_binary_operator_type(op, &op_type)) {
    return MIRAGE_STATUS_UNSUPPORTED;
  }
  auto *created = graph->graph->create_elementbinary_op(
      graph->tensors[lhs], graph->tensors[rhs], op_type);
  return append_single_output(graph, created, out);
}

mirage_status_t mirage_graph_reduction(mirage_graph_t *graph,
                                       mirage_tensor_t input, int32_t dim,
                                       int32_t factor, mirage_tensor_t *out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (input >= graph->tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto *op =
      graph->graph->create_reduction_op(graph->tensors[input], dim, factor);
  return append_single_output(graph, op, out);
}

mirage_status_t mirage_graph_rms_norm(mirage_graph_t *graph,
                                      mirage_tensor_t input,
                                      int32_t normalized_size,
                                      mirage_tensor_t *out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (input >= graph->tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto *op = graph->graph->create_rms_norm_op(graph->tensors[input],
                                              {normalized_size});
  return append_single_output(graph, op, out);
}

mirage_status_t mirage_graph_mark_output(mirage_graph_t *graph,
                                         mirage_tensor_t tensor) {
  if (graph == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (tensor >= graph->tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  graph->graph->mark_output(graph->tensors[tensor]);
  return MIRAGE_STATUS_OK;
}

// ---------------------------------------------------------------------------
// Device handle
// ---------------------------------------------------------------------------

mirage_status_t mirage_device_create(int32_t device_ordinal,
                                     mirage_device_t **out) {
  if (out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  try {
    *out = new mirage_device_t{device_ordinal};
  } catch (std::bad_alloc const &) {
    *out = nullptr;
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }
  return MIRAGE_STATUS_OK;
}

void mirage_device_destroy(mirage_device_t *device) {
  if (device != nullptr) {
    mirage::kernel::DeviceMemoryManager::release_instance();
  }
  delete device;
}

mirage_status_t mirage_device_mem_info(mirage_device_t const * /*device*/,
                                       size_t *out_free, size_t *out_total) {
  if (out_free == nullptr || out_total == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
#ifdef MIRAGE_FINGERPRINT_USE_CUDA
  cudaError_t err = cudaMemGetInfo(out_free, out_total);
  if (err != cudaSuccess) {
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }
  return MIRAGE_STATUS_OK;
#else
  return MIRAGE_STATUS_UNSUPPORTED;
#endif
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

static bool append_search_dim3(int32_t const *src, size_t count,
                               std::vector<mirage::search_c::MInt3> *dst) {
  if (src == nullptr || count == 0) {
    return true;
  }
  // Each dim3 is packed as 3 consecutive int32_t values.
  if (count % 3 != 0) {
    return false;
  }
  size_t n = count / 3;
  dst->reserve(n);
  for (size_t i = 0; i < n; ++i) {
    dst->push_back({src[i * 3], src[i * 3 + 1], src[i * 3 + 2]});
  }
  return true;
}

static bool append_search_udim3(int32_t const *src, size_t count,
                                std::vector<mirage::search_c::MDim3> *dst) {
  if (src == nullptr || count == 0) {
    return true;
  }
  if (count % 3 != 0) {
    return false;
  }
  size_t n = count / 3;
  dst->reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (src[i * 3] < 0 || src[i * 3 + 1] < 0 || src[i * 3 + 2] < 0) {
      return false;
    }
    dst->push_back({static_cast<unsigned>(src[i * 3]),
                    static_cast<unsigned>(src[i * 3 + 1]),
                    static_cast<unsigned>(src[i * 3 + 2])});
  }
  return true;
}

mirage_status_t mirage_search(mirage_device_t *device,
                              mirage_graph_t const *graph,
                              mirage_search_options_t const *options,
                              mirage_search_result_t **out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  // device may be null for CPU-only fallback in the future; currently required.
  (void)device;

  uint32_t max_candidates = k_default_search_capacity;
  bool verbose = false;
  bool formal_verify = false;

  std::vector<mirage::search_c::MInt3> cimaps;
  std::vector<mirage::search_c::MInt3> comaps;
  std::vector<mirage::search_c::MDim3> cgriddims;
  std::vector<mirage::search_c::MDim3> cblockdims;
  std::vector<int> cfmaps;
  std::vector<int> cfranges;

  if (options != nullptr) {
    if (options->max_candidates != 0) {
      max_candidates = options->max_candidates;
    }
    if (max_candidates > k_max_search_capacity) {
      return MIRAGE_STATUS_INVALID_ARGUMENT;
    }
    verbose = options->verbose != 0;
    formal_verify = options->formal_verify != 0;

    if (!append_search_dim3(options->imaps, options->num_imaps, &cimaps) ||
        !append_search_dim3(options->omaps, options->num_omaps, &comaps) ||
        !append_search_udim3(options->grid_dims, options->num_grid_dims,
                             &cgriddims) ||
        !append_search_udim3(options->block_dims, options->num_block_dims,
                             &cblockdims)) {
      return MIRAGE_STATUS_INVALID_ARGUMENT;
    }

    if (options->fmaps != nullptr) {
      cfmaps.assign(options->fmaps, options->fmaps + options->num_fmaps);
    }
    if (options->franges != nullptr) {
      cfranges.assign(options->franges, options->franges + options->num_franges);
    }
  }

  std::vector<mirage::kernel::Graph *> raw_candidates(max_candidates, nullptr);

  int num = mirage::search_c::cython_search(
      graph->graph, "cuda", static_cast<int>(max_candidates),
      raw_candidates.data(), cimaps, comaps, cgriddims, cblockdims, cfmaps,
      cfranges,
      /*filename=*/nullptr, verbose, /*default_config=*/nullptr, formal_verify);

  auto *result = new (std::nothrow) mirage_search_result_t{};
  if (result == nullptr) {
    for (int i = 0; i < num; ++i) {
      delete raw_candidates[i];
    }
    return MIRAGE_STATUS_INTERNAL_ERROR;
  }

  for (int i = 0; i < num; ++i) {
    if (raw_candidates[i] == nullptr) {
      continue;
    }
    // Wrap the candidate graph directly — mirage_graph takes ownership.
    auto *wrapper = new (std::nothrow) mirage_graph_t(raw_candidates[i]);
    if (wrapper == nullptr) {
      delete raw_candidates[i];
      continue;
    }

    // Rebuild the tensor table from the graph's operators.
    for (auto const *op : wrapper->graph->operators) {
      for (auto const &t : op->output_tensors) {
        wrapper->tensors.push_back(t);
      }
    }

    // Reconstruct input_strides from KN_INPUT_OP operators.
    for (auto const *op : wrapper->graph->operators) {
      if (op->op_type == mirage::type::KN_INPUT_OP) {
        auto const *input_op =
            static_cast<mirage::kernel::KNInputOp const *>(op);
        wrapper->input_strides.push_back(input_op->input_strides);
      }
    }

    result->candidates.push_back(wrapper);
  }

  *out = result;
  return MIRAGE_STATUS_OK;
}

void mirage_search_result_destroy(mirage_search_result_t *result) {
  delete result;
}

size_t mirage_search_result_count(mirage_search_result_t const *result) {
  if (result == nullptr) {
    return 0;
  }
  return result->candidates.size();
}

mirage_graph_t const *
mirage_search_result_get(mirage_search_result_t const *result, size_t index) {
  if (result == nullptr || index >= result->candidates.size()) {
    return nullptr;
  }
  return result->candidates[index];
}
