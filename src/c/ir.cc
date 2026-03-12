#include "internal.h"

#include "mirage/c/ir.h"
#include "mirage/c/types.h"

#include "mirage/kernel/customized.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/type.h"

#include <cstdint>
#include <vector>

// mirage_tbgraph wraps a read-only pointer to a threadblock::Graph.
struct mirage_tbgraph {
  mirage::threadblock::Graph const *ptr;
};

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

mirage_epilogue_t from_mirage_epilogue(mirage::type::TBEpilogueType ep) {
  switch (ep) {
  case mirage::type::TB_EPILOGUE_NONE:
    return MIRAGE_EPILOGUE_NONE;
  case mirage::type::TB_EPILOGUE_ALLREDUCE:
    return MIRAGE_EPILOGUE_ALLREDUCE;
  case mirage::type::TB_EPILOGUE_ALLTOALL:
    return MIRAGE_EPILOGUE_ALLTOALL;
  default:
    return MIRAGE_EPILOGUE_NONE;
  }
}

// Sentinel for invalid tensor handles.
constexpr mirage_tensor_t INVALID_TENSOR = UINT32_MAX;

} // namespace

// ---------------------------------------------------------------------------
// Kernel graph walk
// ---------------------------------------------------------------------------

size_t mirage_ir_num_ops(mirage_graph_t const *graph) {
  if (graph == nullptr) {
    return 0;
  }
  return graph->graph->operators.size();
}

mirage_kn_op_type_t mirage_ir_op_type(mirage_graph_t const *graph,
                                      size_t op_index) {
  if (graph == nullptr || op_index >= graph->graph->operators.size()) {
    return static_cast<mirage_kn_op_type_t>(0);
  }
  return static_cast<mirage_kn_op_type_t>(
      graph->graph->operators[op_index]->op_type);
}

size_t mirage_ir_op_num_inputs(mirage_graph_t const *graph, size_t op_index) {
  if (graph == nullptr || op_index >= graph->graph->operators.size()) {
    return 0;
  }
  return graph->graph->operators[op_index]->input_tensors.size();
}

size_t mirage_ir_op_num_outputs(mirage_graph_t const *graph, size_t op_index) {
  if (graph == nullptr || op_index >= graph->graph->operators.size()) {
    return 0;
  }
  return graph->graph->operators[op_index]->output_tensors.size();
}

mirage_tensor_t mirage_ir_op_input(mirage_graph_t const *graph,
                                   size_t op_index, size_t tensor_index) {
  if (graph == nullptr || op_index >= graph->graph->operators.size()) {
    return INVALID_TENSOR;
  }
  auto const *op = graph->graph->operators[op_index];
  if (tensor_index >= op->input_tensors.size()) {
    return INVALID_TENSOR;
  }
  // Find the tensor in our tensor table by GUID match.
  auto const &target = op->input_tensors[tensor_index];
  for (size_t i = 0; i < graph->tensors.size(); ++i) {
    if (graph->tensors[i].guid == target.guid) {
      return static_cast<mirage_tensor_t>(i);
    }
  }
  return INVALID_TENSOR;
}

mirage_tensor_t mirage_ir_op_output(mirage_graph_t const *graph,
                                    size_t op_index, size_t tensor_index) {
  if (graph == nullptr || op_index >= graph->graph->operators.size()) {
    return INVALID_TENSOR;
  }
  auto const *op = graph->graph->operators[op_index];
  if (tensor_index >= op->output_tensors.size()) {
    return INVALID_TENSOR;
  }
  auto const &target = op->output_tensors[tensor_index];
  for (size_t i = 0; i < graph->tensors.size(); ++i) {
    if (graph->tensors[i].guid == target.guid) {
      return static_cast<mirage_tensor_t>(i);
    }
  }
  return INVALID_TENSOR;
}

mirage_status_t mirage_ir_tensor_spec(mirage_graph_t const *graph,
                                      mirage_tensor_t tensor,
                                      mirage_tensor_spec_t *out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (tensor >= graph->tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }

  auto const &t = graph->tensors[tensor];
  *out = {};

  if (!from_mirage_dtype(t.data_type, &out->dtype)) {
    return MIRAGE_STATUS_UNSUPPORTED;
  }

  out->rank = static_cast<uint32_t>(t.num_dims);
  for (int i = 0; i < t.num_dims && i < MIRAGE_MAX_RANK; ++i) {
    out->dims[i] = t.dim[i];
  }
  // DTensor doesn't store strides directly; leave as 0 (row-major default).

  return MIRAGE_STATUS_OK;
}

// ---------------------------------------------------------------------------
// Customized op: threadblock graph
// ---------------------------------------------------------------------------

// Static pool of tbgraph wrappers. These are lightweight (just a pointer) and
// are valid for the lifetime of the parent mirage_graph_t.
// Thread-safety: not required (C API is single-threaded per graph).
static std::vector<mirage_tbgraph_t> s_tbgraph_pool;

mirage_status_t mirage_ir_op_tbgraph(mirage_graph_t const *graph,
                                     size_t op_index,
                                     mirage_tbgraph_t const **out) {
  if (graph == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (op_index >= graph->graph->operators.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }

  auto const *op = graph->graph->operators[op_index];
  if (op->op_type != mirage::type::KN_CUSTOMIZED_OP) {
    return MIRAGE_STATUS_NOT_FOUND;
  }

  auto const *custom_op =
      static_cast<mirage::kernel::KNCustomizedOp const *>(op);

  mirage_tbgraph_t wrapper{&custom_op->bgraph};
  s_tbgraph_pool.push_back(wrapper);
  *out = &s_tbgraph_pool.back();
  return MIRAGE_STATUS_OK;
}

void mirage_ir_tbgraph_grid_dim(mirage_tbgraph_t const *tbg, uint32_t out[3]) {
  if (tbg == nullptr || tbg->ptr == nullptr || out == nullptr) {
    return;
  }
  out[0] = tbg->ptr->grid_dim.x;
  out[1] = tbg->ptr->grid_dim.y;
  out[2] = tbg->ptr->grid_dim.z;
}

void mirage_ir_tbgraph_block_dim(mirage_tbgraph_t const *tbg,
                                 uint32_t out[3]) {
  if (tbg == nullptr || tbg->ptr == nullptr || out == nullptr) {
    return;
  }
  out[0] = tbg->ptr->block_dim.x;
  out[1] = tbg->ptr->block_dim.y;
  out[2] = tbg->ptr->block_dim.z;
}

int32_t mirage_ir_tbgraph_forloop_range(mirage_tbgraph_t const *tbg) {
  if (tbg == nullptr || tbg->ptr == nullptr) {
    return 0;
  }
  return tbg->ptr->forloop_range;
}

int32_t mirage_ir_tbgraph_reduction_dimx(mirage_tbgraph_t const *tbg) {
  if (tbg == nullptr || tbg->ptr == nullptr) {
    return 0;
  }
  return tbg->ptr->reduction_dimx;
}

// ---------------------------------------------------------------------------
// Threadblock operator walk
// ---------------------------------------------------------------------------

size_t mirage_ir_tbgraph_num_ops(mirage_tbgraph_t const *tbg) {
  if (tbg == nullptr || tbg->ptr == nullptr) {
    return 0;
  }
  return tbg->ptr->operators.size();
}

mirage_tb_op_type_t mirage_ir_tbgraph_op_type(mirage_tbgraph_t const *tbg,
                                               size_t op_index) {
  if (tbg == nullptr || tbg->ptr == nullptr ||
      op_index >= tbg->ptr->operators.size()) {
    return static_cast<mirage_tb_op_type_t>(0);
  }
  return static_cast<mirage_tb_op_type_t>(
      tbg->ptr->operators[op_index]->op_type);
}

size_t mirage_ir_tbop_num_inputs(mirage_tbgraph_t const *tbg,
                                 size_t op_index) {
  if (tbg == nullptr || tbg->ptr == nullptr ||
      op_index >= tbg->ptr->operators.size()) {
    return 0;
  }
  return tbg->ptr->operators[op_index]->input_tensors.size();
}

size_t mirage_ir_tbop_num_outputs(mirage_tbgraph_t const *tbg,
                                  size_t op_index) {
  if (tbg == nullptr || tbg->ptr == nullptr ||
      op_index >= tbg->ptr->operators.size()) {
    return 0;
  }
  return tbg->ptr->operators[op_index]->output_tensors.size();
}

static mirage_status_t
fill_stensor_spec(mirage::threadblock::STensor const &st,
                  mirage_stensor_spec_t *out) {
  *out = {};
  if (!from_mirage_dtype(st.data_type, &out->dtype)) {
    return MIRAGE_STATUS_UNSUPPORTED;
  }
  out->rank = static_cast<uint32_t>(st.num_dims);
  for (int i = 0; i < st.num_dims && i < MIRAGE_MAX_RANK; ++i) {
    out->dims[i] = st.dim[i];
  }
  out->smem_offset = st.smem_offset;
  return MIRAGE_STATUS_OK;
}

mirage_status_t mirage_ir_tbop_input_spec(mirage_tbgraph_t const *tbg,
                                          size_t op_index,
                                          size_t tensor_index,
                                          mirage_stensor_spec_t *out) {
  if (tbg == nullptr || tbg->ptr == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (op_index >= tbg->ptr->operators.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto const *op = tbg->ptr->operators[op_index];
  if (tensor_index >= op->input_tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  return fill_stensor_spec(op->input_tensors[tensor_index], out);
}

mirage_status_t mirage_ir_tbop_output_spec(mirage_tbgraph_t const *tbg,
                                           size_t op_index,
                                           size_t tensor_index,
                                           mirage_stensor_spec_t *out) {
  if (tbg == nullptr || tbg->ptr == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (op_index >= tbg->ptr->operators.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto const *op = tbg->ptr->operators[op_index];
  if (tensor_index >= op->output_tensors.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  return fill_stensor_spec(op->output_tensors[tensor_index], out);
}

// ---------------------------------------------------------------------------
// TB input/output mapping
// ---------------------------------------------------------------------------

mirage_status_t mirage_ir_tbop_input_info(mirage_tbgraph_t const *tbg,
                                          size_t op_index,
                                          mirage_tb_input_info_t *out) {
  if (tbg == nullptr || tbg->ptr == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (op_index >= tbg->ptr->operators.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto const *op = tbg->ptr->operators[op_index];
  if (op->op_type != mirage::type::TB_INPUT_OP) {
    return MIRAGE_STATUS_NOT_FOUND;
  }
  auto const *input_op =
      static_cast<mirage::threadblock::TBInputOp const *>(op);

  out->input_map[0] = input_op->input_map.x;
  out->input_map[1] = input_op->input_map.y;
  out->input_map[2] = input_op->input_map.z;
  out->forloop_dim = input_op->forloop_dim;

  return MIRAGE_STATUS_OK;
}

mirage_status_t mirage_ir_tbop_output_info(mirage_tbgraph_t const *tbg,
                                           size_t op_index,
                                           mirage_tb_output_info_t *out) {
  if (tbg == nullptr || tbg->ptr == nullptr || out == nullptr) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  if (op_index >= tbg->ptr->operators.size()) {
    return MIRAGE_STATUS_INVALID_ARGUMENT;
  }
  auto const *op = tbg->ptr->operators[op_index];
  if (op->op_type != mirage::type::TB_OUTPUT_OP) {
    return MIRAGE_STATUS_NOT_FOUND;
  }
  auto const *output_op =
      static_cast<mirage::threadblock::TBOutputOp const *>(op);

  out->output_map[0] = output_op->output_map.x;
  out->output_map[1] = output_op->output_map.y;
  out->output_map[2] = output_op->output_map.z;
  out->forloop_dim = output_op->forloop_dim;
  out->epilogue = from_mirage_epilogue(output_op->epilogue);

  return MIRAGE_STATUS_OK;
}

// ---------------------------------------------------------------------------
// C++ escape hatch
// ---------------------------------------------------------------------------

void *mirage_ir_graph_raw(mirage_graph_t *graph) {
  if (graph == nullptr) {
    return nullptr;
  }
  return graph->graph;
}

void const *mirage_ir_tbgraph_raw(mirage_tbgraph_t const *tbg) {
  if (tbg == nullptr) {
    return nullptr;
  }
  return tbg->ptr;
}
