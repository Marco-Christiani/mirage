/// Layer 3: Read-only inspection of a kernel graph's structure.
///
/// Enables external transpilers to walk the graph and generate their own code
/// (e.g. MLIR) instead of using Mirage's in-tree CUDA transpiler.
#pragma once

#include "mirage/c/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Kernel-level operator types
// ---------------------------------------------------------------------------

typedef enum mirage_kn_op_type {
  MIRAGE_KN_INPUT = 1001,
  MIRAGE_KN_OUTPUT = 1002,
  MIRAGE_KN_MATMUL = 1003,
  MIRAGE_KN_EXP = 1100,
  MIRAGE_KN_SQUARE = 1101,
  MIRAGE_KN_SQRT = 1102,
  MIRAGE_KN_SILU = 1104,
  MIRAGE_KN_SIGMOID = 1105,
  MIRAGE_KN_GELU = 1106,
  MIRAGE_KN_RELU = 1150,
  MIRAGE_KN_CLAMP = 1151,
  MIRAGE_KN_LOG = 1160,
  MIRAGE_KN_ADD = 1200,
  MIRAGE_KN_MUL = 1201,
  MIRAGE_KN_DIV = 1202,
  MIRAGE_KN_POW = 1203,
  MIRAGE_KN_REDUCTION_0 = 1300,
  MIRAGE_KN_REDUCTION_1 = 1301,
  MIRAGE_KN_REDUCTION_2 = 1302,
  MIRAGE_KN_RMS_NORM = 1350,
  MIRAGE_KN_ALLREDUCE = 1900,
  MIRAGE_KN_CUSTOMIZED = 1999,
} mirage_kn_op_type_t;

// ---------------------------------------------------------------------------
// Threadblock-level operator types
// ---------------------------------------------------------------------------

typedef enum mirage_tb_op_type {
  MIRAGE_TB_INPUT = 2001,
  MIRAGE_TB_OUTPUT = 2002,
  MIRAGE_TB_MATMUL = 2003,
  MIRAGE_TB_EXP = 2100,
  MIRAGE_TB_SQUARE = 2101,
  MIRAGE_TB_SQRT = 2102,
  MIRAGE_TB_SILU = 2104,
  MIRAGE_TB_SIGMOID = 2105,
  MIRAGE_TB_GELU = 2106,
  MIRAGE_TB_RELU = 2150,
  MIRAGE_TB_CLAMP = 2151,
  MIRAGE_TB_LOG = 2160,
  MIRAGE_TB_ADD = 2200,
  MIRAGE_TB_MUL = 2201,
  MIRAGE_TB_DIV = 2202,
  MIRAGE_TB_SUB = 2203,
  MIRAGE_TB_POW = 2204,
  MIRAGE_TB_REDUCTION_0 = 2301,
  MIRAGE_TB_REDUCTION_1 = 2302,
  MIRAGE_TB_REDUCTION_2 = 2303,
  MIRAGE_TB_REDUCTION_0_TO_DIMX = 2304,
  MIRAGE_TB_REDUCTION_1_TO_DIMX = 2305,
  MIRAGE_TB_REDUCTION_2_TO_DIMX = 2306,
  MIRAGE_TB_RMS_NORM = 2350,
  MIRAGE_TB_FORLOOP_ACCUM_NO_RED = 2500,
  MIRAGE_TB_FORLOOP_ACCUM_RED_LD_SUM = 2501,
  MIRAGE_TB_FORLOOP_ACCUM_RED_LD_MEAN = 2502,
  MIRAGE_TB_FORLOOP_ACCUM_RED_LD_RMS = 2503,
  MIRAGE_TB_FORLOOP_ACCUM_REDTOX_LD_SUM = 2504,
  MIRAGE_TB_FORLOOP_ACCUM_NO_RED_RESCALE = 2505,
  MIRAGE_TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE = 2506,
  MIRAGE_TB_FORLOOP_ACCUM_MAX = 2507,
  MIRAGE_TB_CUSTOMIZED = 2999,
} mirage_tb_op_type_t;

// ---------------------------------------------------------------------------
// Kernel graph walk
// ---------------------------------------------------------------------------

/// Number of operators in the kernel graph (topological order).
size_t mirage_ir_num_ops(mirage_graph_t const *graph);

/// Operator type at index.
mirage_kn_op_type_t mirage_ir_op_type(mirage_graph_t const *graph,
                                      size_t op_index);

/// Operator input/output tensor handles.
size_t mirage_ir_op_num_inputs(mirage_graph_t const *graph, size_t op_index);
size_t mirage_ir_op_num_outputs(mirage_graph_t const *graph, size_t op_index);

mirage_tensor_t mirage_ir_op_input(mirage_graph_t const *graph,
                                   size_t op_index,
                                   size_t tensor_index);
mirage_tensor_t mirage_ir_op_output(mirage_graph_t const *graph,
                                    size_t op_index,
                                    size_t tensor_index);

/// Tensor metadata.
mirage_status_t mirage_ir_tensor_spec(mirage_graph_t const *graph,
                                      mirage_tensor_t tensor,
                                      mirage_tensor_spec_t *out);

// ---------------------------------------------------------------------------
// Customized op: threadblock graph
// ---------------------------------------------------------------------------

/// Opaque handle to a threadblock graph (read-only, owned by the parent KN op).
typedef struct mirage_tbgraph mirage_tbgraph_t;

/// Get the threadblock graph from a KN_CUSTOMIZED op.
/// Returns NOT_FOUND if the op is not a customized op.
mirage_status_t mirage_ir_op_tbgraph(mirage_graph_t const *graph,
                                     size_t op_index,
                                     mirage_tbgraph_t const **out);

/// Threadblock graph launch configuration.
void mirage_ir_tbgraph_grid_dim(mirage_tbgraph_t const *tbg, uint32_t out[3]);
void mirage_ir_tbgraph_block_dim(mirage_tbgraph_t const *tbg, uint32_t out[3]);
int32_t mirage_ir_tbgraph_forloop_range(mirage_tbgraph_t const *tbg);
int32_t mirage_ir_tbgraph_reduction_dimx(mirage_tbgraph_t const *tbg);

/// Threadblock operator walk.
size_t mirage_ir_tbgraph_num_ops(mirage_tbgraph_t const *tbg);
mirage_tb_op_type_t mirage_ir_tbgraph_op_type(mirage_tbgraph_t const *tbg,
                                               size_t op_index);

/// Shared-memory tensor spec for TB operator inputs/outputs.
typedef struct mirage_stensor_spec {
  mirage_dtype_t dtype;
  uint32_t rank;
  int64_t dims[MIRAGE_MAX_RANK];
  int32_t smem_offset;
} mirage_stensor_spec_t;

size_t mirage_ir_tbop_num_inputs(mirage_tbgraph_t const *tbg, size_t op_index);
size_t mirage_ir_tbop_num_outputs(mirage_tbgraph_t const *tbg, size_t op_index);

mirage_status_t mirage_ir_tbop_input_spec(mirage_tbgraph_t const *tbg,
                                          size_t op_index,
                                          size_t tensor_index,
                                          mirage_stensor_spec_t *out);
mirage_status_t mirage_ir_tbop_output_spec(mirage_tbgraph_t const *tbg,
                                           size_t op_index,
                                           size_t tensor_index,
                                           mirage_stensor_spec_t *out);

// ---------------------------------------------------------------------------
// TB input/output mapping (for external transpilers)
// ---------------------------------------------------------------------------

/// For TB_INPUT ops: how the input tensor maps to grid dimensions.
typedef struct mirage_tb_input_info {
  int32_t input_map[3]; // maps dims to grid.x/y/z (-1 = unmapped)
  int32_t forloop_dim;  // which dim iterates over the forloop (-1 = none)
} mirage_tb_input_info_t;

mirage_status_t mirage_ir_tbop_input_info(mirage_tbgraph_t const *tbg,
                                          size_t op_index,
                                          mirage_tb_input_info_t *out);

/// For TB_OUTPUT ops: how the output maps to grid dimensions + epilogue type.
typedef enum mirage_epilogue {
  MIRAGE_EPILOGUE_NONE = 0,
  MIRAGE_EPILOGUE_ALLREDUCE = 1,
  MIRAGE_EPILOGUE_ALLTOALL = 2,
} mirage_epilogue_t;

typedef struct mirage_tb_output_info {
  int32_t output_map[3];
  int32_t forloop_dim;
  mirage_epilogue_t epilogue;
} mirage_tb_output_info_t;

mirage_status_t mirage_ir_tbop_output_info(mirage_tbgraph_t const *tbg,
                                           size_t op_index,
                                           mirage_tb_output_info_t *out);

// ---------------------------------------------------------------------------
// C++ escape hatch
// ---------------------------------------------------------------------------

#ifdef __cplusplus
/// Returns the underlying mirage::kernel::Graph pointer.
/// The returned pointer is valid for the lifetime of the mirage_graph_t.
/// This couples the consumer to Mirage's C++ ABI.
void *mirage_ir_graph_raw(mirage_graph_t *graph);

/// Returns the underlying mirage::threadblock::Graph pointer.
void const *mirage_ir_tbgraph_raw(mirage_tbgraph_t const *tbg);
#endif

#ifdef __cplusplus
}
#endif
