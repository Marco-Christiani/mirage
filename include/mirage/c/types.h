#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { MIRAGE_MAX_RANK = 4 };

typedef enum mirage_status {
  MIRAGE_STATUS_OK = 0,
  MIRAGE_STATUS_INVALID_ARGUMENT = 1,
  MIRAGE_STATUS_INTERNAL_ERROR = 2,
  MIRAGE_STATUS_UNSUPPORTED = 3,
  MIRAGE_STATUS_NOT_FOUND = 4,
} mirage_status_t;

char const *mirage_status_string(mirage_status_t status);

typedef enum mirage_dtype {
  MIRAGE_DTYPE_F16 = 0,
  MIRAGE_DTYPE_BF16 = 1,
  MIRAGE_DTYPE_F32 = 2,
  MIRAGE_DTYPE_F64 = 3,
} mirage_dtype_t;

typedef enum mirage_unary_op {
  MIRAGE_UNARY_EXP = 0,
  MIRAGE_UNARY_SQRT = 1,
  MIRAGE_UNARY_SILU = 2,
  MIRAGE_UNARY_GELU = 3,
  MIRAGE_UNARY_RELU = 4,
  MIRAGE_UNARY_LOG = 5,
} mirage_unary_op_t;

typedef enum mirage_binary_op {
  MIRAGE_BINARY_ADD = 0,
  MIRAGE_BINARY_MUL = 1,
  MIRAGE_BINARY_DIV = 2,
  MIRAGE_BINARY_POW = 3,
} mirage_binary_op_t;

/// Tensor shape + dtype descriptor. Used for graph inputs and IR inspection.
/// Set stride entries to 0 for row-major default.
typedef struct mirage_tensor_spec {
  mirage_dtype_t dtype;
  uint32_t rank;
  int64_t dims[MIRAGE_MAX_RANK];
  int64_t strides[MIRAGE_MAX_RANK];
} mirage_tensor_spec_t;

/// Opaque handles.
typedef struct mirage_graph mirage_graph_t;
typedef struct mirage_search_result mirage_search_result_t;
typedef struct mirage_device mirage_device_t;
typedef struct mirage_source mirage_source_t;
typedef uint32_t mirage_tensor_t;

#ifdef __cplusplus
}
#endif
