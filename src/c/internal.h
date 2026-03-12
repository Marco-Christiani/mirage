/// Internal shared definitions for the C API implementation files.
/// Not part of the public API.
#pragma once

#include "mirage/kernel/graph.h"

#include <vector>

struct mirage_graph {
  /// The kernel graph. Owned when `owns_graph` is true.
  mirage::kernel::Graph *graph;
  bool owns_graph;
  std::vector<mirage::kernel::DTensor> tensors;
  std::vector<std::vector<size_t>> input_strides;

  /// Construct a new owned graph (for user-created graphs).
  mirage_graph()
      : graph(new mirage::kernel::Graph(dim3(1, 1, 1),
                                        /*disable_fingerprint=*/true)),
        owns_graph(true) {}

  /// Wrap an externally-allocated graph (for search candidates).
  /// Takes ownership of `g`.
  explicit mirage_graph(mirage::kernel::Graph *g)
      : graph(g), owns_graph(true) {}

  ~mirage_graph() {
    if (owns_graph) {
      delete graph;
    }
  }

  // Non-copyable, non-movable.
  mirage_graph(mirage_graph const &) = delete;
  mirage_graph &operator=(mirage_graph const &) = delete;
};
