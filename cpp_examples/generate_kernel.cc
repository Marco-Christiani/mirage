#include "mirage/kernel/graph.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpile.h"
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace mirage;

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <output_cu_file>\n";
    return 1;
  }

  int M = std::atoi(argv[1]);
  int N = std::atoi(argv[2]);
  int K = std::atoi(argv[3]);
  char const *output_file = argv[4];

  std::cout << "Generating kernel for matmul " << M << "x" << N << "x" << K
            << "\n";

  kernel::Graph graph;

  kernel::DTensor A =
      graph.new_input({M, K},
                      {static_cast<size_t>(K), 1}, // row-major strides
                      // type::DT_FLOAT16,
                      type::DT_FLOAT32,
                      layout::DmemRowMajor);

  kernel::DTensor B =
      graph.new_input({K, N},
                      // {1, static_cast<size_t>(K)}, // column-major strides
                      {static_cast<size_t>(N), 1}, // row-major strides
                      // type::DT_FLOAT16,
                      type::DT_FLOAT32,
                      // layout::DmemColumnMajor);
                      layout::DmemRowMajor);

  kernel::DTensor C = graph.matmul(A, B);
  graph.mark_output(C);

  std::vector<std::vector<size_t>> input_strides = {
      {static_cast<size_t>(K), 1}, // A: row-major
      // {1, static_cast<size_t>(K)}  // B: column-major
      {static_cast<size_t>(N), 1}, // B: row-major
  };

  transpiler::TranspilerConfig config;
  config.target_cc = 86;
  config.profiling = false;
  config.num_consumer_wgs = 0;
  config.num_producer_wgs = 0;
  config.pipeline_stages = 0;
  config.enable_online_softmax = false;

  std::cout << "Transpiling...\n";
  transpiler::TranspileResult result =
      transpiler::transpile(&graph, config, input_strides);

  if (result.code.empty()) {
    std::cerr << "❌ Transpilation failed - no code generated\n";
    std::cerr << "Error type: " << static_cast<int>(result.error_type) << "\n";
    return 1;
  }

  std::ofstream out(output_file);
  if (!out) {
    std::cerr << "Failed to open output file: " << output_file << "\n";
    return 1;
  }
  out << result.code;
  out.close();

  std::cout << "✓ Generated: " << output_file << "\n";
  std::cout << "  Workspace buffer size: " << result.buf_size << " bytes";
  if (result.buf_size > 1024 * 1024) {
    std::cout << " (" << (result.buf_size / (1024.0 * 1024.0)) << " MB)";
  }
  std::cout << "\n";
  std::cout << "  Max shared memory: " << result.max_smem_size << " bytes\n";
  std::cout << "  Profiler buffer size: " << result.profiler_buf_size
            << " bytes\n";

  std::cout << "  Output tensors:\n";
  for (size_t i = 0; i < result.output_directives.size(); i++) {
    auto const &dir = result.output_directives[i];
    std::cout << "    [" << i << "] alloc_size=" << dir.alloc_size
              << " elements, shape=[";
    for (size_t j = 0; j < dir.shape.size(); j++) {
      std::cout << dir.shape[j];
      if (j < dir.shape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "], strides=[";
    for (size_t j = 0; j < dir.strides.size(); j++) {
      std::cout << dir.strides[j];
      if (j < dir.strides.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";
  }

  return 0;
}
