import mirage as mi
import torch

if __name__ == "__main__":
    M, N, K = 16, 64, 128

    graph = mi.new_kernel_graph()

    A = graph.new_input(dims=(M, K), dtype=mi.float32)
    B = graph.new_input(dims=(K, N), dtype=mi.float32)
    C = graph.matmul(A, B)
    graph.mark_output(C)

    optimized_graph = graph.superoptimize(
        config="matmul", backend="cuda", warmup_iters=16, profile_iters=1000, save_codes=False
    )

    a_torch = torch.arange(1, M * K + 1, dtype=torch.float32, device="cuda").reshape(M, K)
    b_torch = torch.arange(
        a_torch.numel() + 1, a_torch.numel() + K * N + 1, dtype=torch.float32, device="cuda"
    ).reshape(K, N)

    inputs = [a_torch, b_torch]
    output = optimized_graph(inputs=inputs)

    torch.cuda.synchronize()

    print("Got", len(output), "outputs")
    print(output[0].shape)
    print(output[0].cpu())
    print(float(output[0][-1][-1]))
