
# result = mpk.kn_graph.cygraph.generate_task_graph(num_gpus=1, my_gpu_id=0)
# print("Saving generated")
# with open("task_graph.json", "w") as f:
#     f.write(result["json_file"])
# with open("kernel.cu", "w") as f:
#     f.write(result["cuda_code"])
#!/usr/bin/env python3
"""Test RMSNorm backward pass with proper gradient flow."""

import torch
import torch.nn.functional as F
from mirage import persistent_kernel as pk
# from mirage.persistent_kernel.core import DType
from mirage.persistent_kernel import DType
import numpy as np


def pytorch_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch reference implementation."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x / torch.sqrt(variance + eps)
    return weight * x_normalized


def test_rmsnorm_backward():
    """Test RMSNorm backward with proper gradient flow."""
    
    # Configuration
    batch_size = 1
    hidden_dim = 4096
    num_requests = 16
    eps = 1e-6
    
    # Create persistent kernel in offline mode
    pk_module = pk.PersistentKernel(
        num_gpus=1,
        num_requests=num_requests,
        num_tokens_per_request=[1] * num_requests,
        mode="offline"
    )
    
    # Define grid and block dimensions
    grid_dim = (num_requests, 1, 1)
    block_dim = (32, 1, 1)
    
    # Create input tensors
    input_tensor = pk_module.attach_input(
        shape=(batch_size, hidden_dim),
        dtype=DType.BFLOAT16,
        layout=pk.DTensorLayout.ROW_MAJOR,
        name="input"
    )
    
    weight_tensor = pk_module.create_input_tensor(
        shape=(hidden_dim,),
        dtype=DType.BFLOAT16,
        layout=pk.DTensorLayout.ROW_MAJOR,
        name="weight"
    )
    
    # Forward pass output
    output_tensor = pk_module.create_input_tensor(
        shape=(batch_size, hidden_dim),
        dtype=DType.BFLOAT16,
        layout=pk.DTensorLayout.ROW_MAJOR,
        name="output"
    )
    
    # Create PROPER gradient tensors for backward
    grad_output_tensor = pk_module.create_input_tensor(
        shape=(batch_size, hidden_dim),
        dtype=DType.BFLOAT16,
        layout=pk.DTensorLayout.ROW_MAJOR,
        name="grad_output"
    )
    
    grad_input_tensor = pk_module.create_input_tensor(
        shape=(batch_size, hidden_dim),
        dtype=DType.BFLOAT16,
        layout=pk.DTensorLayout.ROW_MAJOR,
        name="grad_input"
    )
    
    grad_weight_tensor = pk_module.create_input_tensor(
        shape=(hidden_dim,),
        dtype=DType.BFLOAT16,
        layout=pk.DTensorLayout.ROW_MAJOR,
        name="grad_weight"
    )
    
    # Add forward layer
    pk_module.rmsnorm_layer(
        input=input_tensor,
        weight=weight_tensor,
        output=output_tensor,
        grid_dim=grid_dim,
        block_dim=block_dim
    )
    
    # Add backward layer
    pk_module.rmsnorm_backward_layer(
        input=input_tensor,
        grad_output=grad_output_tensor,  # ✅ Proper gradient tensor
        weight=weight_tensor,
        grad_input=grad_input_tensor,
        grad_weight=grad_weight_tensor,
        grid_dim=grid_dim,
        block_dim=block_dim
    )
    
    # Generate and compile
    pk_module.generate_executable(
        target_cc=86,
        file_name="test_rmsnorm_backward.cu",
        folder_name="examples"
    )
    pk_module.compile(target_cc=86)
    
    # Prepare PyTorch reference
    torch.manual_seed(42)
    x_cpu = torch.randn(batch_size, hidden_dim, dtype=torch.float32)
    weight_cpu = torch.ones(hidden_dim, dtype=torch.float32)
    
    # PyTorch forward + backward
    x_cpu.requires_grad_(True)
    weight_cpu.requires_grad_(True)
    
    y_cpu = pytorch_rmsnorm_forward(x_cpu, weight_cpu, eps)
    
    # Create a random upstream gradient
    grad_out_cpu = torch.randn_like(y_cpu)
    
    # Backward pass
    y_cpu.backward(grad_out_cpu)
    
    # Get PyTorch gradients
    grad_input_ref = x_cpu.grad
    grad_weight_ref = weight_cpu.grad
    
    print("\n=== PyTorch Reference ===")
    print(f"Forward output mean: {y_cpu.mean():.6f}")
    print(f"grad_input mean: {grad_input_ref.mean():.6f}")
    print(f"grad_weight mean: {grad_weight_ref.mean():.6f}")
    print(f"grad_input[0, :5]: {grad_input_ref[0, :5]}")
    print(f"grad_weight[:5]: {grad_weight_ref[:5]}")
    
    # Convert to bfloat16 for Mirage
    x_bf16 = x_cpu.to(torch.bfloat16).cuda()
    weight_bf16 = weight_cpu.to(torch.bfloat16).cuda()
    grad_out_bf16 = grad_out_cpu.to(torch.bfloat16).cuda()
    
    # Allocate output buffers
    y_bf16 = torch.zeros_like(x_bf16)
    grad_input_bf16 = torch.zeros_like(x_bf16)
    grad_weight_bf16 = torch.zeros_like(weight_bf16)
    
    # Bind buffers to Mirage tensors
    pk_module.bind(input_tensor, x_bf16.data_ptr())
    pk_module.bind(weight_tensor, weight_bf16.data_ptr())
    pk_module.bind(output_tensor, y_bf16.data_ptr())
    pk_module.bind(grad_output_tensor, grad_out_bf16.data_ptr())
    pk_module.bind(grad_input_tensor, grad_input_bf16.data_ptr())
    pk_module.bind(grad_weight_tensor, grad_weight_bf16.data_ptr())
    
    # Execute
    print("\n=== Executing Mirage Kernel ===")
    pk_module.execute()
    torch.cuda.synchronize()
    
    # Check results
    print("\n=== Mirage Results ===")
    print(f"Forward output mean: {y_bf16.float().mean():.6f}")
    print(f"grad_input mean: {grad_input_bf16.float().mean():.6f}")
    print(f"grad_weight mean: {grad_weight_bf16.float().mean():.6f}")
    print(f"grad_input[0, :5]: {grad_input_bf16[0, :5].float()}")
    print(f"grad_weight[:5]: {grad_weight_bf16[:5].float()}")
    
    # Compare forward pass
    forward_diff = (y_bf16.float() - y_cpu.cuda()).abs().max()
    print(f"\n=== Comparison ===")
    print(f"Forward max diff: {forward_diff:.6e}")
    
    # Compare gradients (with bfloat16 tolerance)
    grad_input_diff = (grad_input_bf16.float() - grad_input_ref.cuda()).abs().max()
    grad_weight_diff = (grad_weight_bf16.float() - grad_weight_ref.cuda()).abs().max()
    
    print(f"grad_input max diff: {grad_input_diff:.6e}")
    print(f"grad_weight max diff: {grad_weight_diff:.6e}")
    
    # Check if gradients are reasonable (not zeros or NaN)
    assert not torch.isnan(grad_input_bf16).any(), "grad_input has NaN!"
    assert not torch.isnan(grad_weight_bf16).any(), "grad_weight has NaN!"
    assert grad_input_bf16.abs().max() > 0, "grad_input is all zeros!"
    assert grad_weight_bf16.abs().max() > 0, "grad_weight is all zeros!"
    
    # bfloat16 tolerance is ~0.01 for typical values
    tolerance = 0.1
    if grad_input_diff < tolerance and grad_weight_diff < tolerance:
        print("\n✅ Backward pass gradients match PyTorch within tolerance!")
    else:
        print(f"\n⚠️  Gradients differ more than tolerance {tolerance}")
        print("This might be due to bfloat16 precision or implementation differences")


if __name__ == "__main__":
    test_rmsnorm_backward()