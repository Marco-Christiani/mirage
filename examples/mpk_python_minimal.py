import torch
import torch.nn.functional as F
import mirage as mi
import numpy as np

# Basic device / sizes
device = torch.device("cuda")
torch.cuda.set_device(0)

# Test dimensions
batch_size = 1
hidden_dim = 4096
max_seq = 16
eps = 1e-6

# Minimal meta tensors (required by PersistentKernel constructor)
max_num_batched_requests = batch_size
max_num_batched_tokens = batch_size
max_num_pages = 1
page_size = 4096

meta_tensors = {
    "step": torch.zeros((max_num_batched_requests,), dtype=torch.int32, device=device),
    "tokens": torch.zeros((max_num_batched_requests, max_seq), dtype=torch.int32, device=device),
    "input_tokens": torch.zeros((max_num_batched_tokens, 1), dtype=torch.int32, device=device),
    "output_tokens": torch.zeros((max_num_batched_tokens, 1), dtype=torch.int32, device=device),
    "num_new_tokens": torch.zeros((max_num_batched_requests,), dtype=torch.int32, device=device),
    "prompt_lengths": torch.zeros((max_num_batched_requests,), dtype=torch.int32, device=device),
    "qo_indptr_buffer": torch.zeros((max_num_batched_requests + 1,), dtype=torch.int32, device=device),
    "paged_kv_indptr_buffer": torch.zeros((max_num_batched_requests + 1,), dtype=torch.int32, device=device),
    "paged_kv_indices_buffer": torch.zeros((max_num_pages,), dtype=torch.int32, device=device),
    "paged_kv_last_page_len_buffer": torch.zeros((max_num_batched_requests,), dtype=torch.int32, device=device),
}

# Get worker/scheduler counts
num_workers, num_local_schedulers = mi.get_configurations_from_gpu(0)

print("="*80)
print("PyTorch Reference Implementation")
print("="*80)

# Create test tensors with requires_grad for PyTorch autograd
input_torch = torch.randn(batch_size, hidden_dim, dtype=torch.float32, device=device, requires_grad=True)
weight_torch = torch.randn(hidden_dim, dtype=torch.float32, device=device, requires_grad=True)

# Forward pass: RMSNorm
def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return weight * x_normalized

# Run forward
output_torch = torch_rmsnorm(input_torch, weight_torch, eps)

# Create upstream gradient
grad_output_torch = torch.randn_like(output_torch)

# Run backward
output_torch.backward(grad_output_torch)

# Extract gradients
grad_input_torch = input_torch.grad.clone()
grad_weight_torch = weight_torch.grad.clone()

print(f"Input shape: {input_torch.shape}")
print(f"Weight shape: {weight_torch.shape}")
print(f"Grad output shape: {grad_output_torch.shape}")
print(f"Grad input shape: {grad_input_torch.shape}")
print(f"Grad weight shape: {grad_weight_torch.shape}")
print(f"\nGrad input stats: min={grad_input_torch.min():.6f}, max={grad_input_torch.max():.6f}, mean={grad_input_torch.mean():.6f}")
print(f"Grad weight stats: min={grad_weight_torch.min():.6f}, max={grad_weight_torch.max():.6f}, mean={grad_weight_torch.mean():.6f}")

print("\n" + "="*80)
print("Mirage Implementation")
print("="*80)

# Construct the PersistentKernel
mpk = mi.PersistentKernel(
    mode="offline",
    world_size=1,
    mpi_rank=0,
    num_workers=num_workers,
    num_local_schedulers=num_local_schedulers,
    num_remote_schedulers=0,
    max_seq_length=max_seq,
    max_num_batched_requests=max_num_batched_requests,
    max_num_batched_tokens=max_num_batched_tokens,
    max_num_pages=max_num_pages,
    page_size=page_size,
    eos_token_id=-1,
    meta_tensors=meta_tensors,
    profiler_tensor=None,
    trace_name="test_rmsnorm_backward",
    spec_decode_config=None,
    use_cutlass_kernel=True,
)

# Convert to bfloat16 for Mirage (matching typical usage)
input_mirage = input_torch.detach().clone().to(torch.bfloat16)
weight_mirage = weight_torch.detach().clone().to(torch.bfloat16)
grad_output_mirage = grad_output_torch.detach().clone().to(torch.bfloat16)

# Allocate output tensors
grad_input_mirage = torch.zeros(batch_size, hidden_dim, dtype=torch.bfloat16, device=device)
grad_weight_mirage = torch.zeros(hidden_dim, dtype=torch.bfloat16, device=device)

# Attach tensors to the persistent kernel graph
input_dt = mpk.attach_input(input_mirage, name="input")
grad_output_dt = mpk.attach_input(grad_output_mirage, name="grad_output")
weight_dt = mpk.attach_input(weight_mirage, name="weight")

grad_input_dt = mpk.attach_input(grad_input_mirage, name="grad_input")
grad_weight_dt = mpk.attach_input(grad_weight_mirage, name="grad_weight")

# Add the backward layer
mpk.rmsnorm_backward_layer(
    input=input_dt,
    grad_output=grad_output_dt,
    weight=weight_dt,
    grad_input=grad_input_dt,
    grad_weight=grad_weight_dt,
    grid_dim=(batch_size, 1, 1),
    block_dim=(256, 1, 1),
)

print("Compiling...")
mpk.compile()
print("Compilation successful!")

print("Executing kernel...")
mpk()
torch.cuda.synchronize()
# err = torch.cuda.cudart().get_last_error()
# if err != torch.cuda.cudaSuccess:
#     print(f"CUDA Error: {err}")
print("Execution complete!")
mpk.finalize()

# Convert back to float32 for comparison
grad_input_mirage_fp32 = grad_input_mirage.to(torch.float32)
grad_weight_mirage_fp32 = grad_weight_mirage.to(torch.float32)

print(f"\nGrad input stats: min={grad_input_mirage_fp32.min():.6f}, max={grad_input_mirage_fp32.max():.6f}, mean={grad_input_mirage_fp32.mean():.6f}")
print(f"Grad weight stats: min={grad_weight_mirage_fp32.min():.6f}, max={grad_weight_mirage_fp32.max():.6f}, mean={grad_weight_mirage_fp32.mean():.6f}")

print("\n" + "="*80)
print("Comparison")
print("="*80)

# Compare gradients
grad_input_diff = torch.abs(grad_input_torch - grad_input_mirage_fp32)
grad_weight_diff = torch.abs(grad_weight_torch - grad_weight_mirage_fp32)

print("\nGrad Input Difference:")
print(f"  Max absolute error: {grad_input_diff.max():.6e}")
print(f"  Mean absolute error: {grad_input_diff.mean():.6e}")
print(f"  Max relative error: {(grad_input_diff / (torch.abs(grad_input_torch) + 1e-8)).max():.6e}")

print("\nGrad Weight Difference:")
print(f"  Max absolute error: {grad_weight_diff.max():.6e}")
print(f"  Mean absolute error: {grad_weight_diff.mean():.6e}")
print(f"  Max relative error: {(grad_weight_diff / (torch.abs(grad_weight_torch) + 1e-8)).max():.6e}")

# Check if results are close (accounting for bfloat16 precision)
grad_input_close = torch.allclose(grad_input_torch, grad_input_mirage_fp32, rtol=1e-2, atol=1e-3)
grad_weight_close = torch.allclose(grad_weight_torch, grad_weight_mirage_fp32, rtol=1e-2, atol=1e-3)

print(f"\n{'='*80}")
if grad_input_close and grad_weight_close:
    print("✓ SUCCESS: Mirage implementation matches PyTorch (within bfloat16 tolerance)")
else:
    print("✗ FAILURE: Mirage implementation differs from PyTorch")
    if not grad_input_close:
        print("  - grad_input mismatch")
    if not grad_weight_close:
        print("  - grad_weight mismatch")
print(f"{'='*80}")
