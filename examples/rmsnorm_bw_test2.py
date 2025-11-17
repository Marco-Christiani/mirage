#!/usr/bin/env python3
"""Test RMSNorm backward using PersistentKernel."""

import torch
import mirage as mi

# Setup
device = torch.device("cuda")
torch.cuda.set_device(0)

# Test dimensions
batch_size = 1
hidden_dim = 4096
max_seq = 16
eps = 1e-6

# Meta tensors required by PersistentKernel
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

num_workers, num_local_schedulers = mi.get_configurations_from_gpu(0)

print("="*80)
print("PyTorch Reference")
print("="*80)

# Create PyTorch reference
torch.manual_seed(42)
input_torch = torch.randn(batch_size, hidden_dim, dtype=torch.float32, device=device, requires_grad=True)
weight_torch = torch.randn(hidden_dim, dtype=torch.float32, device=device, requires_grad=True)

def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return weight * x_normalized

output_torch = torch_rmsnorm(input_torch, weight_torch, eps)
grad_output_torch = torch.randn_like(output_torch)
output_torch.backward(grad_output_torch)

grad_input_ref = input_torch.grad.clone()
grad_weight_ref = weight_torch.grad.clone()

print(f"grad_input: mean={grad_input_ref.mean():.6f}, std={grad_input_ref.std():.6f}")
print(f"grad_weight: mean={grad_weight_ref.mean():.6f}, std={grad_weight_ref.std():.6f}")

print("\n" + "="*80)
print("Mirage Backward Kernel")
print("="*80)

# Create PersistentKernel
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
    trace_name="rmsnorm_backward_test",
    spec_decode_config=None,
    use_cutlass_kernel=True,
)

# Convert to bfloat16
input_bf16 = input_torch.detach().clone().to(torch.bfloat16)
weight_bf16 = weight_torch.detach().clone().to(torch.bfloat16)
grad_output_bf16 = grad_output_torch.detach().clone().to(torch.bfloat16)

# Allocate output buffers
grad_input_bf16 = torch.zeros(batch_size, hidden_dim, dtype=torch.bfloat16, device=device)
grad_weight_bf16 = torch.zeros(hidden_dim, dtype=torch.bfloat16, device=device)

# Attach all tensors (inputs and outputs)
input_dt = mpk.attach_input(input_bf16, name="input")
grad_output_dt = mpk.attach_input(grad_output_bf16, name="grad_output")
weight_dt = mpk.attach_input(weight_bf16, name="weight")
grad_input_dt = mpk.attach_input(grad_input_bf16, name="grad_input")
grad_weight_dt = mpk.attach_input(grad_weight_bf16, name="grad_weight")

# Add backward layer
mpk.rmsnorm_backward_layer(
    input=input_dt,
    grad_output=grad_output_dt,
    weight=weight_dt,
    grad_input=grad_input_dt,
    grad_weight=grad_weight_dt,
    grid_dim=(batch_size, 1, 1),
    block_dim=(128, 1, 1),
)

print("Compiling...")
mpk.compile(output_dir="./generated_code")
print("Generated code saved to ./generated_code/")
print("Executing...")
mpk()
torch.cuda.synchronize()
print("Complete!")

# Convert to float32 for comparison
grad_input_mirage = grad_input_bf16.to(torch.float32)
grad_weight_mirage = grad_weight_bf16.to(torch.float32)

print(f"grad_input: mean={grad_input_mirage.mean():.6f}, std={grad_input_mirage.std():.6f}")
print(f"grad_weight: mean={grad_weight_mirage.mean():.6f}, std={grad_weight_mirage.std():.6f}")

print("\n" + "="*80)
print("Comparison")
print("="*80)

# Check non-zero
grad_input_nonzero = (grad_input_bf16.abs() > 0).sum().item()
grad_weight_nonzero = (grad_weight_bf16.abs() > 0).sum().item()
print(f"Non-zero elements: grad_input={grad_input_nonzero}/{grad_input_bf16.numel()}, grad_weight={grad_weight_nonzero}/{grad_weight_bf16.numel()}")

# Compute errors
grad_input_diff = torch.abs(grad_input_ref - grad_input_mirage)
grad_weight_diff = torch.abs(grad_weight_ref - grad_weight_mirage)

print(f"\ngrad_input error: max={grad_input_diff.max():.6e}, mean={grad_input_diff.mean():.6e}")
print(f"grad_weight error: max={grad_weight_diff.max():.6e}, mean={grad_weight_diff.mean():.6e}")

# Check correctness
grad_input_ok = torch.allclose(grad_input_ref, grad_input_mirage, rtol=1e-2, atol=1e-3)
grad_weight_ok = torch.allclose(grad_weight_ref, grad_weight_mirage, rtol=1e-2, atol=1e-3)

print(f"\n{'='*80}")
if grad_input_ok and grad_weight_ok:
    print("✓ SUCCESS: Gradients match PyTorch!")
else:
    print("✗ MISMATCH")
    if not grad_input_ok:
        print("  - grad_input differs")
    if not grad_weight_ok:
        print("  - grad_weight differs")
print(f"{'='*80}")

mpk.finalize()