#!/usr/bin/env python3
"""Test Linear forward and backward using PersistentKernel."""

import torch
import mirage as mi

# Setup
device = torch.device("cuda")
torch.cuda.set_device(0)

# Test dimensions
# Note: Using 2048 to make grid partitioning work better
# With grid_dim=64, each partition gets 2048/64 = 32 outputs
# Or with grid_dim=32, each partition gets 2048/32 = 64 outputs (better for CUTLASS)
batch_size = 1
in_features = 2048
out_features = 2048
max_seq = 16

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
input_torch = torch.randn(batch_size, in_features, dtype=torch.float32, device=device, requires_grad=True)
weight_torch = torch.randn(out_features, in_features, dtype=torch.float32, device=device, requires_grad=True)

# Forward pass: output = input @ weight.T
output_torch = torch.nn.functional.linear(input_torch, weight_torch)
grad_output_torch = torch.randn_like(output_torch)

# Backward pass
output_torch.backward(grad_output_torch)

grad_input_ref = input_torch.grad.clone()
grad_weight_ref = weight_torch.grad.clone()

print(f"Input shape: {input_torch.shape}")
print(f"Weight shape: {weight_torch.shape}")
print(f"Output shape: {output_torch.shape}")
print(f"output: mean={output_torch.mean():.6f}, std={output_torch.std():.6f}")
print(f"grad_input: mean={grad_input_ref.mean():.6f}, std={grad_input_ref.std():.6f}")
print(f"grad_weight: mean={grad_weight_ref.mean():.6f}, std={grad_weight_ref.std():.6f}")

print("\n" + "="*80)
print("Mirage Forward Kernel")
print("="*80)

# Create PersistentKernel for forward pass
mpk_fwd = mi.PersistentKernel(
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
    trace_name="linear_forward_test",
    spec_decode_config=None,
    use_cutlass_kernel=True,
)

# Convert to bfloat16
input_bf16 = input_torch.detach().clone().to(torch.bfloat16)
weight_bf16 = weight_torch.detach().clone().to(torch.bfloat16)

# Allocate output buffer
output_bf16 = torch.zeros(batch_size, out_features, dtype=torch.bfloat16, device=device)

# Attach tensors
input_dt = mpk_fwd.attach_input(input_bf16, name="input")
weight_dt = mpk_fwd.attach_input(weight_bf16, name="weight")
output_dt = mpk_fwd.attach_input(output_bf16, name="output")

# Add forward layer
# Note: linear_layer expects weight shape (out_features, in_features)
# Grid size of 32 means each partition handles 2048/32 = 64 outputs
mpk_fwd.linear_layer(
    input=input_dt,
    weight=weight_dt,
    output=output_dt,
    grid_dim=(32, 1, 1),
    block_dim=(256, 1, 1),
)

print("Compiling forward pass...")
mpk_fwd.compile(output_dir="examples/generated_code_linear_fwd")
print("Generated code saved to examples/generated_code_linear_fwd/")
print("Executing forward pass...")
mpk_fwd()
torch.cuda.synchronize()
print("Forward complete!")

# Convert to float32 for comparison
output_mirage = output_bf16.to(torch.float32)

print(f"output: mean={output_mirage.mean():.6f}, std={output_mirage.std():.6f}")

# Check forward correctness
output_ref = output_torch.clone()
output_diff = torch.abs(output_ref - output_mirage)
print(f"\nForward output error: max={output_diff.max():.6e}, mean={output_diff.mean():.6e}")

# Debug: print a few values to see what's going on
print(f"\nSample output values (first 5):")
print(f"  PyTorch: {output_ref[0, :5]}")
print(f"  Mirage:  {output_mirage[0, :5]}")

# Bfloat16 has ~3 decimal digits of precision, so we use looser tolerances
# For large values (~100), bfloat16 quantization can cause ~0.5 absolute error
output_ok = torch.allclose(output_ref, output_mirage, rtol=1e-2, atol=0.6)
print(f"\nForward pass: {'✓ SUCCESS' if output_ok else '✗ MISMATCH'}")
if output_ok:
    print(f"  Maximum error: {output_diff.max():.4f} (acceptable for bfloat16)")
else:
    # Find where the biggest errors are
    max_err_idx = torch.argmax(output_diff)
    print(f"  Biggest error at index {max_err_idx}: ref={output_ref.flatten()[max_err_idx]:.4f}, mirage={output_mirage.flatten()[max_err_idx]:.4f}")

mpk_fwd.finalize()

print("\n" + "="*80)
print("Mirage Backward Kernel")
print("="*80)

# For backward, we need to implement:
# grad_input = grad_output @ weight  (grad_output is [batch, out_features], weight is [out_features, in_features])
# grad_weight = grad_output.T @ input  (need to accumulate across batch)

# Create PersistentKernel for backward pass
mpk_bwd = mi.PersistentKernel(
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
    trace_name="linear_backward_test",
    spec_decode_config=None,
    use_cutlass_kernel=True,
)

# Convert to bfloat16
grad_output_bf16 = grad_output_torch.detach().clone().to(torch.bfloat16)

# Allocate output buffers
grad_input_bf16 = torch.zeros(batch_size, in_features, dtype=torch.bfloat16, device=device)
grad_weight_bf16 = torch.zeros(out_features, in_features, dtype=torch.bfloat16, device=device)

# Attach all tensors
input_dt_bwd = mpk_bwd.attach_input(input_bf16, name="input")
weight_dt_bwd = mpk_bwd.attach_input(weight_bf16, name="weight")
grad_output_dt = mpk_bwd.attach_input(grad_output_bf16, name="grad_output")
grad_input_dt = mpk_bwd.attach_input(grad_input_bf16, name="grad_input")
grad_weight_dt = mpk_bwd.attach_input(grad_weight_bf16, name="grad_weight")

# Add backward layer
mpk_bwd.linear_backward_layer(
    input=input_dt_bwd,
    weight=weight_dt_bwd,
    grad_output=grad_output_dt,
    grad_input=grad_input_dt,
    grad_weight=grad_weight_dt,
    grid_dim=(batch_size, 1, 1),
    block_dim=(256, 1, 1),
)

print("Compiling backward pass...")
mpk_bwd.compile(output_dir="examples/generated_code_linear_bwd")
print("Generated code saved to examples/generated_code_linear_bwd/")
print("Executing backward pass...")
mpk_bwd()
torch.cuda.synchronize()
print("Backward complete!")

# Convert to float32 for comparison
grad_input_mirage = grad_input_bf16.to(torch.float32)
grad_weight_mirage = grad_weight_bf16.to(torch.float32)

print(f"grad_input: mean={grad_input_mirage.mean():.6f}, std={grad_input_mirage.std():.6f}")
print(f"grad_weight: mean={grad_weight_mirage.mean():.6f}, std={grad_weight_mirage.std():.6f}")

# Compute errors
grad_input_diff = torch.abs(grad_input_ref - grad_input_mirage)
grad_weight_diff = torch.abs(grad_weight_ref - grad_weight_mirage)

print(f"\ngrad_input error: max={grad_input_diff.max():.6e}, mean={grad_input_diff.mean():.6e}")
print(f"grad_weight error: max={grad_weight_diff.max():.6e}, mean={grad_weight_diff.mean():.6e}")

# Check correctness with bfloat16 tolerances
grad_input_ok = torch.allclose(grad_input_ref, grad_input_mirage, rtol=1e-2, atol=0.6)
grad_weight_ok = torch.allclose(grad_weight_ref, grad_weight_mirage, rtol=1e-2, atol=0.01)

mpk_bwd.finalize()

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Forward pass: {'✓ PASSED' if output_ok else '✗ FAILED'}")
print(f"Backward pass: {'✓ PASSED' if (grad_input_ok and grad_weight_ok) else '✗ FAILED'}")
if grad_input_ok and grad_weight_ok:
    print("  All gradients match PyTorch within bfloat16 precision!")
else:
    if not grad_input_ok:
        print("  - grad_input differs from PyTorch")
    if not grad_weight_ok:
        print("  - grad_weight differs from PyTorch")
print(f"{'='*80}")
