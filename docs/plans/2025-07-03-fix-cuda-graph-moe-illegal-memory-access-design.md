# Fix CUDA Graph Illegal Memory Access in fused_moe_kernel

## Problem

GitHub Issue: https://github.com/sgl-project/sglang/issues/19753

When running DeepSeek-V3 with --enable-torch-compile and FP8 quantization
(block_shape=[128,128]) on multi-GPU setups, we encounter illegal memory access errors.
The root cause is that intermediate buffers (`intermediate_cache2`) are allocated dynamically
inside the chunk loop in `fused_experts_impl`. During CUDA graph capture, these buffers
get specific GPU addresses baked into the graph. When the graph is replayed, re-allocating
these buffers with potentially different addresses causes stale pointer accesses.

## Solution

Pre-allocate all intermediate buffers (`intermediate_cache2`, and moe_align_block_size
buffers) before the chunk loop starts, similar to how `intermediate_cache3` is already
handled. This ensures:

1. **CUDA Graph Compatibility**: All buffer addresses remain stable across graph capture
   and replay.
2. **Performance**: Pre-allocation avoids the overhead of repeated allocations inside
   the loop.

### Key Changes

1. **Buffer Size Calculation**: For gated activations (swiglu style), MoE uses N//2 as
   the hidden dimension; for non-gated activations, it uses N. We compute:
   ```python
   intermediate_cache2_size_n = N // 2 if is_gated else N
   ```

2. **In-Place Operations**: All operations assigning to `curr_intermediate_cache2` must
   use `copy_()` to maintain CUDA graph compatibility. This affects:
   - `gemm1_alpha` and `gemm1_limit` branches in gated silu activation
   - Non-gated activations (silu, gelu, relu2)

3. **Pre-allocated Buffers for moe_align_block_size**: We now pass pre-allocated buffers
   (`_sorted_ids_buf`, `_expert_ids_buf`, `_num_tokens_post_pad_buf`, `_cumsum_buf`)
   to avoid dynamic allocations inside the kernel.

## Testing

- Test with DeepSeek-V3 on multi-GPU with FP8 quantization and CUDA graphs enabled.
- Verify no illegal memory access errors occur during graph replay.
- Verify correctness matches the original implementation.