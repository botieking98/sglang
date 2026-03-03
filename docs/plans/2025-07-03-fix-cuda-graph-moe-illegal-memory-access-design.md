# Fix CUDA Graph Illegal Memory Access in fused_moe_kernel

## Problem

GitHub Issue: https://github.com/sgl-project/sglang/issues/19753

When running DeepSeek-V3 with --enable-torch-compile and FP8 quantization
(block_shape=[128,128]) on multi-GPU setups