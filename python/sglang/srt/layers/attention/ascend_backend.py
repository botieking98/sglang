
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from python.sglang.srt.layers.linear import RowParallelLinear
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

class AscendAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )

        num_tokens = q.shape[0]

        key_cache = kv_cache[0]
        num_blocks, block_size, _ = key_cache.shape

        key_cache = key_cache.view(
            num_blocks, block_size, layer.tp_k_head_num,
            self.qk_rope_head_dim + self.kv_lora_rank)
        
        torch_npu.npu_reshapecache(key=k,
                                    value=None,
                                    keyCache=key_cache,
                                    valueCache=None,
                                    slotMapping=None,
                                    compressType=0,
                                    kvCacheCfg=1)
        
        attn_output = torch.empty(num_tokens,
                                    layer.tp_q_head_num,
                                    layer.v_head_dim,
                                    dtype=q.dtype,
                                    device="npu")

        seq_lens_tensor_cpu = torch.from_numpy(
            np.array(forward_batch.seq_lens.cpu().numpy()).astype(
                np.int32))

        torch_npu.npu_selfattention(query=q,
                                    key=k,
                                    value=v,
                                    kvcacheCfg=0,
                                    mask=None,
                                    maskType=1,
                                    isTriuMask=0,
                                    seqLen=seq_lens_tensor_cpu,
                                    scale=layer.scaling,
                                    qScale=1,
                                    scaleType=0,
                                    headNum=layer.tp_q_head_num,
                                    kvHeadNum=layer.tp_q_head_num,
                                    mlaVHeadSize=0,
                                    calcType=3,
                                    kernelType=0,
                                    clampType=0,
                                    quantType=0,
                                    cacheType=0,
                                    windowSize=0,
                                    clampMin=0,
                                    clampMax=0,
                                    batchRunStatusEnable=False,
                                    inputLayout=0,
                                    outDataType=0,
                                    out=attn_output)
        
        output, _ = RowParallelLinear(attn_output.reshape(num_tokens, -1))
        return output

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        
        return NotImplementedError