"""Diagnostic control only: canonicalize valid routed-token order per expert.

This adds host synchronization and is neither a production fix nor a benchmark
configuration. It leaves inputs, routing scores, expert assignments and weights
unchanged, to isolate floating-point effects of token packing order.
"""
import torch


def install():
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe
    original = marlin_moe.moe_align_block_size

    def canonical(topk_ids, block_size, num_experts, *args, **kwargs):
        result = original(topk_ids, block_size, num_experts, *args, **kwargs)
        sorted_ids, expert_ids, num_padded = result
        valid = int(num_padded.item())
        experts = expert_ids[:valid // block_size]
        assert bool(torch.all(experts[1:] >= experts[:-1]))
        # Every expert retains exactly its original count and padded blocks.
        key = experts.repeat_interleave(block_size).to(torch.int64) * (topk_ids.numel() + 1)
        key = key + sorted_ids[:valid].to(torch.int64)
        order = torch.argsort(key, stable=True)
        sorted_ids[:valid].copy_(sorted_ids[:valid][order])
        return result

    marlin_moe.moe_align_block_size = canonical
    return original
