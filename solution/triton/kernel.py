"""
Triton Kernel Template for FlashInfer Competition - MoE Track.

FP8 block scale MoE operation with DeepSeek-V3 style routing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    # Routing inputs
    routing_logits_ptr,  # [seq_len, num_experts] float32
    routing_bias_ptr,    # [num_experts] bfloat16
    # Hidden states (FP8 quantized)
    hidden_states_ptr,        # [seq_len, hidden_size] float8_e4m3fn
    hidden_states_scale_ptr,  # [num_hidden_blocks, seq_len] float32
    # GEMM1 weights (gate + up projection)
    gemm1_weights_ptr,        # [num_local_experts, gemm1_out_size, hidden_size] float8_e4m3fn
    gemm1_weights_scale_ptr,  # [num_local_experts, num_gemm1_out_blocks, num_hidden_blocks] float32
    # GEMM2 weights (down projection)
    gemm2_weights_ptr,        # [num_local_experts, hidden_size, intermediate_size] float8_e4m3fn
    gemm2_weights_scale_ptr,  # [num_local_experts, num_hidden_blocks, num_intermediate_blocks] float32
    # Scalars
    local_expert_offset,      # int32: offset of local experts in global space
    routed_scaling_factor,    # float32: scaling factor for routing weights
    # Output
    output_ptr,               # [seq_len, hidden_size] bfloat16
    # Dimensions (passed as kernel args)
    seq_len,
    num_experts: tl.constexpr,
    num_local_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    gemm1_out_size: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused MoE kernel with FP8 block-scale quantization and DeepSeek-V3 routing.

    TODO: Implement the kernel according to the MoE track definition.
    """
    pass
