/*
 * Fused MoE Kernel for FlashInfer Bench Contest - B200 Target
 *
 * DeepSeek-V3/R1 FP8 Block-Scale MoE:
 *   - 256 global experts, 32 local (EP=8), hidden=7168, intermediate=2048
 *   - DeepSeek-V3 no-aux routing (sigmoid → group top-2 → group top-4 → global
 * top-8)
 *   - FP8 block-scale dequantization (block=128)
 *   - GEMM1 (hidden→2*intermediate) + SwiGLU + GEMM2 (intermediate→hidden)
 *   - Weighted output accumulation → bfloat16
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

// =========================================================================
// Constants
// =========================================================================
#define HIDDEN_SIZE 7168
#define INTER_SIZE 2048
#define GEMM1_OUT_SIZE 4096 // 2 * INTER_SIZE
#define NUM_GLOBAL_EXP 256
#define NUM_LOCAL_EXP 32
#define TOP_K 8
#define N_GROUP 8
#define TOPK_GROUP 4
#define GROUP_SIZE 32 // NUM_GLOBAL_EXP / N_GROUP
#define BLOCK_SCALE 128
#define NUM_H_BLOCKS 56  // HIDDEN_SIZE / BLOCK_SCALE
#define NUM_I_BLOCKS 16  // INTER_SIZE / BLOCK_SCALE
#define NUM_G1_BLOCKS 32 // GEMM1_OUT_SIZE / BLOCK_SCALE

// GEMM tiling
#define TILE_M 16
#define TILE_N 64
#define TILE_K 128 // matches block scale granularity

// =========================================================================
// FP8 E4M3FN → float conversion
// =========================================================================
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
  uint32_t sign = (val >> 7) & 1;
  uint32_t exp = (val >> 3) & 0xF;
  uint32_t mant = val & 0x7;

  float result;
  if (exp == 0 && mant == 0) {
    return 0.0f;
  } else if (exp == 0) {
    // subnormal: 2^(1-7) * (mant/8) = 2^(-6) * mant * 2^(-3) = mant * 2^(-9)
    result = ldexpf((float)mant, -9);
  } else {
    // normal: 2^(exp-7) * (1 + mant/8)
    result = ldexpf(1.0f + (float)mant * 0.125f, (int)exp - 7);
  }
  return sign ? -result : result;
}

// =========================================================================
// Float → bfloat16 conversion
// =========================================================================
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
  return __float2bfloat16(val);
}

// =========================================================================
// Sigmoid
// =========================================================================
__device__ __forceinline__ float sigmoid_f(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// =========================================================================
// SiLU (Swish)
// =========================================================================
__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + expf(-x));
}

// =========================================================================
// Kernel 1: DeepSeek-V3 Routing
//
// Input:  routing_logits [T, 256] float32, routing_bias [256] bf16
// Output: topk_indices [T, 8] int32, topk_weights [T, 8] float32
//
// One thread block per token, 256 threads (one per expert).
// =========================================================================2
__global__ void
routing_kernel(const float *__restrict__ routing_logits,       // [T, 256]
               const __nv_bfloat16 *__restrict__ routing_bias, // [256]
               int *__restrict__ topk_indices,                 // [T, 8]
               float *__restrict__ topk_weights,               // [T, 8]
               int T, float routed_scaling_factor) {
  int token = blockIdx.x;
  if (token >= T)
    return;
  int tid = threadIdx.x; // 0..255, one per expert

  __shared__ float s_scores[NUM_GLOBAL_EXP];        // sigmoid scores (no bias)
  __shared__ float s_scores_biased[NUM_GLOBAL_EXP]; // sigmoid + bias
  __shared__ float s_group_scores[N_GROUP];
  __shared__ int s_group_top_idx[TOPK_GROUP];
  __shared__ float s_group_mask[N_GROUP];
  __shared__ int s_topk_idx[TOP_K];
  __shared__ float s_topk_val[TOP_K];

  // 1) Compute sigmoid and biased sigmoid
  float logit = routing_logits[token * NUM_GLOBAL_EXP + tid];
  float s = sigmoid_f(logit);
  float bias = __bfloat162float(routing_bias[tid]);
  float s_biased = s + bias;
  s_scores[tid] = s;
  s_scores_biased[tid] = s_biased;
  __syncthreads();

  // 2) Compute group scores: sum of top-2 within each group of 32
  int group_id = tid / GROUP_SIZE;
  int in_group = tid % GROUP_SIZE;

  // Each thread in position 0 of its group computes the top-2 sum
  if (in_group == 0) {
    float top1 = -FLT_MAX, top2 = -FLT_MAX;
    int base = group_id * GROUP_SIZE;
    for (int j = 0; j < GROUP_SIZE; j++) {
      float v = s_scores_biased[base + j];
      if (v > top1) {
        top2 = top1;
        top1 = v;
      } else if (v > top2) {
        top2 = v;
      }
    }
    s_group_scores[group_id] = top1 + top2;
  }
  __syncthreads();

  // 3) Select top-4 groups (thread 0 does this)
  if (tid == 0) {
    for (int g = 0; g < N_GROUP; g++)
      s_group_mask[g] = 0.0f;

    // Simple selection of top TOPK_GROUP groups
    float tmp_scores[N_GROUP];
    for (int g = 0; g < N_GROUP; g++)
      tmp_scores[g] = s_group_scores[g];

    for (int k = 0; k < TOPK_GROUP; k++) {
      int best = 0;
      for (int g = 1; g < N_GROUP; g++) {
        if (tmp_scores[g] > tmp_scores[best])
          best = g;
      }
      s_group_top_idx[k] = best;
      s_group_mask[best] = 1.0f;
      tmp_scores[best] = -FLT_MAX;
    }
  }
  __syncthreads();

  // 4) Global top-8 from kept groups (thread 0 does this)
  if (tid == 0) {
    float vals[TOP_K];
    int idxs[TOP_K];
    for (int k = 0; k < TOP_K; k++) {
      vals[k] = -FLT_MAX;
      idxs[k] = 0;
    }

    for (int e = 0; e < NUM_GLOBAL_EXP; e++) {
      int g = e / GROUP_SIZE;
      if (s_group_mask[g] == 0.0f)
        continue;
      float v = s_scores_biased[e];
      // Insert into sorted top-8
      for (int k = 0; k < TOP_K; k++) {
        if (v > vals[k]) {
          // Shift down
          for (int j = TOP_K - 1; j > k; j--) {
            vals[j] = vals[j - 1];
            idxs[j] = idxs[j - 1];
          }
          vals[k] = v;
          idxs[k] = e;
          break;
        }
      }
    }
    for (int k = 0; k < TOP_K; k++) {
      s_topk_idx[k] = idxs[k];
      s_topk_val[k] = vals[k];
    }
  }
  __syncthreads();

  // 5) Compute normalized weights using s (without bias)
  if (tid == 0) {
    float weight_sum = 0.0f;
    float weights[TOP_K];
    for (int k = 0; k < TOP_K; k++) {
      weights[k] = s_scores[s_topk_idx[k]];
      weight_sum += weights[k];
    }
    if (weight_sum < 1e-20f)
      weight_sum = 1e-20f;

    for (int k = 0; k < TOP_K; k++) {
      topk_indices[token * TOP_K + k] = s_topk_idx[k];
      topk_weights[token * TOP_K + k] =
          (weights[k] / weight_sum) * routed_scaling_factor;
    }
  }
}

// =========================================================================
// Kernel 2: Count tokens per local expert
//
// After routing, we know which global experts each token selected.
// This kernel counts how many tokens go to each local expert and builds
// a mapping: for each (token, slot) → local expert index (or -1 if not local).
// =========================================================================
__global__ void
count_expert_tokens_kernel(const int *__restrict__ topk_indices, // [T, 8]
                           int *__restrict__ expert_counts, // [NUM_LOCAL_EXP]
                           int T, int local_expert_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = T * TOP_K;
  if (tid >= total)
    return;

  int expert_global = topk_indices[tid];
  int expert_local = expert_global - local_expert_offset;
  if (expert_local >= 0 && expert_local < NUM_LOCAL_EXP) {
    atomicAdd(&expert_counts[expert_local], 1);
  }
}

// =========================================================================
// Kernel 3: Build per-expert token lists
//
// For each local expert, build a list of (token_index, slot_index) pairs.
// token_expert_map[expert][i] = original token index
// token_weight_map[expert][i] = routing weight for this token-expert pair
// =========================================================================
__global__ void build_expert_lists_kernel(
    const int *__restrict__ topk_indices,   // [T, 8]
    const float *__restrict__ topk_weights, // [T, 8]
    int *__restrict__ expert_offsets,       // [NUM_LOCAL_EXP] (atomic counters,
                                            // pre-zeroed)
    int *__restrict__ token_ids_per_expert, // [T * TOP_K] flat buffer
    float *__restrict__ weights_per_expert, // [T * TOP_K] flat buffer
    const int *__restrict__ expert_starts, // [NUM_LOCAL_EXP] prefix sum offsets
    int T, int local_expert_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = T * TOP_K;
  if (tid >= total)
    return;

  int expert_global = topk_indices[tid];
  int expert_local = expert_global - local_expert_offset;
  if (expert_local >= 0 && expert_local < NUM_LOCAL_EXP) {
    int token_id = tid / TOP_K;
    float weight = topk_weights[tid];

    int pos = atomicAdd(&expert_offsets[expert_local], 1);
    int offset = expert_starts[expert_local] + pos;
    token_ids_per_expert[offset] = token_id;
    weights_per_expert[offset] = weight;
  }
}

// =========================================================================
// Kernel 4: FP8 Block-Scale GEMM + SwiGLU + GEMM2 + Weighted Accumulation
//
// Process one expert at a time. For each expert:
//   For each token assigned to this expert:
//     1. Dequant FP8 hidden_states using block scales → A [1, H]
//     2. GEMM1: A @ W1^T → [1, 2*I] with block-scale dequant
//     3. SwiGLU: silu(gate) * up → [1, I]
//     4. GEMM2: intermediate @ W2^T → [1, H] with block-scale dequant
//     5. output[token] += weight * result
//
// This is a per-token kernel — each thread block processes one token
// for the given expert. For small token counts, this is launch-efficient.
// =========================================================================
__global__ void moe_expert_compute_kernel(
    const uint8_t *__restrict__ hidden_states,     // [T, H] FP8
    const float *__restrict__ hidden_states_scale, // [H/128, T]
    const uint8_t *__restrict__ gemm1_weights,     // [E_local, 2I, H] FP8
    const float *__restrict__ gemm1_weights_scale, // [E_local, 2I/128, H/128]
    const uint8_t *__restrict__ gemm2_weights,     // [E_local, H, I] FP8
    const float *__restrict__ gemm2_weights_scale, // [E_local, H/128, I/128]
    float *__restrict__ output_f32,                // [T, H] float32 accumulation buffer
    const int *__restrict__ token_ids,             // [num_tokens_for_expert]
    const float *__restrict__ token_weights,       // [num_tokens_for_expert]
    int expert_local, int num_tokens_for_expert, int T) {
  // Each block handles one token for this expert
  int tok_idx = blockIdx.x;
  if (tok_idx >= num_tokens_for_expert)
    return;

  int token_id = token_ids[tok_idx];
  float route_weight = token_weights[tok_idx];
  int tid = threadIdx.x; // thread within block
  int num_threads = blockDim.x;

  // -------------------------------------------------------
  // Step 1: Dequantize hidden_states for this token → A[H]
  // hidden_states_scale layout: [H/128, T], so scale for
  //   block b, token t = hidden_states_scale[b * T + token_id]
  // -------------------------------------------------------
  // We'll compute in chunks distributed across threads.

  // Shared memory for intermediate results
  extern __shared__ float smem[];
  float *A = smem;                                    // [HIDDEN_SIZE]
  float *gemm1_out = smem + HIDDEN_SIZE;              // [GEMM1_OUT_SIZE]
  float *inter = smem + HIDDEN_SIZE + GEMM1_OUT_SIZE; // [INTER_SIZE]

  // Dequant hidden states
  for (int i = tid; i < HIDDEN_SIZE; i += num_threads) {
    int block_idx = i / BLOCK_SCALE;
    float scale = hidden_states_scale[block_idx * T + token_id];
    uint8_t fp8_val = hidden_states[token_id * HIDDEN_SIZE + i];
    A[i] = fp8_e4m3_to_float(fp8_val) * scale;
  }
  __syncthreads();

  // -------------------------------------------------------
  // Step 2: GEMM1 — A[1,H] @ W1[2I,H]^T = C[1,2I]
  // W1 layout: [E_local, 2I, H], scale: [E_local, 2I/128, H/128]
  // -------------------------------------------------------
  for (int n = tid; n < GEMM1_OUT_SIZE; n += num_threads) {
    float acc = 0.0f;
    int n_block = n / BLOCK_SCALE;

    for (int kb = 0; kb < NUM_H_BLOCKS; kb++) {
      float w_scale =
          gemm1_weights_scale[expert_local * NUM_G1_BLOCKS * NUM_H_BLOCKS +
                              n_block * NUM_H_BLOCKS + kb];

      int base_k = kb * BLOCK_SCALE;
      for (int kk = 0; kk < BLOCK_SCALE; kk++) {
        int k = base_k + kk;
        uint8_t w_fp8 =
            gemm1_weights[expert_local * (size_t)GEMM1_OUT_SIZE * HIDDEN_SIZE +
                          n * (size_t)HIDDEN_SIZE + k];
        float w_val = fp8_e4m3_to_float(w_fp8) * w_scale;
        acc += A[k] * w_val;
      }
    }
    gemm1_out[n] = acc;
  }
  __syncthreads();

  // -------------------------------------------------------
  // Step 3: SwiGLU — silu(gate) * up
  // gemm1_out layout: [2I] where first I = up, second I = gate
  // (matching reference: X1 = G1[:, :I], X2 = G1[:, I:])
  //   silu(X2) * X1
  // -------------------------------------------------------
  for (int i = tid; i < INTER_SIZE; i += num_threads) {
    float x1 = gemm1_out[i];              // up projection
    float x2 = gemm1_out[INTER_SIZE + i]; // gate projection
    inter[i] = silu_f(x2) * x1;
  }
  __syncthreads();

  // -------------------------------------------------------
  // Step 4: GEMM2 — inter[1,I] @ W2[H,I]^T = O[1,H]
  // W2 layout: [E_local, H, I], scale: [E_local, H/128, I/128]
  // Then: output[token] += route_weight * O
  // -------------------------------------------------------
  for (int n = tid; n < HIDDEN_SIZE; n += num_threads) {
    float acc = 0.0f;
    int n_block = n / BLOCK_SCALE;

    for (int kb = 0; kb < NUM_I_BLOCKS; kb++) {
      float w_scale =
          gemm2_weights_scale[expert_local * NUM_H_BLOCKS * NUM_I_BLOCKS +
                              n_block * NUM_I_BLOCKS + kb];

      int base_k = kb * BLOCK_SCALE;
      for (int kk = 0; kk < BLOCK_SCALE; kk++) {
        int k = base_k + kk;
        uint8_t w_fp8 =
            gemm2_weights[expert_local * (size_t)HIDDEN_SIZE * INTER_SIZE +
                          n * (size_t)INTER_SIZE + k];
        float w_val = fp8_e4m3_to_float(w_fp8) * w_scale;
        acc += inter[k] * w_val;
      }
    }

    // Weighted accumulation into float32 buffer (experts serialized on stream)
    float weighted = acc * route_weight;
    int out_idx = token_id * HIDDEN_SIZE + n;
    output_f32[out_idx] += weighted;
  }
}

// =========================================================================
// Kernel 5: Zero float32 output buffer
// =========================================================================
__global__ void zero_f32_kernel(float *buf, size_t size) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    buf[tid] = 0.0f;
  }
}

// =========================================================================
// Kernel 5b: Convert float32 accumulator to bfloat16 output
// =========================================================================
__global__ void convert_f32_to_bf16_kernel(const float *__restrict__ src,
                                           __nv_bfloat16 *__restrict__ dst,
                                           size_t size) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    dst[tid] = __float2bfloat16(src[tid]);
  }
}

// =========================================================================
// Kernel 6: Prefix sum (exclusive) on CPU-side small array
// Done on GPU with single block for convenience
// =========================================================================
__global__ void
prefix_sum_kernel(const int *__restrict__ counts, // [NUM_LOCAL_EXP]
                  int *__restrict__ starts,       // [NUM_LOCAL_EXP]
                  int n) {
  // Single thread computes prefix sum for small array
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
      starts[i] = sum;
      sum += counts[i];
    }
  }
}

// =========================================================================
// Host launch function (internal)
// =========================================================================
static void fused_moe_launch(
    const float *routing_logits,      // [T, 256]
    const __nv_bfloat16 *routing_bias, // [256] bf16
    const uint8_t *hidden_states,     // [T, H] fp8
    const float *hidden_states_scale, // [H/128, T]
    const uint8_t *gemm1_weights,     // [32, 4096, 7168] fp8
    const float *gemm1_weights_scale, // [32, 32, 56]
    const uint8_t *gemm2_weights,     // [32, 7168, 2048] fp8
    const float *gemm2_weights_scale, // [32, 56, 16]
    __nv_bfloat16 *output,            // [T, H] bf16
    int T, int local_expert_offset,
    float routed_scaling_factor, cudaStream_t stream) {

  // 1) Allocate workspace
  int *d_topk_indices = nullptr;
  float *d_topk_weights = nullptr;
  int *d_expert_counts = nullptr;
  int *d_expert_offsets = nullptr;
  int *d_expert_starts = nullptr;
  int *d_token_ids = nullptr;
  float *d_token_weights = nullptr;
  float *d_output_f32 = nullptr;

  cudaMalloc(&d_topk_indices, T * TOP_K * sizeof(int));
  cudaMalloc(&d_topk_weights, T * TOP_K * sizeof(float));
  cudaMalloc(&d_expert_counts, NUM_LOCAL_EXP * sizeof(int));
  cudaMalloc(&d_expert_offsets, NUM_LOCAL_EXP * sizeof(int));
  cudaMalloc(&d_expert_starts, NUM_LOCAL_EXP * sizeof(int));
  cudaMalloc(&d_token_ids, T * TOP_K * sizeof(int));
  cudaMalloc(&d_token_weights, T * TOP_K * sizeof(float));
  cudaMalloc(&d_output_f32, (size_t)T * HIDDEN_SIZE * sizeof(float));

  // 0) Zero float32 accumulation buffer
  {
    size_t total = (size_t)T * HIDDEN_SIZE;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    zero_f32_kernel<<<blocks, threads, 0, stream>>>(d_output_f32, total);
  }

  cudaMemsetAsync(d_expert_counts, 0, NUM_LOCAL_EXP * sizeof(int), stream);
  cudaMemsetAsync(d_expert_offsets, 0, NUM_LOCAL_EXP * sizeof(int), stream);

  // 2) Routing
  if (T > 0) {
    routing_kernel<<<T, NUM_GLOBAL_EXP, 0, stream>>>(
        routing_logits, routing_bias, d_topk_indices, d_topk_weights, T,
        routed_scaling_factor);
  }

  // 3) Count tokens per expert
  {
    int total = T * TOP_K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 0) {
      count_expert_tokens_kernel<<<blocks, threads, 0, stream>>>(
          d_topk_indices, d_expert_counts, T, local_expert_offset);
    }
  }

  // 4) Prefix sum to get expert start offsets
  prefix_sum_kernel<<<1, 1, 0, stream>>>(d_expert_counts, d_expert_starts,
                                         NUM_LOCAL_EXP);

  // 5) Build per-expert token lists
  {
    int total = T * TOP_K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 0) {
      build_expert_lists_kernel<<<blocks, threads, 0, stream>>>(
          d_topk_indices, d_topk_weights, d_expert_offsets, d_token_ids,
          d_token_weights, d_expert_starts, T, local_expert_offset);
    }
  }

  // 6) Copy expert counts and starts to host for kernel launch configuration
  int h_expert_counts[NUM_LOCAL_EXP];
  int h_expert_starts[NUM_LOCAL_EXP];
  cudaMemcpyAsync(h_expert_counts, d_expert_counts, NUM_LOCAL_EXP * sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_expert_starts, d_expert_starts, NUM_LOCAL_EXP * sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // 7) For each expert with assigned tokens, launch compute kernel
  // Shared memory: A[H] + gemm1_out[2I] + inter[I] = 7168 + 4096 + 2048 = 13312
  // floats = 53248 bytes
  size_t smem_size =
      (HIDDEN_SIZE + GEMM1_OUT_SIZE + INTER_SIZE) * sizeof(float);
  cudaFuncSetAttribute(moe_expert_compute_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  for (int e = 0; e < NUM_LOCAL_EXP; e++) {
    int num_tokens = h_expert_counts[e];
    if (num_tokens == 0)
      continue;

    // Each block handles one token, threads handle the inner dimensions
    int threads = 256;
    moe_expert_compute_kernel<<<num_tokens, threads, smem_size, stream>>>(
        hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale, d_output_f32,
        d_token_ids + h_expert_starts[e], d_token_weights + h_expert_starts[e],
        e, num_tokens, T);
  }

  // 8) Convert float32 accumulator to bf16 output
  {
    size_t total = (size_t)T * HIDDEN_SIZE;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    convert_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(d_output_f32,
                                                                output, total);
  }

  // Cleanup
  cudaFree(d_topk_indices);
  cudaFree(d_topk_weights);
  cudaFree(d_expert_counts);
  cudaFree(d_expert_offsets);
  cudaFree(d_expert_starts);
  cudaFree(d_token_ids);
  cudaFree(d_token_weights);
  cudaFree(d_output_f32);
}

// =========================================================================
// PyTorch C++ Extension Entry Point (DPS: output is the last argument)
// =========================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void kernel(
    torch::Tensor routing_logits,        // [seq_len, 256] float32
    torch::Tensor routing_bias,          // [256] bfloat16
    torch::Tensor hidden_states,         // [seq_len, 7168] float8_e4m3fn
    torch::Tensor hidden_states_scale,   // [56, seq_len] float32
    torch::Tensor gemm1_weights,         // [32, 4096, 7168] float8_e4m3fn
    torch::Tensor gemm1_weights_scale,   // [32, 32, 56] float32
    torch::Tensor gemm2_weights,         // [32, 7168, 2048] float8_e4m3fn
    torch::Tensor gemm2_weights_scale,   // [32, 56, 16] float32
    int64_t local_expert_offset,         // int32 scalar (passed as Python int)
    double routed_scaling_factor,        // float32 scalar (passed as Python float)
    torch::Tensor output                 // [seq_len, 7168] bfloat16  (DPS output)
) {
  int T = routing_logits.size(0);

  // Get the current CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  fused_moe_launch(
      routing_logits.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16 *>(routing_bias.data_ptr()),
      reinterpret_cast<const uint8_t *>(hidden_states.data_ptr()),
      hidden_states_scale.data_ptr<float>(),
      reinterpret_cast<const uint8_t *>(gemm1_weights.data_ptr()),
      gemm1_weights_scale.data_ptr<float>(),
      reinterpret_cast<const uint8_t *>(gemm2_weights.data_ptr()),
      gemm2_weights_scale.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
      T, (int)local_expert_offset, (float)routed_scaling_factor, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "Fused MoE kernel (CUDA)");
}