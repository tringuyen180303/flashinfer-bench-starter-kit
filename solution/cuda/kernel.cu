/*
 * MoE FP8 Block-Scale Kernel with DeepSeek-V3 Routing
 *
 * Implements: routing (sigmoid + group selection + top-k) → GEMM1 → SwiGLU →
 * GEMM2 Definition:
 * moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
 */

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============================================================================
// DeepSeek-V3 routing constants
// ============================================================================
constexpr int NUM_EXPERTS = 256;
constexpr int N_GROUP = 8;
constexpr int GROUP_SIZE = NUM_EXPERTS / N_GROUP; // 32
constexpr int TOP_K = 8;
constexpr int TOPK_GROUP = 4;
constexpr int NUM_LOCAL_EXPERTS = 32;

// ============================================================================
// 4.1.1  Parallel sigmoid for expert scores
//
// Each block handles one token.  256 threads = one per expert.
// Computes:
//   scores[t][e]           = sigmoid(routing_logits[t][e])
//   scores_with_bias[t][e] = scores[t][e] + bf16_to_f32(routing_bias[e])
//
// Both arrays are needed downstream:
//   - scores           → final weight normalization (without bias)
//   - scores_with_bias → group selection & top-k selection
// ============================================================================
__global__ void
gating_sigmoid_kernel(const float *__restrict__ routing_logits, // [T, 256]
                      const __nv_bfloat16 *__restrict__ routing_bias, // [256]
                      float *__restrict__ scores,           // [T, 256]
                      float *__restrict__ scores_with_bias, // [T, 256]
                      const int T) {
  const int token_idx = blockIdx.x;
  const int expert_idx = threadIdx.x;

  if (token_idx >= T)
    return;

  const int idx = token_idx * NUM_EXPERTS + expert_idx;

  // Load logit (coalesced – consecutive threads read consecutive floats)
  const float logit = routing_logits[idx];

  // Sigmoid: s = 1 / (1 + exp(-x))
  // Using __expf for fast-math (≤2 ULP, fine for routing decisions)
  const float s = 1.0f / (1.0f + __expf(-logit));

  // Bias (bf16→f32).  Only 1 KB total, will sit in L1 after first block.
  const float bias = __bfloat162float(routing_bias[expert_idx]);

  // Store both variants
  scores[idx] = s;
  scores_with_bias[idx] = s + bias;
}

// ============================================================================
// 4.1.2  Top-K selection kernel (DeepSeek-V3 grouped routing)
//
// One block per token, 256 threads (one per expert).
// Steps:
//   1. Load scores into shared memory
//   2. Group scoring: top-2 biased score sum per group (8 groups × 32 experts)
//   3. Group selection: top-4 groups
//   4. Expert selection: top-8 experts from the 4 selected groups (128 cands)
//   5. Weight normalization: unbiased sigmoid scores, normalized & scaled
//
// Outputs:
//   topk_idx     [T, TOP_K]  int32   – global expert indices
//   topk_weights [T, TOP_K]  float32 – normalized routing weights
// ============================================================================
__global__ void topk_gating_kernel(
    const float *__restrict__ scores,           // [T, 256] unbiased sigmoid
    const float *__restrict__ scores_with_bias, // [T, 256] biased
    int *__restrict__ topk_idx,                 // [T, TOP_K]
    float *__restrict__ topk_weights,           // [T, TOP_K]
    const float routed_scaling_factor, const int local_expert_offset,
    const int T) {
  const int token = blockIdx.x;
  const int eid = threadIdx.x; // 0..255, one per expert

  if (token >= T)
    return;

  // ---- Shared memory layout ------------------------------------------------
  __shared__ float smem_s[NUM_EXPERTS];        // unbiased sigmoid
  __shared__ float smem_sb[NUM_EXPERTS];       // biased
  __shared__ float smem_group_scores[N_GROUP]; // top-2 sum per group
  __shared__ int smem_top_groups[TOPK_GROUP];  // indices of selected groups
  __shared__ int smem_topk_idx[TOP_K];         // final expert indices
  __shared__ float smem_topk_weights[TOP_K];   // final normalized weights

  // ---- Step 1: Load scores into shared memory ------------------------------
  const int base = token * NUM_EXPERTS;
  smem_s[eid] = scores[base + eid];
  smem_sb[eid] = scores_with_bias[base + eid];
  __syncthreads();

  // ---- Step 2: Group scoring (threads 0..7, one per group) -----------------
  // Each of the 8 groups has GROUP_SIZE=32 experts.
  // Group score = sum of the top-2 biased scores in that group.
  if (eid < N_GROUP) {
    const int g_start = eid * GROUP_SIZE;
    float top1 = -FLT_MAX, top2 = -FLT_MAX;

    for (int i = 0; i < GROUP_SIZE; i++) {
      float v = smem_sb[g_start + i];
      if (v > top1) {
        top2 = top1;
        top1 = v;
      } else if (v > top2) {
        top2 = v;
      }
    }
    smem_group_scores[eid] = top1 + top2;
  }
  __syncthreads();

  // ---- Step 3: Group selection – top TOPK_GROUP=4 groups (thread 0) --------
  if (eid == 0) {
    // Simple selection sort for top-4 out of 8 groups
    float gs[N_GROUP];
    for (int i = 0; i < N_GROUP; i++)
      gs[i] = smem_group_scores[i];

    for (int k = 0; k < TOPK_GROUP; k++) {
      int best_i = 0;
      float best_v = -FLT_MAX;
      for (int i = 0; i < N_GROUP; i++) {
        if (gs[i] > best_v) {
          best_v = gs[i];
          best_i = i;
        }
      }
      smem_top_groups[k] = best_i;
      gs[best_i] = -FLT_MAX; // exclude from future picks
    }
  }
  __syncthreads();

  // ---- Step 4: Expert selection – top TOP_K=8 from 4 groups (thread 0) -----
  // 4 groups × 32 experts = 128 candidates.
  if (eid == 0) {
    // Collect candidate scores (non-selected groups → -inf)
    // We store the current top-8 in arrays and scan candidates.
    float tk_scores[TOP_K];
    int tk_indices[TOP_K];
    for (int k = 0; k < TOP_K; k++) {
      tk_scores[k] = -FLT_MAX;
      tk_indices[k] = -1;
    }

    for (int gi = 0; gi < TOPK_GROUP; gi++) {
      int g = smem_top_groups[gi];
      int g_start = g * GROUP_SIZE;
      for (int i = 0; i < GROUP_SIZE; i++) {
        float v = smem_sb[g_start + i];
        int idx = g_start + i;

        // Insert into top-k if larger than the current minimum
        // Find the minimum element in the top-k
        int min_k = 0;
        float min_v = tk_scores[0];
        for (int k = 1; k < TOP_K; k++) {
          if (tk_scores[k] < min_v) {
            min_v = tk_scores[k];
            min_k = k;
          }
        }
        if (v > min_v) {
          tk_scores[min_k] = v;
          tk_indices[min_k] = idx;
        }
      }
    }

    // ---- Step 5: Weight normalization ------------------------------------
    // Use unbiased sigmoid scores for the selected experts
    float weight_sum = 0.0f;
    for (int k = 0; k < TOP_K; k++) {
      tk_scores[k] = smem_s[tk_indices[k]]; // replace biased with unbiased
      weight_sum += tk_scores[k];
    }

    // Normalize and scale
    float inv_sum = routed_scaling_factor / (weight_sum + 1e-20f);
    for (int k = 0; k < TOP_K; k++) {
      smem_topk_idx[k] = tk_indices[k];
      smem_topk_weights[k] = tk_scores[k] * inv_sum;
    }
  }
  __syncthreads();

  // ---- Write results to global memory (threads 0..TOP_K-1) -----------------
  if (eid < TOP_K) {
    int out_base = token * TOP_K + eid;
    topk_idx[out_base] = smem_topk_idx[eid];
    topk_weights[out_base] = smem_topk_weights[eid];
  }
}

// ============================================================================
// 4.2  Token sorting & indexing
//
// Goal: rearrange the T×TOP_K (token, expert) assignments so that all tokens
// assigned to the same local expert are contiguous.  This lets section 4.3
// run batched GEMMs per expert.
//
// Two tiny kernels + a host-side prefix sum:
//
//   Kernel A – token_counting_kernel
//     Each thread processes one slot in topk_idx[T*TOP_K].
//     If the expert is local, atomicAdd on expert_counts[local_expert_id].
//
//   Host – exclusive prefix sum on expert_counts → expert_offsets
//     expert_offsets[e] = sum of expert_counts[0..e-1]
//     total_sorted = expert_offsets[NUM_LOCAL_EXPERTS]
//
//   Kernel B – token_scatter_kernel
//     Same thread mapping.  Each thread that has a local expert atomicAdds
//     on a running offset array to get its write position, then writes
//     its token_id and weight into the sorted arrays.
//
// Outputs:
//   sorted_token_ids [total_sorted]  int32   – original token index
//   sorted_weights   [total_sorted]  float32 – routing weight for the slot
//   sorted_expert_ids[total_sorted]  int32   – local expert index (0..31)
//   expert_offsets   [E_local + 1]   int32   – boundaries in sorted arrays
// ============================================================================

__global__ void token_counting_kernel(
    const int *__restrict__ topk_idx, // [T * TOP_K] global expert ids
    int *__restrict__ expert_counts,  // [NUM_LOCAL_EXPERTS] zeroed
    const int local_expert_offset,
    const int num_slots) // T * TOP_K
{
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= num_slots)
    return;

  const int global_eid = topk_idx[slot];
  const int local_eid = global_eid - local_expert_offset;

  if (local_eid >= 0 && local_eid < NUM_LOCAL_EXPERTS) {
    atomicAdd(&expert_counts[local_eid], 1);
  }
}

__global__ void token_scatter_kernel(
    const int *__restrict__ topk_idx,       // [T * TOP_K]
    const float *__restrict__ topk_weights, // [T * TOP_K]
    const int *__restrict__ expert_offsets, // [NUM_LOCAL_EXPERTS + 1]
    int *__restrict__ scatter_counters,     // [NUM_LOCAL_EXPERTS] zeroed
    int *__restrict__ sorted_token_ids,     // [total_sorted]
    float *__restrict__ sorted_weights,     // [total_sorted]
    int *__restrict__ sorted_expert_ids,    // [total_sorted]
    const int local_expert_offset,
    const int num_slots) // T * TOP_K
{
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= num_slots)
    return;

  const int global_eid = topk_idx[slot];
  const int local_eid = global_eid - local_expert_offset;

  if (local_eid >= 0 && local_eid < NUM_LOCAL_EXPERTS) {
    // Atomically claim a write position within this expert's segment
    const int pos = atomicAdd(&scatter_counters[local_eid], 1);
    const int write_idx = expert_offsets[local_eid] + pos;

    const int token_id = slot / TOP_K; // original token index

    sorted_token_ids[write_idx] = token_id;
    sorted_weights[write_idx] = topk_weights[slot];
    sorted_expert_ids[write_idx] = local_eid;
  }
}

// ============================================================================
// Host entry point
// ============================================================================
void kernel(torch::Tensor routing_logits, // [T, 256]           float32
            torch::Tensor routing_bias,   // [256]               bfloat16
            torch::Tensor hidden_states,  // [T, 7168]           float8_e4m3fn
            torch::Tensor hidden_states_scale, // [56, T]             float32
            torch::Tensor gemm1_weights, // [32, 4096, 7168]    float8_e4m3fn
            torch::Tensor gemm1_weights_scale, // [32, 32, 56]        float32
            torch::Tensor gemm2_weights, // [32, 7168, 2048]    float8_e4m3fn
            torch::Tensor gemm2_weights_scale, // [32, 56, 16]        float32
            int64_t local_expert_offset,       // scalar int32
            double routed_scaling_factor,      // scalar float32
            torch::Tensor output) // [T, 7168]           bfloat16 (DPS)
{
  const int T = routing_logits.size(0);
  const int num_slots = T * TOP_K; // total (token, expert) assignments

  // ----- Tensor options ----------------------------------------------------
  auto opts_f32 = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(routing_logits.device());
  auto opts_i32 = torch::TensorOptions()
                      .dtype(torch::kInt32)
                      .device(routing_logits.device());

  // ----- Intermediate buffers for routing ----------------------------------
  auto scores = torch::empty({T, NUM_EXPERTS}, opts_f32);
  auto scores_with_bias = torch::empty({T, NUM_EXPERTS}, opts_f32);
  auto topk_idx_tensor = torch::empty({T, TOP_K}, opts_i32);
  auto topk_wt_tensor = torch::empty({T, TOP_K}, opts_f32);

  // ----- 4.1.1  Parallel sigmoid -------------------------------------------
  {
    dim3 grid(T);
    dim3 block(NUM_EXPERTS); // 256 threads per block

    gating_sigmoid_kernel<<<grid, block>>>(
        routing_logits.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(routing_bias.data_ptr()),
        scores.data_ptr<float>(), scores_with_bias.data_ptr<float>(), T);
  }

  // ----- 4.1.2  Top-K selection --------------------------------------------
  // ----- 4.1.3  Local expert filtering (embedded in topk_gating_kernel) -----
  {
    dim3 grid(T);
    dim3 block(NUM_EXPERTS); // 256 threads per block

    topk_gating_kernel<<<grid, block>>>(
        scores.data_ptr<float>(), scores_with_bias.data_ptr<float>(),
        topk_idx_tensor.data_ptr<int>(), topk_wt_tensor.data_ptr<float>(),
        static_cast<float>(routed_scaling_factor),
        static_cast<int>(local_expert_offset), T);
  }

  // ----- 4.2  Token sorting & indexing -------------------------------------
  // Phase 1: Count tokens per local expert
  auto expert_counts = torch::zeros({NUM_LOCAL_EXPERTS}, opts_i32);
  {
    const int threads = 256;
    const int blocks = (num_slots + threads - 1) / threads;

    token_counting_kernel<<<blocks, threads>>>(
        topk_idx_tensor.data_ptr<int>(), expert_counts.data_ptr<int>(),
        static_cast<int>(local_expert_offset), num_slots);
  }

  // Phase 2: Exclusive prefix sum → expert_offsets[E_local + 1]
  //   expert_offsets[e] = sum(expert_counts[0..e-1])
  //   expert_offsets[E_local] = total_sorted
  // We do this on CPU (only 32 ints to move) to avoid a CUB dependency.
  cudaDeviceSynchronize();
  auto counts_cpu = expert_counts.cpu();
  auto offsets_cpu = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::kInt32);
  int *c_ptr = counts_cpu.data_ptr<int>();
  int *o_ptr = offsets_cpu.data_ptr<int>();
  o_ptr[0] = 0;
  for (int e = 0; e < NUM_LOCAL_EXPERTS; e++) {
    o_ptr[e + 1] = o_ptr[e] + c_ptr[e];
  }
  const int total_sorted = o_ptr[NUM_LOCAL_EXPERTS];
  auto expert_offsets = offsets_cpu.to(routing_logits.device());

  // Phase 3: Scatter tokens into expert-sorted arrays
  auto sorted_token_ids = torch::empty({total_sorted}, opts_i32);
  auto sorted_weights = torch::empty({total_sorted}, opts_f32);
  auto sorted_expert_ids = torch::empty({total_sorted}, opts_i32);
  auto scatter_counters = torch::zeros({NUM_LOCAL_EXPERTS}, opts_i32);

  if (total_sorted > 0) {
    const int threads = 256;
    const int blocks = (num_slots + threads - 1) / threads;

    token_scatter_kernel<<<blocks, threads>>>(
        topk_idx_tensor.data_ptr<int>(), topk_wt_tensor.data_ptr<float>(),
        expert_offsets.data_ptr<int>(), scatter_counters.data_ptr<int>(),
        sorted_token_ids.data_ptr<int>(), sorted_weights.data_ptr<float>(),
        sorted_expert_ids.data_ptr<int>(),
        static_cast<int>(local_expert_offset), num_slots);
  }

  // ----- 4.3    Fused expert computation (TODO) ----------------------------
  // At this point we have:
  //   sorted_token_ids  [total_sorted] – which token each slot belongs to
  //   sorted_weights    [total_sorted] – routing weight for the slot
  //   sorted_expert_ids [total_sorted] – local expert id (0..31)
  //   expert_offsets    [33]           – start/end boundaries per expert
  //
  // For each local expert e (0..31):
  //   tokens = sorted_token_ids[expert_offsets[e] .. expert_offsets[e+1])
  //   weights = sorted_weights[expert_offsets[e] .. expert_offsets[e+1])
  //   Run: GEMM1[e] → SwiGLU → GEMM2[e], accumulate weighted into output

  // Zero output until expert computation is implemented
  output.zero_();
}

// ============================================================================
// PyBind11 binding
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "MoE FP8 Block-Scale Kernel (DeepSeek-V3 routing)");
}
