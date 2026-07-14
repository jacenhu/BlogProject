# GLM 5.2 Attention 机制

> **KV Cache 系列文章**
> - [KV Cache 基础知识](sglang_kvcache_knowhow.md)
> - [GLM 5.2 KV Cache](sglang_kvcache_glm5.2.md)
> - [GLM 5.2 KV Cache 对接原理](sglang_glm52_kv_cache_integration.md)
> - **GLM 5.2 Attention 机制** ← 当前文章
> - [DeepSeek V4 KV Cache](sglang_kvcache_deeepseekv4.md)
> - [KV Cache 传输](sglang_glm52_kv_cache_transfer.md)

## 目录

1. [概述](#1-概述)
2. [MLA：Multi-head Latent Attention](#2-mlamulti-head-latent-attention)
3. [DSA：DeepSeek Sparse Attention](#3-dsadeepseek-sparse-attention)
4. [MLA + DSA 融合架构](#4-mla--dsa-融合架构)
5. [核心数学公式](#5-核心数学公式)
6. [完整前向传播流程](#6-完整前向传播流程)
7. [Indexer 机制详解](#7-indexer-机制详解)
8. [多后端实现](#8-多后端实现)
9. [与传统 Attention 对比](#9-与传统-attention-对比)
10. [源码索引](#10-源码索引)

---

## 1. 概述

GLM 5.2 的 attention 机制是 **MLA (Multi-head Latent Attention)** + **DSA (DeepSeek Sparse Attention)** 的融合，是目前 LLM 领域最激进的内存-计算联合优化方案之一。

### 一句话概括

> 用低秩压缩把 KV cache 缩小约 18 倍，再用稀疏索引把 attention 计算量缩小约 60 倍（128K 上下文下），二者叠加实现超长上下文的高效推理。

### 核心思路

```
传统 Attention 的成本:
  存储: O(n x num_heads x head_dim)  -- n 是序列长度
  计算: O(n^2 x num_heads x head_dim)

MLA 解决存储:
  KV 压缩到低秩潜在空间: [heads, head_dim] -> [1, kv_lora_rank + rope]
  存储降至 ~1/18

DSA 解决计算:
  两阶段稀疏化: 先用小 key 粗筛 topk 位置，再精算
  计算降至 ~topk/n (n=128K 时 ~1.6%)
```

### 架构全景图

```
                             hidden_states [B, 7168]
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              q_latent         kv_latent         k_pe_raw
             [B, 1536]         [B, 512]          [B, 64]
                    │               │               │
      ┌─────────────┼───────┐       │               │
      │             │       │       │               │
      ▼             ▼       ▼       ▼               ▼
   layernorm    Indexer  q_b_proj  layernorm      RoPE
      │          .wq()     │         │               │
      ▼             │       ▼         ▼               ▼
   q_b_proj         │    q_nope    k_nope          k_rope
      │             │  [B,heads,128] [B,1,512]    [B,1,64]
      ▼             │       │         │               │
   q_nope           │       │         │               │
  [B,heads,128]     │       │         │               │
      │             │       │         │               │
      ▼             │       │         │               │
  bmm(W_kc)         │       │         │               │
   (Q吸收投影)       │       │         │               │
      │             │       │         │               │
      ▼             │       │         │               │
  q_nope_out        │       │   写入 KV Cache:        │
  [B,heads,512]     │       │   set_mla_kv_buffer(    │
      │             │       │     k_nope, k_rope)     │
      │             │       │         │               │
      │             │       │   写入 Indexer Cache:   │
      │             │       │   set_index_k_scale_    │
      │             │       │   buffer(index_key,     │
      │             │       │     index_scale)        │
      │             │       │                        │
      │    ┌────────┘       │                        │
      │    │                │                        │
      │    │ 读取 Indexer KV Cache                    │
      │    │ fp8_index(q_index, K_index)              │
      │    │   -> topk_indices [B, 2048]              │
      │    │                │                        │
      ▼    ▼                ▼                        │
  ┌────────────────────────────────┐                 │
  │  q_full = concat(q_nope_out,   │                 │
  │                   q_rope)      │                 │
  │  [B, heads, 576]              │                 │
  └────────────┬───────────────────┘                 │
               │                                      │
               ▼                                      │
  ┌────────────────────────────────────────┐         │
  │  Sparse Attention                      │         │
  │  flash_mla_sparse_fwd(                 │         │
  │    q_full,                             │         │
  │    kv_cache [size, 1, 576],            │ <───────┘
  │    page_table[topk_only]               │
  │  )                                     │
  │  -> attention_output [B, heads, 512]  │
  └────────────────────────────────────────┘
               │
               ▼
          o_proj -> output [B, 7168]
```

---

## 2. MLA：Multi-head Latent Attention

### 2.1 设计动机

标准 MHA 中，每个 token 的 K 和 V 是完整的 `[num_kv_heads, head_dim]` 矩阵。但 attention 的输出只需 `[num_heads, head_dim]` 的加权和，这意味着 K/V 存在大量冗余。

MLA 将 K/V 压缩到一个**低秩潜在空间**，仅在需要时才通过升维矩阵解压到完整维度。核心思想来源于 LoRA 的低秩分解：`W_kv ≈ W_kv_a x W_kv_b`。

```
标准 MHA:
  hidden -> K_proj -> [num_kv_heads, head_dim]    -- 直接投影到高维
  hidden -> V_proj -> [num_kv_heads, head_dim]    -- K 和 V 独立

MLA:
  hidden -> kv_a_proj -> [kv_lora_rank]             -- 压缩到低秩潜在空间
           kv_b_up   -> [num_heads, head_dim]      -- 需要时才解压
```

### 2.2 关键投影矩阵

GLM 5.2 的 `kv_lora_rank = 512`，`num_heads = 128`：

```python
# 源码: deepseek_v2.py, DeepseekV2AttentionMLA.__init__

# 融合投影: hidden -> q_compress + kv_latent + k_rope_raw
self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
    hidden_size,                                        # 7168
    q_lora_rank + kv_lora_rank + qk_rope_head_dim,     # 1536 + 512 + 64 = 2112
)

# Q 解压投影: q_latent -> num_heads x qk_head_dim
self.q_b_proj = ColumnParallelLinear(
    q_lora_rank,                       # 1536
    num_heads * qk_head_dim,           # 128 x 192 = 24576
)

# KV 解压投影: kv_latent -> num_heads x (nope + v)
self.kv_b_proj = ColumnParallelLinear(
    kv_lora_rank,                      # 512
    num_heads * (qk_nope_head_dim + v_head_dim),  # 128 x (128 + 512) = 81920
)
```

### 2.3 投影流程详解

```
Step 1: 融合下投影 (fused_qkv_a_proj_with_mqa)
  hidden_states [B, 7168]
      │  x weight [2112, 7168]
      ▼
  qkv_latent [B, 2112]
      │
      ├── q_latent:   [B, 0:1536]      -- Q 的压缩表示
      ├── kv_latent:  [B, 1536:2048]   -- KV 的压缩表示 (512维)
      └── k_pe_raw:   [B, 2048:2112]   -- 未旋转的 RoPE 键 (64维)

Step 2: Q 路径
  q_latent [B, 1536]
      │
      ├── q_a_layernorm(RMSNorm)
      │
      ├── q_b_proj [1536 -> 24576]
      │     reshape -> [B, num_local_heads, 192]
      │          (TP切分后: num_local_heads = num_heads / tp_size)
      │
      └── split:
            q_nope: [B, heads, 128]    -- 与内容相关的部分
            q_rope: [B, heads, 64]     -- 用于 RoPE 的位置部分

Step 3: KV 路径
  kv_latent [B, 512]
      │
      ├── kv_a_layernorm(RMSNorm)
      │     unsqueeze(1) -> [B, 1, 512]
      │
      └── k_nope: [B, 1, 512]         -- 压缩后的 K/V 内容
                                        -- 注意: 同时充当 K 的内容部分和 V!

  k_pe_raw [B, 64]
      │
      ├── RoPE(positions, k_pe_raw)
      │     unsqueeze(1) -> [B, 1, 64]
      │
      └── k_rope: [B, 1, 64]          -- 位置编码后的 K
```

### 2.4 V 就是 K_nope —— MLA 最核心的设计

```python
# 源码: memory_pool.py:2230
def get_value_buffer(self, layer_id: int):
    # MLA 中 V = k_nope
    return self.kv_buffer[layer_id - self.start_layer][..., :self.kv_lora_rank]
    #      取 kv_buffer 的前 kv_lora_rank=512 维
```

完整 K 存储在 `kv_buffer` 的一维 `[576]` 中：

```
kv_buffer[slot_id, 0, :]:
  ┌─────────────────────────────────┬──────────────────┐
  │  k_nope (latent, 同时也是 V)      │  k_rope (位置编码) │
  │  kv_lora_rank = 512              │  64                │
  │  用途: 既做 K 的内容部分           │  用途: 只做 K 的    │
  │        又做 V (加权求和的目标)      │       位置编码部分  │
  └─────────────────────────────────┴──────────────────┘
  ◄──────────────── 576 elements ──────────────────────►
```

### 2.5 Q 的 BMM 吸收变换

为了让 Q 能与压缩后的 K（只有 512 维，没有 128 个头的维度）做 attention，需要将 Q 投影到 `kv_lora_rank` 空间：

```
q_nope: [B, heads, 128]
      │
      ├── transpose(0, 1) -> [heads, B, 128]
      │
      ├── bmm(q_nope^T, W_kc)
      │     W_kc: [heads, 128, kv_lora_rank]
      │     来源: kv_b_proj.weight 中 qk_nope_head_dim 对应的部分
      │     物理含义: 将 Q 的内容部分旋转到与 K_nope 同维的空间
      │     输出: [heads, B, 512]
      │
      └── transpose(0, 1) -> [B, heads, 512]

最终 q_full:
  q_full = concat(q_nope_out, q_rope) = [B, heads, 512 + 64]
                                       = [B, heads, 576]
```

**W_kc 的由来**: `kv_b_proj` 的权重形状是 `[num_heads x (nope + v), kv_lora_rank] = [128 x 640, 512]`。其中前 `nope=128` 维是 K_nope 的"解压矩阵"，后 `v=512` 维是 V 的"解压矩阵"。`W_kc` 取的是前半部分，reshape 为 `[heads, nope, kv_lora_rank] = [128, 128, 512]`。Q 与 `W_kc` 做 BMM 后变成 `[heads, B, 512]`，这样就与 K_nope 处于同一空间了。

### 2.6 MLA Attention Score 计算

在 MLA 空间中，attention score 的计算方式如下：

```
对于请求 i，序列位置 t 和 j:

K_full[j] = concat(kv_cache[j].nope, kv_cache[j].rope)
           = [1, kv_lora_rank + qk_rope_head_dim]
           = [1, 576]

Q_full[t]  = concat(q_nope_absorbed[t], q_rope[t])
           = [heads, kv_lora_rank + qk_rope_head_dim]
           = [heads, 576]

Score[t][j] = (Q_full[t] dot K_full[j]) / sqrt(576)

Attention[t] = softmax_j(Score[t][j]) x V[j]
             = softmax_j(Score[t][j]) x kv_cache[j].nope
             = softmax_j(Score[t][j]) x kv_cache[j, :, :512]
```

### 2.7 存储对比

```
GLM-4 (标准 MHA, 40 heads, head_dim=128):
  K per token: 40 x 128 = 5120 elements
  V per token: 40 x 128 = 5120 elements
  总计: 10240 elements x 2 bytes (BF16) = 20,480 bytes ~ 20 KB

GLM 5.2 (MLA, 1 kv head, kv_lora_rank=512, rope=64):
  KV per token: 512 + 64 = 576 elements
  总计: 576 x 2 bytes (BF16, 主KV) + 132 bytes (FP8, Indexer KV)
       = 1,152 + 132 = 1,284 bytes ~ 1.25 KB

存储压缩比: 20,480 / 1,284 ~ 16x
```

---

## 3. DSA：DeepSeek Sparse Attention

### 3.1 设计动机

MLA 解决了 KV cache 的存储问题，但 attention 计算仍然是 O(n^2 x d)。对于超长上下文（如 128K tokens），每层每个 token 都要与 128K 个历史 token 计算 attention score。

DSA 观察到 attention 本质上是**稀疏的**：在长序列中，绝大多数 token 的 attention score 接近于 0，真正有贡献的只有少数关键位置。如果能低成本地找到这些关键位置，就可以跳过无关 token 的计算。

### 3.2 两阶段稀疏化

```
┌─────────────────────────────────────────────────────────┐
│                  DSA 两阶段 Attention                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Stage 1: 粗筛 (Indexer)                                 │
│    ┌─────────────────────────────────────────┐           │
│    │  Index Query:  q_index [1, 128]         │           │
│    │  Index Key:    K_index_cache [n, 128]   │           │
│    │  Score:        q_index x K_index^T      │           │
│    │              -> [1, n]                  │           │
│    │  TopK:         topk(score, k=2048)      │           │
│    │              -> selected_indices [2048] │           │
│    └─────────────────────────────────────────┘           │
│                         │                                │
│                         ▼                                │
│  Stage 2: 精算 (Sparse Attention)                        │
│    ┌─────────────────────────────────────────┐           │
│    │  Full Query:  Q_full [heads, 576]       │           │
│    │  Full Key:    K_full[topk_indices, 576] │           │
│    │  Full Value:  V_full[topk_indices, 512] │           │
│    │  Score:       Q_full x K_full^T         │           │
│    │              -> [heads, 2048]           │           │
│    │  Output:      softmax(score) x V_full   │           │
│    └─────────────────────────────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.3 为什么 DSA 有效

DSA 的有效性建立在两个关键理论上：

**理论 1: Attention 的稀疏性。** 对于语言模型，attention 通常集中在少数 token 上（如标点、关键词、BOS token），大部分 token 的 attention score 对数值贡献极小。DSA 利用一个小的"索引模型"（128 维的 indexer）来近似找出这些"关键 token"。

**理论 2: 近似 Attention Score 保持 TopK 序。** 虽然 Indexer 用的是 128 维的量化 key（FP8），不是完整的 576 维，但实验表明，用 128 维近似选出的 topk 位置，与用完整 576 维计算后选出的 topk 位置，重叠率超过 95%。这是因为 attention 的稀疏性使得"谁是重要的"比"精确的重不重要程度"更容易判定。

### 3.4 Indexer 的 K Cache（第二层 KV Cache）

DSA 需要每层额外存储一份小 key：

```
Full KV (主 cache):
  kv_buffer[slot]: [1, 576]  -- BF16 or FP8
  用途: 精确 attention 计算

Indexer KV (第二层 cache):
  index_k_with_scale_buffer[page]:
    page = 64 tokens
    每个 token: 128 bytes (FP8 key) + 4 bytes (FP32 scale)
  用途: 快速粗筛 topk 位置
```

Indexer K 的存储格式（每页 64 tokens, CUDA）：

```
Page i (8448 bytes = 64 x 132):
  ┌──────────────────────────────┬──────────────────────────┐
  │  Token 0..63 的 FP8 index K  │  Token 0..63 的 scale   │
  │  64 x 128 = 8192 bytes       │  64 x 4 = 256 bytes      │
  │  uint8[8192]                 │  uint8[256]              │
  └──────────────────────────────┴──────────────────────────┘

其中:
  FP8 K:    block_size=128, 每 token 1 个 block, 128个FP8元素=128 bytes
  Scale:    每 token 1 个 FP32 scale因子 = 4 bytes
  总计:     132 bytes / token
```

---

## 4. MLA + DSA 融合架构

### 4.1 为什么二者是正交优化

```
                │  存储优化      │  计算优化
────────────────┼───────────────┼────────────────
MLA (低秩压缩)   │  KV cache -18x │  无直接影响
DSA (稀疏索引)   │  额外 +132B/tok│  attention -60x
────────────────┼───────────────┼────────────────
MLA + DSA       │  综合 -16x    │  综合 -60x
```

MLA 和 DSA 互相独立、正交优化：
- MLA 降低了 K/V 的维度，使 DSA 的 Indexer 能用更小的 key 做粗筛
- DSA 的稀疏化使 MLA 的 576 维 K 无需对每个历史 token 计算

### 4.2 融合后的完整 Attention 管道

```
输入: hidden_states [B, 7168]

1. 下投影
   fused_qkv_a -> q_latent [B,1536], kv_latent [B,512], k_pe [B,64]

2. Q 路径
   layernorm(q_latent) -> q_b_proj -> q_nope [B,heads,128], q_rope [B,heads,64]

3. K/V 路径
   layernorm(kv_latent) -> k_nope [B,1,512]
   RoPE(k_pe)           -> k_rope [B,1,64]

4. [MLA] Q 吸收投影
   q_nope_out = bmm(q_nope^T, W_kc) -> [B, heads, 512]

5. [DSA] Indexer 执行 (如果 skip_topk=False)
   a) 写入 Indexer Cache:
      index_key = indexer.wk(hidden) [B, 1, 128]
      FP8量化 -> 写入 index_k_with_scale_buffer

   b) 读取 + 计算 topk:
      q_index = indexer.wq(q_latent) [B, 1, 128]
      遍历所有历史 token，读取 Indexer K Cache
      score = fp8_mqa_logits(q_index, K_index_cache, weights)
      topk_indices = score.topk(k=2048) -> [B, 2048]

6. [MLA+DSA] 稀疏 Attention
   q_full = concat(q_nope_out, q_rope) [B, heads, 576]
   构建稀疏页表: page_table = req_to_token[topk_indices]
   只对 topk=2048 个位置计算:
     K_full = kv_cache[topk_indices] -> [B, 2048, 576]
     V = K_full[..., :512]
     score = Q_full x K_full^T / sqrt(576)
     output = softmax(score) x V

7. 输出投影
   o_proj(output) -> [B, 7168]
```

---

## 5. 核心数学公式

### 5.1 符号定义

| 符号 | 含义 | 典型值 |
|------|------|--------|
| d | 隐藏维度 hidden_size | 7168 |
| d_q | Q 低秩维度 q_lora_rank | 1536 |
| d_kv | KV 低秩维度 kv_lora_rank | 512 |
| d_R | RoPE 维度 qk_rope_head_dim | 64 |
| d_n | Q 非位置维度 qk_nope_head_dim | 128 |
| h | 注意力头数 num_heads | 128 |
| n | 序列长度 | 可变, max 128K |
| k | DSA topk | 2048 |

### 5.2 投影矩阵

```
W^A in R^{(d_q + d_kv + d_R) x d}    融合下投影
W^UQ in R^{(h x (d_n + d_R)) x d_q}   Q 上投影
W^UKV in R^{(h x (d_n + d_v)) x d_kv} KV 上投影
W^O in R^{d x (h x d_v)}             输出投影
W^IQ in R^{d_idx x d_q}              Indexer Q 投影 (d_idx=128)
W^IK in R^{d_idx x d}                Indexer K 投影 (d_idx=128)
w in R^{1 x d_idx}                   Indexer 权重向量
```

### 5.3 MLA 公式

```
给定 token t 的输入 h_t in R^d:

Step 1: 融合下投影
  [q_t^C; c_t^KV; k_t^R] = W^A x h_t

  q_t^C  in R^{d_q}     Q 的压缩表示
  c_t^KV in R^{d_kv}    KV 的压缩表示 (潜在向量)
  k_t^R  in R^{d_R}     RoPE 键 (未旋转)

Step 2: Q 上投影
  q_t = W^UQ x RMSNorm(q_t^C) in R^{h x (d_n + d_R)}

  分解:
    q_t^C,nope in R^{h x d_n}    Q 的内容部分
    q_t^C,rope in R^{h x d_R}    Q 的位置部分

Step 3: Q 吸收投影
  W^UKV 分解为两个子矩阵:
    W^UK = W^UKV[:, :d_n, :]         in R^{h x d_n x d_kv}
    W^UV = W^UKV[:, d_n:, :]         in R^{h x d_v x d_kv}

  Q 吸收:
    q_t^Abs = q_t^C,nope x W^UK      in R^{h x d_kv}
    (batch matmul: [B,h,d_n] x [h,d_n,d_kv] -> [B,h,d_kv])

Step 4: RoPE
  q_t^R = RoPE(q_t^C,rope)
  k_t^R = RoPE(k_t^R)

Step 5: Attention Score
  sc_{t,j} = (q_t^Abs dot c_j^KV + q_t^R dot k_j^R) / sqrt(d_kv + d_R)
           = (q_t^Abs x c_j^KV^T + q_t^R x k_j^R^T) / sqrt(512 + 64)
           = (q_t^Abs x c_j^KV^T + q_t^R x k_j^R^T) / sqrt(576)

  紧凑形式:
    Q_full = [q_t^Abs || q_t^R] in R^{h x (d_kv + d_R)}
    K_full = [c_j^KV || k_j^R]     in R^{1 x (d_kv + d_R)}
    Score  = Q_full x K_full^T / sqrt(d_kv + d_R)

Step 6: Output
  V_j = c_j^KV in R^{d_kv}   (V 就是 KV 潜在向量本身)

  o_t = sum_j(softmax(sc_{t,j}) x V_j) in R^{h x d_kv}
  output_t = W^O x o_t
```

### 5.4 DSA 公式（Indexer）

```
给定 token t 的输入 h_t 和压缩 Q q_t^C:

Step 1: Indexer Key 投影
  k_t^idx = W^IK x h_t in R^{d_idx}

Step 2: Indexer Query 投影
  q_t^idx = W^IQ x q_t^C in R^{d_idx}

Step 3: 量化 Indexer Key (写入 Indexer KV Cache)
  k_t^idx_fp8, scale_t = act_quant(k_t^idx, block_size=128)

Step 4: 计算近似 Attention Score
  sc_{t,j}^idx = (q_t^idx dot k_j^idx) x w
               = (q_t^idx x k_j^idx^T) x w

  其中 w in R^{d_idx} 是可学习的逐通道权重

Step 5: TopK 选择
  I_t = topk_indices(sc_t^idx, k=k)
  I_t in N^k  选出的 k=2048 个最重要位置
```

### 5.5 融合公式

```
DSA Attention 最终输出:

给定 I_t = topk(q_t^idx x K_idx_cache^T x w, k=2048):

  K_selected = {c_j^KV || k_j^R | j in I_t} in R^{k x (d_kv + d_R)}
  V_selected = {c_j^KV | j in I_t} in R^{k x d_kv}

  o_t = softmax(
    (q_t^Abs x K_selected_nope^T + q_t^R x K_selected_rope^T) / sqrt(576)
  ) x V_selected
```

---

## 6. 完整前向传播流程

### 6.1 分步详解

以下是 `DeepseekV2AttentionMLA.forward()` 的完整流程（源码: `deepseek_v2.py:1840`）：

```
forward(hidden_states, positions, forward_batch, ...)
  │
  ├─ forward_prepare()  ———— 产生 K, V, Q, topk_indices
  │   │
  │   ├─ dispatch_attn_forward_method(forward_batch)
  │   │   决定使用 MLA / MHA / MHA_ONE_SHOT 等哪个路径
  │   │
  │   ├─ forward_absorb_prepare()  ———— MLA 路径入口
  │   │   │
  │   │   ├─ fused_qkv_a_proj_with_mqa(hidden_states)
  │   │   │    -> qkv_latent [B, 2112]
  │   │   │   split:
  │   │   │     q_latent  [B, 1536]
  │   │   │     kv_latent [B, 512]
  │   │   │     k_pe_raw  [B, 64]
  │   │   │
  │   │   ├─ layernorm(q_latent) -> q_b_proj -> q [B, heads, 192]
  │   │   │   split -> q_nope [B, heads, 128], q_rope [B, heads, 64]
  │   │   │
  │   │   ├─ layernorm(kv_latent) -> k_nope [B, 1, 512]
  │   │   │
  │   │   ├─ RoPE(q_rope, k_pe_raw) -> q_rope, k_rope
  │   │   │
  │   │   ├─ bmm(q_nope^T, W_kc) -> q_nope_out [B, heads, 512]
  │   │   │
  │   │   └─ if use_dsa and not skip_topk:
  │   │         indexer(x=hidden_states, q_lora=q_latent,
  │   │                 positions, forward_batch, layer_id)
  │   │         |
  │   │         ├─ index_key = indexer.wk(hidden_states) [B,1,128]
  │   │         ├─ FP8 量化 index_key
  │   │         ├─ 写入 index_k_with_scale_buffer
  │   │         ├─ q_index = indexer.wq(q_latent) [B,1,128]
  │   │         ├─ 读取 Indexer K Cache
  │   │         ├─ fp8_mqa_logits(q_index, K_index, weights)
  │   │         └─ topk -> topk_indices [B, 2048]
  │   │
  │   └─ return (q_rope, k_rope, q_nope_out, k_nope,
  │              forward_batch, ..., topk_indices, ...)
  │
  └─ forward_core()  ———— 执行 Attention
      │
      └─ forward_absorb_core()
          │
          └─ self.attn_mqa.forward(q_nope, None, None, ...)
              │
              └─ DSA Backend.forward_decode()  (或 forward_extend)
                  │
                  ├─ set_mla_kv_buffer(layer, out_cache_loc,
                  │                     k_nope, k_rope)
                  │   └─ 写入主 KV Cache
                  │
                  ├─ get_key_buffer(layer_id)
                  │   └─ 获取完整 KV buffer 引用
                  │
                  ├─ transform_index_page_table_decode(
                  │     page_table, topk_indices)
                  │   └─ 将 topk 位置转换为物理 slot 索引
                  │
                  └─ flash_mla_sparse_fwd(
                        q, kv_cache, topk_page_table)
                      └─ 稀疏 Attention 内核
```

### 6.2 Decode vs Prefill 的差异

| 维度 | Decode | Extend (Prefill) |
|------|--------|-----------------|
| 新 token 数 | 1 per request | 可变 (通常 1-32K+) |
| Q 维度 | `[B, heads, 576]` | `[total_new_tokens, heads, 576]` |
| K/V 新增 | 1 slot/request | num_new_tokens slots |
| KV 写入 | `set_mla_kv_buffer` 写 1 个 slot | 批量写入 |
| KV 读取范围 | 全历史 (topk) | 全历史 + 新 token 间互 attention |
| 因果 mask | 无需 (每个 Q 只有 1 个位置) | 需要 (prefill chunk 内因果) |
| Indexer | 增量: 只写 1 个 index key | 批量: 写 num_new_tokens 个 |
| DSA 后端 | `dsa_decode_impl` | `dsa_prefill_impl` (可不同!) |

---

## 7. Indexer 机制详解

### 7.1 Indexer 的网络结构

```python
# 源码: dsa_indexer.py class Indexer
class Indexer:
    def __init__(self, ...):
        # K 投影: hidden -> 1 head x 128 dim
        self.wk = ReplicatedLinear(hidden_size=7168, output=128)

        # Q 投影: q_latent -> 1 head x 128 dim
        self.wq = ReplicatedLinear(input=1536, output=128)

        # 可学习的通道权重
        self.weights = Parameter([1, 128])

        # 量化参数
        self.block_size = 128   # FP8 量化块大小
        self.scale_fmt = "ue8m0"  # 微缩放格式
```

### 7.2 Indexer 前向传播

```python
def forward_cuda(self, x, q_lora, positions, forward_batch, layer_id):
    # x: hidden_states [B, 7168]
    # q_lora: 压缩后的 Q latent [B, 1536]

    # 1. 产生并存储 Indexer Key
    index_key = self.wk(x)  # [B, 1, 128]
    self._store_index_k_cache(
        forward_batch, layer_id, index_key
    )
    # -> 写入 DSATokenToKVPool.index_k_with_scale_buffer

    # 2. 产生 Indexer Query
    index_query = self.wq(q_lora)  # [B, 1, 128]

    # 3. 使用 deep_gemm.fp8_mqa_logits 计算近似 score
    #    遍历每个 batch 元素的所有历史 token
    topk_indices = self.forward_indexer(
        q_fp8=index_query,
        weights=self.weights,
        forward_batch=forward_batch,
        topk=self.index_topk,  # 2048
        layer_id=layer_id,
    )

    return topk_indices  # [B, 2048]
```

### 7.3 跨层 TopK 共享

每层都运行 Indexer 开销很大。DSA 利用 attention 的层间相似性，让多数层**跳过 Indexer 计算**，直接复用上层的 topk 结果：

```python
# 源码: deepseek_v2.py:1671-1679
# skip_topk: True → 跳过本层 Indexer, 复用 prev_topk_indices
# next_skip_topk: True → 下一层也将跳过

# MTP (NextN) 层: 总是共享
if is_nextn:
    self.skip_topk = True
    self.next_skip_topk = True

# 普通层: 根据配置决定
else:
    self.skip_topk = dsa_layer_skips_topk(config, layer_id)
    self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)

# 运行时:
if not self.skip_topk:
    topk_indices = self.indexer(...)     # 计算新的 topk
else:
    topk_indices = prev_topk_indices     # 复用上层结果
```

通常的共享策略是每 2-4 层执行一次 Indexer。例如，如果 `skip_topk` 配置为 `[True, False, True, False, True, False, ...]`，则只有不到一半的层执行 Indexer，减少约 60% 的索引开销。

### 7.4 Indexer 的 CUDA Graph 优化

在 CUDA Graph 捕获模式下，Indexer 被**拆分为独立算子**（split op），以支持 breakable/piecewise CUDA graph：

```python
# Indexer split op 将 forward 拆为两部分:
# Op 1: 写入 Indexer K Cache (必须在 graph 内执行)
# Op 2: 读取 Indexer K Cache + 计算 topk (可在 graph 间执行)
```

这使得 Indexer 的 topk 计算可以在 CUDA graph 的"断点"处独立运行，不破坏 graph 的完整性。

### 7.5 Prefill 下的 CP (Context Parallel) 支持

DSA 支持 Prefill Context Parallelism，将长序列按 round-robin 分配给多个 rank：

```python
# dsa_indexer.py -> is_dsa_enable_prefill_cp()
# 当启用时:
# - 每个 CP rank 只负责序列的一部分 tokens
# - Indexer K Cache 只在各自的 rank 上存储本地 tokens
# - topk 结果需要通过 all-gather 汇总
```

---

## 8. 多后端实现

DSA 支持多种底层实现，通过 `dsa_decode_impl` 和 `dsa_prefill_impl` 选择：

| 实现 | 适用场景 | 内核 | 平台 |
|------|---------|------|------|
| `flashmla_sparse` | NVIDIA, MLA 稀疏 decode | flash_mla_sparse_fwd | CUDA |
| `flashmla_kv` | NVIDIA, MLA 全量 (无 topk) | flash_mla_with_kvcache | CUDA |
| `fa3` | NVIDIA, FlashAttention-3 | flash_attn_varlen_func | CUDA |
| `tilelang` | NVIDIA/ROCm, 动态生成 | tilelang 代码生成 | CUDA/ROCm |
| `trtllm` | NVIDIA SM100+, Blackwell | TensorRT-LLM MLA | CUDA |
| `aiter` | AMD ROCm | AITER mla_decode_fwd | ROCm |
| `cutlass_mla` | NVIDIA, CUTLASS | CUTLASS MLA | CUDA |

### 8.0 如何进入 DSA 后端入口

**Q: 模型是怎么选择进入 DSA 后端还是普通后端的？**

答案是通过**模型注册名称映射**自动完成的。整个选择链如下：

```python
# 1. ModelRunner 装载模型时，根据 config.architectures[0] 确定模型类名
model_runner.py :: load_model()
  -> ModelRegistry.load_model(config)
    -> 查找 config.architectures[0] 对应的模型类
       (如 "Glm4MoeForCausalLM" -> Glm4MoeForCausalLM 类)

# 2. 该模型类在 __init__ 中读取 HuggingFace config 的 DSA 相关字段
model_config.py :: is_deepseek_dsa(config)
  -> 检查 config 中是否存在 DSA 标记:
     config.index_n_heads, config.index_head_dim, config.index_topk
  -> 返回 True -> self.use_dsa = True

# 3. 每层构造 DeepseekV2AttentionMLA 时传入 use_dsa
deepseek_v2.py :: DeepseekV2DecoderLayer.__init__
  self_attn = DeepseekV2AttentionMLA(config, ...)
    self.use_dsa = is_deepseek_dsa(config)  # True for GLM 5.2

# 4. Attention 层根据 use_dsa 创建 Indexer
deepseek_v2.py :: DeepseekV2AttentionMLA.__init__
    if self.use_dsa:
        self.indexer = Indexer(...)   # 创建 DSA 索引器

# 5. 后端初始化: ModelRunner 根据模型类型选择后端
model_runner.py :: init_attention_backends()
  -> 检测模型是否使用 DSA -> 选择 DSA 后端
  -> _get_attention_backend_from_str("dsa") 或自动检测
  -> 返回 DeepseekSparseAttnBackend 实例

# 6. Attention 前向传播时的路由
deepseek_v2.py :: dispatch_attn_forward_method(forward_batch)
  -> 返回 AttnForwardMethod.MLA  (MLA + DSA 路径)
  -> 进入 forward_absorb_prepare() -> forward_absorb_core()
    -> self.attn_mqa(q, k, v, forward_batch, ...)
      -> RadixAttention.forward()
        -> get_attn_backend().forward(...)  # 已绑定为 DSA 后端
          -> DeepseekSparseAttnBackend.forward_decode/forward_extend
```

**关键点**: 模型通过 HuggingFace config 中的 `index_n_heads`、`index_head_dim`、`index_topk` 等字段自动标识为 DSA 模型。`model_config.py:is_deepseek_dsa()` 检测这些字段后返回 `True`，后续所有组件（模型层、attention 层、后端选择）都基于 `self.use_dsa` 自动走 DSA 路径。**不需要任何手动配置**。

### 8.1 FlashMLa Sparse 路径（默认，NVIDIA）

```python
# dsa_backend.py, line 1855-1864
def _forward_flashmla_sparse(self, q_all, kv_cache, page_table_1,
                               sm_scale, v_head_dim):
    o = flash_mla_sparse_fwd(
        q=q_all,                 # [B, heads, 576]  (q_nope_absorbed + q_rope)
        kv_cache=kv_cache,        # [num_slots, 1, 576]
        topk_indices=page_table_1, # [B, 2048]
        sm_scale=sm_scale,        # 1/sqrt(576)
        v_head_dim=v_head_dim,   # 512
    )
    return o
```

### 8.2 TileLang 路径（NVIDIA/ROCm 通用）

```python
# dsa_backend.py, line 1878-1891
def _forward_tilelang(self, q_all, kv_cache, page_table_1,
                        sm_scale, v_head_dim):
    # TileLang 在运行时生成优化的 CUDA/ROCm 内核
    o = self.tilelang_kernel(
        q=q_all,
        kv_cache=kv_cache,
        page_table=page_table_1,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )
    return o
```

### 8.3 Prefill 和 Decode 可混合后端

SGLang 支持 prefill 和 decode 使用不同的后端：

```python
# 例如: prefill 用 FA3, decode 用 FlashMLa
prefill_backend = "fa3"
decode_backend = "flashmla_sparse"

# HybridAttnBackend 根据 forward_mode 自动路由
```

### 8.4 Decode 和 Prefill 的特殊路径差异

当 `save_kv_cache=True` 且使用 `tilelang` 后端时，decode 和 prefill 有重要差异：

**Decode 路径 (dsa_backend.py:1878-1891)**：
```python
# Decode 时:
# - q_all 通过 'q' 参数传入，q_rope=None (调用者传入完整的 fused_q)
# - backend 使用 q_all 的 zero-copy view 避免冗余 concat
# - page_table_1 = transform_index_page_table_decode(page_table, topk_indices, page_size=1)
# - 使用 dsa_decode_impl 对应的内核
```

**Prefill 路径 (dsa_backend.py:1622-1661)**：
```python
# Prefill 时:
# - q_rope 必须不为 None (调用者传入分离的 q_nope + q_rope)
# - backend 内部组装 q_nope 和 q_rope:
#     q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)      # [T, heads, 512]
#     q_rope = q_rope.view(-1, layer.tp_q_head_num, 576 - 512)         # [T, heads, 64]
# - 构建 page_table:
#     if RAGGED:
#         topk_indices = topk_indices + topk_indices_offset
#     elif PAGED:
#         page_table_1 = transform_index_page_table_prefill(...)
# - 使用 dsa_prefill_impl (可能与 dsa_decode_impl 不同!)
```

**Prefill 的特殊处理**:
1. Prefill 需要处理 ragged batch（不同请求的 prefill chunk 长度不同）
2. `topk_indices_offset` 用于将 topk 索引从 token 内部偏移转换为全局 KV cache slot
3. DSA prefill 需要将 `cache_seq_lens` 裁剪为 `dsa_cache_seqlens_int32`
4. Prefill 的 Q 维度是 `[total_new_tokens, heads, 576]`，需要计算 `cu_seqlens_q` 和 `cu_seqlens_k`

---

## 9. 与传统 Attention 对比

### 9.1 维度对比

| 维度 | MHA (GLM-4) | MLA (标准) | MLA + DSA (GLM 5.2) |
|------|------------|-----------|---------------------|
| K 形状 | `[heads, head_dim]` | `[1, kv_lora_rank + rope]` | 同左 |
| K 元素数 | 40 x 128 = 5120 | 512 + 64 = 576 | 同左 |
| V 存储 | 独立 `[heads, head_dim]` | K 的前 kv_lora_rank 部分 | 同左 |
| KV / token (BF16) | ~20 KB | ~1.13 KB | ~1.13 KB + 132B index |
| Attention 类型 | 全局 dense | 全局 dense | 稀疏 topk=2048 |
| 计算复杂度 | O(n^2 x h x d_h) | O(n^2 x (kvr + rope)) | O(n x 128 + k x (kvr + rope)) |
| 128K 上下文单层耗时 | 不可行 | ~10 ms | ~0.3 ms |
| 额外参数 | 无 | fused_qkv_a, q_b, kv_b | + Indexer (wk, wq, weights) |
| 额外存储 | 无 | 无 | Indexer KV Cache |

### 9.2 Q 路径对比

```
MHA (GLM-4):
  hidden -> q_proj -> [B, heads, head_dim]
  hidden -> k_proj -> [B, heads, head_dim]
  hidden -> v_proj -> [B, heads, head_dim]
  Q, K, V 完全独立

MLA (GLM 5.2, 无 DSA):
  hidden -> fused_a_proj -> [q_latent, kv_latent, k_rope]
  q_latent -> q_b_proj -> [q_nope, q_rope]
  kv_latent -> layernorm -> k_nope
  q_nope -> bmm(W_kc) -> q_nope_absorbed
  共享: W_kc 来自 kv_b_proj 的部分权重

MLA + DSA (GLM 5.2):
  额外: indexer.wk(hidden) -> index_key
        indexer.wq(q_latent) -> index_query
        fp8_index(q_index, K_index_cache) -> topk_indices
```

### 9.3 内存占用对比

```
假设 128K tokens, 60 层, BF16 存储:

MHA (GLM-4):
  主 KV: 128K x 60 x 10240 x 2 = 156 GB

MLA (无 DSA):
  主 KV: 128K x 60 x 576 x 2 = 8.8 GB

MLA + DSA (GLM 5.2):
  主 KV:    128K x 60 x 576 x 2 = 8.8 GB
  Index KV: 128K x 60 x 132 = 1.0 GB
  总计: ~9.8 GB

节省: (156 - 9.8) / 156 = 93.7%
```

### 9.4 计算量对比

```
假设 128K tokens, 60 层, 单 token decode:

MHA (GLM-4):
  每层: 128K x 40 x 128 x 2 = 1.31 GFLOPs
  60层: 78.6 GFLOPs

MLA + DSA (GLM 5.2):
  每层 (Indexer): 128K x 1 x 128 = 16.4 MFLOPs
  每层 (Sparse Attn): 2048 x 1 x 576 = 1.18 MFLOPs
  每层合计: 17.6 MFLOPs
  60层 (Indexer 执行 25%): 15x16.4 + 60x1.18 = 317 MFLOPs

计算节省: (78600 - 317) / 78600 = 99.6%
```

---

## 10. 源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| | | **MLA 架构** |
| `python/sglang/srt/models/deepseek_v2.py` | 1546 | `DeepseekV2AttentionMLA` 类定义 |
| `python/sglang/srt/models/deepseek_v2.py` | 1559-1606 | MLA 关键参数定义 |
| `python/sglang/srt/models/deepseek_v2.py` | 1613-1620 | `fused_qkv_a_proj_with_mqa` 融合下投影 |
| `python/sglang/srt/models/deepseek_v2.py` | 1621-1630 | `q_a_layernorm` + `q_b_proj` Q 解压 |
| `python/sglang/srt/models/deepseek_v2.py` | 1648-1689 | `kv_a_proj_with_mqa` 或 `kv_b_proj` KV 解压 |
| `python/sglang/srt/models/deepseek_v2.py` | 1701 | `kv_a_layernorm` KV 归一化 |
| `python/sglang/srt/models/deepseek_v2.py` | 1721-1730 | `attn_mqa` MLA 路径的 RadixAttention |
| `python/sglang/srt/models/deepseek_v2.py` | 1744-1756 | `attn_mha` MHA 回退路径 |
| `python/sglang/srt/models/deepseek_v2.py` | 1840-1859 | `forward()` 主入口 |
| `python/sglang/srt/models/deepseek_v2.py` | 1861-1954 | `forward_prepare()` 调度 + dispatch |
| `python/sglang/srt/models/deepseek_v2.py` | 1984-2003 | `prepare_qkv_latent()` 融合投影 |
| | | **MLA 前向实现** |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 167 | `DeepseekMLAForwardMixin` 类定义 |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 230 | `forward_absorb_prepare()` Q/K/V 产生 |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 252-258 | qkv_latent split 核心代码 |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 260-327 | q/k layernorm (含 AMD 融合路径) |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 399-403 | 非 MLA 路径 (MHA 回退) |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 407-408 | q_nope/q_pe/k_pe 拆分 |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 424-513 | Q BMM 吸收投影 (多种后端路径) |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 527-534 | RoPE 位置编码 |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 581 | `forward_absorb_core()` Attention 执行 |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 596-665 | `_skip_rope_for_dsa_tilelang_fused` 融合路径 |
| `.../deepseek_common/attention_forward_methods/forward_mla.py` | 666+ | 其他后端路径 (FA3, TRTLLM, etc.) |
| | | **DSA Backend** |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 298 | `_DSA_IMPL_T` 实现类型定义 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 141 | `DSAMetadata` 元数据结构 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 298-406 | `DeepseekSparseAttnBackend.__init__` 初始化 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1529-1606 | `forward_extend()` Prefill 路径 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1577-1590 | Prefill KV 写入 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1608-1625 | Prefill Q 重组 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1622-1653 | Prefill 稀疏页表构建 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1764-1815 | `forward_decode()` Decode 路径 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1802-1815 | Decode KV 写入 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1817-1833 | Decode Q 重组 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1839-1853 | Decode 稀疏页表构建 |
| | | **Indexer** |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 300 | `Indexer` 类定义 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1192-1277 | `forward_indexer()` TopK 计算 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1219-1225 | 构建 page block table |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1244-1253 | 读取 Indexer KV Cache |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1258-1270 | FP8 index 计算 + topk |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1279-1357 | `_store_index_k_cache()` 写入 Indexer KV |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1300-1318 | Fused store 快速路径 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1321-1342 | AITER 快速路径 (AMD) |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1345-1357 | 标准路径: act_quant + set_index_k_scale_buffer |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1372-1391 | `forward_cuda()` Indexer 主入口 |
| | | **KV Cache 存储** |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2130 | `MLATokenToKVPool` 类定义 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2157-2171 | MLA 存储参数 (kv_lora_rank, qk_rope_head_dim, kv_cache_dim) |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2184-2199 | `_create_buffers()` kv_buffer 分配 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2221-2228 | `get_key_buffer()` 读取 K |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2230-2238 | `get_value_buffer()` V = k_nope |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2243-2270 | `set_kv_buffer()` 写入 (FP8 路径) |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2272-2322 | `set_mla_kv_buffer()` 分两部分写入 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2529 | `DSATokenToKVPool` 类定义 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2592-2611 | `index_k_with_scale_buffer` 分配 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2618-2628 | DSA `move_kv_cache()` 同步移动两层缓存 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2630-2633 | `get_index_k_with_scale_buffer()` |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2635-2646 | `get_index_k_continuous()` |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2648-2659 | `get_index_k_scale_continuous()` |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2692-2702 | `set_index_k_scale_buffer()` 写入 Indexer KV |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2704-2727 | `get_cpu_copy()` Indexer KV 的 CPU offload |
| | | **模型配置检测** |
| `python/sglang/srt/configs/model_config.py` | - | `is_deepseek_dsa()` 检测 DSA 模型 |
| `python/sglang/srt/configs/model_config.py` | - | `get_dsa_index_n_heads()` 获取 index head 数 |
| `python/sglang/srt/configs/model_config.py` | - | `get_dsa_index_head_dim()` 获取 index head dim |
| `python/sglang/srt/configs/model_config.py` | - | `get_dsa_index_topk()` 获取 topk 值 |
| `python/sglang/srt/configs/model_config.py` | - | `dsa_layer_skips_topk()` 跨层共享配置 |
| | | **后端选择** |
| `python/sglang/srt/layers/attention/attention_registry.py` | - | `register_attention_backend("dsa")` 注册 |
| `python/sglang/srt/model_executor/model_runner.py` | ~871 | `init_attention_backends()` 后端初始化 |
| | | **内存分配** |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | - | `_init_pools()` Pool 创建决策 |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | - | `init_memory_pool()` 内存池初始化 |
| `python/sglang/srt/mem_cache/common.py` | - | `alloc_for_extend()` / `alloc_for_decode()` |
| `python/sglang/srt/mem_cache/radix_cache.py` | - | `RadixCache` 前缀缓存 |

---

## 附录: 关键设计决策

1. **V 就是 K_nope**: 不需要单独存储 V，`get_value_buffer()` 返回 `kv_buffer[..., :kv_lora_rank]`，节省一半存储。

2. **Q 吸收投影 (W_kc) 来自 kv_b_proj**: 不需要额外参数，Q 的投影矩阵共享 KV 的解压矩阵，参数量不增加。

3. **Indexer 只有 1 个 head**: 稀疏索引用 128 维的 1 个 head，而不是 128 个 head，大幅减少索引计算量。

4. **Indexer 跨层共享**: 大多数层不执行 Indexer，复用上层的 topk，减少 ~60% 的索引计算。

5. **FP8 量化索引**: Indexer 的 K cache 强制使用 FP8 存储，128 维的 key 只需 128+4=132 字节/token。

6. **Prefill 和 Decode 可用不同后端**: `dsa_prefill_impl != dsa_decode_impl` 可以独立优化两个阶段。

7. **DSA CP (Context Parallel)**: 支持 prefill 阶段的序列并行，将长序列分片到多个 GPU。

8. **兼容 MHA 回退**: 通过 `self.use_mha` 和 `self.attn_mha`，在需要时（如某些层 tp_q_head_num == tp_k_head_num）可回退到标准 MHA。
