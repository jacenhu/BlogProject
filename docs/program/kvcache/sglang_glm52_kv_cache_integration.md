# sglang GLM 5.2 KV Cache 对接原理

> 首次编写：2026-07-14 | 最后更新：2026-07-14

> **KV Cache 系列文章**
> - [KV Cache 基础知识](sglang_kvcache_knowhow.md)
> - [GLM 5.2 KV Cache](sglang_kvcache_glm5.2.md)
> - **GLM 5.2 KV Cache 对接原理** ← 当前文章
> - [GLM 5.2 Attention 机制](sglang_glm52_attention_mechanism.md)
> - [DeepSeek V4 KV Cache](sglang_kvcache_deeepseekv4.md)
> - [KV Cache 传输](sglang_glm52_kv_cache_transfer.md)

## 目录

1. [总览](#1-总览)
2. [模型架构：MLA + DSA](#2-模型架构mla--dsa)
3. [K 和 V 是如何产生的](#3-k-和-v-是如何产生的)
4. [底层 Tensor 存储](#4-底层-tensor-存储)
5. [KV Cache 写入流程](#5-kv-cache-写入流程)
6. [KV Cache 读取流程](#6-kv-cache-读取流程)
7. [DSA 稀疏注意力完整链路](#7-dsa-稀疏注意力完整链路)
8. [分配与回收机制](#8-分配与回收机制)
9. [完整生命周期](#9-完整生命周期)
10. [关键源码索引](#10-关键源码索引)

---

## 1. 总览

GLM 5.2 在 SGLang 中的模型实现文件是 `deepseek_v2.py`，其 attention 机制是 **MLA (Multi-head Latent Attention) + DSA (DeepSeek Sparse Attention)** 的组合。KV Cache 也因此分为**两层存储**：

```
┌──────────────────────────────────────────────────────────────────┐
│                      GLM 5.2 KV Cache                             │
├──────────────────────────────────────────────────────────────────┤
│  第一层: MLATokenToKVPool  (主 KV Cache)                          │
│     kv_buffer[layer]: [num_slots + page_size, 1, 576]             │
│     存储压缩后的 KV latent (512) + RoPE 位置编码 (64)               │
│                                                                   │
│  第二层: DSATokenToKVPool  (DSA 索引 KV Cache)                     │
│     index_k_with_scale_buffer[layer]:                             │
│       [num_pages, 64 * (128 + 4)]  uint8                          │
│     存储 DSA 稀疏注意力所需的 index key (FP8 + scale)               │
│                                                                   │
│  页表: ReqToTokenPool.req_to_token                                │
│     [max_running_requests, max_context_len]                       │
│     逻辑位置 → 物理 slot 的映射表                                   │
└──────────────────────────────────────────────────────────────────┘
```

与标准 MHA（如 GLM-4）的核心区别：

| 特性 | GLM-4 (MHA) | GLM 5.2 (MLA + DSA) |
|------|-------------|---------------------|
| K/V 维度 | `[heads, head_dim]` × 2 | `[1, kv_lora_rank + rope]` |
| K/V 元素数（典型值） | ~10240 / token | 576 / token（压缩约 18×） |
| V 存储 | 独立 tensor | K 的前 `kv_lora_rank` 部分即 V |
| Attention 类型 | 全局 dense attention | DSA 稀疏 attention（只算 topk 位置） |
| 第二层 cache | 无 | Indexer KV Cache (FP8, 128 维) |
| 后端实现 | `FlashAttentionBackend` / `FlashInferAttnBackend` | `DeepseekSparseAttnBackend` |
| 页表粒度 | page_size 可变 | page_size=1（token 级索引）+ page_size=64（indexer） |

---

## 2. 模型架构：MLA + DSA

### 2.1 源码位置

```
python/sglang/srt/models/deepseek_v2.py          — 模型主体 & Attention 层
  └── DeepseekV2DecoderLayer                      — 每层 Transformer
        └── DeepseekV2AttentionMLA               — MLA Attention 模块 (line 1546)
              ├── RadixAttention (attn_mqa)       — MLA 路径，kv_heads=1
              ├── RadixAttention (attn_mha)       — MHA 回退路径
              └── Indexer                         — DSA 索引器 (line 1653)

python/sglang/srt/layers/attention/dsa_backend.py — DSA 后端（KV 读写 + attention 计算）
python/sglang/srt/layers/attention/dsa/dsa_indexer.py — Indexer（稀疏索引生成 + index KV 管理）
python/sglang/srt/mem_cache/memory_pool.py        — 物理存储分配
  ├── MLATokenToKVPool  (line 2130)               — 主 KV Cache 存储
  └── DSATokenToKVPool (line 2529)                — DSA 索引 KV Cache 存储
```

### 2.2 Attention 模块的关键参数

```python
# deepseek_v2.py DeepseekV2AttentionMLA.__init__
self.kv_lora_rank   = 512     # KV 压缩后的低秩维度
self.qk_rope_head_dim = 64    # RoPE 位置编码维度
self.qk_nope_head_dim = 128   # Q 的非位置部分维度
self.v_head_dim      = 512    # V 的维度 = kv_lora_rank（关键！V 就是 K_nope）
self.q_lora_rank     = 1536   # Q 的压缩低秩维度
self.num_heads       = 128    # 总注意力头数
self.scaling         = head_dim^(-0.5)
self.use_dsa         = True   # 启用 DSA 稀疏注意力
```

### 2.3 关键权重矩阵

```
fused_qkv_a_proj_with_mqa:
  输入: hidden_size (7168)
  输出: q_lora_rank + kv_lora_rank + qk_rope_head_dim (1536 + 512 + 64 = 2112)

q_b_proj:
  输入: q_lora_rank (1536)
  输出: num_heads × qk_head_dim (128 × 192 = 24576, TP 切分后)

kv_b_proj:
  输入: kv_lora_rank (512)
  输出: num_heads × (qk_nope_head_dim + v_head_dim) (128 × (128 + 512) = 81920, TP 切分后)
```

---

## 3. K 和 V 是如何产生的

### 3.1 MLA 低秩分解

GLM 5.2 不使用传统的 Q、K、V 投影，而是用低秩压缩方案。整个过程的入口在 `forward_absorb_prepare`：

```
hidden_states: [B, hidden_size=7168]
      │
      ▼
fused_qkv_a_proj_with_mqa(hidden_states)
      │
      ▼
qkv_latent: [B, 2112]
  ├── q_latent:   [B, 1536]         ← Q 的压缩表示
  ├── kv_latent:  [B, 512]          ← KV 的压缩表示（关键！）
  └── k_pe_raw:   [B, 64]           ← 未旋转的 RoPE 位置编码
```

### 3.2 Q 的处理路径

```
q_latent: [B, 1536]
  │
  ├── q_a_layernorm(RMSNorm)
  │
  ├── q_b_proj(q_latent): 1536 → 24576
  │     reshape → [B, num_local_heads, 192]
  │
  └── split:
        q_nope: [B, heads, 128]   ← 与内容相关的 Q 部分
        q_rope: [B, heads, 64]    ← 与位置相关的 Q 部分
```

### 3.3 K/V 的处理路径（核心压缩）

```
kv_latent: [B, 512]
  │
  ├── kv_a_layernorm(RMSNorm)
  │     unsqueeze(1) → [B, 1, 512]
  │
  └── k_nope: [B, 1, 512]       ← 这就是 K 的内容部分！
                                   同时也是 V！

k_pe_raw: [B, 64]
  │
  ├── RoPE(positions, k_pe_raw)
  │     unsqueeze(1) → [B, 1, 64]
  │
  └── k_rope: [B, 1, 64]        ← K 的位置部分
```

**核心设计**：

```
完整 K = concat(k_nope, k_rope) = [B, 1, 512 + 64] = [B, 1, 576]
V      = k_nope = [B, 1, 512]

即 Value 就是 K 的非位置部分。get_value_buffer() 直接返回 kv_buffer[..., :kv_lora_rank]。
```

### 3.4 Q 的 BMM 吸收变换

为了让 Q 能与压缩后的 K 做 attention，需要将 Q 投影到 `kv_lora_rank` 维度：

```
q_nope: [B, heads, 128]
  │
  ├── transpose(0,1) → [heads, B, 128]
  │
  ├── bmm(W_kc): [heads, B, kv_lora_rank=512]
  │     W_kc 是 kv_b_proj 中与 Q 相乘的部分
  │     来源: kv_b_proj.weight 的"前半部分"
  │
  └── transpose(0,1) → [B, heads, 512]

最终 Q:
  q_full = concat(q_nope_bmm, q_rope) = [B, heads, 512 + 64] = [B, heads, 576]
```

**Attention Score 计算**（在 MLA 空间中）：

```
score = (q_nope_absorbed × k_nope^T) / scale  +  (q_rope × k_rope^T) / scale
      = Q_full · K_full^T / sqrt(192)
```

---

## 4. 底层 Tensor 存储

### 4.1 主 KV Cache：MLATokenToKVPool.kv_buffer

```python
# 源码: memory_pool.py:2192
kv_buffer[layer]: torch.Tensor

形状:
    [num_slots + page_size, 1, kv_lora_rank + qk_rope_head_dim]
    [size + 64,             1, 512 + 64 = 576]

dtype:
    取决于 kv_cache_dtype 配置:
    - "auto" / None → BF16/FP16（与模型 dtype 一致）
    - "fp8_e4m3"   → torch.float8_e4m3fn

设备: GPU (cuda:0)
```

**物理布局**（在最后一维 `[576]` 上）：

```
slot_id
  │
  ▼
  ┌─────────────────────────────────┬──────────────────┐
  │  k_nope (latent, 同时作为 V)      │  k_rope (位置编码) │
  │  kv_lora_rank = 512              │  64                │
  │  dtype: BF16 或 FP8              │  dtype: BF16      │
  └─────────────────────────────────┴──────────────────┘
  ◄──────────────── 576 elements ──────────────────────►

一个 token 的内存占用:
  BF16: 576 × 2 bytes = 1,152 bytes ≈ 1.13 KB
  FP8:  576 × 1 byte  =   576 bytes ≈ 0.56 KB
```

**对比 MHA（以 GLM-4, 40 heads, head_dim=128 为例）**：

```
MHA 每 token KV = (num_kv_heads × head_dim × 2) × dtype_size
                 = (40 × 128 × 2) × 2 = 20,480 bytes ≈ 20 KB
MLA 每 token KV = (576) × 2 = 1,152 bytes ≈ 1.13 KB

压缩比: 20,480 / 1,152 ≈ 17.8×
```

### 4.2 DSA 索引 KV Cache：DSATokenToKVPool.index_k_with_scale_buffer

```python
# 源码: memory_pool.py:2592
index_k_with_scale_buffer[layer]: torch.Tensor

形状:
    [num_pages, page_size * (index_head_dim + index_head_dim / quant_block_size × 4)]
    [num_pages, 64 * (128 + 4)]
    [num_pages, 8448]

dtype: torch.uint8 (字节级存储)
page_size: 64 (CUDA) 或 16 (ROCm 新路径) 或 1 (ROCm 旧路径)
```

**每页内部布局（64 tokens, CUDA）**：

```
Page i: 8448 bytes = 64 × (128 + 4)

  ┌──────────────────────────────────┬──────────────────────┐
  │  Token 0..63 的 FP8 index key    │  Token 0..63 的 scale │
  │  64 × 128 bytes = 8192 bytes     │  64 × 4 bytes = 256  │
  ├──────────────────────────────────┼──────────────────────┤
  │  0     ...     8191              │  8192   ...   8447   │
  │  uint8 (FP8 数据)                │  uint8 (FP32 scale)  │
  └──────────────────────────────────┴──────────────────────┘

每个 token 的 index KV:
  FP8 key:   128 bytes (128 个 FP8 元素)
  Scale:     4 bytes (1 个 FP32，因为 block_size=128, 128/128=1 个 scale)
  总计:      132 bytes/token
```

**为什么需要 Index KV Cache？**

DSA 的 Indexer 需要一个小的 key（128 维）来快速计算近似 attention score，选出 topk 个位置再做精确 attention。这个 128 维的 index key 与主 KV cache 中的 576 维完整 key 是分开存储的，因为：

1. 精度不同：index key 强制 FP8 量化，主 KV 是可选的
2. 用途不同：index key 只用于粗筛（topk 选择），不参与最终 attention 计算
3. 存储格式不同：page 对齐 vs slot 对齐

### 4.3 页表：ReqToTokenPool.req_to_token

```python
req_to_token: torch.Tensor
形状:    [max_running_requests + 1, max_context_len]
含义:    req_to_token[req_pool_idx, position] = kv_cache_slot_index
         slot 0 保留为 CUDA graph 填充 token 的哑槽位
```

这是**所有 backend 共享的二级索引表**，是 prefix cache 的基础设施。两个请求共享同一段前缀时，它们对应的 `req_to_token` 条目指向相同的物理 slot。

### 4.4 KV Cache 存储层级总结

```
                    ┌─────────────────────┐
                    │   ReqToTokenPool    │  ← 逻辑 → 物理映射
                    │   req_to_token      │
                    │   [max_req, max_len]│
                    └──────┬──────────────┘
                           │ slot_index
              ┌────────────┼────────────┐
              ▼            │            ▼
  ┌─────────────────────┐  │  ┌──────────────────────────┐
  │ MLATokenToKVPool    │  │  │ DSATokenToKVPool         │
  │ (主 KV Cache)       │  │  │ (索引 KV Cache)           │
  │                     │  │  │                          │
  │ kv_buffer[layer]:   │  │  │ index_k_with_scale[l]:   │
  │ [slots+64, 1, 576]  │  │  │ [pages, 64×132] uint8   │
  │                     │  │  │                          │
  │ slot[s]:            │  │  │ page[p][t*132:(t+1)*132]│
  │ [0:512] = k_nope/V  │  │  │  [0:128]    = FP8 key   │
  │ [512:576]= k_rope   │  │  │  [128:132]  = FP32 scale│
  └─────────────────────┘  │  └──────────────────────────┘
                           │
                  slot_index 相同！
               （index key 与 KV 使用相同位置）
```

---

## 5. KV Cache 写入流程

### 5.1 整体时序

以一次 decode 为例：

```
Scheduler
  │
  ├── alloc_for_decode(forward_batch)
  │     ├── 从分配器中申请新 slot
  │     ├── req_to_token[req_id, seq_len] = new_slot_index
  │     └── forward_batch.out_cache_loc = [slot_0, slot_1, ...]
  │
  ├── Model Forward
  │     └── DeepseekV2DecoderLayer.forward()
  │           └── self_attn.forward()
  │                 ├── forward_absorb_prepare()
  │                 │     ├── fused_qkv_a_proj → qkv_latent (split)
  │                 │     ├── q_latent → layernorm → q_b_proj → q_nope, q_rope
  │                 │     ├── kv_latent → layernorm → k_nope [B,1,512]
  │                 │     ├── k_pe_raw → RoPE → k_rope [B,1,64]
  │                 │     ├── q_nope → bmm(W_kc) → [B,heads,512]
  │                 │     └── Indexer.forward() → 写入 index KV cache + 产生 topk_indices
  │                 │
  │                 └── forward_absorb_core()
  │                       └── attn_mqa.forward(q_nope, None, None, ...)
  │                             └── DSA backend.forward_decode()
  │                                   ├── set_mla_kv_buffer(k_nope, k_rope)
  │                                   │     └── 写入主 KV cache
  │                                   └── sparse attention 计算
  │
  └── 请求结束
        ├── cache_finished_req() → 插入 RadixCache
        └── release_kv_cache() → 释放 slot
```

### 5.2 写入主 KV Cache

```python
# dsa_backend.py forward_decode, line 1802-1815
if save_kv_cache:
    cache_loc = forward_batch.out_cache_loc
    self.token_to_kv_pool.set_mla_kv_buffer(
        layer,
        cache_loc,     # [B] int64 tensor, 每个 token 的目标 slot
        k,             # k_nope: [B, 1, 512], KV latent
        k_rope,        # k_rope: [B, 1, 64],  位置编码
    )
```

底层执行（`memory_pool.py:2272-2322`）：

```python
def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
    # 将 k_nope 和 k_rope 分别写入 kv_buffer 的不同区域
    # kv_buffer[loc, :, 0:512] = k_nope
    # kv_buffer[loc, :, 512:576] = k_rope

    # FP8 路径: 在线量化 BF16 → FP8
    if self.dsa_kv_cache_store_fp8:
        cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
            cache_k_nope, cache_k_rope
        )
        set_mla_kv_buffer_triton(
            kv_buffer[layer_id], loc, cache_k_nope_fp8, cache_k_rope_fp8
        )
    else:
        set_mla_kv_buffer_triton(
            kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
        )
```

`set_mla_kv_buffer_triton` 是一个 Triton kernel，本质上做：

```
for i in range(num_tokens):
    slot = loc[i]
    kv_buffer[slot, 0, 0:kv_lora_rank] = k_nope[i]
    kv_buffer[slot, 0, kv_lora_rank:] = k_rope[i]
```

### 5.3 写入 Indexer KV Cache

```python
# dsa_indexer.py _store_index_k_cache, line 1279
def _store_index_k_cache(self, forward_batch, layer_id, key, ...):
    # key: [B, 1, 128],  Indexer 的 wk(hidden_states) 输出

    # 快速路径: fused kernel 一站式量化+写入
    if can_use_dsa_fused_store(key.dtype, ...):
        buf = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
        fused_store_index_k_cache(
            key, buf, out_cache_loc, page_size=64
        )
        return

    # 标准路径: 先量化再写入
    k_fp8, k_scale = act_quant(key, block_size=128, scale_fmt="ue8m0")
    token_to_kv_pool.set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=out_cache_loc,
        index_k=k_fp8,         # [B, 128] FP8
        index_k_scale=k_scale, # [B, 1]   FP32 scale
    )
```

`set_index_k_scale_buffer` 的实现（`memory_pool.py:2692-2702`）：

```python
def set_index_k_scale_buffer(self, layer_id, loc, index_k, index_k_scale):
    buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
    # buf: [num_pages, 8448] uint8
    # 将 index_k 和 index_k_scale 打包写入对应 page 的对应位置
    index_buf_accessor.SetKAndS.execute(
        pool=self, buf=buf, loc=loc,
        index_k=index_k, index_k_scale=index_k_scale
    )
```

---

## 6. KV Cache 读取流程

### 6.1 获取 KV Buffer

```python
# dsa_backend.py forward_decode, line 1818
kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
# 返回 kv_buffer[layer_id] 的引用: [num_slots, 1, kv_lora_rank + qk_rope_head_dim]
```

```python
# memory_pool.py:2221
def get_key_buffer(self, layer_id: int):
    # 对于 MLA，返回完整的 kv_buffer（包含 k_nope + k_rope）
    # 调用方自行拆分 k_nope 和 k_rope
    return self.kv_buffer[layer_id - self.start_layer]

def get_value_buffer(self, layer_id: int):
    # 对于 MLA，value = k_nope = kv_buffer[..., :kv_lora_rank]
    return self.kv_buffer[layer_id - self.start_layer][..., :self.kv_lora_rank]
```

### 6.2 构建 DSA 稀疏页表

这是 DSA 与普通 MLA 最大的不同。DSA 只对 topk 个 token 做精确 attention：

```python
# dsa_backend.py forward_decode, line 1849-1853
page_table_1 = transform_index_page_table_decode(
    page_table=metadata.page_table_1,   # 原始全序列页表 [B, max_seq_len]
    topk_indices=topk_indices,          # Indexer 选出的 topk 位置 [B, topk]
    page_size=1,
)
# page_table_1: [B, topk]
# page_table_1[b][j] = 第 b 个请求的第 j 个 topk token 的 KV cache slot index
```

### 6.3 内核选择与调用

DSA backend 支持多种底层实现，通过 `dsa_decode_impl` 选择：

| 实现 | 适用场景 | 内核 |
|------|---------|------|
| `flashmla_sparse` | NVIDIA, MLA 稀疏路径 | flash_mla_sparse_fwd |
| `flashmla_kv` | NVIDIA, MLA 全量路径 | flash_mla_with_kvcache |
| `tilelang` | NVIDIA/ROCm | tilelang 生成的内核 |
| `fa3` | NVIDIA, FlashAttention-3 | flash_attn_varlen_func |
| `trtllm` | NVIDIA SM100+, Blackwell | TensorRT-LLM MLA |
| `aiter` | AMD ROCm | AITER mla_decode_fwd |

以 `flashmla_sparse` 为例（最常用路径）：

```python
# dsa_backend.py, line 1855-1864
def _forward_flashmla_sparse(self, q_all, kv_cache, page_table_1,
                               sm_scale, v_head_dim):
    # q_all:     [B, heads, 576]  — q_nope_bmm + q_rope 的拼接
    # kv_cache:  [num_slots, 1, 576] — 整个 KV buffer
    # page_table_1: [B, topk] — 只包含 topk 个 slot index

    # kernel 内部：
    # for each batch b:
    #   q = q_all[b]  # [heads, 576]
    #   for j in range(topk):
    #     slot = page_table_1[b, j]
    #     k_nope = kv_cache[slot, 0, 0:v_head_dim]      # [512]
    #     k_rope = kv_cache[slot, 0, v_head_dim:]        # [64]
    #     k = concat(k_nope, k_rope)                     # [576]
    #     score[b][j] = q · k / sm_scale
    #   output[b] = softmax(scores) × k_nope[topk]       # V = k_nope

    o = flash_mla_sparse_fwd(
        q=q_all,
        kv_cache=kv_cache,
        topk_indices=page_table_1,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )
    return o
```

### 6.4 Indexer KV Cache 的读取（生成 topk 索引）

```python
# dsa_indexer.py forward_indexer, line 1192
def forward_indexer(self, q_fp8, weights, forward_batch, topk, layer_id):
    # q_fp8:   indexer 的 Q（从 q_latent 投影而来）[B, 1, 128] FP8
    # weights: indexer 的权重 [B, 128]
    # topk:    通常 2048

    for i in range(batch_size):
        seq_len = forward_batch.seq_lens[i]

        # 1. 从 Indexer KV Cache 读取
        block_tables = req_to_token[...] // page_size
        k_fp8 = token_to_kv_pool.get_index_k_continuous(
            layer_id, seq_len, block_tables[i]
        )  # [seq_len, 128] FP8
        k_scale = token_to_kv_pool.get_index_k_scale_continuous(
            layer_id, seq_len, block_tables[i]
        )  # [seq_len, 1] FP32

        # 2. 计算近似 attention score: Q_index · K_index^T
        index_score = fp8_index(q_fp8, weights, k_fp8, k_scale)
        # [1, seq_len]

        # 3. 取 topk 个位置
        topk_indices = index_score.topk(min(topk, seq_len), dim=-1)[1]
        # [1, topk]

    return topk_indices  # [B, topk]
```

---

## 7. DSA 稀疏注意力完整链路

### 7.1 核心思想

```
传统 Attention:
  Q_full × K_full^T → [B, seq_len] → softmax → × V
  (序列中所有 token 都参与计算，O(seq_len²))

DSA Attention (两阶段):
  Stage 1 (粗筛): Q_index × K_index^T → [B, topk]
      用 128 维的小 key 快速计算近似 attention score
      选出 topk=2048 个最重要的 token

  Stage 2 (精算): 只在这 2048 个位置做完整的 576 维精确 attention
      Q_full × K_full[topk]^T → [B, topk] → softmax → × V[topk]

  复杂度: O(seq_len × 128 + topk × 576) vs O(seq_len × 576)
  当 seq_len=128K 时: DSA 节省约 2048/128K ≈ 98.4% 的计算量
```

### 7.2 Indexer 的权重

每个 decoder layer 都有一个独立的 Indexer，包含：

```python
# dsa_indexer.py Indexer
self.wk: ReplicatedLinear
  输入: hidden_size (7168)
  输出: 1 × index_head_dim (128)
  # 只产生 1 个 index head 的 key

self.wq: ReplicatedLinear
  输入: q_lora_rank (1536)
  输出: 1 × index_head_dim (128)
  # 只产生 1 个 index head 的 query

self.weights: Parameter
  形状: [1, index_head_dim] = [1, 128]
  # 可学习的通道加权系数
```

### 7.3 Indexer 的 TopK 跨层共享

为了减少 Indexer 的执行开销，DSA 支持 **topk 跨层共享**：

```python
# deepseek_v2.py, line 1671-1679
# skip_topk: 当前层跳过 Indexer 计算，复用上一层的 topk_indices
# next_skip_topk: 下一层将跳过 Indexer

if is_nextn:
    self.skip_topk = True        # MTP 层始终共享
    self.next_skip_topk = True
else:
    self.skip_topk = dsa_layer_skips_topk(config, layer_id)
    self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)
```

通常每 2-4 层才执行一次完整的 Indexer，中间层复用结果。

### 7.4 DSA 与 MHA 的混合策略

GLM 5.2 的某些层（或某些场景下）可能回退到标准 MHA：

```python
# dsa_backend.py _set_impl
# 当 tp_q_head_num == tp_k_head_num 且配置启用时
if self.use_mha:
    # 走标准 MHA 路径（dense attention, 不使用稀疏）
    # kv_heads = q_heads > 1
```

---

## 8. 分配与回收机制

### 8.1 KV Cache 池大小计算

```python
# model_runner_kv_cache_mixin.py
def init_memory_pool(self):
    # 1. 获取模型加载后剩余 GPU 内存
    available_gpu_memory = get_available_gpu_memory()

    # 2. 减除非 KV 开销
    usable_memory = available_gpu_memory - non_kv_overhead

    # 3. 根据模型配置计算池大小
    pool_sizes = MemoryPoolConfigurator.calculate_pool_sizes(
        model_config=model_config,
        usable_memory=usable_memory,
        page_size=64,
    )

    # 4. 创建 DSATokenToKVPool
    token_to_kv_pool = DSATokenToKVPool(
        size=max_total_num_tokens,
        page_size=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=num_layers,
        index_head_dim=128,
        ...
    )
```

### 8.2 Slot 分配

```
alloc_for_decode(forward_batch):
  ├── 检查 free_pages 是否足够
  │     ├── 不足 → evict_from_tree_cache(need_num)
  │     │           └── 从 RadixCache 中驱逐最少使用的叶子节点
  │     └── 足够 → 继续
  │
  ├── 从 PagedTokenToKVPoolAllocator 分配新 slot
  │     allocated_indices = allocator.alloc(num_tokens)
  │
  ├── 写入 ReqToTokenPool
  │     req_to_token[req_pool_indices, seq_lens] = allocated_indices
  │
  └── 设置 forward_batch.out_cache_loc = allocated_indices
```

### 8.3 Slot 回收

```
请求完成时:
  release_kv_cache(req_pool_idx):
    ├── 从 req_to_token 中读取该请求的所有 slot indices
    ├── 将 slot indices 返还给分配器 (free)
    ├── 将 KV cache 内容插入 RadixCache（供后续请求前缀匹配）
    └── 释放 req_pool_idx
```

### 8.4 前缀缓存 (RadixCache)

```
RadixCache 是一个 Radix Tree，节点存储连续的 token 序列及其 KV cache slot:

match_prefix(token_ids):
  ├── 从根节点开始遍历
  ├── 找到最长匹配前缀
  └── 返回该前缀对应的 KV slot indices

insert(token_ids, kv_indices):
  ├── 在树中插入新的 token 序列
  ├── 关联对应的 KV cache slot indices
  └── 如果节点被驱逐，释放对应的 slot
```

---

## 9. 完整生命周期

```
┌──────────────────────────────────────────────────────────────────────┐
│                      请求生命周期（Decode 步骤）                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. RadixCache.match_prefix(token_ids)                               │
│     └── 检查是否有已缓存的前缀，返回已有 slot indices                    │
│                                                                      │
│  2. alloc_for_decode(forward_batch)                                  │
│     ├── 如有必要: evict_from_tree_cache()                             │
│     ├── allocator.alloc(num_tokens) → new_slots                      │
│     └── req_to_token[req_id, pos] = new_slots                        │
│                                                                      │
│  3. Model Forward ─────────── 每层循环 ───────────┐                   │
│     │                                              │                 │
│     │  Layer.forward(hidden_states, ...)           │                 │
│     │  ├── input_layernorm                         │                 │
│     │  │                                           │                 │
│     │  ├── self_attn.forward()                     │                 │
│     │  │   ├── forward_absorb_prepare()            │                 │
│     │  │   │   ├── fused_qkv_a → qkv_latent        │                 │
│     │  │   │   │   split(q_latent, kv_latent, k_pe)│                 │
│     │  │   │   ├── q = q_b_proj(layernorm(q_latent))│               │
│     │  │   │   ├── k_nope = layernorm(kv_latent)   │                 │
│     │  │   │   ├── k_pe = RoPE(k_pe_raw)            │                 │
│     │  │   │   ├── q_nope = bmm(q_nope, W_kc)      │                 │
│     │  │   │   │                                   │                 │
│     │  │   │   └── Indexer.forward()               │                 │
│     │  │   │       ├── index_key = wk(hidden)      │                 │
│     │  │   │       ├── 量化 + 写入 Indexer KV Cache │                │
│     │  │   │       └── 读取 Indexer K Cache        │                 │
│     │  │   │           fp8_index(q_index, k_index) │                 │
│     │  │   │           → topk_indices [B, 2048]    │                 │
│     │  │   │                                       │                 │
│     │  │   └── forward_absorb_core()               │                 │
│     │  │       └── attn_mqa.forward()              │                 │
│     │  │           └── DSA backend.forward_decode()│                 │
│     │  │               ├── set_mla_kv_buffer(      │                 │
│     │  │               │     kv_buffer,            │                 │
│     │  │               │     out_cache_loc,        │                 │
│     │  │               │     k_nope, k_rope)       │  ← 写入主 KV    │
│     │  │               │                           │                 │
│     │  │               ├── get_key_buffer(layer)   │  ← 读取主 KV    │
│     │  │               ├── transform_page_table(   │                 │
│     │  │               │     page_table,           │                 │
│     │  │               │     topk_indices)         │  ← 稀疏化页表   │
│     │  │               │                           │                 │
│     │  │               └── flash_mla_sparse_fwd(   │                 │
│     │  │                     q, kv_cache,          │                 │
│     │  │                     topk_page_table)      │  ← 稀疏注意力   │
│     │  │                                           │                 │
│     │  ├── post_attention_layernorm                │                 │
│     │  └── mlp(hidden_states)  (MoE)               │                 │
│     │                                              │                 │
│     └──────────────────────────────────────────────┘                 │
│                                                                      │
│  4. 请求结束                                                         │
│     ├── cache_finished_req() → 插入 RadixCache                       │
│     └── release_kv_cache(req_pool_idx)                               │
│         ├── 释放 req_to_token 条目                                    │
│         └── 释放 KV slot indices → allocator.free()                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `python/sglang/srt/models/deepseek_v2.py` | 1546 | `DeepseekV2AttentionMLA` 类定义 |
| `python/sglang/srt/models/deepseek_v2.py` | 1614 | `fused_qkv_a_proj_with_mqa` 投影（产生 KV latent） |
| `python/sglang/srt/models/deepseek_v2.py` | 1681 | `kv_b_proj`（吸收 Q_nope 的权重来源） |
| `python/sglang/srt/models/deepseek_v2.py` | 1721 | `RadixAttention(attn_mqa)` MLA 路径的 attention 模块 |
| `python/sglang/srt/models/deepseek_v2.py` | 1744 | `RadixAttention(attn_mha)` MHA 回退路径 |
| `python/sglang/srt/models/deepseek_v2.py` | 1653 | `Indexer` 构造（DSA 索引器） |
| `python/sglang/srt/models/deepseek_v2.py` | 1678 | `skip_topk` 跨层共享逻辑 |
| `python/sglang/srt/models/deepseek_v2.py` | 1840 | `DeepseekV2AttentionMLA.forward()` |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | 230 | `forward_absorb_prepare()` — MLA 前向准备（产生 K/V） |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | 581 | `forward_absorb_core()` — MLA 前向核心（调用 backend） |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 298 | `_DSA_IMPL_T` DSA 实现类型 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1529 | `forward_extend()` — DSA prefill 路径 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1764 | `forward_decode()` — DSA decode 路径 |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1802 | KV 写入: `set_mla_kv_buffer()` |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1818 | KV 读取: `get_key_buffer()` |
| `python/sglang/srt/layers/attention/dsa_backend.py` | 1849 | 稀疏页表: `transform_index_page_table_decode()` |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 300 | `Indexer` 类定义 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1192 | `forward_indexer()` — TopK 选择 |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1279 | `_store_index_k_cache()` — 写入 Indexer KV Cache |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` | 1372 | `forward_cuda()` — Indexer 主入口 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2130 | `MLATokenToKVPool` — 主 KV Cache 存储 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2192 | `kv_buffer` 创建 — Tensor 分配 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2221 | `get_key_buffer()` — 读取完整 KV |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2230 | `get_value_buffer()` — V = k_nope |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2272 | `set_mla_kv_buffer()` — 分两部分写入 KV |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2529 | `DSATokenToKVPool` — DSA 索引 KV Cache 存储 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2592 | `index_k_with_scale_buffer` 创建 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2630 | `get_index_k_with_scale_buffer()` — 读取索引 KV |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2692 | `set_index_k_scale_buffer()` — 写入索引 KV |
| `python/sglang/srt/mem_cache/memory_pool.py` | 2704 | `get_cpu_copy()` — 索引 KV 的 CPU offload |
| `python/sglang/srt/mem_cache/common.py` | - | `alloc_for_extend` / `alloc_for_decode` |
| `python/sglang/srt/mem_cache/radix_cache.py` | - | `RadixCache` — 前缀缓存 (Radix Tree) |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | - | Pool 初始化和大小计算 |

---

## 附录：关键设计要点

1. **V 就是 K_nope**：MLA 的核心设计，`get_value_buffer()` 直接返回 `kv_buffer[..., :kv_lora_rank]`，不需要单独存储 V。

2. **K 分两部分存储**：`k_nope`（内容，512 维）和 `k_rope`（位置，64 维）分别通过 `set_mla_kv_buffer()` 的两个参数写入。

3. **双层 KV Cache**：主 KV cache（576 维）用于精确 attention；Indexer KV cache（128 维 FP8）用于快速粗筛。

4. **Page 对齐的索引存储**：主 KV 是 slot 粒度（page_size=1）；Indexer KV 是 page 粒度（page_size=64），以便批量化读取。

5. **topk 跨层共享**：并非每层都执行 Indexer，多数层复用上层的 topk 结果，减少索引计算开销。

6. **FP8 可选**：主 KV 可选 BF16/FP8 存储；Indexer KV 强制 FP8 存储。

7. **Prefix Cache 透明**：所有 KV 操作都通过 `req_to_token` 间接索引，前缀匹配对模型 forward 完全透明。