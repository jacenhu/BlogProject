# 深度解析 SGLang KV Cache：RadixTree 前缀共享、全生命周期流转、跨设备传输与 HiCache 多级缓存工程优化

## 前言
在大模型推理体系中，KV Cache 是制约 **TTFT、吞吐、显存上限、长上下文能力** 的核心瓶颈。传统 PagedAttention 仅实现「块级显存复用」，无法解决多请求公共前缀重复计算问题。

SGLang 相较于 vLLM 最大架构革新是 **RadixTree（基数树）全局前缀 KV 共享机制**，结合三层内存池（`ReqToTokenPool` → `TokenToKVPoolAllocator` → `KVCache`）、精细化请求级元数据管理、**跨设备 KV 传输链路**（GPU↔CPU offload、Disaggregation PD、HiCache 多后端存储），实现了「计算复用+显存复用+跨设备数据流转+硬件层级扩容」四重优化。

> **源码版本说明**：本文基于 SGLang 主分支源码分析。GLM-5.2 模型尚未合入 SGLang 主线，文中涉及 GLM-5.2 的适配分析属于架构推演，与 SGLang 通用机制严格区分标注。但 GLM-5.2 所依赖的 **DSA 稀疏注意力、MLA 低秩压缩 KV、DeepSeek V4 HiSparse 等基础设施已在 SGLang 代码库中存在**，可据此推演 GLM-5.2 的适配路径。

本文将从**底层数据结构 → RadixCache 前缀共享内核 → KV 完整生命周期 → 跨设备 KV Cache 传输专项 → HiCache 多级缓存工程扩容 → 源码路径导读 → 性能与展望**逐层递进，完成 SGLang KV 全套技术栈深度拆解。

## 目录

- [第1章 SGLang KV Cache 底层数据结构与内存池机制](#第1章-sglang-kv-cache-底层数据结构与内存池机制)
    - [1.1 传统 KV Cache 架构缺陷（PyTorch 原生连续缓存）](#11-传统-kv-cache-架构缺陷pytorch-原生连续缓存)
    - [1.2 SGLang 三层内存池架构核心设计](#12-sglang-三层内存池架构核心设计)
        - [1.2.1 `ReqToTokenPool`：请求级逻辑映射池（逻辑层）](#121-reqtotokenpool请求级逻辑映射池逻辑层)
        - [1.2.2 `TokenToKVPoolAllocator`：Token 映射物理显存池（物理层）](#122-tokentokvpoolallocatortoken-映射物理显存池物理层)
        - [1.2.3 `KVCache`：物理存储层（`MHATokenToKVPool` / `MLATokenToKVPool` / `DSATokenToKVPool`）](#123-kvcache物理存储层mhatokentokvpool-mlatokentokvpool-dsatokentokvpool)
        - [1.2.4 三层解耦优势：逻辑请求自由伸缩、物理显存统一池化](#124-三层解耦优势逻辑请求自由伸缩、物理显存统一池化)
    - [1.3 `req_to_token` 统一页表机制（per-token 直接映射）](#13-req_to_token-统一页表机制per-token-直接映射)
    - [1.4 KV Cache 精细化元数据体系](#14-kv-cache-精细化元数据体系)
    - [1.5 Chunked Prefill：长文本分段与 `req_pool_idx` 复用机制](#15-chunked-prefill长文本分段与-req_pool_idx-复用机制)
    - [1.6 工程踩坑与源码细节](#16-工程踩坑与源码细节)
- [第2章 SGLang RadixCache 基数树：全局前缀 KV 共享核心](#第2章-sglang-radixcache-基数树全局前缀-kv-共享核心)
    - [2.1 RadixAttention 演进背景：PagedAttention 只能单请求复用](#21-radixattention-演进背景pagedattention-只能单请求复用)
    - [2.2 SGLang 中多种 Cache 类型的全景图](#22-sglang-中多种-cache-类型的全景图)
    - [2.3 `TreeNode` 核心源码字段全解析](#23-treenode-核心源码字段全解析)
    - [2.4 最长公共前缀匹配算法流程](#24-最长公共前缀匹配算法流程)
    - [2.5 树节点分裂、新建、挂载、EvictionPolicy 多策略淘汰](#25-树节点分裂、新建、挂载、evictionpolicy-多策略淘汰)
    - [2.6 RadixCache 与 `req_to_token` + `token_to_kv_pool_allocator` 三者关系](#26-radixcache-与-req_to_token-token_to_kv_pool_allocator-三者关系)
    - [2.7 引用计数 RC 联动机制：`lock_ref` + `host_ref_counter`](#27-引用计数-rc-联动机制lock_ref-host_ref_counter)
    - [2.8 线上问题：前缀失效、树内存泄漏、共享缓存脏数据](#28-线上问题前缀失效、树内存泄漏、共享缓存脏数据)
- [第3章 KV Cache 生成机制：Prefill / Decode 双阶段](#第3章-kv-cache-生成机制prefill-decode-双阶段)
    - [3.1 Prefill 预填充全量生成逻辑](#31-prefill-预填充全量生成逻辑)
    - [3.2 Decode 增量单 Token 生成](#32-decode-增量单-token-生成)
    - [3.3 惰性显存分配策略：不预占显存、随用随分](#33-惰性显存分配策略不预占显存、随用随分)
    - [3.4 多并行架构下 KV 分片生成：TP 按 head 切分 / PP 按 layer 隔离](#34-多并行架构下-kv-分片生成tp-按-head-切分-pp-按-layer-隔离)
    - [3.5 [GLM-5.2 适配] DSA 稀疏注意力的 token_mask 选择性 KV 生成](#35-glm-52-适配-dsa-稀疏注意力的-token_mask-选择性-kv-生成)
- [第4章 KV Cache 写入机制：显存固化与数据落地](#第4章-kv-cache-写入机制显存固化与数据落地)
    - [4.1 GPU 原地零拷贝写入主路径：`KVCache.set_kv_buffer()`](#41-gpu-原地零拷贝写入主路径kvcacheset_kv_buffer)
    - [4.2 跨设备写入链路：`get_cpu_copy()` / `load_cpu_copy()` 同步 offload](#42-跨设备写入链路get_cpu_copy-load_cpu_copy-同步-offload)
    - [4.3 追加写入（Decode 增量）vs 覆盖写入（上下文重置/窗口刷新）](#43-追加写入decode-增量vs-覆盖写入上下文重置窗口刷新)
    - [4.4 Batch 并发写入锁竞争：RadixCache 全局读写锁与 Block 空闲池竞争规避](#44-batch-并发写入锁竞争radixcache-全局读写锁与-block-空闲池竞争规避)
    - [4.5 [GLM-5.2 适配] 超长文本分段写入与 SWA 滑动窗口截断写入](#45-glm-52-适配-超长文本分段写入与-swa-滑动窗口截断写入)
- [第5章 KV Cache 读取机制：Attention 计算核心链路](#第5章-kv-cache-读取机制attention-计算核心链路)
    - [5.1 标准读取全流程：Query 计算 → `req_to_token` 查页表 → `create_flashinfer_kv_indices_triton` 转换为 paged KV → Attention kernel](#51-标准读取全流程query-计算-req_to_token-查页表-create_flashinfer_kv_indices_triton-转换为-paged-kv-attention-kernel)
    - [5.2 `IndicesUpdater`：`req_to_token` → flashinfer `paged_kv_indices` 的 Triton 转换 kernel](#52-indicesupdaterreq_to_token-flashinfer-paged_kv_indices-的-triton-转换-kernel)
    - [5.3 零拷贝读取、in-place 视图复用、预取优化](#53-零拷贝读取、in-place-视图复用、预取优化)
    - [5.4 多卡跨分片 KV 聚合读取：TP AllGather / EP 路由分发](#54-多卡跨分片-kv-聚合读取tp-allgather-ep-路由分发)
    - [5.5 多级缓存命中分支：L1 (GPU) 直接命中 / L2 (CPU HostKVCache) → `load_cpu_copy` / L3 (Storage Backend) → `PrefetchOperation`](#55-多级缓存命中分支l1-gpu-直接命中-l2-cpu-hostkvcache-load_cpu_copy-l3-storage-backend-prefetchoperation)
    - [5.6 [GLM-5.2 适配] RoPE 位置偏移修正与稀疏 Token 精准读取](#56-glm-52-适配-rope-位置偏移修正与稀疏-token-精准读取)
- [第6章 KV Cache 淘汰与内存回收机制](#第6章-kv-cache-淘汰与内存回收机制)
    - [6.1 显存水位线分级管控：软阈值降级 Swap / 硬阈值强制淘汰](#61-显存水位线分级管控软阈值降级-swap-硬阈值强制淘汰)
    - [6.2 `evict_policy.py`：可插拔淘汰策略（LRU / LFU / SLRU / FIFO）](#62-evict_policypy可插拔淘汰策略lru-lfu-slru-fifo)
    - [6.3 [GLM-5.2 适配] SWA 窗口外 KV 强制过期 + DSA 无效 Token KV 主动释放](#63-glm-52-适配-swa-窗口外-kv-强制过期-dsa-无效-token-kv-主动释放)
    - [6.4 细粒度 slot 回收 vs 粗粒度整会话回收](#64-细粒度-slot-回收-vs-粗粒度整会话回收)
    - [6.5 双层 RC 防误删：`TreeNode.lock_ref` + `host_ref_counter`](#65-双层-rc-防误删treenodelock_ref-host_ref_counter)
    - [6.6 淘汰后置：树分支修剪、`req_to_token` 索引刷新、slot 归还 `free_pages`、HiCache 下沉传输标记](#66-淘汰后置树分支修剪、req_to_token-索引刷新、slot-归还-free_pages、hicache-下沉传输标记)
- [第7章 KV Cache 跨设备传输体系设计](#第7章-kv-cache-跨设备传输体系设计)
    - [7.1 传输场景全景：SGLang 中实际存在的三类传输](#71-传输场景全景sglang-中实际存在的三类传输)
    - [7.2 GPU↔CPU Offload 详细链路](#72-gpucpu-offload-详细链路)
    - [7.3 Disaggregation PD 传输链路](#73-disaggregation-pd-传输链路)
    - [7.4 HiCache 层级传输：HostKVCache ↔ Storage Backend](#74-hicache-层级传输hostkvcache-storage-backend)
    - [7.5 [GLM-5.2 适配] 大 KV 量下的传输优化方向](#75-glm-52-适配-大-kv-量下的传输优化方向)
    - [7.6 传输链路常见故障与排坑](#76-传输链路常见故障与排坑)
- [第8章 非标准 Attention 架构对 KV Cache 的强约束](#第8章-非标准-attention-架构对-kv-cache-的强约束)
    - [8.1 SGLang 中已有的非标准 KV Cache 实现全景](#81-sglang-中已有的非标准-kv-cache-实现全景)
    - [8.2 MLA（Multi-Head Latent Attention）KV Cache 存储对比分析](#82-mlamulti-head-latent-attentionkv-cache-存储对比分析)
    - [8.3 DSA（Dense-Sparse Attention）稀疏窗口机制](#83-dsadense-sparse-attention稀疏窗口机制)
    - [8.4 RoPE 位置编码偏移引发的索引修正原理](#84-rope-位置编码偏移引发的索引修正原理)
    - [8.5 Continuous Batch 动态批处理资源调度](#85-continuous-batch-动态批处理资源调度)
    - [8.6 FP8 量化 KV Cache：`store_dtype=torch.uint8` 的数值对齐与精度兼容](#86-fp8-量化-kv-cachestore_dtypetorchuint8-的数值对齐与精度兼容)
    - [8.7 MoE 专家并行 EP 下多卡 KV 分布与路由](#87-moe-专家并行-ep-下多卡-kv-分布与路由)
    - [8.8 [GLM-5.2 推演] 结合 MLA + DSA + MoE 的综合 KV Cache 架构设计方向](#88-glm-52-推演-结合-mla-dsa-moe-的综合-kv-cache-架构设计方向)
- [第9章 SGLang 端到端全推理链路时序闭环](#第9章-sglang-端到端全推理链路时序闭环)
    - [9.1 请求接入、分词、`Req` 对象元信息初始化](#91-请求接入、分词、req-对象元信息初始化)
    - [9.2 RadixCache `match_prefix()` 前缀匹配判定](#92-radixcache-match_prefix-前缀匹配判定)
    - [9.3 Prefill 全量 / Chunked Prefill 增量双分支执行流程](#93-prefill-全量-chunked-prefill-增量双分支执行流程)
    - [9.4 Decode 循环生成 + 增量 KV 持续挂载](#94-decode-循环生成-增量-kv-持续挂载)
    - [9.5 多轮对话缓存复用加速逻辑](#95-多轮对话缓存复用加速逻辑)
    - [9.6 会话超时/结束资源回收链路：`release_kv_cache()` 完整流程](#96-会话超时结束资源回收链路release_kv_cache-完整流程)
    - [9.7 显存超限→触发淘汰→`evict()` → 可能触发 HiCache 下沉传输](#97-显存超限触发淘汰evict-可能触发-hicache-下沉传输)
    - [9.8 下级缓存命中→`load_back()` → `load_cpu_copy` 回灌 GPU](#98-下级缓存命中load_back-load_cpu_copy-回灌-gpu)
    - [9.9 GLM-5.2 场景推演：工具调用 / 长摘要 / 记忆裁剪特殊链路](#99-glm-52-场景推演工具调用-长摘要-记忆裁剪特殊链路)
- [第10章 SGLang HiCache 多级缓存架构原理](#第10章-sglang-hicache-多级缓存架构原理)
    - [10.1 三级存储层级定义](#101-三级存储层级定义)
    - [10.2 多池类型管理：`PoolName` 枚举与 `PoolTransfer` 传输抽象](#102-多池类型管理poolname-枚举与-pooltransfer-传输抽象)
    - [10.3 `HiCacheController` + `HybridCacheController`：升降级调度与预取](#103-hicachecontroller-hybridcachecontroller升降级调度与预取)
    - [10.4 与第 7 章传输链路的联动](#104-与第-7-章传输链路的联动)
    - [10.5 HiCache 与 Disaggregation PD 的协同：`StorageMedium` 标记](#105-hicache-与-disaggregation-pd-的协同storagemedium-标记)
    - [10.6 工程稳定性方案](#106-工程稳定性方案)
- [第11章 核心源码路径导读](#第11章-核心源码路径导读)
    - [11.1 内存池与页表](#111-内存池与页表)
    - [11.2 RadixCache 前缀匹配与淘汰](#112-radixcache-前缀匹配与淘汰)
    - [11.3 跨设备传输](#113-跨设备传输)
    - [11.4 Attention Backend 中的 paged KV 转换](#114-attention-backend-中的-paged-kv-转换)
    - [11.5 调度入口与工具函数](#115-调度入口与工具函数)
- [第12章 多框架横向性能对比](#第12章-多框架横向性能对比)
    - [12.1 测试基线：vLLM (PagedAttention) / SGLang (RadixCache) / SGLang + HiCache](#121-测试基线vllm-pagedattention-sglang-radixcache-sglang-hicache)
    - [12.2 核心指标：TTFT、TPOT、QPS、显存占用、缓存命中率、传输时延](#122-核心指标ttft、tpot、qps、显存占用、缓存命中率、传输时延)
- [第13章 架构局限与未来演进](#第13章-架构局限与未来演进)
    - [13.1 当前短板](#131-当前短板)
    - [13.2 未来方向](#132-未来方向)
- [附录](#附录)
    - [A.1 传统原版 PagedAttention 原理回顾](#a1-传统原版-pagedattention-原理回顾)
    - [A.2 `KVCache` 子类全景参考](#a2-kvcache-子类全景参考)
    - [A.3 `TokenToKVPoolAllocator` 子类全景参考](#a3-tokentokvpoolallocator-子类全景参考)


---

# 第一部分：底层内核架构（数据结构基础）
## 第1章 SGLang KV Cache 底层数据结构与内存池机制
### 1.1 传统 KV Cache 架构缺陷（PyTorch 原生连续缓存）
- 1.1.1 四维张量排布：`[batch, seq_len, n_head, head_dim]` 存储范式

四维张量 [batch, seq_len, n_head, head_dim] 是 Transformer 架构（如多头注意力机制）中处理 Q、K、V 及最终输出时的核心逻辑排布。这种排布旨在将序列数据映射到不同的子空间，实现多角度特征的并行捕捉。

维度详解与物理意义:
batch (批次大小 / 序列条数)：表示一次并行处理的独立样本数量。
seq_len (序列长度)：单个样本包含的 Token 数量或时间步数。
n_head (注意力头数)：模型并行的注意力子空间数量。
总特征维度通常由 (n_head * head_dim) 决定。
head_dim (单头维度)：每个注意力头负责提取的特征向量长度。

在代码中的流转过程（PyTorch 为例）
在多头注意力机制（Multi-Head Attention）内部，数据排布需经过多次变换以满足计算要求：
线性投影：通常输入的特征维度为 [batch, seq_len, embed_dim]。经过投影层展开为 Q、K、V 后，形状仍保持为 [batch, seq_len, embed_dim]。拆分与重塑 (view)：将最后一个维度 embed_dim 拆分为 n_head 和 head_dim，张量变为四维：[batch, seq_len, n_head, head_dim]。转置与排列 (transpose 或 permute)：为了能对每个头进行点积注意力计算，需要将 seq_len 和 n_head 维度进行交换，变成 [batch, n_head, seq_len, head_dim]。这样计算时，能保证每个 head 独立处理序列信息。

参考： https://medium.com/@kavierim/transformers-from-scratch-part-3-multi-head-attention-d1a3a061ba89


- 1.1.2 固定连续内存导致：显存碎片化严重、无法局部复用、长序列 OOM
- 1.1.3 批量推理无法共享公共 Prompt，Prefill 算力严重浪费

### 1.2 SGLang 三层内存池架构核心设计
#### 1.2.1 `ReqToTokenPool`：请求级逻辑映射池（逻辑层）

**定位**：

SGLang 内存池体系分为三层，`ReqToTokenPool` 是**逻辑层**——只做请求到 token 位置的索引映射，不管物理显存：

```
ReqToTokenPool           ← 逻辑层：请求 → token 位置映射（本类）
TokenToKVPoolAllocator   ← 分配层：物理 slot 的分配/释放/碎片整理
KVCache                  ← 物理层：GPU 上真实的 K/V 张量
```

源码位于 `python/sglang/srt/mem_cache/memory_pool.py:235-302`，总共不到 70 行。

---

**核心数据结构**：

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size, max_context_len, device, enable_memory_saver):
        self.size = size
        self._alloc_size = size + 1    # +1: 第 0 行是 CUDA graph padding
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len),
            dtype=torch.int32, device=device
        )
        self.free_slots = list(range(1, self._alloc_size))
```

只有四个实例属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `size` | `int` | 最大并发请求数 |
| `max_context_len` | `int` | 单请求最大上下文长度 |
| `req_to_token` | `torch.Tensor` | GPU 上的二维页表，`[req_pool_idx, pos] → kv_slot` |
| `free_slots` | `list[int]` | 空闲行号的列表，从 1 开始 |

**页表解读**：`req_to_token[3][127] = 2048` 表示"第 3 号请求槽位中，该请求的第 127 个 token，存储在 KV cache 的第 2048 号物理 slot"。

---

**三个方法**：

`alloc(reqs: list[Req]) -> Optional[List[int]]`

```python
def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
    # 1. 先挑出已有 req_pool_idx 的请求（chunked prefill 复用）
    reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
    assert all(
        reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
        for i in reusing
    ), "reusing request must be chunked or have committed KV"

    # 2. 为新请求分配行号
    need_size = len(reqs) - len(reusing)
    if need_size > len(self.free_slots):
        return None                    # 不够 → 返回 None，上层调度阻塞
    select_index = self.free_slots[:need_size]
    self.free_slots = self.free_slots[need_size:]

    # 3. 把行号写回 Req 对象
    offset = 0
    for r in reqs:
        if r.req_pool_idx is None:
            r.req_pool_idx = select_index[offset]
            offset += 1
    return [r.req_pool_idx for r in reqs]
```

**关键设计点**：

- **只分配行号**，不填页表。填页表由 `common.py:write_cache_indices()` 通过 Triton kernel 完成。
- **chunked prefill 复用**：同一请求跨多个 chunk 时不重新分配，复用已有的 `req_pool_idx`。
- **返回 `None` 表示资源不足**，上层 scheduler 据此阻塞批次。

`free(req: Req)`

```python
def free(self, req: Req):
    assert req.req_pool_idx is not None, "request must have req_pool_idx"
    self.free_slots.append(req.req_pool_idx)
    req.req_pool_idx = None
```

行号归还到 `free_slots`，同时清除 `Req` 对象上的引用。

`write(indices, values)`

```python
def write(self, indices, values):
    self.req_to_token[indices] = values
```

直接写 GPU 张量。由 `common.py:write_cache_indices()` 调用，底层通过 Triton kernel `write_req_to_token_pool_triton` 批量写入 prefix + extended token 的物理 slot 映射。

**`clear()`**

```python
def clear(self):
    self.free_slots = list(range(1, self._alloc_size))
```

重置所有空闲槽位。

`available_size()`

```python
def available_size(self):
    return len(self.free_slots)
```

返回当前可用槽位数。

---

**索引 0 的约定**

```
行 0:  CUDA graph padding（dummy 读写落在这里，无害）
行 1~size: 实际请求
```

`free_slots` 初始化为 `list(range(1, self._alloc_size))`，永远不从 0 分配。CUDA graph 的 padded batch 把无效请求的 `req_pool_indices` 置 0，attention kernel 读写的都是 padding 行，不会污染真实数据。KV Canary (`jit_kernel/kv_canary/consts.py:8`) 也显式记录了这条约定：

```python
# Mirrors SGLang's ReqToTokenPool contract: req_pool_idx 0 is the CUDA-graph padding row
```

---

**在整个推理流程中的位置**

```
Scheduler
  │
  ├─ alloc_req_slots()       ──→  ReqToTokenPool.alloc(reqs)    分配行号
  ├─ write_cache_indices()   ──→  ReqToTokenPool.write()        Triton kernel 填页表
  │
  ▼
Attention Backend
  │
  └─ init_forward_metadata()
       req_to_token_pool.req_to_token  ──→  构建 page table 传给 GPU kernel
```

**Prefill 阶段**

1. `alloc_req_slots()` 调用 `ReqToTokenPool.alloc(reqs)` 为每个请求分配 `req_pool_idx`
2. `TokenToKVPoolAllocator` 为每个 token 分配物理 slot
3. `write_cache_indices()` 调用 `ReqToTokenPool.write()` 把 `(req_pool_idx, pos) → kv_slot` 写入页表
4. Attention kernel 读取 `req_to_token` 页表，按物理地址查 K/V

**Decode 阶段**

1. 新 token 的物理 slot 被分配后，追加一条映射到页表对应行末尾
2. Attention kernel 读取整行 page table 完成全序列 attention

**释放阶段**

1. `release_kv_cache()` 从 `req_to_token[req_pool_idx]` 读出该请求所有物理 slot，归还给 `TokenToKVPoolAllocator`
2. `ReqToTokenPool.free(req)` 归还行号到 `free_slots`

---

**全局访问方式**

通过 forward context 获取：

```python
# forward_context.py:74
def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool
```

池在服务启动时由 `ModelRunner._init_pools()` 一次性创建，然后被所有 attention backend、radix cache、schedule batch 共享引用。

---

**实例化分支逻辑**

在 `model_runner_kv_cache_mixin.py:_init_pools()` 中，根据模型类型和部署模式选择具体的池类：

| 条件 | 池类 |
|---|---|
| 分离式 decode + Mamba 模型 | `HybridMambaDecodeReqToTokenPool` |
| 分离式 decode + 无 Mamba | `DecodeReqToTokenPool` |
| 普通模式 + Mamba 模型 | `HybridReqToTokenPool` |
| 普通模式 + DSV4 on NPU | `DSV4NPUReqToTokenPool` |
| 普通模式（默认） | `ReqToTokenPool` |

---

**层次**

```
ReqToTokenPool                         (基类, ~70行)
  ├── HybridReqToTokenPool             (+Mamba conv/temporal state 池)
  │     └── HybridMambaDecodeReqToTokenPool  (+分离式 decode pre-alloc)
  ├── DSV4NPUReqToTokenPool            (+c4/c128 压缩 KV 页表)
  └── MlxAuxiliaryStateReqToTokenPool  (+MLX 辅助状态池)

DecodeReqToTokenPool                   (分离式 decode, 兄弟类, 非继承)
```

---

**Triton Kernel 支持**

`python/sglang/srt/mem_cache/triton_ops/common.py` 中两个关键 kernel：

| Kernel | 功能 |
|---|---|
| `write_req_to_token_pool_triton` | 批量写入 prefix + extended token 的物理 slot 映射到 `req_to_token` |
| `get_last_loc_triton` | 从 `req_to_token` 读取每个请求最后一个 token 的物理位置 |

---

**一句话总结**

一张 GPU 上的二维 int32 张量，行是请求槽位、列是 token 位置、值是物理 slot 索引，外加一个空闲行号列表。不存请求元数据、不管物理显存、不感知前缀缓存——纯粹就是一张页表。

同一条请求的多个 chunk 共享一行页表，每个 chunk 往同一行追加新的 slot 映射，KV cache 自然累积，attention 时只需读这一行就能看到所有 token 的 K/V。

#### 1.2.2 `TokenToKVPoolAllocator`：Token 映射物理显存池（物理层）

**定位**：

SGLang 内存池体系三层中的**分配层**——管理 KV cache 物理 slot 的分配和释放，但不持有真实的 K/V 数据：

```
ReqToTokenPool           ← 逻辑层：请求 → token 位置映射（页表）
TokenToKVPoolAllocator   ← 分配层：物理 slot 的分配/释放/碎片整理（本类）
KVCache                  ← 物理层：GPU 上真实的 K/V 张量
```

源码位于 `python/sglang/srt/mem_cache/allocator/` 包。

---

**类层次**

```
BaseTokenToKVPoolAllocator          (抽象基类, allocator/base.py:27)
  ├── TokenToKVPoolAllocator         (per-token 分配, allocator/token.py:28)
  ├── PagedTokenToKVPoolAllocator    (per-page 分配, allocator/paged.py:105)
  ├── SWATokenToKVPoolAllocator      (Hybrid SWA, allocator/swa.py:20)
  ├── HiSparseTokenToKVPoolAllocator (稀疏注意力, allocator/hisparse.py:15)
  │     └── DeepSeekV4HiSparseTokenToKVPoolAllocator
  └── (NPU 变体)
```

---

**核心数据结构**

基类 (`allocator/base.py`)

```python
class BaseTokenToKVPoolAllocator(abc.ABC):
    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        self.size = size              # 物理 slot 总数
        self.page_size = page_size    # 分配粒度: 1=per-token, >1=per-page
        self._kvcache = kvcache       # 反向引用 KVCache，用于 CPU offload 等
        self.need_sort = need_sort    # 是否对空闲页排序（前缀缓存命中率高时关闭）

        self.free_pages = None        # 空闲物理索引（GPU tensor, int64）
        self.release_pages = None     # 延迟释放队列（待排序合并）
        self.free_group = []          # 批量释放暂存
        self.is_not_in_free_group = True
```

`TokenToKVPoolAllocator`（`allocator/token.py:28-84`，总共 ~55 行）

```python
class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(self, size, dtype, device, kvcache, need_sort):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # 索引 0 是 padded token 的 dummy 写入位置
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, device=device)
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=device)
```

两个 GPU tensor：

| 属性 | 类型 | 含义 |
|---|---|---|
| `free_pages` | `torch.Tensor[int64]` | 空闲物理 slot 索引，按需排序 |
| `release_pages` | `torch.Tensor[int64]` | 延迟释放队列，等 `merge_and_sort_free()` 时合并 |

**为什么都在 GPU 上？** 分配/释放操作发生在每个 batch 的 forward 路径上，tensor 在 GPU 上可以直接参与 Triton kernel、CUDA kernel 的操作，避免 CPU↔GPU 同步。

---

**核心方法**：

`alloc(need_size: int) -> Optional[torch.Tensor]`（`token.py:55-64`）

```python
def alloc(self, need_size: int):
    # 不够且需要排序 → 先合并延迟释放队列再排序
    if self.need_sort and need_size > len(self.free_pages):
        self.merge_and_sort_free()

    if need_size > len(self.free_pages):
        return None              # 不够 → 返回 None，上层抛异常

    select_index = self.free_pages[:need_size]
    self.free_pages = self.free_pages[need_size:]
    return select_index          # 返回 GPU tensor，直接给 kernel 用
```

与 `ReqToTokenPool.alloc()` 结构完全对称，但操作的是 **GPU tensor** 而非 Python list。返回的 `select_index` 就是物理 slot 编号，后续 `write_cache_indices()` 把它写入 `req_to_token` 页表。

`free(free_index: torch.Tensor)`（`token.py:66-76`）

```python
def free(self, free_index: torch.Tensor):
    if free_index.numel() == 0:
        return

    if self.is_not_in_free_group:
        if self.need_sort:
            # 需要排序 → 先丢进延迟释放队列
            self.release_pages = torch.cat((self.release_pages, free_index))
        else:
            # 不需要排序 → 直接追加到 free_pages 末尾
            self.free_pages = torch.cat((self.free_pages, free_index))
    else:
        # 批量释放模式 → 暂存到 free_group
        self.free_group.append(free_index)
```

两阶段释放机制：
- **`need_sort=True`**（前缀缓存命中率高）→ 先放到 `release_pages`，攒到 `alloc` 不够时再 `merge_and_sort_free()` 合并排序。排序保证碎片整理——连续 free 的 slot 不连续，排序后大块连续分配效率更高。
- **`need_sort=False`**（命中率低、释放少）→ 直接追加到 `free_pages` 末尾，免排序开销。

`merge_and_sort_free()`（`base.py:78-84`）

```python
def merge_and_sort_free(self):
    if len(self.release_pages) > 0:
        self.free_pages = torch.cat((self.free_pages, self.release_pages))
        self.free_pages, _ = torch.sort(self.free_pages)
        self.release_pages = torch.empty((0,), dtype=..., device=...)
```

把延迟释放队列和现有空闲 page 合并后排序，保证碎片整理。

`available_size()`（`token.py:51-53`）

```python
def available_size(self):
    return len(self.free_pages) + len(self.release_pages)
```

批量释放：`free_group_begin()` / `free_group_end()`（`base.py:69-76`）

```python
def free_group_begin(self):
    self.is_not_in_free_group = False
    self.free_group = []

def free_group_end(self):
    self.is_not_in_free_group = True
    if self.free_group:
        self.free(torch.cat(self.free_group))
```

批量释放场景（如一次释放多个请求的 KV）先把所有待释放索引收集到 `free_group`，最后一次性 `torch.cat` + `free`，减少 `torch.cat` 调用次数。

---

`PagedTokenToKVPoolAllocator`（`allocator/paged.py`）

当 `page_size > 1` 时使用。core 思路同 `TokenToKVPoolAllocator`，但以**页**为最小分配单元。

关键差异：

| 方面 | `TokenToKVPoolAllocator` | `PagedTokenToKVPoolAllocator` |
|---|---|---|
| 粒度 | per-token (`page_size=1`) | per-page (`page_size=64/128/...`) |
| `free_pages` 存什么 | slot 索引 | page 编号 |
| `alloc()` 返回 | slot 索引 | `page_num * page_size + offset` 展开为 slot 索引 |
| `free()` | 直接释放 slot | `torch.unique(free_index // page_size)` 去重后释放 page |

额外方法：

| 方法 | 用途 |
|---|---|
| `alloc_extend()` | Prefill 阶段分配，通过 Triton kernel `alloc_extend_kernel` 在同页内顺序追加 |
| `alloc_decode()` | Decode 阶段每个请求分配 1 个新 slot，通过 `alloc_decode_kernel` |

这两个 kernel 都利用了 page 内连续分配的特性——extend/decode 时如果当前 page 还有空位就直接用，否则才从 `free_pages` 拿新 page。

alloc_extend 就是一个 GPU Triton kernel 驱动的页分配器——知道每个请求已经占了多少页（前缀缓存），算出还需要多少页，用三段填充法把已有页剩余空位、
  完整新页、最后不完整页串联起来，返回一段物理上不连续但逻辑上无缝的光滑 slot 序列。

---

`SWATokenToKVPoolAllocator`（`allocator/swa.py`）

Hybrid SWA（Sliding Window Attention）模型的分配器。内部组合**两个**子分配器：

```
SWATokenToKVPoolAllocator
  ├── full_attn_allocator   ← 管理全 attention 层的 KV slot
  └── swa_attn_allocator    ← 管理 SWA 层的 KV slot
```

核心：`full_to_swa_index_mapping` —— 一个映射表，`full_idx → swa_idx`。
- `alloc()` / `alloc_extend()` / `alloc_decode()` 都同时分配 full 和 SWA 两套 slot
- `free()` 从 full index 查出 swa index，两边一起释放
- `translate_loc_from_full_to_swa()` 供 attention backend 把 full pool 的 `last_loc` 转为 swa pool 的位置

---

索引 0 的约定

与 `ReqToTokenPool` 一致：

```python
# token.py:44
self.free_pages = torch.arange(1, self.size + 1, ...)

# paged.py:279
self.free_pages = torch.arange(1, self.num_pages + 1, ...)
```

索引 0 留给 padded token 的 dummy 写入，永远不分配给真实请求。

---

在推理流程中的位置

```
alloc_for_extend()                        (common.py:456)
  │
  ├── alloc_req_slots()                  分配请求行号 (ReqToTokenPool)
  │
  ├── alloc_token_slots(tree_cache)  ←── 分配物理 slot (TokenToKVPoolAllocator)
  │     │                                  tree_cache 内部调用
  │     └── token_to_kv_pool_allocator.alloc(need_size)
  │           → 返回 select_index: [1024, 1025, ..., 2047]
  │
  └── write_cache_indices()              把 select_index 写入 req_to_token 页表

释放:
release_kv_cache()                        (common.py:685)
  │
  ├── 从 req_to_token[req_pool_idx] 读出待释放的物理 slot
  │
  ├── token_to_kv_pool_allocator.free(indices_to_free)
  │
  └── req_to_token_pool.free(req)
```

关键点：**token_to_kv_pool_allocator 的调用入口不在 allocator 本身，而是通过 tree_cache（PrefixCache）代理**：

```python
# common.py:487
out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
# 内部: tree_cache.token_to_kv_pool_allocator.alloc(need_size)
```

因为前缀缓存命中时可能不需要分配新 slot——tree_cache 先把命中部分复用，缺口部分才调用 allocator 分配。

---

**总结**

`TokenToKVPoolAllocator` 就是一个 **GPU 上的空闲 slot 管理器**：

- 一个 `free_pages` int64 tensor 存所有空闲物理索引
- `alloc()` 从头部切，返回连续的 GPU tensor 直接给 kernel 用
- `free()` 追加到尾部（或延迟队列），按需排序做碎片整理
- 不持有 KV 数据本身（数据在 `KVCache` 里），只管"哪些位置可用"
- 与 `ReqToTokenPool` 镜像设计——一个管行号、一个管 slot，两者通过页表 `req_to_token` 关联


#### 1.2.3 `KVCache`：物理存储层（`MHATokenToKVPool` / `MLATokenToKVPool` / `DSATokenToKVPool`）

**定位**：

三层内存池体系中的**物理层**——真正持有 GPU 上 K/V 张量、负责显存布局与读写落地的类：

```
ReqToTokenPool           ← 逻辑层：请求 → token 位置映射（页表）
TokenToKVPoolAllocator   ← 分配层：物理 slot 的分配/释放/碎片整理
KVCache                  ← 物理层：GPU 上真实的 K/V 张量（本节）
```

它**只管物理存储**：在哪块显存上、以什么布局存 K/V。它**不知道"哪个 token 放哪"**——那是分配层 `TokenToKVPoolAllocator` + 逻辑层 `ReqToTokenPool` 的职责；中间由 `loc`（槽位索引）解耦。源码位于 `python/sglang/srt/mem_cache/memory_pool.py`。

---

**公共基类 `KVCache`（`memory_pool.py:978`）**

所有物理池的抽象基类，定义存储层共性：

```python
class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(self, size, page_size, dtype, layer_num, device,
                 enable_memory_saver, start_layer=None, end_layer=None):
        self.size = size                      # 池容量（token 数，不含 padding）
        self.page_size = page_size            # 分页大小
        self.dtype = dtype                    # 计算 dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, ...):
            # fp8 存为 uint8：Tensor.index_put 不支持 fp8（L995-999）
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0   # 该池负责的层范围（支持多池分层）
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(...)
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None    # 分层传输同步钩子
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)   # disagg nvlink 独立显存池
        )
```

抽象接口：`get_key_buffer` / `get_value_buffer` / `get_kv_buffer` / `set_kv_buffer`。

核心字段含义：

| 字段 | 含义 |
|---|---|
| `size` / `page_size` | 池容量与分页粒度 |
| `store_dtype` | 实际存储 dtype；fp8 存为 `uint8` 以绕过 `index_put` 限制 |
| `start_layer`/`end_layer` | 该池负责的层范围（SWA/Dense 多池分层时各管一段） |
| `memory_saver_adapter` | CPU offload 友好的显存分配适配器 |
| `custom_mem_pool` | disagg 场景独立 CUDA memory pool（Nvlink 链路传输） |
| `layer_transfer_counter` | 逐层传输同步：`get_key_buffer` 会 `wait_until(layer_id)` |
| `cpu_offloading_chunk_size` | D2H/H2D 分块粒度（默认 8192 token） |

**通用约定：所有池都多分配 `+ self.page_size` 个槽位**作为 padding slot 0，用于写 padded/dummy token 的输出（不污染真实数据）。配合 `maybe_detect_oob(loc, 0, self.size + self.page_size, ...)` 越界检查，防止 stale slot id 导致静默 KV 损坏（由 `SGLANG_ENABLE_ASYNC_ASSERT` 控制开关）。

---

**`MHATokenToKVPool`（`memory_pool.py:1074`）—— 标准 MHA**

适用 Llama / Qwen / GLM 等标准多头注意力模型，每 token 存完整 K 和 V。

张量布局（每层一对独立 K/V buffer，共 `layer_num` 对）：

```python
# 默认 NHD 布局（L1262）
self.k_buffer[i] = torch.zeros(size + page_size, head_num, head_dim)       # [token, head, dim]
self.v_buffer[i] = torch.zeros(size + page_size, head_num, v_head_dim)     # 支持 K/V 维度不等
```

可选 AITER 5D SHUFFLE 布局（仅 ROCm AITER，`SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`，L1112-1144）：

```
K: (num_blocks, H, D_k // X, page, X)
V: (num_blocks, H, page // X, D_v, X)     # X = 16 / dtype_itemsize，fp8=16，bf16/fp16=8
```

为 aiter `mha_batch_prefill_func` / `pa_decode_gluon` 原生消费的 SHUFFLE 物理布局。

元数据加速（创建时预计算 GPU 端指针表，供 JIT kernel 直接取址，L1285-1302）：

```python
self.data_ptrs    = torch.cat([k_data_ptrs, v_data_ptrs])     # 各层 K/V 的 data_ptr (uint64)
self.data_strides = [np.prod(x.shape[1:]) * x.dtype.itemsize] # 每层单 token 字节步幅
```

`enable_kv_cache_copy=True` 时还会 `_init_kv_copy_and_warmup()`（L1168）预热 `copy_all_layer_kv_cache_tiled` 跨层拷贝 kernel，用于 disagg 整池搬迁，按 stride 自适应 tile 大小（8192/4096 阈值切 512/256/128 tile）。

写入 `set_kv_buffer()`（L1409）分三条路径：

1. **`dcp_kv_mask` 路径**（L1440）—— context parallel 的 masked 写入 kernel `masked_set_kv_buffer_kernel`
2. **vectorized_5d 路径**（L1460）—— 调 `launch_reshape_and_cache_shuffle_5d`
3. **NHD 默认路径**（L1482）—— 调 `_set_kv_buffer_impl`，支持 `alt_stream`（异步写流）与 `same_kv_dim` 优化

辅助能力：`set_kv_buffer_prefix_valid`（按 `commit_lens` 部分写入，draft/prefix commit）、`move_kv_cache(tgt, src)`（槽位搬迁）/、`get_cpu_copy`/`load_cpu_copy`（分块 `cpu_offloading_chunk_size=8192` 异步 CPU 拷贝，用于 offload disagg）。

---

**`MLATokenToKVPool`（`memory_pool.py:2130`）—— DeepSeek MLA**

适用 DeepSeek-V2/V3 的 Multi-head Latent Attention。核心省显存思想：**每 token 只存一个低秩潜在向量**，而非完整多头 K/V。

张量布局（每层只有一个 `kv_buffer`，K/V 合一，L2192）：

```python
self.kv_buffer[i] = torch.zeros(size + page_size, 1, kv_cache_dim)
# kv_cache_dim = kv_lora_rank + qk_rope_head_dim   (默认 512 + 64 = 576)
```

- `head_num=1`：所有查询头共享同一份 latent，头维坍缩为 1
- latent 分两段：前 `kv_lora_rank` 是压缩的 V（nope 部分），后 `qk_rope_head_dim` 是带 RoPE 的 K
- 对 128 头 × 128 维的模型，MHA 每 token 存 `128×128×2(K+V)`；MLA 只存 `576`，约 **57× 压缩**

逻辑 K/V 拆分（物理一块连续，对外仍暴露 key/value，L2221-2238）：

```python
def get_value_buffer(self, layer_id):
    return self.kv_buffer[...][:, :, :self.kv_lora_rank]   # V = 前半
# key 取整块（含 rope 段）
```

写入：`set_kv_buffer(loc, cache_k, cache_v)` 把整块 latent 写入（支持 context parallel 的 `dcp` mask，L2254）；MLA 专用 `set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)`（L2272）分别接收 nope/rope 两段，可走 FP8 量化路径（`dsa_kv_cache_store_fp8`、HIP triton 量化 `set_mla_kv_buffer_triton_fp8_quant`）。

DSA 钩子：`use_dsa=True` 时不立即打印分配日志（`_finalize_allocation_log` 推迟到 DSA 子类，因为 DSA 还要再分配 indexer 缓冲，L2180）。

---

**`DSATokenToKVPool`（`memory_pool.py:2529`）—— DeepSeek V3.2 DSA**

继承自 `MLATokenToKVPool`，适用 DeepSeek-V3.2 的 DeepSeek Attention (DSA)。在 MLA latent KV 之上**额外增加一个索引缓存**用于稀疏注意力 Top-K 路由。

双缓冲结构：

```
DSA 池 = MLA 的 latent kv_buffer  +  index_k_with_scale_buffer（每层各一）
```

`index_k_with_scale_buffer` 布局（L2592）：

```python
shape = (num_pages, page_size * (index_head_dim + index_head_dim//quant_block_size * 4))
#      = (num_pages, 64 * (128 + 4))   # page_size=64, head_dim=128, fp32 scale=4字节
# dtype = uint8
```

每个页内：前 `64×128` 字节是 fp8 索引 K，后 `64×4` 字节（view 成 float32）是 per-block 量化 scale。K 数据与 scale 打包在同一块连续内存，便于 DSA 稀疏 Top-K 索引 kernel 直接读取。

平台约束（L2576）：

```python
if _is_hip:
    # ROCm: page_size 须为 16 倍数（preshuffle）否则 ==1（legacy）
else:
    assert page_size == 64   # CUDA 强制页大小 64
```

索引访问 API（通过 `index_buf_accessor` 提供融合读写）：

| 方法 | 作用 |
|---|---|
| `get_index_k_with_scale_buffer` | 原始 buffer |
| `get_index_k_continuous` / `get_index_k_scale_continuous` | 分取 K / scale |
| `get_index_k_scale_buffer` | **融合**一次取 K+scale（Triton，比分开调高效） |
| `set_index_k_scale_buffer` | 融合写入 K+scale |

一致性维护（关键）：DSA 有两块缓冲，所有"搬移/卸载"操作都必须**成对更新**，否则读脏数据。

- `move_kv_cache(tgt, src)`（L2618）：先 `super().move_kv_cache` 搬 latent，再逐层搬 `index_k_with_scale_buffer`
- `get_cpu_copy` 返回 `{"kv":..., "index_k":...}` 字典。注释明确指出（L2704-2709）：retract 释放的页会被别的请求 `set_index_k_scale_buffer` 复用，若不同步 offload 索引缓存，resume 时会恢复 latent 却留下**别人的 index/scale**，导致 DSA 注意力读到错位的垃圾数据

---

**三者对比总览**

| 维度 | MHATokenToKVPool | MLATokenToKVPool | DSATokenToKVPool |
|---|---|---|---|
| 适用模型 | Llama/Qwen/GLM 等标准 MHA | DeepSeek-V2/V3 MLA | DeepSeek-V3.2 DSA |
| 每层 buffer | K + V 两个 | kv_buffer 一个 | kv_buffer + index_k_with_scale |
| token 存储量 | `head_num·head_dim·2` | `kv_lora_rank+qk_rope` | MLA 量 + 索引页 |
| 头维度 | `head_num` | `1`（共享 latent） | `1` + 独立 index head |
| 布局选项 | NHD / AITER 5D SHUFFLE | 单块 | 单块 + 打包 uint8 索引页 |
| 继承关系 | ← KVCache | ← KVCache | ← MLATokenToKVPool |
| 额外能力 | alt_stream 异步写、kv_copy JIT | DSA 钩子 | 索引 K/scale 融合读写、双缓冲一致性 |

附录 A.2 还会覆盖子类家族：`NoOpMHATokenToKVPool`（L1642，空池，all-SWA 模型的 full sub-pool）、`MHATokenToKVPoolFP4`（L1752）、`MLATokenToKVPoolFP4`（L2389）、`HybridLinearKVPool`（L1902，SWA/Dense 混合双池），此处先不展开。

---

**与上层的协作关系（物理层 vs 逻辑层）**

```
逻辑层  RadixCache / ReqToTokenPool      ← 决定 token→槽位映射、前缀复用
   ↓ 提供 slot loc
物理层  MHATokenToKVPool / MLA / DSA      ← 本节，管显存布局与读写
   ↓ data_ptrs / JIT kernel
执行层  Attention Backend                 ← 读 K/V 算 attention，写新 K/V
```

调度器调 `pool.set_kv_buffer(layer, loc, k, v)` 写入；attention backend 调 `pool.get_key_buffer(layer_id)` 读出。中间由 `loc`（槽位索引）解耦——这种分层让同一套前缀缓存逻辑能复用于 MHA/MLA/DSA 三种截然不同的物理布局。

**一句话总结**：三个类是 KV cache 的"显存布局器"——MHA 存全量多头 K/V，MLA 把多头压成单 latent（57× 省显存），DSA 在 MLA 之上再加一份打包的 fp8 索引页支撑稀疏 Top-K 路由；三者都靠 `loc` 槽位索引与上层解耦，靠多出的 `page_size` padding slot 0 兜底 dummy 写入。

#### 1.2.4 三层解耦优势：逻辑请求自由伸缩、物理显存统一池化

把 `ReqToTokenPool`（逻辑层）、`TokenToKVPoolAllocator`（分配层）、`KVCache`（物理层）拆成三层，不是单纯的代码整洁，而是为了以下四个解耦带来的工程红利。它们两两之间都通过**索引张量**耦合，绝不直接持有对方对象的数据。

---

**优势 1：请求逻辑可任意伸缩，物理显存零感知**

`ReqToTokenPool` 只管"第 N 号请求的第 i 个 token → 哪个物理 slot"的映射（一张 `[req_pool_idx, pos]` 的 int32 页表）。它：

- **不感知模型结构**：不知道 head_num、head_dim、是 MHA 还是 MLA；
- **不感知物理 dtype**：页表里只存 int32 索引，K/V 是 fp8/bf16/fp4 都与它无关；
- **不感知显存水位**：slot 够不够是 `TokenToKVPoolAllocator.alloc()` 返回 `None` 时才由上层决策阻塞。

于是 chunked prefill 的"同一请求跨多 chunk 复用一行页表"、continuous batching 的"请求随时进出"都只在逻辑层发生，物理层张量纹丝不动——只是页表里的某些行被追加、某些行被归还。

---

**优势 2：物理显存统一池化，碎片交给分配层专管**

`KVCache` 一次性预分配整池张量（`size + page_size`），之后**永不再 grow/shrink**，只原地读写。所有"哪块 slot 空闲、何时碎片整理、要不要排序"都被收进 `TokenToKVPoolAllocator`：

- `free_pages` / `release_pages` 两个 GPU tensor 管空闲与延迟释放；
- `need_sort` 控制是否攒一批再 `merge_and_sort_free()` 做碎片整理；
- per-token（`page_size=1`）与 per-page（`page_size>1`）是两个子类，分配策略不同但 `KVCache` 无感。

物理层因此可以专心做布局优化——MHA 的 NHD / AITER 5D SHUFFLE、MLA 的 latent 合一、DSA 的打包索引页——完全不被分配策略污染。

---

**优势 3：一套前缀缓存逻辑，复用三种物理布局**

RadixCache 的前缀共享、引用计数、淘汰策略，操作的全是"slot 索引"这一抽象：

```
树节点 value  ──→  一段物理 slot 索引序列  ──→  通过 allocator.free() 归还
                                          ──→  通过 KVCache.move_kv_cache() 搬迁
                                          ──→  通过 get_cpu_copy/load_cpu_copy 卸载/回灌
```

无论底层是 MHA 要搬"K+V 两 buffer"、MLA 要搬"单 kv_buffer"、还是 DSA 要搬"latent + index_k_with_scale"成对缓冲（L2618 的锁步搬迁），对 RadixCache 而言都只是"搬一段 loc"。这是 DSA/MLA 这类强约束架构能直接复用通用缓存体系的关键。

---

**优势 4：跨设备传输与计算路径互不阻塞**

三层分层让传输可以挂在每一层而不互相干扰：

| 传输场景 | 挂在哪一层 | 机制 |
|---|---|---|
| 逐层 KV 加载（disagg） | 物理层 | `layer_transfer_counter.wait_until(layer_id)`，`get_key_buffer` 同步（L1391） |
| 请求级 CPU offload | 物理层 | `get_cpu_copy` / `load_cpu_copy`，分块 8192（L1349） |
| 整池跨卡搬迁 | 物理层 | `data_ptrs` + `copy_all_layer_kv_cache_tiled` JIT kernel（L1208） |
| 索引 slot 复用 | 分配层 | `free_pages` 回收，下次 `alloc` 再切出 |
| 请求映射回收 | 逻辑层 | `ReqToTokenPool.free(req)` 归还行号 |

计算路径只依赖 `loc`，传输路径只改 `KVCache` 内部张量数据或 `allocator` 的空闲集合，二者在 forward 路径上天然解耦——这才有 alt_stream 异步写 KV（L1494）与 attention 计算重叠的可能。

---

**一句话总结**：逻辑层管"谁映射到哪"、分配层管"哪块空闲/何时整理"、物理层管"数据以什么布局落地"——三者用一张 int32 页表 + 一组 int64 空闲索引解耦，让请求逻辑、显存碎片、存储布局三件事各自独立演进，是 SGLang 能一套代码同时支撑 MHA/MLA/DSA/SWA/Hybrid 多种 KV 架构的架构根基。

### 1.3 `req_to_token` 统一页表机制（per-token 直接映射）

`req_to_token` 是三个池类互通的**中央页表**，承载 `(req_pool_idx, pos) → kv_slot` 的全部映射。它是 SGLang 与 vLLM **最根本的架构差异**之一，直接决定了 KV 分配/释放/读取的粒度与灵活性。

- 1.3.1 `page_size=1`：per-token 粒度的 slot 映射

默认 `page_size=1`（大多数 MHA 模型）下，`req_to_token` 是一张 `[max_batch, max_context_len]` 的 `int32` GPU 张量。其创建逻辑在 `ReqToTokenPool.__init__`（`memory_pool.py:244`）：

```python
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len),   # _alloc_size = size + 1
    dtype=torch.int32, device=device
)
```

**物理含义**：`req_to_token[3][127] = 2048` 表示"第 3 号请求的第 127 个 token，其 KV 存储在物理 slot #2048"。

**写入由 Triton kernel 驱动**。`write_req_to_token_pool_triton`（`triton_ops/common.py:9`）每个请求一个 block，分两步：

```python
@triton.jit
def write_req_to_token_pool_triton(req_to_token_ptr, req_pool_indices,
    prefix_tensors, pre_lens, seq_lens, extend_lens, out_cache_loc, ...):
    pid = tl.program_id(0)
    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # Step 1: 写入前缀命中段
    for i in range(tl.cdiv(pre_len, BLOCK_SIZE)):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
                 value, mask=mask)

    # Step 2: 写入新分配的 extend 段
    cumsum_start = 0  # prefix sum of extend_lens for prior requests
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)
    for i in range(tl.cdiv(seq_len - pre_len, BLOCK_SIZE)):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(req_to_token_ptr + req_pool_index * req_to_token_ptr_stride
                 + offset + pre_len, value, mask=mask)
```

两步写入：前缀部分从 `prefix_tensors[i]` 读出树缓存命中的 slot 序列（长度 `pre_len`），延申部分从 `out_cache_loc` 读出本轮新分配的 slot 段（长度 `extend_len`）。两步写完后，页表行 `[req_pool_index, 0:seq_len]` 完整覆盖该请求全部已确认 token 的 KV 物理位置。

**读取由 `get_last_loc` 驱动**。`get_last_loc_triton`（`triton_ops/common.py:144`）从页表取每请求最后一个 token 的物理位置：`req_to_token[req_pool_indices, prefix_lens - 1]`。decode 阶段用它找"当前页的最后一槽"，判断是否需要申请新页（`alloc_paged_token_slots_decode`）。HIP 上 Triton 的 `int32→int64` store 有 bug，所以 HIP 路径走 `get_last_loc_triton_safe`（L91）——中间存 int32 结果再统一 cast。

- 1.3.2 `page_size>1`：`PagedTokenToKVPoolAllocator` 隐式分页

`page_size>1` 时 `req_to_token` 的**格式不变**（仍是 `[req_pool_idx, pos] → kv_slot`），变化只发生在分配层：

| 方面 | `page_size=1` | `page_size>1` |
|---|---|---|
| allocator 粒度 | per-token slot | per-page（页编号 `page_num * page_size + offset` → slot） |
| `alloc` 返回 | slot 索引序列 | 页级展开后的 slot 索引序列 |
| `free` | 直接释放 slot | `torch.unique(free_index // page_size)` 去重后按页释放 |
| 分配 kernel | `allocator.alloc()` | `alloc_paged_token_slots_extend` 三段填充（已有页剩余 + 完整新页 + 最后不完整页） |
| 页表内容 | slot 是连续的 | slot 在 page 内连续，跨 page 不连续——但页表行统一存 slot，**不暴露 page 概念** |

核心设计：**页表行对 attention backend 永远是一段 `int32` slot 序列**，backend 不需要知道 page 边界。`PagedTokenToKVPoolAllocator` 在分配时做三段填充串成"逻辑连续、物理可能多次跨越 page 边界"的 slot 序列，写进页表后对 downstream 透明。这使得 flashinfer 等 backend 可以把 `page_size=1` 和 `>1` 统一处理——只需要每请求一个 `seq_len` 和对应的 slot 序列。

- 1.3.3 与 vLLM BlockTable 的架构差异对比

| 维度 | vLLM BlockTable | SGLang `req_to_token` |
|---|---|---|
| 粒度 | per-block（block_id → KV block） | **per-token**（pos → KV slot） |
| 形状 | `[max_batch, max_blocks]` | `[max_batch, max_context_len]` |
| block 概念 | 显式：block table 存 block id，kernel 内部 `block_id * block_size + offset` 算 token 位置 | 隐式：当 `page_size>1` 时在分配层做 page 映射再展开，页表本身不存 page |
| 跨请求前缀共享 | block table 是 per-request，无法共享 | `req_to_token` 各请求前缀段直接指向同一段 slot（RadixCache 管理） |
| 内存开销 | `O(batch × max_blocks)` | `O(batch × max_context_len)` |

SGLang 的 per-token 映射虽然有更高的页表存储开销（`max_batch × max_context_len × 4` 字节），但换来了**前缀共享的天然支持**：请求 A 和 B 共享 1000 token 前缀时，它们页表行前 1000 列的值完全相同，直接指向同一段物理 slot。vLLM 的 block table 无法做到这一点——block 是块级单位，块内 token 顺序固定，必须整块复制页表。

- 1.3.4 flashinfer 后端中 `req_to_token` → paged KV 格式的转换

flashinfer 的 attention kernel 要求 paged KV 格式输入：`paged_kv_indices`（一维 int32，把所有请求的 slot 序列按 `kv_indptr` 偏移拼接）和 `kv_indptr`（每请求起始偏移）。`create_flashinfer_kv_indices_triton`（`triton_ops/kv_indices.py:9`）把 `req_to_token` 转换为这种格式：

```python
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,      # [max_batch, max_context_len]
    req_pool_indices_ptr,  # [batch_size]
    page_kernel_lens_ptr,  # [batch_size] 每个请求需要读的 token 数
    kv_indptr,             # [batch_size+1] 累积偏移
    kv_start_idx,          # [batch_size] optional，SWA 窗口的起始偏移
    kv_indices_ptr,        # [total_tokens] 输出缓冲区
    req_to_token_ptr_stride: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)       # 本请求在输出中的起始位
    kv_start = tl.load(kv_start_idx + pid) if kv_start_idx else 0
    kv_end = kv_start + tl.load(page_kernel_lens_ptr + pid)

    for i in range(tl.cdiv(kv_end - kv_start, BLOCK_SIZE)):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(req_to_token_ptr + req_pool_index * req_to_token_ptr_stride
                       + kv_start + offset, mask=mask)
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
```

每个请求一个 block，从 `req_to_token[req_pool_index, kv_start:kv_end]` 逐行拷贝到 `kv_indices` 的对应偏移段。`kv_start_idx` 可选参数支持 SWA 的"仅读取窗口内 KV"——SWA 混合模型下 full attention 层取全量 token，SWA 层只取 `kv_start_idx` 到末尾的窗口内 token。这个转换是 zero-copy 的——只拷贝 int32 索引（每 token 4 字节），不碰 KV 数据本身。

### 1.4 KV Cache 精细化元数据体系

`Req` 对象（`schedule_batch.py:700`）承载了请求级 KV 管理的全部元数据，按功能分为四个维度。每个字段在 Prefill/Decode/释放/淘汰四个阶段被不同模块读写，构成 SGLang 请求状态机。

- 1.4.1 请求维度：`kv_committed_len`、`kv_allocated_len`、`req_pool_idx`、`priority`、`time_stats`

| 字段 | 类型 | 含义 | 读写时机 |
|---|---|---|---|
| `req_pool_idx` | `Optional[int]` | 在 `ReqToTokenPool.req_to_token` 中的行号（L788） | `alloc_req_slots` 分配，`release_kv_cache` 释放为 `None` |
| `kv_committed_len` | `int` | 已确认提交的 KV token 数——即页表行中有效 slot 数的下界（L738） | decode 每步 +1；chunk 结束时累加本 chunk 新 token；`pop_committed_kv_cache` 读取后置 `kv_committed_freed=True` |
| `kv_allocated_len` | `int` | 已分配的 KV token 数——包括尚未确认的 draft token slot（L739） | prefill 时 `= seq_len`；spec decode 时 `>= kv_committed_len`（含待验证 draft slots）；`pop_overallocated_kv_cache` 返回 `(committed, allocated)` 区间并把超出部分释放 |
| `priority` | `Optional[int]` | 淘汰优先级，传给 `PriorityStrategy`（L820） | 请求创建时设定，insert 进树时作为 `TreeNode.priority` |
| `time_stats` | `SchedulerReqTimeStats` | 全生命周期时间戳（L973） | 记录 queue/prefill/decode 各阶段耗时，用于可观测性 |

`kv_allocated_len - kv_committed_len` 就是该请求的"悬空 slot"——spec decode 中已被分配但尚未验证的 draft token slot。`release_kv_cache`（`common.py:645`）先调 `cache_finished_req` 把已确认 KV 插树，再调 `pop_overallocated_kv_cache` 释放悬空 slot，最后 `req_to_token_pool.free(req)` 归还行号。这一"确认→插树→释放悬空→归还行号"的四段释放确保不会有内存泄漏。

- 1.4.2 前缀缓存维度：`prefix_indices`、`num_matched_prefix_tokens`、`host_hit_length`、`cache_protected_len`

| 字段 | 类型 | 含义 | 来源 |
|---|---|---|---|
| `prefix_indices` | `torch.Tensor[int64]` | 设备侧命中前缀对应的 KV slot 序列（L845） | `match_prefix` 返回的 `MatchResult.device_indices`；chunked prefill 时每次 `cache_unfinished_req` 后刷新 |
| `last_node` | `Any` | 该请求当前持有锁的最深树节点（L847） | `match_prefix` 或 `insert` 返回的终止节点；用于 `inc_lock_ref` / `dec_lock_ref` 生命周期管理 |
| `num_matched_prefix_tokens` | `int` | 总缓存命中 token 数（L858） | `= len(prefix_indices) + host_hit_length`，用于调度器按"未命中 token 数"排序 batch |
| `host_hit_length` | `int` | L2 主机侧命中 token 数，HiCache 专用（L851） | `HiRadixCache.match_prefix` 返回的 `host_hit_length`；产生 `load_back` 操作从 CPU 回灌 GPU |
| `cache_protected_len` | `int` | 已插入树且在树里受保护的前缀长度（L866） | `cache_unfinished_req` 更新（见第 2 章 L542）——`page_size>1` 时可能 < `len(prefix_indices)`，因为 partial page 的 slot 在页表里但未进树，需在后续 chunk/结束释放 |
| `best_match_node` | `Any` | `match_prefix` 的完整匹配结果节点（L849） | 供 HiCache 的 `init_load_back` 锚定 load-back 来源 |

`cache_protected_len` 是"防泄漏"的关键。`page_size>1` 下，partial page 尾部的 slot 被写入页表供 attention 读取，但由于长度不满足 page 对齐，不能插入 RadixTree——它们被记到 `cache_protected_len` 的额外尾部，在下一次 `cache_unfinished_req` 和最终 `cache_finished_req` 中释放（见 `radix_cache.py:538-542` 的注释）。这个机制防止"页表里引用了但树里没记录"的 slot 永远无法回收。

- 1.4.3 SWA 维度：`swa_evicted_seqlen`、`sliding_window_size`

`swa_evicted_seqlen`（L751）追踪 SWA 池中已被逻辑淘汰的 KV 长度。它在两类 cache 下的行为不同（见 L747-750 注释）：
- **RadixCache**：`[cache_protected_len, swa_evicted_seqlen)` 的 KV 由 `ScheduleBatch.maybe_evict_swa` 手动释放；`[0, cache_protected_len)` 由 radix cache 淘汰时释放。
- **ChunkCache**：`[0, swa_evicted_seqlen)` 全部由 `maybe_evict_swa` 手动释放。

`free_swa_out_of_window_slots`（`common.py:68`）计算窗口外应淘汰量：`evict_threshold = pre_len - sliding_window_size - page_size`（保留至少一页 margin 供树存储非 tombstone 节点，L95），然后从 `req_to_token[req_pool_idx, swa_evicted_seqlen:new_swa_evicted_seqlen]` 取出 slot 调 `allocator.free_swa` 释放。SWA 池**只释放 slot 映射、不修改页表**（页表仍保留完整 slot 序列供 full attention 层读取），这是通过 `kv_start_idx` 在 flashinfer 转换 kernel 中裁剪窗口实现（1.3.4 节）。

- 1.4.4 生命周期维度：`kv_committed_freed`、`kv_overallocated_freed`、`inflight_middle_chunks`

| 字段 | 含义 | 保护机制 |
|---|---|---|
| `kv_committed_freed`（L740） | 已确认 KV 是否已释放 | `pop_committed_kv_cache` 断言 `kv_committed_freed` 为 False，释放后置 True——**防重复释放** |
| `kv_overallocated_freed`（L741） | 悬空 KV 是否已释放 | `pop_overallocated_kv_cache` 同理；spec decode 路径允许 committed < allocated，非 spec 路径断言二者相等（L677） |
| `inflight_middle_chunks`（L871） | chunked prefill 中的未完成 chunk 数 | 每新增一个 chunk +1，每处理完成一个 chunk -1；调度器据此判断"该请求还有未完成的 chunk，不能释放页表行" |
| `is_retracted` / `retracted_stain`（L874-876） | 请求是否被 retract（回退）/ 是否曾被 retract | retract 时把已分配但未确认的 decode slot 释放，`retracted_stain` 标记历史上被 retract 过，影响 scheduling priority |
| `extend_batch_idx` / `decode_batch_idx`（L754-755） | 当前 batch 中的索引 | overlap scheduler 用 `decode_batch_idx >= 1` 判断 decode 是否已脱离 extend 阶段，决定何时可以 evict SWA（`maybe_evict_swa` L2881）

### 1.5 Chunked Prefill：长文本分段与 `req_pool_idx` 复用机制

Chunked prefill 把长 prompt 按 `chunked_prefill_size`（`CacheInitParams` 中的参数，`cache_init_params.py:44`）切成多个 chunk 分批次 prefill，核心价值是避免单条长 prompt 阻塞调度、牺牲 TTFT。

**`req_pool_idx` 复用是 chunked prefill 的关键设计**。同一请求跨多 chunk 保持同一个 `req_pool_idx`（页表行），各 chunk 往同一页表行**追加写入**新 token 的 slot 映射：

```
Chunk 1: req_to_token[3, 0:1024]   ← prefix match 命中 800 token + 新分配 224 slot
Chunk 2: req_to_token[3, 1024:2048]← prefix match 命中 1024 token（含 Chunk1 已算）+ 新分配 1024 slot
Chunk 3: req_to_token[3, 2048:3072]← 同上，最终 seq_len = 第 1 章整体 extend 的长度
```

`ReqToTokenPool.alloc`（`memory_pool.py:270`）的第一行逻辑就是复用检测：

```python
def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
    reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
    assert all(
        reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
        for i in reusing
    ), "reusing request must be chunked or have committed KV"
    need_size = len(reqs) - len(reusing)
    ...
```

复用条件严格约束：`inflight_middle_chunks > 0`（中间 chunk，还有后续）或 `kv_committed_len > 0`（至少已有部分 KV 已落盘）——不允许空请求复用已分配的行号。

**chunk 间的 KV 持久化**：每 chunk 结束后调 `cache_unfinished_req`（`radix_cache.py:493`），把本 chunk 新算的 KV slot 插入 RadixTree 使之能被后续 chunk（或其他请求）命中。关键流程（已在第 2 章阐述，此处聚焦 chunked 视角）：

1. `cache_unfinished_req` 插入新 KV 到树 → 树返回更新后的 `new_indices`
2. `self.req_to_token_pool.write(...)` 把页表行中受保护段刷新为树返回的新 slot 映射（`radix_cache.py:533`）
3. `req.cache_protected_len = len(new_indices)` 更新受保护长度
4. `req.prefix_indices` 设为 `new_indices + 未进树的 tail`（`radix_cache.py:550-554`）
5. `inc_lock_ref(new_last_node)` 锁住新节点，`dec_lock_ref(req.last_node)` 释放旧节点

**Chunk 间交互的 `inflight_middle_chunks`**：每切一个新 chunk +1，每处理完一个 chunk -1。当 `inflight_middle_chunks == 0` 时该 chunk 是最后一个——此时 `cache_finished_req`（而非 `cache_unfinished_req`）被调用，把最终 KV 完整插树并释放行号。

Chunked prefill 的额外收益是**与其他请求的 batching**：调度器可以把多个请求的不同 chunk 混编进同一个 prefill batch，长 prompt 的中间 chunk 和短 prompt 的第一个 chunk 共享 GPU forward，提高 batch 利用率。

### 1.6 工程踩坑与源码细节

- 1.6.1 `_alloc_size = size + 1`：索引 0 的 CUDA graph padding 约定

`ReqToTokenPool` 的 `_alloc_size = size + 1`（`memory_pool.py:244`），`free_slots` 从 1 开始（`list(range(1, _alloc_size))`），索引 0 永远不被分配。`TokenToKVPoolAllocator` 同样：`free_pages = torch.arange(1, size + 1)`（`allocator/token.py:44`），slot 0 永远空闲。约定被多处依赖：

- **CUDA graph padding**：CUDA graph 的 batch 大小固定，但实际请求数可能不足——padded 空请求的 `req_pool_indices` 被填 0，attention kernel 读 `req_to_token[0, *]`（全零行），写 KV 也落在 slot 0（dummy 位置），不污染真实请求数据。
- **KV Canary 显式引用**：`jit_kernel/kv_canary/consts.py:8` 注释 `# Mirrors SGLang's ReqToTokenPool contract: req_pool_idx 0 is the CUDA-graph padding row`——外部工具也遵循此约定。
- **`maybe_detect_oob` 越界检测**（`memory_pool.py` 的 `set_kv_buffer` / `move_kv_cache`）：范围 check 上限是 `size + page_size`（不是 `size`），因为 `page_size` 个 padding slot 在多 page_size 场景还要额外容纳 page 对齐的空余。`loc == 0` 的写入永远合法——它是 padding slot。

这个约定让 CUDA graph 的 padded batch 与真实 KV 管理完全解耦——padding 请求不需要任何 `ReqToTokenPool` / `TokenToKVPoolAllocator` 分配，也不需要额外的 `if is_padding` 判断。

- 1.6.2 `need_sort` 与 `merge_and_sort_free`：碎片整理时机选择

`BaseTokenToKVPoolAllocator.need_sort`（`allocator/base.py:49`）控制 slot 释放策略：

```python
self.need_sort = need_sort          # 是否需要排序
self.free_pages = None              # 可用 slot 池（GPU int64 tensor）
self.release_pages = None           # 延迟释放队列
self.free_group = []                # 批量释放暂存
self.is_not_in_free_group = True
```

两阶段释放（`allocator/token.py:66`）：

```python
def free(self, free_index):
    if free_index.numel() == 0: return
    if self.is_not_in_free_group:
        if self.need_sort:
            self.release_pages = torch.cat((self.release_pages, free_index))  # 延迟
        else:
            self.free_pages = torch.cat((self.free_pages, free_index))         # 直接追加
    else:
        self.free_group.append(free_index)                                     # 攒批
```

两种模式的选择逻辑：
- **`need_sort=True`**（默认，前缀缓存命中率高的场景）：释放的 slot 先放进 `release_pages` 延迟队列，**只在 `alloc` 发现 `need_size > len(free_pages)` 时才执行** `merge_and_sort_free()`——把 `free_pages` 与 `release_pages` 合并后 `torch.sort`。排序保证大块连续分配、减少碎片。
- **`need_sort=False`**（命中率低、释放频繁的场景）：直接 `cat` 到 `free_pages` 尾部，免排序开销。

为什么延迟排序更优？前缀缓存被大量请求共享时，树节点淘汰不是逐 token、而是整段 slot 一起 free——这些释放是**批量的而且位置随机**。如果不排序，`free_pages` 很快变成乱序片段，大段 extend 分配时要在碎片化的空闲 set 中反复跳过。排序一次的成本 O(n log n) 远低于 n 次碎片化分配中每步扫描的隐性开销。延迟则把多个小释放攒成一次排序，摊还掉频繁 sort 的峰值 GPU/CPU 同步延迟。

批量释放 API `free_group_begin` / `free_group_end`（`base.py:69`）用于同一 batch 内释放多个请求 KV 的场景——先把所有 `free_index` 收集到 `self.free_group`，最后一次性 `torch.cat + free`，减少 `torch.cat` 调用次数（每次 cat 都是 GPU memory allocator 开销）。

## 第2章 SGLang RadixCache 基数树：全局前缀 KV 共享核心

### 2.1 RadixAttention 演进背景：PagedAttention 只能单请求复用

- 2.1.1 分页缓存只能「单请求复用」，无法「跨请求前缀复用」

vLLM 的 PagedAttention 解决的是**单请求内部**的显存碎片化：把一个请求的 KV 切成等大 block，用 block table 映射，使显存利用率接近 100%。但它的 block table 是 **per-request** 的——请求 A 和请求 B 即使共享完全相同的前 1000 个 system prompt token，各自的 KV cache 仍然各算一份、各存一份、各 prefill 一遍。KV 复用只发生在一个请求自己的 decode 期间，不跨越请求边界。

- 2.1.2 企业级固定系统 Prompt 场景 70% 以上 KV 计算冗余

真实服务里，大量请求共享同一份 system prompt（角色设定、工具描述、RAG 检索上下文、few-shot 示例）。这部分 prompt 动辄数千 token，其 KV 计算成本高昂却逐请求重复。SGLang 的 RadixAttention（源自 SOSP'24 论文）把"全局共享前缀 KV"作为一等公民：用一棵基数树（radix tree）记录所有已计算过的 token 序列到 KV slot 的映射，新请求**先查树、命中则直接复用**，只对命中点之后的新 token 计算 KV。

这就是 SGLang 相对 vLLM 的核心架构红利：把 PagedAttention 的"单请求 block 复用"升级为"全集群跨请求前缀复用"。

### 2.2 SGLang 中多种 Cache 类型的全景图

`python/sglang/srt/mem_cache/` 下并非只有一个 cache 类，而是一族，按是否多级存储、是否混合 SSM/SWA、是否分片组合区分。它们都实现 `PrefixCacheTrait` 协议（`base_prefix_cache.py:35`）：暴露 `req_to_token_pool` / `token_to_kv_pool_allocator` / `page_size` / `disable` 四个字段，以及 `match_prefix` / `insert` / `evict` / `inc_lock_ref` / `dec_lock_ref` 等统一方法。

| 类 | 文件:行号 | 定位 |
|---|---|---|
| `BasePrefixCache` | `base_prefix_cache.py:211` | 抽象基类，实现 `PrefixCacheTrait` 协议 |
| `RadixCache` | `radix_cache.py:286` | 基础基数树，GPU 单级 KV 共享 |
| `HiRadixCache` | `hiradix_cache.py:75` | 继承 `RadixCache`，叠加 L2/L3 多级存储（HostKVCache + storage backend） |
| `UnifiedRadixCache` | `unified_radix_cache.py:305` | SSM(Mamba) + Attention 混合模型统一缓存 |
| `SWARadixCache` | `swa_radix_cache.py:343` | 滑动窗口注意力，维护 SWA 尾部 + 全量双池 |
| `MambaRadixCache` | `mamba_radix_cache.py:421` | Mamba/SSM 状态缓存 |
| `HiMambaRadixCache` | `hi_mamba_radix_cache.py:97` | Mamba + HiCache 多级 |
| `ChunkCache` / `SWAChunkCache` | `chunk_cache.py:35`/`113` | chunked prefill 用的轻量缓存 |
| `SessionRadixCacheMixin` | `session_radix_cache.py:23` | 会话级缓存复用，作为 mixin 混入 `RadixCache` |

- 2.2.1 `RadixCache`（基础基数树缓存）

`RadixCache(SessionRadixCacheMixin, KVCacheEventMixin, BasePrefixCache)`（`radix_cache.py:286`）。构造接收 `CacheInitParams`（`cache_init_params.py:18`），从中取出三层引用：`req_to_token_pool` / `token_to_kv_pool_allocator` / `page_size`，以及 `is_eagle`（EAGLE 投机解码的 bigram 视图）、`eviction_policy`（默认 `"lru"`）。`reset()` 时创建根节点 `root_node`，其 `lock_ref=1` 永不淘汰、`priority=-sys.maxsize` 保证任何真实优先级都覆盖它。

- 2.2.2 `HiRadixCache`（带 HiCache 多级存储的基数树）

直接继承 `RadixCache`（`hiradix_cache.py:75`），在其基础上装配 `HostKVCache`（L2）和可选的 storage backend（L3）。它引入 `host_value` / `host_ref_counter` / `write_through_pending_id` 等主机侧字段，并托管 `HiCacheController` 做 write-through / write-back / load-back 调度。`write_through_threshold`（write_through=1，其余=2）和 `load_back_threshold=10` 控制下沉/回灌触发点。详见第 10 章。

- 2.2.3 `UnifiedRadixCache`（统一混合模型缓存，SSM + Attention）

`unified_radix_cache.py:305`。面向 Mamba/SSM + Attention 混合架构（如 Jamba），一棵树同时挂 Attention 的 KV slot 和 SSM 的 conv/state，`InsertParams` / `EvictParams` 都带 `mamba_*` 字段区分两类分量。

- 2.2.4 `SWARadixCache` / `MambaRadixCache` / `ChunkCache` / `SessionRadixCache`

`SWARadixCache` 处理滑动窗口：窗口外的 KV 被逻辑淘汰但 slot 未必立即释放，靠 `swa_evicted_seqlen` 追踪。`MambaRadixCache` 管理 SSM 的时序状态而非 KV。`ChunkCache` 是 chunked prefill 路径上的轻量替代。`SessionRadixCacheMixin` 给 `RadixCache` 增加会话级 `_tag_session_leaf` / `_discard_session_leaf`，实现多轮对话同一会话的 KV 跨轮复用。

### 2.3 `TreeNode` 核心源码字段全解析

`TreeNode`（`radix_cache.py:223`）是基数树的节点，承载一段 token 序列及其 KV slot。源码：

```python
class TreeNode:
    counter = 0
    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)      # child_key -> TreeNode
        self.parent: TreeNode = None
        self.key: RadixKey = None                  # 该节点承载的 token 序列
        self.value: Optional[torch.Tensor] = None  # 设备侧 KV slot 索引段
        self.lock_ref = 0                          # 引用计数（保护免被淘汰）
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        self.host_ref_counter = 0                  # 主机侧引用计数
        self.host_value = None                     # 主机侧 KV slot 索引段
        self.write_through_pending_id = None       # 待 write-through 的存储操作 id
        self.hash_value = None                     # 各 page 的 SHA256，用于存储寻址
        self.priority = priority                   # priority-aware 淘汰优先级
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
```

- 2.3.1 `key: RadixKey`（含 `token_ids` + `extra_key` + `is_bigram`）

`RadixKey`（`radix_cache.py:57`）用 `__slots__` 压成四个字段：`token_ids`（`array[int]`，原始 token）、`extra_key`（可选 str，**命名空间隔离**——不同 LoRA id / cache_salt 的 KV 即便 token 相同也分树存放，绝不共享，见 `match_prefix` docstring L363-372）、`is_bigram`（EAGLE 投机解码时启用，token_ids 存 N+1 个原始 token 表示 N 个 bigram，相邻 bigram 共享一个边界 token）、`limit`（O(1) 虚拟截断，避免切片拷贝）。

`match()`（L159）用**指数搜索 + 二分**找首个分歧 token——倍增窗口整段切片比较（C 层级），再在分歧窗口内二分，长公共前缀不退化为逐 token Python 循环。`page_aligned()`（L133）把 key 截到 `page_size` 整数倍，`child_key()`（L195）生成 `page_size` 个逻辑单元的可哈希键用于 `children` dict 查找。

- 2.3.2 `value` / `host_value`：设备侧与主机侧 KV 数据指针

`value` 是一个 `torch.Tensor`，存的是**这段 token 对应的物理 slot 索引序列**（int64），不是 K/V 数据本身——K/V 数据在第 1 章的 `KVCache` 物理池里。`value` 的长度 `== len(key)`。`host_value` 同理，是 L2 主机侧的 slot 索引，仅 `HiRadixCache` 使用。节点被淘汰时 `value` 置 `None`（`evicted` 属性 L252）；下沉到主机后 `host_value` 非空（`backuped` 属性 L256）。

- 2.3.3 `children` / `parent`：树拓扑结构

`children` 是 `defaultdict(TreeNode)`，键是 `child_key`（按 `page_size` 个逻辑单元命名空间化）。`parent` 反向指针。根节点 `parent=None`、`lock_ref=1`。基数树的性质：从根到任一节点的路径上各节点 key 拼接 = 该节点的完整 token 前缀；公共前缀在树里只存一遍。

- 2.3.4 `lock_ref` / `host_ref_counter`：双层引用计数防误删

`lock_ref` 是设备侧引用计数：请求正在使用某分支的 KV 时 `inc_lock_ref`，用完 `dec_lock_ref`。`lock_ref>0` 的节点及其祖先受保护，**不进 `evictable_leaves`**（见 L793 `_update_leaf_status`）。`host_ref_counter` 是主机侧引用计数，`protect_host()`/`release_host()`（L259/263）增减，保护 `host_value` 不被 L2 淘汰——存储操作（write-through/load-back）引用主机数据期间不允许回收。双层计数分别守护 GPU 与 CPU 两份数据的生命周期。

- 2.3.5 `last_access_time` / `hit_count` / `priority`：淘汰决策元数据

三者都是 `EvictionStrategy` 的输入：`last_access_time`（LRU/MRU 用）、`hit_count`（LFU/SLRU 用）、`priority`（PriorityStrategy 用，`_insert_helper` 里沿路径 `max` 传播 L723）。`__lt__`（L282）按 `last_access_time` 排序，供 `heapq` 优先队列。

- 2.3.6 `hash_value` / `write_through_pending_id` / `creation_time`

`hash_value: List[str]` 存该节点各 page 的 SHA256（`hash_page()` L207），作用是**存储后端寻址**——L3 backend 用前缀哈希链定位 KV 数据块，无需 token 内容。`write_through_pending_id` 标记该节点有一个尚未完成的 write-through 下沉操作，避免重复下发。`creation_time` 供 FIFO/FILO 策略。`split_node_hash_value`（`utils.py`）在节点分裂时正确切分哈希链。

### 2.4 最长公共前缀匹配算法流程

- 2.4.1 新请求从根节点逐层比对前缀 Token

`match_prefix()`（`radix_cache.py:360`）的流程：

```python
key = params.key
key, _ = key.maybe_to_bigram_view(self.is_eagle)   # EAGLE: 翻 bigram flag，O(1)
if self.disable or len(key) == 0:
    return self._empty_match_result
key = key.page_aligned(self.page_size)             # 截到 page 整数倍
if len(key) == 0:
    return self._empty_match_result
value, last_node = self._match_prefix_helper(self.root_node, key)
if value:
    value = torch.cat(value)
return MatchResult(device_indices=value, last_device_node=last_node, ...)
```

核心在 `_match_prefix_helper`（`radix_cache.py:653`）：从根节点出发，用 `key.child_key(page_size)` 查 `children` dict。

```python
def _match_prefix_helper(self, node, key):
    access_time = time.monotonic()
    node.last_access_time = access_time            # 刷新淘汰元数据
    child_key = key.child_key(self.page_size)
    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):            # 命中在 child 内部 → 分裂
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:                                       # 完整吃掉 child，继续往下
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
    return value, node
```

匹配过程中**同时刷新沿途节点的 `last_access_time`**，所以"查一次前缀"顺带让命中路径在 LRU 里变热。

- 2.4.2 完全匹配 / 部分匹配 / 零匹配三种分支处理逻辑

- **完全匹配**：`prefix_len == len(child.key)`，整个 child 被吃掉，`value` 累加 `child.value`，`key` 截掉已匹配前段，继续用新的 `child_key` 往下一层走。循环直到 `key` 耗尽。
- **部分匹配**：`prefix_len < len(child.key)`，命中点落在 child 内部。此时调 `_split_node` **把 child 一分为二**：前 `prefix_len` 个 token 成为新节点 `new_node`（继承 value 前段），剩余成为原 child。匹配到此结束，返回 `new_node`。这是基数树的"分裂暴露精确边界"。
- **零匹配**：`child_key not in node.children`，循环不进入，直接返回当前累计的 `value`（可能为空）与当前 `node`。`value` 为空时返回 `_empty_match_result`（以 root_node 为终止节点）。

### 2.5 树节点分裂、新建、挂载、EvictionPolicy 多策略淘汰

- 2.5.1 前缀部分重合触发节点分裂

`_split_node`（`radix_cache.py:679`）把一个节点切成"公共前缀 + 私有后缀"父子两层：

```python
def _split_node(self, key, child, split_len):
    new_node = TreeNode(priority=child.priority)     # 新节点继承优先级（代表公共前缀）
    new_node.hit_count = child.hit_count
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref               # 继承引用计数
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone() # 注意 clone，避免与原 child 共享存储
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node
    new_node.hash_value, child.hash_value = split_node_hash_value(...)
    return new_node
```

注意 `value` 用 `.clone()` 切分——`TreeNode.value` 是独立 tensor，分裂后父子各自持有自己的 slot 段拷贝，物理 slot 不动（K/V 数据没被复制，只是索引被重新划分）。这让后续两个不同后缀的分支可以独立淘汰/复用。

- 2.5.2 后缀增量生成叶子节点延伸与 Insert 挂载

`insert()`（`radix_cache.py:420`）调 `_insert_helper`（L709）。它先沿已有前缀走（必要时同样分裂），走到 key 剩余非空且无对应 child 时，**新建叶子挂载**：

```python
# _insert_helper 末尾（L749）
if len(key):
    new_node = TreeNode(priority=priority)
    new_node.parent = node
    new_node.key = key
    new_node.value = value.clone()
    self._inc_hit_count(new_node, chunked)
    node.children[child_key] = new_node
    self.evictable_size_ += len(key)               # 新增可淘汰量
    self._update_leaf_status(node)
    self._update_leaf_status(new_node)
    self._record_store_event(new_node)
    node = new_node
```

chunked prefill 请求会跳过 `hit_count` 自增（`_inc_hit_count` L701：`if chunked: return`），避免一个请求在自己的多个 chunk 间自我引用吹高热度。

- 2.5.3 `evict_policy.py`：可插拔淘汰策略（不限于 LRU+LFU）

`evict_policy.py` 定义抽象基类 `EvictionStrategy`（L10），`get_priority(node)` 返回可比较的值（越小越先淘汰）。注册表在 `utils.py:55`：

```python
_EVICTION_POLICY_FACTORIES = {
    "lru": LRUStrategy,      # last_access_time
    "lfu": LFUStrategy,      # (hit_count, last_access_time)
    "fifo": FIFOStrategy,    # creation_time
    "mru": MRUStrategy,      # -last_access_time
    "filo": FILOStrategy,    # -creation_time
    "priority": PriorityStrategy,  # (priority, last_access_time) 低优先级先淘汰
    "slru": SLRUStrategy,    # (segment, last_access_time)，hit>=阈值的进 Protected 段
}
```

`get_eviction_strategy(policy)`（`utils.py:66`）按名查表实例化。`SLRUStrategy`（L49）仿 S3 FIFO/分段 LRU：`hit_count < protected_threshold`(默认2) 的进 Probationary 段、≥ 的进 Protected 段，元组比较保证 Protected 段整体晚于 Probationary 段被淘汰。策略可插拔：新增策略只需继承 `EvictionStrategy` 并注册进工厂表。

- 2.5.4 `EvictParams` / `EvictResult`：逐出控制与后置处理

`evict()`（`radix_cache.py:568`）用小顶堆按 `eviction_strategy.get_priority(node)` 排序 `evictable_leaves`，逐个 pop：

```python
def evict(self, params: EvictParams) -> EvictResult:
    num_tokens = params.num_tokens
    leaves = list(self.evictable_leaves)
    eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
    heapq.heapify(eviction_heap)
    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)   # ★ slot 归还到分配层
        num_evicted += len(x.value)
        self._delete_leaf(x)
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(eviction_heap, (self.eviction_strategy.get_priority(x.parent), x.parent))
        self._record_remove_event(x)
    return EvictResult(num_tokens_evicted=num_evicted)
```

关键链路：**树节点 `value`（slot 索引段）→ `token_to_kv_pool_allocator.free()` → slot 回到 `free_pages`**。叶子被删后，若父节点因此变成无子叶且无锁，它也被推进堆继续淘汰——这实现了"整条失效分支回收到根"。`EvictParams` 还带 `swa_num_tokens` / `mamba_num` 供 SWA/Mamba 池分别控制。

### 2.6 RadixCache 与 `req_to_token` + `token_to_kv_pool_allocator` 三者关系

- 2.6.1 树节点 value → 物理 slot 索引 → `allocator.free()` 释放链路

三者职责再次明确：

```
RadixCache (逻辑共享层)      ← 用 TreeNode.value 存"token 序列 → slot 段"映射，做前缀复用/淘汰
   │ value 是 int64 slot 索引 tensor
   ▼
TokenToKVPoolAllocator(分配层) ← free(value) 把 slot 还回 free_pages；alloc() 切出新 slot
   │
   ▼
KVCache (物理层)              ← 真正的 K/V 显存，set_kv_buffer / get_key_buffer 读写
```

`cache_finished_req`（`radix_cache.py:442`）是最完整的链路示例：请求结束时，把 `req_to_token_pool.req_to_token[req_pool_idx, :kv_committed_len]` 读出作为该请求的全部 slot，构造成 `RadixKey` 插入树（`insert` 带走引用），**重复部分的 slot 立即 `free`**（`kv_indices[cache_protected_len:result.prefix_len]`，L475），未对齐尾部也 `free`（L485）。插入树的那段 slot 的所有权从"请求"转移给"树节点"。

- 2.6.2 `PrefixCacheTrait` 协议：`req_to_token_pool` + `token_to_kv_pool_allocator` + `page_size`

`PrefixCacheTrait`（`base_prefix_cache.py:35`）是个 `Protocol`，约束所有 cache 类必须暴露这三个字段：

```python
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool
```

这让上层调度代码可以**面向协议编程**——无论下游是 `RadixCache` / `HiRadixCache` / `UnifiedRadixCache`，都通过同一组 `match_prefix` / `insert` / `evict` / `inc_lock_ref` 接口操作，三类内部布局差异（SWA 双池、Mamba 状态、HiCache 多级）被封装在各自实现里。

### 2.7 引用计数 RC 联动机制：`lock_ref` + `host_ref_counter`

- 2.7.1 `IncLockRefResult` / `DecLockRefParams`：引用计数增减 API

`inc_lock_ref`（`radix_cache.py:597`）从某节点**沿父指针一路加到根**，`dec_lock_ref`（L612）对称地一路减：

```python
def inc_lock_ref(self, node):
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:                     # 0→1：脱离可淘汰
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        self._update_leaf_status(node)
        node = node.parent
    return IncLockRefResult(delta=delta)

def dec_lock_ref(self, node, params=None):
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 1:                     # 1→0：变回可淘汰
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
            delta += len(node.key)
        node.lock_ref -= 1
        self._update_leaf_status(node)
        node = node.parent
    return DecLockRefResult(delta=delta)
```

`IncLockRefResult`（`base_prefix_cache.py:102`）除 `delta`（protected 配额变化量，供上层调整水位）外，还带 `swa_uuid_for_lock` / `skip_lock_node_ids` 等 SWA/HiCache 专用字段，并通过 `to_dec_params()` 生成配对的 `DecLockRefParams`——**保证 inc 和 dec 的对象集合严格对应**，防止 lock 泄漏。

- 2.7.2 计数归零触发分支销毁与 slot 回收

`lock_ref` 从 1 降到 0 时（`dec_lock_ref` 内 `if node.lock_ref == 1` 分支），节点从 `protected_size_` 转入 `evictable_size_`，并由 `_update_leaf_status`（L793）重新评估是否进 `evictable_leaves`：

```python
def _update_leaf_status(self, node):
    if node.evicted or node.lock_ref > 0:          # 被淘汰过 或 加锁 → 不可淘汰
        if node in self.evictable_leaves: self.evictable_leaves.remove(node)
        return
    for child in node.children.values():           # 还有未淘汰子节点 → 不是叶
        if not child.evicted:
            if node in self.evictable_leaves: self.evictable_leaves.remove(node)
            return
    if node not in self.evictable_leaves:          # 真正的"可淘汰叶"
        self.evictable_leaves.add(node)
```

只有"非淘汰、未加锁、且无未淘汰子节点"的节点才是可淘汰叶。计数归零让节点**有资格被淘汰**，但真正释放 slot 发生在后续 `evict()` 被调度触发时（显存超水位）——`evict` 调 `allocator.free(x.value)` 才归还物理 slot，并 `_delete_leaf` 把节点从树里摘除。这是"标记可淘汰"与"实际回收"的两段式设计，避免在 hot path 同步释放。

请求生命周期里 RC 的典型用法见 `cache_unfinished_req`（L493）：每个 chunk 结束 `dec_lock_ref(req.last_node)` 释放上一段、`inc_lock_ref(new_last_node)` 锁住新匹配到的节点，确保正在使用的分支不会被淘汰。

### 2.8 线上问题：前缀失效、树内存泄漏、共享缓存脏数据

- **前缀失效**：`extra_key` 命名空间用错会导致本该隔离的 KV 被共享。例如多 LoRA 服务若忘记传 `lora_id` 作 extra_key，不同 adapter 的请求会命中同一前缀节点，读到错误的 KV。`match_prefix` docstring（L363-372）明确强调 extra_key 不同的条目必须分树。

- **树内存泄漏**：`cache_protected_len` 机制（L538-542）专为 `page_size>1` 设计——partial page 的 slot 被加进 `req.prefix_indices` 但**未进树**，必须在下一次 `cache_unfinished_req` 和最终 `cache_finished_req` 里释放，否则泄漏。`write_through_pending_id` 未正确清理则会让主机侧数据无法回收。

- **共享缓存脏数据**：DSA 池的 `index_k_with_scale_buffer`（第 1 章 1.2.3）在 retract 释放页后被别的请求复用，若 `get_cpu_copy` 没同步卸载 index 缓存，resume 时会恢复 latent 却留下别家的 index/scale（`memory_pool.py:2704-2709` 注释）。`ReqToTokenPool` 索引 0 padding 约定、`maybe_detect_oob` 越界检查都是防 stale slot id 写入造成静默脏数据。

- **并发安全**：`RadixCache` 通过调度器串行化访问（单 scheduler 线程）、`HiRadixCache` 的 `ongoing_write_through`/`ongoing_load_back` 字典追踪异步操作，避免 match/evict 与后台传输竞争。分布式场景下 `HiRadixCache` 用 `_all_reduce_attn_groups` / `_pp_sync` 在 TP/PP 组间同步淘汰决策。


# 第二部分：KV Cache 完整生命周期原理
## 第3章 KV Cache 生成机制：Prefill / Decode 双阶段

生成（generation）指"为新 token 分配物理 KV slot 并把映射写入页表"这一阶段——K/V 的实际数值由 attention kernel 在 forward 中算出后写入第 4 章的 `set_kv_buffer`，而本章管的是"在哪分配槽位、怎么登记进页表"。入口全部在 `python/sglang/srt/mem_cache/common.py`。

### 3.1 Prefill 预填充全量生成逻辑

- 3.1.1 张量并行批量 Prefill 计算流程图解

一个 prefill/extend batch 的 KV 生成链路：

```
Scheduler.run_batch (extend)
  │
  ├─ alloc_for_extend(batch)                         common.py:456
  │    ├─ batch.maybe_evict_swa()                    先回收 SWA 窗口外 slot
  │    ├─ alloc_req_slots(...)                       common.py:410  分配 req_pool_idx 行号
  │    │     └─ req_to_token_pool.alloc(reqs)
  │    ├─ alloc_token_slots / alloc_paged_token_slots_extend   分配物理 slot → out_cache_loc
  │    │     └─ 内部先 evict_from_tree_cache(...)    common.py:300  淘汰凑够
  │    │           └─ tree_cache.evict(EvictParams(...))
  │    └─ write_cache_indices(...)                   common.py:122  Triton kernel 填页表
  │           └─ write_req_to_token_pool_triton      把 prefix + extend slot 写入 req_to_token
  │
  ▼
forward() → 每层 attention 算出 K/V → set_kv_buffer(layer, loc=out_cache_loc, k, v)  落地
```

prefill 是"批量、连续段"分配：一个 extend batch 把所有请求要算的新 token 一次性算出 `extend_num_tokens`，统一向 allocator 要一段连续 slot 段 `out_cache_loc`，再按请求拆开写进各自页表行。

- 3.1.2 `alloc_for_extend()`：KV slot 分配 + `write_cache_indices()` 写入页表

`alloc_for_extend`（`common.py:456`）的核心步骤：

```python
def alloc_for_extend(batch):
    batch.maybe_evict_swa()                       # 回收 SWA 窗口外 slot（第6章）
    prefix_tensors = [r.prefix_indices for r in batch.reqs]   # 前缀命中复用的 slot
    # ... 构造 prefix_lens / extend_lens 的 cpu+device 张量 ...

    req_pool_indices = alloc_req_slots(...)       # ① 分配请求行号
    if _alloc_page_size(batch) == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)  # ②a per-token
    else:
        out_cache_loc = alloc_paged_token_slots_extend(...)    # ②b per-page（三段填充）
    write_cache_indices(out_cache_loc, ..., batch.req_to_token_pool)  # ③ 填页表
    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu
```

三步对应三层：①`ReqToTokenPool.alloc`（逻辑层行号）→ ②`allocator.alloc`（分配层 slot）→ ③`write_cache_indices`（写页表）。`alloc_token_slots`（L272）内部先 `evict_from_tree_cache` 把树缓存淘汰到腾出足够空闲，再 `allocator.alloc(num_tokens)`，`None` 则抛 `Out of memory`。

`write_cache_indices`（`common.py:122`）支持两条路径：attention backend 支持 triton 时用 `write_req_to_token_pool_triton` kernel 一次性批量写入（把每个请求的 `prefix_tensors[i]` 指针表上送 GPU），否则循环 `req_to_token_pool.write` 逐请求写——前者写 `[req_idx, 0:prefix_len]` = prefix slot、`[req_idx, prefix_len:seq_len]` = 新分配的 extend slot。

per-page 路径 `alloc_paged_token_slots_extend` 用"三段填充法"：已有页剩余空位 + 完整新页 + 最后不完整页，串成一段逻辑无缝但物理不连续的 slot 序列，正是第 1 章 1.2.2 提到的页分配器行为。

- 3.1.3 前缀命中分支：`match_prefix()` → 跳过重复计算，复用历史树节点

`prefix_tensors = [r.prefix_indices for r in batch.reqs]` 里的 `prefix_indices` 来自第 2 章 RadixCache 的 `match_prefix` 结果——它返回命中前缀对应的物理 slot 序列（`MatchResult.device_indices`）。这部分 slot **不再重新分配、不再重新计算 KV**：它们在页表里直接复用树节点 `value` 指向的已存 slot，只为命中点之后的 `extend_len` 个新 token 走 ② 分配。命中越高，`extend_num_tokens` 越小，prefill 计算量越省——这是 RadixCache 的直接收益。`cache_unfinished_req`（第 2 章 2.6）在 chunk 边界把新算的 KV 插回树，使后续 chunk/请求能继续命中。

### 3.2 Decode 增量单 Token 生成

- 3.2.1 单步 forward 产出 KV 切片

decode 阶段每个请求每步只生成 1 个新 token（投机解码除外），KV 生成量小但频率高。`alloc_for_decode`（`common.py:590`）为整个 batch 的所有请求各分配 `token_per_req` 个新 slot：

```python
def alloc_for_decode(batch, token_per_req):
    batch.maybe_evict_swa()
    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]
    if _alloc_page_size(batch) == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)  # 整批一次性
    else:
        last_loc = batch.req_to_token_pool.req_to_token[batch.req_pool_indices, seq_lens_gpu - 1]
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(...)   # per-page decode kernel
    locs = seq_lens_gpu.clone()   # encoder-decoder 则 batch.encoder_lens + seq_lens_gpu
    batch.req_to_token_pool.write((batch.req_pool_indices, locs), out_cache_loc.to(torch.int32))
    return out_cache_loc
```

decode 的页表写入是**追加**：`locs = seq_lens_gpu`（当前已确认长度位置），把新 slot 写到页表行末尾。per-page 路径用 `alloc_decode_kernel`：每个请求看当前最后一页 `last_loc` 还有没有空位，有就续用、没有才申请新页。

- 3.2.2 `alloc_decode()`：每 token 追加一个 slot，`kv_committed_len += 1`

物理 slot 分配由 `allocator.alloc_decode`（`allocator/paged.py`，第 1 章 1.2.2）完成。投机解码下每步要分配多个 slot：`get_alloc_len_per_decode`（`common.py:213`）算出每请求每步的预留量——非 spec 为 1，EAGLE spec 为 `max(spec_steps * spec_topk, num_draft_tokens)`；`page_size>1 且 topk>1` 的 spec v2 树还要按最坏页对齐足迹估算（L238-241）。`get_alloc_reserve_per_decode`（L244）再 ×2 做 double-buffer 吸收 overlap 模式下 `kv_committed_len` 的滞后（`eagle_prepare_for_decode`）。每步 forward 后调度器把确认接受的 token 数累加进 `req.kv_committed_len`，被拒的草稿 token 对应 slot 随即回收——这就是"每 token 追加一个 slot，committed_len 递增"的精确含义。

### 3.3 惰性显存分配策略：不预占显存、随用随分

`KVCache` 物理池在服务启动时一次性 `torch.zeros` 预分配满 `size + page_size` 槽位的显存（第 1 章），这是**池级预占**。但**请求级是惰性的**：请求到来才 `alloc` 取 slot，结束就 `free` 归还，永不为"可能到来的请求"预先占位。`alloc_token_slots` 里 `allocator.alloc` 返回 `None` 时不会静默失败，而是先 `evict_from_tree_cache` 触发淘汰凑空间，凑不够才抛 OOM——显存始终在"已分配给在跑请求 + 树缓存里的历史前缀 + 空闲"三态间动态流转，没有为未来预扣。`need_sort`/延迟释放（第 1 章 1.2.2）让 free 的 slot 不立即排序，攒批再整理，进一步贴合"随用随分、延迟整理"的惰性风格。

### 3.4 多并行架构下 KV 分片生成：TP 按 head 切分 / PP 按 layer 隔离

- **TP（张量并行）**：同一请求的 KV 按注意力头切分到各 TP rank。`MHATokenToKVPool` 的 `head_num` 是本 rank 的本地头数（第 1 章），每个 rank 只存自己那部分头的 K/V。`req_to_token` 页表是 per-rank 的，slot 索引指向本 rank 物理池。attention kernel 的 TP AllGather 在 Q 维度做，KV 各 rank 独立读写本地池。MLA 因 `head_num=1`（latent 共享），TP 下 latent 按 `kv_lora_rank` 切分。
- **PP（流水线并行）**：各 PP rank 只持有自己那一段 layer 的 KV。`KVCache.start_layer`/`end_layer`（第 1 章 1.2.3 基类）标注本 rank 负责的层范围，`get_key_buffer(layer_id)` 用 `layer_id - start_layer` 索引本地 buffer。请求跨 PP stage 时 KV 不迁移——每个 stage 各算各的层，激活值在 stage 间传递，KV 留在本地池。
- **EP（专家并行）**：MoE 下专家层是 FFN 不产生 KV，attention 层的 KV 分布同 TP；专家路由在 FFN 内部，不影响 attention 的 KV 分片（第 8 章 8.7 详述）。

`CacheInitParams`（第 2 章 2.2）带的 `tp_cache_group` / `pp_cache_group` / `attn_cp_cache_group` 用于跨组同步淘汰决策——`HiRadixCache._all_reduce_attn_groups` / `_pp_sync` 保证各 rank 的树缓存淘汰一致。

### 3.5 [GLM-5.2 适配] DSA 稀疏注意力的 token_mask 选择性 KV 生成

GLM-5.2 若采用 DSA（Dense-Sparse Attention，深求 V3.2 同源架构），稀疏层的 KV 生成有两点特殊性，映射到 SGLang 已有基础设施：

1. **latent KV 用 `DSATokenToKVPool`**（第 1 章 1.2.3）：稀疏层的潜在 KV 仍按 MLA 方式存单 latent（`kv_lora_rank + qk_rope_head_dim`），生成路径同标准 MLA，`set_mla_kv_buffer` 写入。

2. **索引 K 只对"有效 token"生成**：DSA 用一份 `index_k_with_scale_buffer`（打包 fp8 K + scale）支撑 Top-K 路由。稀疏层并非对所有历史 token 维护索引 KV，而是按稀疏 mask 选择的 token 子集。SGLang 侧的 `set_index_k_scale_buffer`（`memory_pool.py:2692`）只对传入的 `loc` 写索引，天然支持"选择性生成"——只需让调度器把稀疏命中的 token 位置作为 `loc` 传入即可，无需为全量 token 生成索引 KV。这把 DSA 的稀疏性直接转化为索引 KV 的存储/生成节省。

适配要点：GLM-5.2 的 attention backend 需在 forward 中产出稀疏 mask，按 mask 收集要生成索引的 token 位置，再调 `set_index_k_scale_buffer`。latent KV 的生成与标准 MLA 一致，可零改动复用。这一节属架构推演——SGLang 主线已有 DSA 物理池与索引读写 API，GLM-5.2 的差异主要在 backend 如何决定稀疏 token 集合，而非物理存储层。


## 第4章 KV Cache 写入机制：显存固化与数据落地

写入（write）指把 attention kernel 算出的 K/V 数值真正落到物理池显存里。第 3 章解决了"分配哪个 slot、登记进页表"，本章解决"把张量数据写进那个 slot 对应的显存"。核心入口是 `KVCache.set_kv_buffer`，源码在第 1 章 1.2.3 已展开，本章聚焦写入的链路、语义与并发。

### 4.1 GPU 原地零拷贝写入主路径：`KVCache.set_kv_buffer()`

- 4.1.1 `KVWriteLoc`：full pool + SWA pool 的二元写入目标

`KVWriteLoc`（`memory_pool.py:959`）是一个打包结构，把"完整 pool 的写位置"和"SWA pool 的预翻译位置"绑在一起：

```python
class KVWriteLoc:
    """Write target(s) for KVCache.set_kv_buffer.
    loc is the full-pool write location; swa_loc is the pre-translated
    full->SWA location for hybrid SWA pools (None otherwise)."""
    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None

def unwrap_write_loc(loc_info):
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc
    return loc_info, None
```

普通模型 backend 只传一个裸 `loc` tensor；Hybrid SWA 模型（`SWATokenToKVPoolAllocator`，第 1 章 1.2.2）的 full pool 和 SWA pool 是两个独立物理池，slot 索引不通用，backend 需要同时知道"写到 full 池哪"和"写到 SWA 池哪"。`KVWriteLoc` 让 backend 无论什么池类型都只发一次 `set_kv_buffer` 调用，由 `unwrap_write_loc`（L971）拆成 `(loc, swa_loc)`——后者交由 `SWAKVPool` 的写入路径处理。这是"二元写入目标"的设计意图：对调用方屏蔽池类型差异。

- 4.1.2 CUDA kernel 直接写入 Block 显存，张量视图复用，无 `clone/copy`

`MHATokenToKVPool.set_kv_buffer`（`memory_pool.py:1409`）的写入不做任何数据拷贝中间体，而是直接把 `cache_k`/`cache_v` 写进 `k_buffer[layer_id - start_layer][loc]` 的现有显存视图。三条路径：

```python
def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, ...):
    loc, _ = unwrap_write_loc(loc_info)
    maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")  # 越界防护
    ...
    if dcp_kv_mask is not None:            # 路径1: context parallel masked 写
        masked_set_kv_buffer_kernel[(N,)](...)
        return
    if self.kv_cache_layout == "vectorized_5d":   # 路径2: AITER SHUFFLE 5D
        launch_reshape_and_cache_shuffle_5d(cache_k, cache_v, self.k_buffer[...], self.v_buffer[...], loc)
        return
    _set_kv_buffer_impl(cache_k, cache_v, self.k_buffer[...], self.v_buffer[...], loc,
                        row_dim=..., store_dtype=..., device_module=...,
                        size_limit=self.size + self.page_size,
                        alt_stream=self.alt_stream,        # 异步写流
                        same_kv_dim=self.same_kv_dim)      # K/V 等维优化
```

要点：
- **dtype 视图复用**：fp8 存储时 `store_dtype=uint8`，`cache_k = cache_k.view(self.store_dtype)` 后直接写，不重新分配（L1436-1438）。`_get_key_buffer` 读出时再 `view(self.dtype)` 还原。
- **`alt_stream` 异步写**：`enable_alt_stream=True`（CUDA）时 KV 写入走独立 CUDA stream（L1151），与 attention 计算流重叠，避免写 KV 阻塞下一步计算。
- **`same_kv_dim` 优化**：当 `head_dim == v_head_dim`，K/V 写入 kernel 可合并特化，省一次 kernel 调度。
- **越界防护**：`maybe_detect_oob`（受 `SGLANG_ENABLE_ASYNC_ASSERT` 控制）在写前校验 `loc` 范围，把"stale slot id 导致的静默 KV 损坏/非法地址"变成可定位的断言。

MLA 的写入（`MLATokenToKVPool.set_kv_buffer` L2243）更简单——单 `kv_buffer`，`self.kv_buffer[layer_id-start_layer][loc] = cache_k` 直接索引赋值；`set_mla_kv_buffer`（L2272）分 nope/rope 两段并可走 fp8 量化。DSA 的索引写入 `set_index_k_scale_buffer`（L2692）通过 `index_buf_accessor.SetKAndS.execute` 融合写 K+scale 到打包页。

### 4.2 跨设备写入链路：`get_cpu_copy()` / `load_cpu_copy()` 同步 offload

这是"把 GPU 显存 KV 写到 CPU 内存"的反向链路，用于 CPU offload 式 disagg 与 HiCache 下沉。`MHATokenToKVPool.get_cpu_copy`（`memory_pool.py:1346`）：

```python
def get_cpu_copy(self, indices, mamba_indices=None):
    current_platform.synchronize()
    kv_cache_cpu = []
    chunk_size = self.cpu_offloading_chunk_size   # 8192
    for layer_id in range(self.layer_num):
        kv_cache_cpu.append([])
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i : i + chunk_size]
            k_cpu = self.k_buffer[layer_id][chunk_indices].to("cpu", non_blocking=True)
            v_cpu = self.v_buffer[layer_id][chunk_indices].to("cpu", non_blocking=True)
            kv_cache_cpu[-1].append([k_cpu, v_cpu])
    current_platform.synchronize()
    return kv_cache_cpu
```

按 `cpu_offloading_chunk_size=8192` 分块、`non_blocking=True` 异步 D2H、首尾 `synchronize`。分块是为控制单次 DMA 的 pinned memory 占用与峰值显存波动。`load_cpu_copy`（L1364）对称地 H2D 回写 `self.k_buffer[layer_id][chunk_indices] = k_chunk`。

DSA 重写了 `get_cpu_copy`（`memory_pool.py:2704`）返回 `{"kv":..., "index_k":...}` 字典——latent KV 走 `super().get_cpu_copy`，索引页按 `page_indices = indices[::page_size] // page_size` 转 page 索引后单独 offload。注释（L2704-2709）强调：retract 释放的页会被别的请求 `set_index_k_scale_buffer` 复用，若不同步 offload 索引缓存，resume 时会恢复 latent 却留下别家的 index/scale → DSA 注意力读到错位垃圾。这是 DSA 双缓冲一致性的写入侧保障。

### 4.3 追加写入（Decode 增量）vs 覆盖写入（上下文重置/窗口刷新）

- **追加写入（decode）**：每步 decode 的 `loc` 是页表行末尾的新 slot（第 3 章 3.2），`set_kv_buffer` 写的是**从未被写过的空 slot**，语义上是 append。K/V 只增不改，历史 slot 数据保留供后续 attention 全序列读取。
- **覆盖写入（上下文重置/窗口刷新）**：发生在 chunked prefill 回退、SWA 窗口滚动、KV 被 evict 后重新 load_back 等场景。此时 `loc` 指向的 slot 可能已有旧数据，`set_kv_buffer` 直接 `k_buffer[..][loc] = cache_k` 覆盖。因为 `loc` 是 allocator 重新分配的（evict 过的 slot 已 `free` 归 `free_pages`，重新 `alloc` 才会用到），覆盖的是"逻辑上已失效"的旧数据，不破坏正在被引用的 KV。
- **SWA 窗口截断写入**：SWA 池只保留滑动窗口内的 KV，窗口外的 slot 由 `maybe_evict_swa`（`common.py` 第 3 章开头调）回收。窗口滚动时新 token 写入窗口尾，旧 token 的 slot 被释放——这既非纯 append 也非覆盖，而是"环形窗口"语义。`SWATokenToKVPoolAllocator` 的 full/SWA 双池使窗口截断只影响 SWA 池，full 池仍保留全量用于 full attention 层。

### 4.4 Batch 并发写入锁竞争：RadixCache 全局读写锁与 Block 空闲池竞争规避

- **RadixCache 访问串行化**：SGLang 单 scheduler 线程模型下，`match_prefix`/`insert`/`evict`/`inc_lock_ref` 在同一线程内顺序执行，天然无并发竞争，不需要显式 `tree_lock`。`HiRadixCache` 引入后台 write-through/load-back 异步任务后，用 `ongoing_write_through` / `ongoing_load_back` 字典追踪在途操作，节点级 `host_ref_counter`（第 2 章 2.3）保护主机数据不被异步传输与淘汰同时触碰。
- **Block 空闲池竞争规避**：`TokenToKVPoolAllocator` 的 `free_pages` 是单一 GPU tensor，alloc 从头切、free 追加尾（或延迟队列），同一 batch 内一次 `alloc` 切出整段 slot，不存在多请求争抢同一 free 槽位。`free_group_begin/end`（第 1 章 1.2.2）把一批释放攒成一次 `torch.cat` + `free`，减少碎片化操作频率。`maybe_evict_swa` 在 alloc 前先把 SWA 窗口外 slot 回收，避免 alloc 时才发现不够再回退淘汰的反复。
- **写入流与计算流分离**：`alt_stream` 让 KV 写入与下一层 attention 计算在不同 CUDA stream 上并发，从 GPU 执行层面规避"写 KV 阻塞读 KV"的锁等待。

### 4.5 [GLM-5.2 适配] 超长文本分段写入与 SWA 滑动窗口截断写入

GLM-5.2 面向超长上下文，写入侧有两点适配方向，均映射到 SGLang 已有机制：

1. **超长文本分段写入**：长上下文必然走 chunked prefill，每个 chunk 的 KV 由 `cache_unfinished_req` 写入树（第 2 章 2.6）后再由后续 chunk 命中复用。写入端的关键是 `set_kv_buffer_prefix_valid`（`memory_pool.py:1498`）——它按 `commit_lens` 只写"本 chunk 确认提交"的部分 token 的 KV，draft/未确认部分不落盘，避免 chunk 边界回退时写入脏 KV。GLM-5.2 的长上下文只需正确设置 chunk_size 与 commit 语义，写入路径零改动复用。

2. **SWA 滑动窗口截断写入**：若 GLM-5.2 部分层采用 SWA 控制长上下文的 KV 总量，写入侧由 `SWAKVPool`（`kv_cache_layout="nhd"`，第 1 章 1.2.3 注释 L1119-1121 指出 SWA sub-pool 强制 legacy NHD 布局）承接。`KVWriteLoc.swa_loc` 让 backend 一次调用同时写 full 池和 SWA 池；窗口滚动时 `maybe_evict_swa` 释放窗口外 SWA slot。GLM-5.2 适配只需在模型 config 标注哪些层是 SWA 层、配置 `sliding_window_size`，物理写入层自动按 full/SWA 双池分别落地。

综合：GLM-5.2 超长上下文的写入不要求新物理池，复用 MHA（Dense 层）+ SWA（滑窗层）的既有双池写入路径与 `set_kv_buffer_prefix_valid` 的分段提交语义即可。若叠加 DSA，再走 `DSATokenToKVPool` 的 latent + 索引页双写路径（4.2 节所述一致性保障）。


## 第5章 KV Cache 读取机制：Attention 计算核心链路

读取（read）指 attention kernel 在执行 Q×K^T 运算时，用页表把"token→物理 slot"的映射转换成 GPU kernel 能直接索引的 paged KV 地址，再从 `KVCache.get_key_buffer/get_value_buffer` 读出的 K/V 张量上进行 attention 计算。源码入口：`python/sglang/srt/layers/attention/` 下的 flashinfer/triton attention backend，页表转换 kernel 在 `python/sglang/srt/layers/attention/triton_ops/kv_indices.py`。

### 5.1 标准读取全流程：Query 计算 → `req_to_token` 查页表 → `create_flashinfer_kv_indices_triton` 转换为 paged KV → Attention kernel

一次 decode batch 的读取链路：

```
forward_batch.req_pool_indices → FlashInferIndicesUpdaterDecode.update()
  │
  ├─ [1] create_flashinfer_kv_indices_triton          kv_indices.py:9
  │       从 req_to_token[req_pool_idx, 0:seq_len] 逐请求拉出 slot 序列
  │       写入 kv_indices buffer（一维 int32/int64 连续数组）
  │
  ├─ [2] flashinfer wrapper.begin_forward()
  │       绑定 kv_indices + page_table_layout → paged KV buffer (k_buffer/v_buffer 的 data_ptr)
  │
  └─ [3] flashinfer decode kernel
          每 thread block 从 kv_indices 读 slot → 从 K/V buffer 读对应行 → 算 attention
```

Pre-前言：forward metadata 在 `init_forward_metadata`（`flashinfer_backend.py:739`）阶段构建——`seq_lens`、`kv_indptr`（前缀和指针数组表示每个请求 paged KV 段的起始偏移）、`kv_last_page_len`（最后一个 page 的实际长度）、`kv_indices_buf`（CUDA graph 专用固定大小 buffer）。这些 metadata 在后续 IndicesUpdater 与 flashinfer wrapper 之间流转。

### 5.2 `IndicesUpdater`：`req_to_token` → flashinfer `paged_kv_indices` 的 Triton 转换 kernel

`FlashInferIndicesUpdaterDecode`（`flashinfer_backend.py:1141`）是 decode 读取的网关。核心字段：

```python
self.req_to_token = model_runner.req_to_token_pool.req_to_token  # GPU 页表
self.kv_indptr = attn_backend.kv_indptr                           # 请求段前缀和
self.kv_last_page_len = attn_backend.kv_last_page_len
```

`update_single_wrapper`（L1186）调用 `call_begin_forward` → 内部 `create_flashinfer_kv_indices_triton[(bs,)]`：

```python
# kv_indices.py:9
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,         # [max_batch, max_context_len] GPU 页表
    req_pool_indices_ptr,      # [bs] 本 batch 各请求的 req_pool_idx
    page_kernel_lens_ptr,      # [bs] 各请求 seq_len（需读取的 token 数）
    kv_indptr,                 # [bs+1] 前缀和：请求 i 的 KV 段从 kv_indptr[i] 开始
    kv_start_idx,              # [bs] 可选起始偏移（chunked prefill 时用）
    kv_indices_ptr,            # [total_tokens] 输出：平铺的 slot 索引序列
    req_to_token_ptr_stride,
):
    pid = tl.program_id(axis=0)           # 每个请求一个 CT
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)
    kv_start = tl.load(kv_start_idx + pid) if kv_start_idx else 0
    kv_end = kv_start + tl.load(page_kernel_lens_ptr + pid)

    for i in range(num_loop):              # 每 512 token 一批
        offset = tl.arange(0, 512).to(tl.int64) + i * 512
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + kv_start + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
```

核心算法：**per-request one CTA，从 `req_to_token` 行按 seq_len 做 gather → scatter 到 `kv_indices` 紧凑数组**。`kv_indptr` 是前缀和定位数组——请求 i 的 KV 序列从 `kv_indices[kv_indptr[i]]` 开始连续存放。这是从"per-request 行式页表"到"per-kernel 紧凑 flat 索引"的形态转换，flashinfer decode kernel 只接受紧凑 flat 格式。

MLA 版本 `create_flashmla_kv_indices_triton`（kv_indices.py:99）输出 `/ PAGED_SIZE`（L150）——MLA 内部存 slot 也是按 token 粒度，但 flashMLA kernel 要求 page 编号而非 slot 编号，此处做除法转换。`get_num_page_per_block_flashmla` / `get_num_kv_index_blocks_flashmla`（L84/89）算 page block 布局。

### 5.3 零拷贝读取、in-place 视图复用、预取优化

- **零拷贝**：`get_key_buffer(layer_id)`（`memory_pool.py:1387`）返回的是 `k_buffer[layer_id - start_layer]` 的**直接视图**——与物理池是同一块显存，不做 clone/contiguous/reshape 拷贝。`store_dtype != dtype` 时只做 `.view(dtype)` 类型转换，同样零拷贝。`data_ptrs` 直接存各层 buffer 的指针，传入 flashinfer wrapper 的 `begin_forward` 时也就是传指针——kernel 内部 `load` 直接命中显存对应地址。

- **CUDA graph 固定 buffer**：`flashinfer_backend.py:836-861` 的 `cuda_graph_kv_indices` 是一组固定大小的 `kv_indices` 缓冲，在 graph 捕获时一次性分配，后续 replay 时不重新分配——免去每次 batch 的 `torch.zeros`/`torch.empty` 开销。`kv_indices` 的 `copy_` 只在 graph 需要更新时发生。

- **alt_stream 读/写重叠**：第 4 章 4.1 的 `alt_stream` 用于写 KV 到物理池，读 KV 的 attention kernel 在 default stream 上跑——两流并行：forward 输出一边写到 KV cache（alt_stream），下一层 attention 一边读 KV cache（default stream），硬件级别重叠读写延迟。

- **预取**：flashinfer wrapper 的 `begin_forward` 内预取 `k_buffer` / `v_buffer` 的 data_ptr 并绑定 page table layout，CUDA kernel launch overhead 被 reduce 到每 batch 一次 wrapper begin 而非每 request 一次。

### 5.4 多卡跨分片 KV 聚合读取：TP AllGather / EP 路由分发

- **TP（张量并行）**：KV 按 head 切分在各 TP rank 本地。attention kernel 每个 rank 只读自己本地 head 的 K/V 计算部分注意力，Q 维度通过 AllGather 或 replicated Q 同步。`get_key_buffer` 返回的 `head_num` 是本地头数（第 1 章），kernel 只对本地 K/V 做 batch matmul，Q×K^T 的结果需要 AllReduce 汇总。
- **EP（专家并行）**：MoE 层（FFN）不产生 KV，attention 层的 KV 分布同 TP。EP 路由在 MoE FFN 内进行，attention 读取路径不变。
- **CP（Context Parallel）**：`attn_cp_cache_group` 标定 context parallel 组，KV 按序列长度切分到各 CP rank。`dcp_kv_mask` 在第 4 章 set_kv_buffer 的写入路径用 mask 分片写；读取时 `dcp_size > 1` 可能替换 allocator 为 `PagedTokenToKVPoolAllocator`（`_alloc_page_size` common.py:444 逻辑），`kv_indices` 通过 CP group 做 partial gather。

### 5.5 多级缓存命中分支：L1 (GPU) 直接命中 / L2 (CPU HostKVCache) → `load_cpu_copy` / L3 (Storage Backend) → `PrefetchOperation`

读取比写入多了一条"可能需要回灌"的路径——KV 数据可能在 L2/L3 而不是 L1：

```
L1 (GPU KVCache)    ← get_key_buffer → 直接读，零延迟
L2 (HostKVCache)    ← HiRadixCache._init_host_load_back → load_cpu_copy(indices) 回灌 GPU
L3 (Storage Backend) ← PrefetchOperation 预取 → L2 → load_cpu_copy
```

`HiRadixCache`（第 10 章详述）在 `match_prefix` 后若发现匹配节点有 `host_value` 而 `value` 为空（数据在 L2），触发 `init_load_back` → `cache_controller.load_back_backend` 按 `host_value` 指定的 slot 段将 CPU 数据 `load_cpu_copy` 回灌 GPU 后 `set_kv_buffer` 复写——之后读取路径同 L1。`HostKVCache`（`pool_host/base.py`）用 `pin_memory` + DMA 做 H2D 拷贝，`LayerDoneCounter` 做逐层等待（第 7 章 7.2.3）。

L3 命中时 `cache_controller` 在后台发出 `PrefetchOperation`——先 H2D host→GPU 再 GPU 写入，`PrefetchTimeoutConfig`（第 10 章 10.4）在超时后直接跳过预取让请求重算 KV。

### 5.6 [GLM-5.2 适配] RoPE 位置偏移修正与稀疏 Token 精准读取

1. **RoPE 位置偏移**：GLM-5.2 继承 ChatGLM 系列的多阶段 RoPE（双位置 ID），在 KV cache 复用场景下，被命中的前缀 KV 的 RoPE 位置与当前请求的实际位置存在 **偏移**——前缀部分以它在树中首次插入时的原始位置编码，当前生成部分的 RoPE 位置从 `prefix_len` 起算。SGLang 的 `RadixStorage` 只存 slot 索引不存位置 ID，偏移修正由 attention backend 在 forward 时根据 `cache_protected_len`（前缀长度）动态计算位置偏移量，无需改动 KV cache 物理层。

2. **稀疏 Token 精准读取（DSA）**：DSA 的 sparse 层只对 Top-K 选中的 token 做 attention，其余 token 不读。读取时 `DSATokenToKVPool.get_index_k_with_scale_buffer` 取出索引 K+scale，由 DSA kernel 计算 Top-K 后**仅对被选中的 token 从 kv_buffer 读取 latent KV**。关键是索引 K 的 read 和 latent KV 的 read 是两阶段：先极快地扫 fp8 索引挑出 Top-K，再精准读那 K 个 token 的 latent。这比全量读所有 token 的 KV 做 attention 节省数倍显存带宽——DSA 的读取路径天然是稀疏的。

GLM-5.2 适配要点：RoPE 位置偏移量需在 attention backend 中按 `cache_protected_len` 动态调整；DSA 读取两阶段已在 `DSATokenToKVPool` 的 index accessor API 中实现（`get_index_k_scale_buffer` 融合取 K+scale → kernel → 按 mask 取 latent），backend 侧复用即改。



## 第6章 KV Cache 淘汰与内存回收机制

淘汰（eviction）是"把树缓存中不再需要或优先级最低的 KV 前缀逐出、将其物理 slot 归还给 allocator"的系统级反压机制。不是每个请求结束时才回收——那叫"释放"（release）。淘汰发生在显存不足时才触发，是 SGLang 保证长服务永远不 OOM 的关键。

### 6.1 显存水位线分级管控：软阈值降级 Swap / 硬阈值强制淘汰

SGLang 的淘汰触发链是**按需触发**，不是定时轮询：

```
alloc_token_slots(need_size)                          common.py:272
  ├─ evict_from_tree_cache(tree_cache, need_size)     common.py:300
  │    └─ allocator.available_size() < need_size?
  │         └─ tree_cache.evict(EvictParams(num_tokens=need_size - available))
  └─ allocator.alloc(need_size)
       └─ if None → Out of memory (淘汰后仍不够)
```

`evict_from_tree_cache`（`common.py:300`）计算出缺口：`num_tokens_missing = need_size - allocator.available_size()`，然后调 `tree_cache.evict(EvictParams(num_tokens=num_tokens_missing))`。这就是"硬阈值强制淘汰"——不够就逐出凑够，逐出完后还是不够 → 抛 OOM。

对于 Hybrid SWA 池，`evict_from_tree_cache` 同时检查 full 与 SWA 两个 allocator 的可用量（L309-318），向 `evict` 传 `swa_num_tokens` 额外参数。逐出过程本身通过 `evict` 方法（`radix_cache.py:568`）落地，Section 2.5.4 已详述——小顶堆排序 `evictable_leaves`、逐个 free 后 `_delete_leaf`、父节点若无子叶且 lock_ref=0 也进堆继续逐出。

HiCache 体系（第 10 章）增加**软阈值降级**：在显存未到硬阈值但接近时，先把低频节点写穿到 L2/L3 存储层，写入成功后把该节点的 `value` 清空（`backuped` 为 True）并标记 `host_value`，下次需要时 `load_back` 回灌——这就是"软阈值 Swap"比 evict 轻量之处：数据未丢，只是从 GPU 搬到了主机。

### 6.2 `evict_policy.py`：可插拔淘汰策略（LRU / LFU / SLRU / FIFO）

策略定义在 `evict_policy.py`，注册表在 `utils.py:55`。第 2 章 2.5.3 已列全策略与其 `get_priority` 返回值。于此强调几点工程细节：

- **LRU 是默认**：`CacheInitParams.eviction_policy = "lru"`（`cache_init_params.py:29`）。对于大多数服务场景，LRU 行为最直观——最久未用的系统提示词才被淘汰，hot 对话 KV 永远留在树里。
- **SLRU 分段保护**：hit_count≥2 就进 Protected 段，无论访问时间都晚于 Probationary 段被淘汰。适合有固定 system prompt + 可变 user prompt 的场景——system prompt 命中高自动进保护段，user prompt 随用随蒸。
- **Priority 策略**：`(priority, last_access_time)` 元组，低优先级先蒸发。适配 QoS 级别区分——VIP 请求设高 priority，其前缀节点在整条路径上 `max` 传播后（`_insert_helper` L723），整批前缀优先于普通请求存活。
- **可插拔性**：新增策略只需继承 `EvictionStrategy` 实现 `get_priority`，再在 `_EVICTION_POLICY_FACTORIES` 注册名字。`get_eviction_strategy(policy)` 按名查表实例化，`evict` 方法直接 `self.eviction_strategy.get_priority(node)` 使用，不感知策略具体类型。

### 6.3 [GLM-5.2 适配] SWA 窗口外 KV 强制过期 + DSA 无效 Token KV 主动释放

- **SWA 窗口外 KV 强制过期**

`free_swa_out_of_window_slots`（`common.py:68`）是 SWA 层淘汰的精确入口。它判断当前 token 位置 `pre_len` 超过滑动窗口 `sliding_window_size` 后，把窗口外的 slot 强制释放：

```python
evict_threshold = pre_len - sliding_window_size - page_size  # 默认保留一个 page margin
new_swa_evicted_seqlen = max(req.swa_evicted_seqlen, evict_threshold)
if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
    free_slots = req_to_token_pool.req_to_token[
        req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
    ]
    token_to_kv_pool_allocator.free_swa(free_slots)
    req.swa_evicted_seqlen = new_swa_evicted_seqlen
```

关键设计点：`swa_evicted_seqlen` 是惰性推进——每步 decode/extend 时才检查是否需要滚窗，窗口超出量一次释放。默认保留一个 `page_size` margin（`drop_page_margin=False` 不丢弃 margin，L92-95），确保始终有一页 SWA KV 在滑动窗口外保留在树里作为非 tombstone 节点，防止多轮对话场景下的 SWA 内存泄漏（与 `swa_radix_cache.py` 的 `_insert_helper` case 3 联动，注释见 L88-91）。

`maybe_evict_swa`（`schedule_batch.py:2864`）在 batch 级控制触发：decode 模式下记录 `swa_maintenance_step` 按 `SGLANG_SWA_EVICTION_INTERVAL`（默认 1）控制频率，避免每步都查；overlap 模式下 req 的 `decode_batch_idx>=1` 才触发（确保前一个 extend batch 已完成）。同时有一个优化路径 `SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW`：decode 位置滑出窗口后把 SWA 部分的树锁降级（`dec_swa_lock_only` L2899），让 SWA evictable 叶子可在 LRU 压力下被回收。

- **DSA 无效 Token KV 主动释放**

DSA 稀疏注意力会在 sparse 层只保留部分"有效 token"的 KV（由 Top-K 路由选择），其余 token 对该层无注意力贡献。这对淘汰的影响是：DSA 的 `index_k_with_scale_buffer` 按页存索引 K+scale，evict 时不仅要释放 latent `kv_buffer` 的页，还要释放对应索引页——`move_kv_cache`（`memory_pool.py:2618`）与 `get_cpu_copy`（L2704）的锁步搬迁/卸载已保障这一点。SGLang 目前的淘汰按 `TreeNode.value` 的 slot 段粒度进行（一段对应整页），DSA 下的正确性要求是：**被淘汰的页同时在 kv_buffer 和 index_k_with_scale_buffer 两层里被一致释放**——`DSATokenToKVPool` 继承 MLA 的 `kv_buffer`（第 1 章）并在 `_clear_buffers` 中同时删除两个 buffer（L2614-2616），但单 token 粒度的淘汰精准性有赖于调度层按 page 对齐处理。

GLM-5.2 的适配方向：若采用 DSA，稀疏层按 token_mask 维护有效 token 集合，淘汰策略应优先逐出"不在此 mask 内且不在当前窗口内"的 token KV，最大化稀疏层利用率。现有基础设施已支撑——`free_swa_out_of_window_slots` 与 `evict` 的组合可满足窗口 + LRU 双重筛选。

### 6.4 细粒度 slot 回收 vs 粗粒度整会话回收

- **细粒度 slot 回收**：`TokenToKVPoolAllocator.free(free_index)`（第 1 章 1.2.2）按任意 slot 索引集释放，不要求连续、也无需整页。一个请求结束只释放属于它的 slot，其他请求的 slot 毫发无损。RadixCache 淘汰也按节点 `value` 的 slot 段为单位，一段对应一次 `free(x.value)`。

- **粗粒度整会话回收**：`release_kv_cache`（`common.py:645`）是一次请求完成后的全链路回收：

```python
def release_kv_cache(req, tree_cache, is_insert=True):
    tree_cache.cache_finished_req(req, is_insert=is_insert)
    # cache_finished_req: 把 committed KV 插入树（带 priority），释放重复+未对齐尾部
    if req.req_pool_idx is None: return
    start_p, end_p = req.pop_overallocated_kv_cache()  # 投机解码可能分配多了
    # ...
    if start_p < end_p:
        indices_to_free = req_to_token_pool.req_to_token[req_pool_idx][start_p:end_p]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)  # 回收 over-allocate
    tree_cache.req_to_token_pool.free(req)  # 归还 req_pool_idx 行号
```

链路分三步：①已确认 KV 交给树（插入带走所有权，重复部分释放）；②over-allocate 部分（投机解码拒绝的 draft token 对应 slot）由 `pop_overallocated_kv_cache` 读出并 free；③`ReqToTokenPool.free` 归还页表行号。三步覆盖了"请求的 committed + draft + 页表"三重资源，保证请求结束后不泄漏任何 slot/行号。

- **SessionRadixCache 的多轮回收**：`SessionRadixCacheMixin`（`session_radix_cache.py:23`）在 `cache_finished_req` 中通过 `_tag_session_leaf` 将会话级叶子挂钩 `req.session_id`，超时或被 `_discard_session_leaf` 清理，但不立即淘汰——它只是**标记**，实际 evict 仍由显存压力触发。

### 6.5 双层 RC 防误删：`TreeNode.lock_ref` + `host_ref_counter`

第 2 章 2.7 已详述 `inc_lock_ref`/`dec_lock_ref` 沿 parent 路径遍历 + `_update_leaf_status` 维护 `evictable_leaves` + `evictable_size_/protected_size_` 配额的机制。此处补充两个防误删保障：

1. **请求级 lock 保证正在使用的 KV 不被逐出**：`cache_unfinished_req`（第 2 章 2.6 末尾）中的 `dec_lock_ref(req.last_node)` → `inc_lock_ref(new_last_node)` 确保请求的活跃节点永远 `lock_ref>0`→在 `_update_leaf_status` 的 L794 分支被跳过→不进 `evictable_leaves`→不被 evict。

2. **host_ref_counter 保证后台传输不读脏数据**：HiCache 的 write-through 异步线程把 KV 从 GPU 写到 L2/L3，整个过程节点的主机引用 `host_ref_counter>0`（`TreeNode.protect_host()` / `release_host()`，L259/263）。`host_ref_counter>0` 虽然不阻止 evict，但 HiCache 的 `cache_controller` 在 write-through 期间持有 `host_value` 的所有权，后台流不完成不释放——这就是第 4 章 4.4 的"并发安全：异步传输与同步淘汰不互撞"。

3. **radix_tree_node ref 和 allocator slot ref 的双向独立**：`TreeNode.lock_ref` 管的是"这个节点代表的 token 序列是否被引用"，管不了物理 slot 的分配。两者通过 `evict` 方法桥接：`lock_ref` 归零 → 节点进 `evictable_leaves` → 淘汰压力下 `evict` 调用 `allocator.free(x.value)` 归还 slot。"引用的粒度"是 token 序列段，"分配/释放的粒度"是物理 slot——两个系统通过节点 value 对 slot 的持有权串联。

### 6.6 淘汰后置：树分支修剪、`req_to_token` 索引刷新、slot 归还 `free_pages`、HiCache 下沉传输标记

一次 evict 操作（`radix_cache.py:568`）的后置：

1. **树分支修剪**：`_delete_leaf`（L782）把节点从 `parent.children` 中摘除，`evictable_size_ -= len(node.key)`，并从 `evictable_leaves` 移除。若父节点因所有子节点被逐出而变"空壳"，在 `evict` 的 loop 中被推回堆继续逐出（L588-590）——实现"失效分支整条回收"。

2. **slot 归还**：`self.token_to_kv_pool_allocator.free(x.value)`（L584）是逐出落地的物理操作——这行代码把 `TreeNode.value` 储存的 slot 索引段还给 `free_pages` tensor（或 `release_pages` 延迟队列），`num_evicted += len(x.value)` 计数。物理层 `KVCache` 完全无感——它只管张量布局，不知道 slot 已经"可用"。

3. **`req_to_token` 索引刷新**：被逐出的节点所对应的 token KV 已经无效，但**页表的首部（prefix match 部分的 slot 索引来自该节点 `value`）不会被自动清零**。这是因为调度器在每个新的 forward 前会重新 `match_prefix`——新版匹配结果会覆盖页表。若节点被逐出后请求还没重新匹配，旧 slot 索引残留在页表里意味着请求读到 stale slot 的脏数据——调度器必须保证在逐出后任何引用该节点 slot 的请求已重新走 `match_prefix→write_cache_indices` 流，或在逐出前把请求的 `last_node` 移到 `root_node`。

4. **HiCache 下沉传输标记**：`HiRadixCache` 的淘汰策略可选"先落盘再逐出"——`HiCacheController` 在淘汰前检查 `TreeNode` 是否有可落盘的 `host_value` 或应写回的存储后端标记。`write_through_pending_id` 追踪在途写穿操作；`cache_controller` 的 `PrefetchTimeoutConfig`（第 10 章 10.4）控制后台下沉的超时策略，超时则放弃落盘直接 evict。

5. **事件记录**：`self._record_remove_event(x)`（L592）记录淘汰事件，供 metrics 和 debug trace 用（KVCacheEventMixin 提供）。

总结链路：释放操作按粒度从细到粗依次是"free_swa_out_of_window_slot 逐 token 释放窗口外 → evict 按节点段逐出 → release_kv_cache 整请求回收 → cache reset 全清"。RC 双层保护（请求 lock + host 后台上锁）和延迟释放排序让这条链路在并发运行时保持 KV 不脏、不丢、不泄漏。



# 第三部分：专项模块——KV Cache 跨设备传输机制
## 第7章 KV Cache 跨设备传输体系设计

前六章都在 GPU 单卡内讨论 KV 存储、分配、读写、淘汰。本章处理 KV 数据**离开 GPU 显存**的一切路径。源码分布在 `python/sglang/srt/mem_cache/memory_pool.py`（GPU↔CPU offload）、`python/sglang/srt/disaggregation/`（PD 分离传输）、`python/sglang/srt/mem_cache/hicache_storage.py` 与 `pool_host/` 和 `storage/`（HiCache 多级）。

### 7.1 传输场景全景：SGLang 中实际存在的三类传输

- 7.1.1 GPU↔CPU 请求级 Offload：`Req.offload_kv_cache()` / `load_kv_cache()`

单个请求的 KV 从 GPU 搬移到 CPU 内存，对应 `KVCache.get_cpu_copy()` / `load_cpu_copy()`（`memory_pool.py:1346/1364`）。用于两个场景：（1）`TorchMemorySaverAdapter` 的显存压缩——请求排队期间暂时 offload 到 CPU 释放 GPU 压力；（2）PD 分离式 decode 端的 KV 暂存——prefill 生成后把 KV 卸载到 CPU，decode 端按需加载。

- 7.1.2 Prefill→Decode 分离式传输（Disaggregation PD）：NCCL / NIXL 跨节点

prefill 节点生成 KV → 传输给 decode 节点消费。核心在 `python/sglang/srt/disaggregation/` 目录。传输路径有三种：NCCL AllReduce/AllGather（同机多卡）、NIXL（跨机点对点）、mooncake RDMA（跨机零拷贝）。`DecodeReqToTokenPool` 为传输预分配 slot 缓存。

- 7.1.3 HiCache 层级传输：HostKVCache ↔ Storage Backend 的后台数据流转

L2（HostKVCache）↔ L3（Storage Backend）的优先级升降级与后台上传下发，由 `HiCacheController` 和 `CacheController` 管理（第 10 章详述）。

### 7.2 GPU↔CPU Offload 详细链路

- 7.2.1 `KVCache.get_cpu_copy()` / `load_cpu_copy()`：同步 D2H / H2D 拷贝

第 4 章 4.2 已详细分析源码，此处提三个要点：
- **分块大小 8192**：受 `cpu_offloading_chunk_size`（`KVCache.__init__` L1009）控制，控制单次 DMA 的 pinned memory 峰值。
- **按层遍历外层、按 chunk 遍历内层**：外层 `for layer_id` 保证整层数据连续传输，内层 `for chunk` 控制显存峰值。
- **DSA 双缓冲**：`DSATokenToKVPool.get_cpu_copy`（L2704）返回 `{"kv":..., "index_k":...}`，索引页单独 offload 防止 resume 读脏。

- 7.2.2 `TorchMemorySaverAdapter`：显存压缩与 Memory Saver 机制

`TorchMemorySaverAdapter.create(enable=enable_memory_saver)`（`KVCache.__init__` L1003）按 `enable_memory_saver` 开关控制 `_create_buffers` 在专属显存区域内分配，使 offload 时能安全释放批量显存块。

- 7.2.3 `LayerDoneCounter`：layer-wise 传输控制（`register_layer_transfer_counter`）

`KVCache.register_layer_transfer_counter(counter)`（L1061）注册一个逐层完成的计数器，`get_key_buffer` 和 `get_value_buffer` 在返回 buffer 前调用 `counter.wait_until(layer_id - start_layer)`（L1391）同步等待。这使 disagg 场景下的**逐层 KV 加载**成为可能——decode 端起 layer 0 的 KV 传输完成即可开始 attention，无需等全部 layer 传输完毕。`LayerDoneCounter` 本身定义在 `python/sglang/srt/utils/layer_transfer_counter.py`。

### 7.3 Disaggregation PD 传输链路

- 7.3.1 `DecodeReqToTokenPool`：预分配 slot + 传输 slot 的分离池设计

`DecodeReqToTokenPool`（`memory_pool.py` 的 alloc 分支，`model_runner_kv_cache_mixin.py` 的条件路由）为 decode 端预分配 `req_pool_idx`，与 prefill 端的 slot 分开——传输完的 KV 直接落到预分配 slot，decode 端无需重新分配。

- 7.3.2 NCCL 集合通信 vs NIXL 点对点传输 vs RDMA 零拷贝

| 方式 | 文件/类 | 场景 | 特点 |
|---|---|---|---|
| NCCL | 内置 `torch.distributed` | 同机多卡 | 集合通信，AllReduce/AllGather |
| NIXL | `disaggregation/nixl/` | 跨机点对点 | 基于 libfabric 的点对点 |
| mooncake RDMA | `disaggregation/mooncake/` 或 `mooncake_store` | 跨机零拷贝 | 走定制 memory pool、绕过 CPU |

`custom_mem_pool`（`KVCache.__init__` L1015）为 mooncake 等定制后端提供独立显存池，传输直达 GPU 显存不经过 CPU bounce buffer。

- 7.3.3 `kv_cache_builder.py`：KV 数据序列化与反序列化

`python/sglang/srt/mem_cache/kv_cache_builder.py` 定义跨节点传输时的 KV 打包格式和序列化/反序列化原语，配合 `disaggregation/utils.py` 的传输位姿计算。

- 7.3.4 SWA allocator 的 `alloc_extend_swa_tail`：decode 端仅传输 SWA 尾部

SWA 层只需滑动窗口内的 KV，decode 端 `SWATokenToKVPoolAllocator.alloc_extend_swa_tail` 仅分配窗口内所需 slot，减少传输量。

### 7.4 HiCache 层级传输：HostKVCache ↔ Storage Backend

- 7.4.1 `PoolTransfer` / `PoolName`：多池类型的传输抽象

在 `pool_host/` 和 `hicache_storage.py` 中定义，`PoolName` 枚举（attention_kv、mamba_state、swa_kv 等）区分传输的是 MHA KV、MLA latent、SSM state 还是 SWA KV。`PoolTransfer` 封装分页大小、dtype、各层 buffer info（data_ptr + nbytes）。

- 7.4.2 `GetPageContext` / `SetPageContext`：分页传输 API

`GetPageContext` 指定"从哪个池（PoolName）、哪段 page（start_page→end_page）、哪个 storage 后端"取页数据；`SetPageContext` 对称写回。分页粒度使得 L2↔L3 传输无需整请求搬移——只搬最近访问的 page。

- 7.4.3 RDMA Batch 操作与 `STORAGE_BATCH_SIZE` 批量化

`STORAGE_BATCH_SIZE`（常量定义在 `hicache_storage.py`）控制单次 RDMA DMA 的 batch 页数。多页聚合为单次 RDMA 提交，分摊`ibv_post_send`/硬件门铃的开销。

- 7.4.4 `PrefetchTimeoutConfig`：超时控制的线性策略

`PrefetchTimeoutConfig`（`hicache_storage.py` 或 `cache_controller` 中定义）控制 L3→L2 的预取超时：若预取操作在超时门内未完成，直接放弃预取让请求走正常 token 生成路径（重算 KV），避免预取拖慢请求。策略有线性退避等。

### 7.5 [GLM-5.2 适配] 大 KV 量下的传输优化方向

- 7.5.1 MLA 低秩压缩 KV 减少传输字节量

MLA 每 token 只存 `kv_lora_rank + qk_rope_head_dim`（~576 float）而非全量多头 KV（如 128×128×2=32768 float）。跨设备传输时，MLA 的 `get_cpu_copy` 只需传单层单 `kv_buffer`，字节量约为 MHA 的 1/57——这对 PD 传输和 HiCache 下沉都直接减少网络带宽压力。

- 7.5.2 DSA 稀疏 Mask 过滤仅传输有效 Token

DSA 的 `DSATokenToKVPool` 在 offload 时独传 `index_k_with_scale_buffer`。但稀疏注意力可利用 token_mask 进行进一步优化：不是所有 token 的索引页都有用，只传输"当前 Top-K 集内 token 对应的索引页"。这需要调度层在 offload 前知道稀疏命中集合——GLM-5.2 adaption 这里需要在 DSA backend 与传输调度间增加一个稀疏过滤器。

### 7.6 传输链路常见故障与排坑

- **stale slot 静默脏数据**：offload 期间 slot 被 retract 释放 + 重新分配 → load_back 回灌时 slot 已被新请求占用 → 数据错乱。`maybe_detect_oob` 检查 + DSA 双缓冲锁步传输是既有防御。
- **不同步的层间传输**：`LayerDoneCounter` 的 `wait_until` 若卡死（某个 layer 传输线程挂了），后续所有 attention 计算阻塞 → 需要 per-layer 超时 fallback。
- **HiCache write-through 与 evict 竞态**：`host_ref_counter` 保护 + `write_through_pending_id` 去重 + `ongoing_write_through` dict 追踪在途操作（第 6 章 6.5）。

# 第四部分：模型专属适配与深度交互
## 第8章 非标准 Attention 架构对 KV Cache 的强约束

第 1 章 1.2.3 已详述 MHA/MLA/DSA 三种物理池的结构。本章从"架构约束"视角看非标 attention 对 KV 体系的全链影响——不只是物理布局不同，而是从生成、写入、读取、传输到淘汰都需要适配。

### 8.1 SGLang 中已有的非标准 KV Cache 实现全景

- 8.1.1 `MLATokenToKVPool`：MLA 低秩压缩 KV 的专用物理池

`memory_pool.py:2130`。核心约束：只存一个 latent 向量，head_num=1，get_value_buffer 返回 kv_buffer 前半（nope 段），get_key_buffer 返回全 latent。这意味着**所有以下接口需要感知"只有 kv_buffer 一个 buffer"**：`get_contiguous_buf_infos`（L2212 只返回 kv_buffer 的信息而非 K+V 分开）、`set_kv_buffer`（单 buffer 索引赋值而非 K/V 分别写）。

- 8.1.2 `DSATokenToKVPool`：DSA 稀疏注意力的专用物理池

`memory_pool.py:2529`。继承 MLA，增加 `index_k_with_scale_buffer`。核心约束：双缓冲一致性——`move_kv_cache`、`get_cpu_copy`、`load_cpu_copy` 三者都必须同时搬运 latent + index 两份数据。平台强约束：CUDA `page_size=64`，HIP `page_size=1` 或 `page_size%16==0`。

- 8.1.3 `HiSparseDSATokenToKVPool` + `HiSparseTokenToKVPoolAllocator`：稀疏二级池

`hisparse_memory_pool.py` 定义。DeepSeek V4 HiSparse 引入二级稀疏池——主池存全量，辅助池只存高贡献 KV（按 attention score 筛选）。`HiSparseTokenToKVPoolAllocator`（`allocator/hisparse.py`）管理主辅助两套空闲 slot，分配/释放需同时操作两套 allocator。

- 8.1.4 `DeepSeekV4TokenToKVPool`：c4/c128 多级压缩池体系

`deepseek_v4_memory_pool.py:28`（推断）。DeepSeek V4 用多级压缩——c4（高压缩比，存潜在上下文摘要）+ c128（标准压缩，存当前窗口细节）。多级池的 allocator 需同时管理 c4/c128 两套 page 集合，evict 时两套 slot 交叉释放，`get_contiguous_buf_infos` 返回四组（c4_k/c4_v/c128_k/c128_v）而非两组。

### 8.2 MLA（Multi-Head Latent Attention）KV Cache 存储对比分析

- 8.2.1 `kv_lora_rank` 压缩 vs 全量 KV 的存储/传输差异

| 指标 | MHA (128 头 × 128 维) | MLA (kv_lora_rank=512, rope=64) |
|---|---|---|
| 每 token 存储量 | head_num × head_dim × 2 = 32768 元素 | kv_lora_rank + qk_rope = 576 元素 |
| 压缩比 | — | ~57× |
| 每层 buffer 数 | 2 (K + V) | 1 (latent) |
| get_value_buffer 返回 | 整 v_buffer | kv_buffer 前半 (nope) |
| 跨设备传输 | 传 K+V 各层各 chunk | 传单层单 buffer |
| Attention 读 | 按 head 读全量 KV | 读 latent 后 MLA kernel 内投影展开 |

- 8.2.2 flashinfer_mla_backend 中的 paged KV 转换适配

`create_flashmla_kv_indices_triton`（`kv_indices.py:99`）与标准 MHA 版本的差异：输出 `/ PAGED_SIZE`（L150），将 slot 索引转为 page 编号，因为 flashMLA kernel 的 paged KV 索引粒度是 page 而非 token。`get_num_page_per_block_flashmla` / `get_num_kv_index_blocks_flashmla`（L84/89）计算 CTA 维度的 page block 大小。

### 8.3 DSA（Dense-Sparse Attention）稀疏窗口机制

- 8.3.1 Dense Layer + Sparse Layer 交替架构下的双缓存设计

DSA 模型部分层是全注意力（Dense），部分层是稀疏注意力（Sparse）。Sparse 层需要两个缓存：主 KV latent（`kv_buffer`，与 MLA 一致）和索引 K（`index_k_with_scale_buffer`，用于 Top-K 路由）。Dense 层只需 latent KV。两者共用 `kv_buffer` 的存取逻辑——物理 buf 布局一致，区别只在 Sparse 层额外多一个 index buffer 且 idx 访问走 get_index_k_scale_buffer 而非 get_kv_buffer。

- 8.3.2 `sparsity/` 目录下的稀疏索引与压缩状态管理

`python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` 等文件管理稀疏状态的压缩与同步。DSA 的稀疏路由需要在 forward 后更新 compress state，在下一个 forward 前同步 mask 位置——这些状态与 KV cache 物理池分离但在 attention kernel 内联动：mask 决定从 `kv_buffer` 读取哪些 token、从 `index_k_with_scale_buffer` 的哪些 page 进行 Top-K。

### 8.4 RoPE 位置编码偏移引发的索引修正原理

前缀缓存复用时，被命中段 token 在树缓存中的 RoPE 位置与当前请求的真实位置可能不同。SGLang 的 KV cache 存的是原始 K/V（RoPE 已施加），**复用时的位置偏移需要 attention kernel 在 Q 侧或 RoPE 后处理中修正**，而非 KV 侧重写。`cache_protected_len` 是请求维度记录"本请求从树缓存中复用了多少 token 的 KV"，由 attention backend 的 `init_forward_metadata` 用于计算每个 token 的实际 RoPE 位置偏移。

### 8.5 Continuous Batch 动态批处理资源调度

Continuous batching 让请求可随时加入/离开 batch，不要求同批请求同步完成。这对 KV cache 的影响是：
- **chunked prefill**：长请求的 prefill 拆成多个 chunk，每个 chunk 在页表同一行追加 slot→`req_pool_idx` 跨 chunk 复用（第 3 章 3.1.2）。
- **decode 混合 batch**：同 batch 可能混有 extend 和 decode 两类请求，`alloc_for_extend` / `alloc_for_decode` 分别处理各自的 KV 分配。
- **batching 粒度不影响物理池**：物理池是全局平面的，batch 内请求乱序和跨 batch 交错通过 allocator 的 `free_pages` / `release_pages` 解决碎片。

### 8.6 FP8 量化 KV Cache：`store_dtype=torch.uint8` 的数值对齐与精度兼容

`KVCache.__init__`（`memory_pool.py:995-999`）的核心处理：

```python
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
    self.store_dtype = torch.uint8   # index_put 不支持 fp8，存为 uint8
else:
    self.store_dtype = dtype
```

FP8 KV 存储用 uint8 做 `index_put`，读写时 view(fp8_dtype) 还原。`set_kv_buffer` 中 `cache_k.div_(k_scale)` 后 quantize 到 fp8，`view(store_dtype)` 写 uint8；`get_key_buffer` 中 `view(dtype)` 从 uint8 回到 fp8。MHATokenToKVPoolFP4 和 MLATokenToKVPoolFP4（`memory_pool.py:1752/2389`）进一步支持 FP4 存储，写入路径需 per-block scale。

### 8.7 MoE 专家并行 EP 下多卡 KV 分布与路由

MoE 的专家层是 FFN，不产生 KV——KV 的分布与 MoE EP 路由解耦。attention 层在 TP 组内按 head 切 KV，MoE FFN 的 EP 在 attention layer 间穿插，各 EP rank 的 attention 层 KV 各存各的。路由 token 从 KV_rank_A 的 attention 输出后可能路由到 FFN_rank_B，但返回 attention_rank_A 时 KV 仍原地——"token 路由走、KV 原地留"。

### 8.8 [GLM-5.2 推演] 结合 MLA + DSA + MoE 的综合 KV Cache 架构设计方向

GLM-5.2 若同时采用 MLA（低秩压缩）+ DSA（稀疏注意力）+ MoE（专家混合），KV cache 体系需要以下几层组合：

```
GLM-5.2 KV Pool = MSGPTokenToKVPool(MHATokenToKVPool, MLA-like latent, DSA index buffer)
  ├── Dense Attention Layers   → MLATokenToKVPool（只有 latent kv_buffer）
  ├── Sparse Attention Layers  → DSATokenToKVPool（latent + index_k_with_scale）
  ├── MoE FFN Layers          → 无 KV 缓存，EP 路由独立
  └── (可选) SWA Layers       → SWAKVPool（full + swa 双池）
```

SGLang 已实现全部子组件：MLATokenToKVPool / DSATokenToKVPool / SWATokenToKVPoolAllocator / MoE EP。适配需组装而非重写——在 `pool_configurator.py`（`model_runner_kv_cache_mixin.py` 调用）中按 GLM-5.2 的 config 将各层分配到正确的 pool 类别，`HybridLinearKVPool`（`memory_pool.py:1902`）模式可直接复用作为模板。

## 第9章 SGLang 端到端全推理链路时序闭环

本章把前八章节串联成一次完整请求的全时序——从请求到达到全部资源回收。

### 9.1 请求接入、分词、`Req` 对象元信息初始化

```
HTTP/GRPC Request → TokenizerManager → tokenize → Req 对象
  req.origin_input_ids = tokenized_ids
  req.output_ids = []
  req.kv_committed_len = 0
  req.kv_allocated_len = 0
  req.req_pool_idx = None   ← 尚未分配页表行
  req.last_node = None
  req.extra_key = lora_id / cache_salt  ← 命名空间隔离
```

### 9.2 RadixCache `match_prefix()` 前缀匹配判定

每次 `get_new_batch_prefill` 启动：

```
tree_cache.match_prefix(MatchPrefixParams(key=radix_key))
  → MatchResult(device_indices=matched_slot_seq, last_device_node=matched_node)
  → req.prefix_indices = matched_slot_seq   ← 存在树里的 slot 段
  → req.last_node = matched_node
```

匹配上则跳过该段的 KV 计算与 slot 分配。

### 9.3 Prefill 全量 / Chunked Prefill 增量双分支执行流程

```
alloc_for_extend(batch) → 分配 extend_num_tokens 个新 slot → write_cache_indices 填页表
  → model forward → attention kernel 算新 token 的 K/V
  → set_kv_buffer(layer, loc=extend_slot, k, v) 落盘

Chunked:
  → 剩余 token 留到下次 chunk
  → cache_unfinished_req 插入树
  → dec_lock_ref(old_last_node) + inc_lock_ref(new_last_node)
  → prefix_indices 更新
```

### 9.4 Decode 循环生成 + 增量 KV 持续挂载

```
每步: alloc_for_decode(batch) → 对 batch 中每个请求分配 1 个 slot
  → write (req_pool_idx, seq_lens, loc) 追加到页表
  → model forward → 产出 1 token logit → sample 1 token
  → req.kv_committed_len += 1
  → set_kv_buffer(layer, loc=decode_slot, k, v)
  → 若 EOS 或 max_len → 触发 release_kv_cache
```

### 9.5 多轮对话缓存复用加速逻辑

`SessionRadixCacheMixin._tag_session_leaf(req, radix_key, node)`（`session_radix_cache.py`）把请求的终端叶子节点与 `req.session_id` 关联。同 session 的下一轮请求的 `match_prefix` 会从该会话叶子开始匹配——无需重新插入已存在的公共前缀。Session 级别缓存不跨 session 共享，各 session 独立树路径。

### 9.6 会话超时/结束资源回收链路：`release_kv_cache()` 完整流程

第 6 章 6.4 已详述三步回收：
1. `cache_finished_req` 把 committed KV 交树（带走所有权，释放重复）
2. `pop_overallocated_kv_cache` 回收草稿 token slot
3. `ReqToTokenPool.free(req)` 归还页表行号

### 9.7 显存超限→触发淘汰→`evict()` → 可能触发 HiCache 下沉传输

`alloc_token_slots(need_size)` 的 `evict_from_tree_cache` → `tree_cache.evict(EvictParams(num_tokens=missing))` → 小顶堆逐出 → `allocator.free(x.value)` → slot 回 free_pages → 若 HiCache write-through 启用：逐出前 `cache_controller.write_through(node)` 下沉到 L2/L3。

### 9.8 下级缓存命中→`load_back()` → `load_cpu_copy` 回灌 GPU

HiCache `match_prefix` 命中 `host_value` 非空而 `value` 为空 → `init_load_back(InitLoadBackParams(best_match_node, host_hit_length))` → `cache_controller.load_back_backend` → `load_cpu_copy(kv_cache_cpu_dict, indices)` 分块 H2D → `set_kv_buffer` 写回 GPU → 后续正常读取。

### 9.9 GLM-5.2 场景推演：工具调用 / 长摘要 / 记忆裁剪特殊链路

- **工具调用**：每次 tool call 返回结果当作新 prompt 追加，`extra_key=tool_name` 隔离不同工具的 KV 前缀。
- **长摘要**：chunked prefill 下多个 chunk 轮转 `cache_unfinished_req → match_prefix` 复用前一 chunk 的结果。
- **记忆裁剪**：用 SWA 滑窗或 DSA 稀疏 mask 自动裁剪低价值 KV，由 `free_swa_out_of_window_slots` 或 `DSATokenToKVPool.move_kv_cache` 锁步搬迁维护一致性。

# 第五部分：HiCache 多级缓存工程优化
## 第10章 SGLang HiCache 多级缓存架构原理

HiCache 给 RadixCache 增加了 GPU(L1)→CPU(L2)→Storage(L3) 的升降级能力。核心在 `python/sglang/srt/mem_cache/hiradix_cache.py`、`hicache_storage.py`、`pool_host/`、`storage/`。

### 10.1 三级存储层级定义

- 10.1.1 L1：GPU 显存（`KVCache` 物理池，原生读写，零拷贝）

即是 `MHATokenToKVPool` / `MLATokenToKVPool` / `DSATokenToKVPool` 持有的 `k_buffer`/`v_buffer`/`kv_buffer`。读延迟 ≈ 几十纳秒，吞吐 ≈ TB/s。树节点 `value` 非空时数据在 L1。

- 10.1.2 L2：CPU 内存（`HostKVCache`，`pool_host/base.py`，pin_memory + DMA）

`HostKVCache` 在 `HiRadixCache.__init__` 中按 `hicache_ratio` × `kv_cache_size` 分配 CPU pin_memory 池，每层一个连续 pinned buffer。`pin_memory` 加速 DMA 传输（D2H/H2D 无需 intermediate bounce buffer），树节点 `host_value` 非空时数据在 L2。

- 10.1.3 L3：多后端存储层（`storage/`：file / mooncake_store / hf3fs / lmcache / nixl / eic / simm / aibrix_kvcache）

`python/sglang/srt/mem_cache/storage/` 下每个后端一个子模块，实现统一的 `StorageBackend` 协议（`get`/`set`/`evict` 接口）。各后端定位：

| 后端 | 适用场景 |
|---|---|
| `file` | 本地 NVMe SSD，单机持久化 HiCache |
| `mooncake_store` | 分布式 KV 存储，PD 分离专用 |
| `hf3fs` | 3FS 分布式文件系统，大规模离线 |
| `lmcache` | LM 原生缓存格式，与 vLLM 互操作 |
| `nixl` | 基于 RDMA/NIXL 的远端访问 |
| `eic` / `simm` / `aibrix_kvcache` | 云厂商 KV 存储后端 |

### 10.2 多池类型管理：`PoolName` 枚举与 `PoolTransfer` 传输抽象

`PoolName` 枚举（`hicache_storage.py` 或 `pool_host/` 中）按"缓冲类别"区分传输上下文：

- `attention_kv` — MHA 全量 KV
- `attention_latent_kv` — MLA latent KV
- `attention_index_k` — DSA index_k_with_scale
- `mamba_state` / `swa_kv` — SSM 状态 / 滑窗 KV

`PoolTransfer` 封装分页大小、dtype、各层 buffer 信息（data_ptr + nbytes），使 L2↔L3 传输可一次 API 覆盖多池多层的 KV 数据。

### 10.3 `HiCacheController` + `HybridCacheController`：升降级调度与预取

- 10.3.1 `PrefetchOperation`：预判后续访问路径，提前下发 H2D 传输

`PrefetchOperation` 由 `HiCacheController` 在后台管理——它监视树缓存的热度，对"可能在接下来的 batch 被请求"的节点主动从 L2→L1 或 L3→L2 发起预取。预取成功则请求直享 L1 命中；失败则由 `PrefetchTimeoutConfig` 超时放弃。

- 10.3.2 `PoolHitPolicy`：命中策略与降级触发条件

`PoolHitPolicy` 控制降级阈值——当 `host_hit_length` 超过阈值且 `mem_quota` 充足时触发 `load_back`。`init_load_back`（`base_prefix_cache.py:146`）携带 `host_hit_length` 与 `mem_quota` 供策略决策。升降级可配置为 `write_through`（写穿：每次 GPU 出新 KV 同时下沉 L2）、`write_back`（写回：淘汰时再下沉）、`write_through_selective`（选择性写穿）。

### 10.4 与第 7 章传输链路的联动

HiCache 下沉与第 7 章传输共用同一套物理链路——`get_cpu_copy`/`load_cpu_copy` 做 D2H/H2D、`LayerDoneCounter` 做逐层同步、`storage/` backend 做持久化。HiSparse 带来新的协同问题：c4/c128 各有自己的 eviction policy 和传输调度，`HybridCacheController` 去协调。

### 10.5 HiCache 与 Disaggregation PD 的协同：`StorageMedium` 标记

PD 分离式传输与 HiCache 下沉共享 L2/L3 通道。`StorageMedium` 标记每块 KV 数据当前在三级存储中的实际位置（L1/L2/L3），PD 的 prefill→decode 传输直接把数据写入 L1（绕过 L2），HiCache 的 write-through 则写入 L2/L3。两者在同一进程内互不阻塞——`StorageMedium` 使判断"这块数据是否需要额外搬运"成为 O(1) 查表。

### 10.6 工程稳定性方案

- **IO 限流**：`STORAGE_BATCH_SIZE` 分页批量传输 + 分块 8192 的 D2H/H2D，使 IO 步幅可控不会突发打满 PCIe。
- **脏数据校验**：`hash_value`（第 2 章 2.3）提供节点级别 SHA256 链，L2/L3 读取前校验哈希一致性防止存储端位翻转。
- **过期会话自动清理**：`cache_ttl_seconds`（`CacheInitParams:49`）为每个树节点设 TTL，超时未访问自动逐出——防止无会话命中的历史数据永久占 L2/L3。

# 第六部分：源码导读、性能测评与技术展望
## 第11章 核心源码路径导读

按本文章节对应关系直接列出精确文件路径与起止行号：

### 11.1 内存池与页表

| 模块 | 文件 | 关键类/函数 | 行号 |
|---|---|---|---|
| 逻辑层 | `memory_pool.py` | `ReqToTokenPool` | 235-302 |
| 分配层（per-token） | `allocator/token.py` | `TokenToKVPoolAllocator` | 28-84 |
| 分配层（per-page） | `allocator/paged.py` | `PagedTokenToKVPoolAllocator` | 105- |
| 分配层（Hybrid SWA） | `allocator/swa.py` | `SWATokenToKVPoolAllocator` | 20- |
| 分配层（HiSparse） | `allocator/hisparse.py` | `HiSparseTokenToKVPoolAllocator` | 15- |
| 物理层（MHA） | `memory_pool.py` | `MHATokenToKVPool` | 1074-1641 |
| 物理层（MLA） | `memory_pool.py` | `MLATokenToKVPool` | 2130-2388 |
| 物理层（DSA） | `memory_pool.py` | `DSATokenToKVPool` | 2529- |
| 公共入口 | `common.py` | `alloc_for_extend / alloc_for_decode / release_kv_cache / write_cache_indices` | 456-701 |

### 11.2 RadixCache 前缀匹配与淘汰

| 模块 | 文件 | 关键类/函数 | 行号 |
|---|---|---|---|
| 基数树 | `radix_cache.py` | `RadixCache/TreeNode/RadixKey` | 57-818 |
| 多级缓存 | `hiradix_cache.py` | `HiRadixCache` | 75- |
| 混合模型 | `unified_radix_cache.py` | `UnifiedRadixCache` | 305- |
| 淘汰策略 | `evict_policy.py` | LRU/LFU/SLRU/FIFO/MRU/FILO/Priority | 1-65 |
| 策略工厂 | `utils.py` | `get_eviction_strategy` | 66-74 |
| 协议 | `base_prefix_cache.py` | `PrefixCacheTrait / MatchPrefixParams / InsertParams / EvictParams` | 35- |

### 11.3 跨设备传输

| 模块 | 文件 | 关键类/函数 |
|---|---|---|
| D2H/H2D | `memory_pool.py` | `KVCache.get_cpu_copy / load_cpu_copy` |
| 分离式传输 | `disaggregation/decode.py` / `decode_kvcache_offload_manager.py` | decode 端传输逻辑 |
| KV 序列化 | `kv_cache_builder.py` | KV 打包/解包 |
| L2 主机池 | `pool_host/base.py` | `HostKVCache` |
| L3 存储 | `storage/` 下各子模块 | file/mooncake/hf3fs/lmcache/nixl backend |
| 传输控制 | `hicache_storage.py` | PoolTransfer/PoolName/GetPageContext/SetPageContext |

### 11.4 Attention Backend 中的 paged KV 转换

| 模块 | 文件 | 关键类/函数 | 行号 |
|---|---|---|---|
| Triton 转换 | `layers/attention/triton_ops/kv_indices.py` | `create_flashinfer_kv_indices_triton / create_flashmla_kv_indices_triton` | 9-152 |
| FlashInfer 后端 | `layers/attention/flashinfer_backend.py` | `FlashInferAttnBackend / FlashInferIndicesUpdaterDecode` | 291-1320+ |
| MLA 后端 | `layers/attention/flashinfer_mla_backend.py` | MLA attention kernel 入口 | — |

### 11.5 调度入口与工具函数

| 模块 | 文件 | 关键函数 | 行号 |
|---|---|---|---|
| 通用 alloc/release | `common.py` | `alloc_for_extend / alloc_for_decode / release_kv_cache / write_cache_indices / alloc_token_slots / evict_from_tree_cache / get_last_loc / get_alloc_len_per_decode` | 68-701 |
| 调度入口 | `managers/schedule_batch.py` | `ScheduleBatch.maybe_evict_swa` | 2864- |
| 初始化路由 | `model_runner_kv_cache_mixin.py` | `_init_pools` 按模型类型选 pool 类 | — |
| Pool 组装 | `pool_configurator.py` | 按 config 将各层分配到正确的 KVCache 子类 | — |

## 第12章 多框架横向性能对比

### 12.1 测试基线：vLLM (PagedAttention) / SGLang (RadixCache) / SGLang + HiCache

- **vLLM**：per-request block table 管理 KV，无跨请求前缀复用。同 system prompt 下每个请求独立 prefill。
- **SGLang (RadixCache)**：全局基数树共享前缀 KV。命中前缀的请求免 prefill，只计算增量。
- **SGLang + HiCache**：基数树 + L2/L3 持久化。热点 KV 在 GPU、温数据在 CPU、冷数据在存储后端。支持跨重启恢复。

### 12.2 核心指标：TTFT、TPOT、QPS、显存占用、缓存命中率、传输时延

| 指标 | vLLM 基线 | SGLang 优势区 | SGLang+HiCache 优势区 |
|---|---|---|---|
| TTFT | 基准 | 前缀命中时大幅降低（免 prefill） | 跨重启恢复 + 预取，冷启动 TTFT 远低于纯 SGLang |
| TPOT | 相当 | 相当 | 相当 |
| QPS | 1× | 缓存命中率高时 1.5~3×（减少 prefill 计算量） | 更高（更大有效 KV 容量→更高命中） |
| 显存占用 | PagedAttention 池化 | 同为池化，额外树节点开销 <5% | GPU 显存可更低（冷数据迁 L2/L3 释放显存） |
| 缓存命中率 | 0%（无跨请求前缀复用） | 系统提示词场景 70~95% | 跨重启保留 → 命中保持 |
| 传输时延 | 0（无跨节点 KV 搬移） | PD 分离增加 NCCL/RDMA 时延 | PCIe D2H/H2D 20~50ms per 1000 token；RDMA 更低 |

## 第13章 架构局限与未来演进

### 13.1 当前短板

- **树深度过高**：超长上下文下基数树深度 ≈ 序列长度，`inc_lock_ref/dec_lock_ref` 沿 parent 走到根 O(depth) 开销在深度 100K+ 时不可忽略。解决方案：树分段压缩或跳表索引。
- **`req_to_token` 对超长上下文的存储开销**：`max_batch × max_context_len` 的 int32 页表在 batch=256、context=1M 时占 1 GB 显存（仅页表不包含 KV 数据）。压缩方案：以 page 为粒度改用 page table 替代 per-token 页表。
- **PD offload 同步延迟**：第 7 章 GPU↔CPU 分块 offload 在 8192 分块下长序列需要多轮 D2H/H2D，延迟累积。
- **DSA 双缓冲传输开销**：DSA 的 `index_k_with_scale_buffer` 为 KV latent 带来了额外 128+4 的 per-page 传输负担，稀疏优化不充分。

### 13.2 未来方向

- **Chunked Prefill 优化**：按 chunk 热度预取前缀 KV，减少 chunk 间 match_prefix 阻塞。
- **分布式全局 KV Cache 集群**：L3 扩展为 RDMA 可达的全集群 KV 池，请求的 KV 就近服务于任何节点。
- **RDMA 零拷贝跨机传输**：mooncake RDMA 路径绕 CPU 直写 GPU 显存池（`custom_mem_pool` 机制），消除 CPU bounce buffer 开销。
- **自适应多级缓存调度**：`HybridCacheController` 演化成基于访问模式的 ML 驱动的分层缓存调度器——根据节点历史命中、优先级、显存水位自动调优 eviction policy 和 prefetch 深度。

## 附录

### A.1 传统原版 PagedAttention 原理回顾

论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（arXiv:2309.06180）

传统原版 PagedAttention = 把 LLM 推理的 KV 缓存做成操作系统虚拟内存分页系统，用离散固定大小显存块 + 页表映射替代整块连续内存预分配，根治 KV 缓存显存碎片化，大幅提升大模型在线服务并发吞吐量。

SGLang 在 PagedAttention 的物理分页之上构建了 RadixTree 前缀共享、HiCache 多级缓存、MLA/DSA 压缩物理池三层扩展，使 KV cache 从"单一请求的显存管理器"变成"全集群的语义知识缓存"。

### A.2 `KVCache` 子类全景参考

| 类 | 文件:行号 | 适用场景 | 核心特点 |
|---|---|---|---|
| `MHATokenToKVPool` | `memory_pool.py:1074` | Llama/Qwen/GLM 标准 MHA | K+V 两 buffer，NHD 布局，可选 AITER 5D |
| `NoOpMHATokenToKVPool` | `memory_pool.py:1642` | all-SWA 模型的 full sub-pool | 空池，所有 KV 都在 SWA pool |
| `MHATokenToKVPoolFP4` | `memory_pool.py:1752` | FP4 量化 KV | 写入路径 extra quantize/dequant per-block scale |
| `HybridLinearKVPool` | `memory_pool.py:1902` | SWA + Dense 混合 | 组合 full_kv_pool + swa_kv_pool，write_loc 双写 |
| `MLATokenToKVPool` | `memory_pool.py:2130` | DeepSeek-V2/V3 MLA | 单 latent kv_buffer，57× 压缩 |
| `MLATokenToKVPoolFP4` | `memory_pool.py:2389` | FP4 MLA | MLA latent + FP4 量化，set_mla_kv_buffer FP4 特化 |
| `DSATokenToKVPool` | `memory_pool.py:2529` | DeepSeek-V3.2 DSA | MLA + index_k_with_scale_buffer 双缓冲 |
| `DeepSeekV4TokenToKVPool` | `deepseek_v4_memory_pool.py:28` | DeepSeek V4 多级压缩 | c4 高压缩 + c128 标准压缩双池 |

### A.3 `TokenToKVPoolAllocator` 子类全景参考

| 类 | 文件:行号 | 粒度 | 核心差异 |
|---|---|---|---|
| `BaseTokenToKVPoolAllocator` | `allocator/base.py:27` | 抽象 | free_pages/release_pages/need_sort 基类 |
| `TokenToKVPoolAllocator` | `allocator/token.py:28` | per-token (page_size=1) | alloc 返回 slot seq，free 直接追加 |
| `PagedTokenToKVPoolAllocator` | `allocator/paged.py:105` | per-page (page_size>1) | alloc/alloc_extend/alloc_decode 三段填充，free 去重 |
| `SWATokenToKVPoolAllocator` | `allocator/swa.py:20` | per-token HW | 组合 full+swa 两套 allocator，full_to_swa_index_mapping |
| `HiSparseTokenToKVPoolAllocator` | `allocator/hisparse.py:15` | 主/辅双池 | 主池全量+辅助池高贡献 KV，双 allocator 同步 |