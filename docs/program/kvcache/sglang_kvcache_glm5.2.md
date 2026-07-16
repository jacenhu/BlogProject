# 深度解析 SGLang KV Cache：RadixTree 前缀共享、全生命周期流转、跨设备传输与 HiCache 多级缓存工程优化

## 前言
在大模型推理体系中，KV Cache 是制约 **TTFT、吞吐、显存上限、长上下文能力** 的核心瓶颈。传统 PagedAttention 仅实现「块级显存复用」，无法解决多请求公共前缀重复计算问题。

SGLang 相较于 vLLM 最大架构革新是 **RadixTree（基数树）全局前缀 KV 共享机制**，结合三层内存池（`ReqToTokenPool` → `TokenToKVPoolAllocator` → `KVCache`）、精细化请求级元数据管理、**跨设备 KV 传输链路**（GPU↔CPU offload、Disaggregation PD、HiCache 多后端存储），实现了「计算复用+显存复用+跨设备数据流转+硬件层级扩容」四重优化。

> **源码版本说明**：本文基于 SGLang 主分支源码分析。GLM-5.2 模型尚未合入 SGLang 主线，文中涉及 GLM-5.2 的适配分析属于架构推演，与 SGLang 通用机制严格区分标注。但 GLM-5.2 所依赖的 **DSA 稀疏注意力、MLA 低秩压缩 KV、DeepSeek V4 HiSparse 等基础设施已在 SGLang 代码库中存在**，可据此推演 GLM-5.2 的适配路径。

本文将从**底层数据结构 → RadixCache 前缀共享内核 → KV 完整生命周期 → 跨设备 KV Cache 传输专项 → HiCache 多级缓存工程扩容 → 源码路径导读 → 性能与展望**逐层递进，完成 SGLang KV 全套技术栈深度拆解。

## 目录

- [前言](#前言)

**第一部分：底层内核架构（数据结构基础）**
- [第1章 SGLang KV Cache 底层数据结构与内存池机制](#第1章-sglang-kv-cache-底层数据结构与内存池机制)
  - [1.1 传统 KV Cache 架构缺陷](#11-传统-kv-cache-架构缺陷)
  - [1.2 SGLang 三层内存池架构核心设计](#12-sglang-三层内存池架构核心设计)
    - [1.2.1 `ReqToTokenPool`：请求级逻辑映射池（逻辑层）](#121-reqtotokenpool请求级逻辑映射池逻辑层)
    - [1.2.2 `TokenToKVPoolAllocator`：Token 映射物理分配池（分配层）](#122-tokentokvpoolallocatortoken-映射物理分配池分配层)
    - [1.2.3 `KVCache`：物理存储层（`MHATokenToKVPool` / `MLATokenToKVPool` / `DSATokenToKVPool`）](#123-kvcache物理存储层)
    - [1.2.4 三层解耦优势：逻辑请求自由伸缩、物理显存统一池化](#124-三层解耦优势)
  - [1.3 `req_to_token` 统一页表机制（per-token 直接映射）](#13-req_to_token-统一页表机制per-token-直接映射)
    - [1.3.1 `page_size=1`：per-token 粒度的 slot 映射](#131-page_size1per-token-粒度的-slot-映射)
    - [1.3.2 `page_size>1`：`PagedTokenToKVPoolAllocator` 隐式分页](#132-page_size1pagedtokentokvpoolallocator-隐式分页)
    - [1.3.3 与 vLLM BlockTable 的架构差异对比](#133-与-vllm-blocktable-的架构差异对比)
    - [1.3.4 flashinfer 后端中 `req_to_token` → paged KV 格式的转换](#134-flashinfer-后端中-req_to_token--paged-kv-格式的转换)
  - [1.4 KV Cache 精细化元数据体系](#14-kv-cache-精细化元数据体系)
    - [1.4.1 请求维度：`kv_committed_len`、`kv_allocated_len`、`req_pool_idx`、`priority`、`time_stats`](#141-请求维度kv_committed_lenkv_allocated_lenreq_pool_idxprioritytime_stats)
    - [1.4.2 前缀缓存维度：`prefix_indices`、`num_matched_prefix_tokens`、`host_hit_length`、`cache_protected_len`](#142-前缀缓存维度prefix_indicesnum_matched_prefix_tokenshost_hit_lengthcache_protected_len)
    - [1.4.3 SWA 维度：`swa_evicted_seqlen`、`sliding_window_size`](#143-swa-维度swa_evicted_seqlensliding_window_size)
    - [1.4.4 生命周期维度：`kv_committed_freed`、`kv_overallocated_freed`、`inflight_middle_chunks`](#144-生命周期维度kv_committed_freedkv_overallocated_freedinflight_middle_chunks)
  - [1.5 Chunked Prefill：长文本分段与 `req_pool_idx` 复用机制](#15-chunked-prefill长文本分段与-req_pool_idx-复用机制)
  - [1.6 工程踩坑与源码细节](#16-工程踩坑与源码细节)
    - [1.6.1 `_alloc_size = size + 1`：索引 0 的 CUDA graph padding 约定](#161-_alloc_size--size--1索引-0-的-cuda-graph-padding-约定)
    - [1.6.2 `need_sort` 与 `merge_and_sort_free`：碎片整理时机选择](#162-need_sort-与-merge_and_sort_free碎片整理时机选择)

- [第2章 SGLang RadixCache 基数树：全局前缀 KV 共享核心](#第2章-sglang-radixcache-基数树全局前缀-kv-共享核心)
  - [2.1 RadixAttention 演进背景：PagedAttention 只能单请求复用](#21-radixattention-演进背景pagedattention-只能单请求复用)
  - [2.2 SGLang 中多种 Cache 类型的全景图](#22-sglang-中多种-cache-类型的全景图)
    - [2.2.1 `RadixCache`（基础基数树缓存）](#221-radixcache基础基数树缓存)
    - [2.2.2 `HiRadixCache`（带 HiCache 多级存储的基数树）](#222-hiradixcache带-hicache-多级存储的基数树)
    - [2.2.3 `UnifiedRadixCache`（统一混合模型缓存，SSM + Attention）](#223-unifiedradixcache统一混合模型缓存ssm--attention)
    - [2.2.4 `SWARadixCache` / `MambaRadixCache` / `ChunkCache` / `SessionRadixCache`](#224-swaradixcache--mambaradixcache--chunkcache--sessionradixcache)
  - [2.3 `TreeNode` 核心源码字段全解析](#23-treenode-核心源码字段全解析)
    - [2.3.1 `key: RadixKey`（含 `token_ids` + `extra_key` + `is_bigram`）](#231-key-radixkey含-token_ids--extra_key--is_bigram)
    - [2.3.2 `value` / `host_value`：设备侧与主机侧 KV 数据指针](#232-value--host_value设备侧与主机侧-kv-数据指针)
    - [2.3.3 `children` / `parent`：树拓扑结构](#233-children--parent树拓扑结构)
    - [2.3.4 `lock_ref` / `host_ref_counter`：双层引用计数防误删](#234-lock_ref--host_ref_counter双层引用计数防误删)
    - [2.3.5 `last_access_time` / `hit_count` / `priority`：淘汰决策元数据](#235-last_access_time--hit_count--priority淘汰决策元数据)
    - [2.3.6 `hash_value` / `write_through_pending_id` / `creation_time`](#236-hash_value--write_through_pending_id--creation_time)
  - [2.4 最长公共前缀匹配算法流程](#24-最长公共前缀匹配算法流程)
    - [2.4.1 新请求从根节点逐层比对前缀 Token](#241-新请求从根节点逐层比对前缀-token)
    - [2.4.2 完全匹配 / 部分匹配 / 零匹配三种分支处理逻辑](#242-完全匹配--部分匹配--零匹配三种分支处理逻辑)
  - [2.5 树节点分裂、新建、挂载、EvictionPolicy 多策略淘汰](#25-树节点分裂新建挂载evictionpolicy-多策略淘汰)
    - [2.5.1 前缀部分重合触发节点分裂](#251-前缀部分重合触发节点分裂)
    - [2.5.2 后缀增量生成叶子节点延伸与 Insert 挂载](#252-后缀增量生成叶子节点延伸与-insert-挂载)
    - [2.5.3 `evict_policy.py`：可插拔淘汰策略（不限于 LRU+LFU）](#253-evict_policypy可插拔淘汰策略不限于-lruulfu)
    - [2.5.4 `EvictParams` / `EvictResult`：逐出控制与后置处理](#254-evictparams--evictresult逐出控制与后置处理)
  - [2.6 RadixCache 与 `req_to_token` + `token_to_kv_pool_allocator` 三者关系](#26-radixcache-与-req_to_token--token_to_kv_pool_allocator-三者关系)
    - [2.6.1 树节点 value → 物理 slot 索引 → `allocator.free()` 释放链路](#261-树节点-value--物理-slot-索引--allocatorfree-释放链路)
    - [2.6.2 `PrefixCacheTrait` 协议：`req_to_token_pool` + `token_to_kv_pool_allocator` + `page_size`](#262-prefixcachetrait-协议req_to_token_pool--token_to_kv_pool_allocator--page_size)
  - [2.7 引用计数 RC 联动机制：`lock_ref` + `host_ref_counter`](#27-引用计数-rc-联动机制lock_ref--host_ref_counter)
    - [2.7.1 `IncLockRefResult` / `DecLockRefParams`：引用计数增减 API](#271-inclockrefresult--declockrefparams引用计数增减-api)
    - [2.7.2 计数归零触发分支销毁与 slot 回收](#272-计数归零触发分支销毁与-slot-回收)
  - [2.8 线上问题：前缀失效、树内存泄漏、共享缓存脏数据](#28-线上问题)

**第二部分：KV Cache 完整生命周期原理**
- [第3章 KV Cache 生成机制：Prefill / Decode 双阶段](#第3章-kv-cache-生成机制prefill--decode-双阶段)
  - [3.1 Prefill 预填充全量生成逻辑](#31-prefill-预填充全量生成逻辑)
    - [3.1.1 张量并行批量 Prefill 计算流程图解](#311-张量并行批量-prefill-计算流程图解)
    - [3.1.2 `alloc_for_extend()`：KV slot 分配 + `write_cache_indices()` 写入页表](#312-alloc_for_extendkv-slot-分配--write_cache_indices-写入页表)
    - [3.1.3 前缀命中分支：`match_prefix()` → 跳过重复计算，复用历史树节点](#313-前缀命中分支match_prefix--跳过重复计算复用历史树节点)
  - [3.2 Decode 增量单 Token 生成](#32-decode-增量单-token-生成)
    - [3.2.1 单步 forward 产出 KV 切片](#321-单步-forward-产出-kv-切片)
    - [3.2.2 `alloc_decode()`：每 token 追加一个 slot，`kv_committed_len += 1`](#322-alloc_decode每-token-追加一个-slotkv_committed_len--1)
  - [3.3 惰性显存分配策略：不预占显存、随用随分](#33-惰性显存分配策略不预占显存随用随分)
  - [3.4 多并行架构下 KV 分片生成：TP 按 head 切分 / PP 按 layer 隔离](#34-多并行架构下-kv-分片生成tp-按-head-切分--pp-按-layer-隔离)
  - [3.5 [GLM-5.2 适配] DSA 稀疏注意力的 token_mask 选择性 KV 生成](#35-glm-52-适配-dsa-稀疏注意力的-token_mask-选择性-kv-生成)

- [第4章 KV Cache 写入机制：显存固化与数据落地](#第4章-kv-cache-写入机制显存固化与数据落地)
  - [4.1 GPU 原地零拷贝写入主路径：`KVCache.set_kv_buffer()`](#41-gpu-原地零拷贝写入主路径kvcacheset_kv_buffer)
    - [4.1.1 `KVWriteLoc`：full pool + SWA pool 的二元写入目标](#411-kvwritelocfull-pool--swa-pool-的二元写入目标)
    - [4.1.2 CUDA kernel 直接写入 Block 显存，张量视图复用，无 `clone/copy`](#412-cuda-kernel-直接写入-block-显存张量视图复用无-clonecopy)
  - [4.2 跨设备写入链路：`get_cpu_copy()` / `load_cpu_copy()` 同步 offload](#42-跨设备写入链路get_cpu_copy--load_cpu_copy-同步-offload)
  - [4.3 追加写入（Decode 增量）vs 覆盖写入（上下文重置/窗口刷新）](#43-追加写入decode-增量vs-覆盖写入上下文重置窗口刷新)
  - [4.4 Batch 并发写入锁竞争：RadixCache 全局读写锁与 Block 空闲池竞争规避](#44-batch-并发写入锁竞争radixcache-全局读写锁与-block-空闲池竞争规避)
  - [4.5 [GLM-5.2 适配] 超长文本分段写入与 SWA 滑动窗口截断写入](#45-glm-52-适配-超长文本分段写入与-swa-滑动窗口截断写入)

- [第5章 KV Cache 读取机制：Attention 计算核心链路](#第5章-kv-cache-读取机制attention-计算核心链路)
  - [5.1 标准读取全流程：Query 计算 → `req_to_token` 查页表 → `create_flashinfer_kv_indices_triton` 转换为 paged KV → Attention kernel](#51-标准读取全流程)
  - [5.2 `IndicesUpdater`：`req_to_token` → flashinfer `paged_kv_indices` 的 Triton 转换 kernel](#52-indicesupdaterreq_to_token--flashinfer-paged_kv_indices-的-triton-转换-kernel)
  - [5.3 零拷贝读取、in-place 视图复用、预取优化](#53-零拷贝读取in-place-视图复用预取优化)
  - [5.4 多卡跨分片 KV 聚合读取：TP AllGather / EP 路由分发](#54-多卡跨分片-kv-聚合读取tp-allgather--ep-路由分发)
  - [5.5 多级缓存命中分支：L1 (GPU) 直接命中 / L2 (CPU HostKVCache) → `load_cpu_copy` / L3 (Storage Backend) → `PrefetchOperation`](#55-多级缓存命中分支l1-gpu-直接命中--l2-cpu-hostkvcache--load_cpu_copy--l3-storage-backend--prefetchoperation)
  - [5.6 [GLM-5.2 适配] RoPE 位置偏移修正与稀疏 Token 精准读取](#56-glm-52-适配-rope-位置偏移修正与稀疏-token-精准读取)

- [第6章 KV Cache 淘汰与内存回收机制](#第6章-kv-cache-淘汰与内存回收机制)
  - [6.1 显存水位线分级管控：软阈值降级 Swap / 硬阈值强制淘汰](#61-显存水位线分级管控软阈值降级-swap--硬阈值强制淘汰)
  - [6.2 `evict_policy.py`：可插拔淘汰策略（LRU / LFU / SLRU / FIFO）](#62-evict_policypy可插拔淘汰策略lru--lfu--slru--fifo)
  - [6.3 [GLM-5.2 适配] SWA 窗口外 KV 强制过期 + DSA 无效 Token KV 主动释放](#63-glm-52-适配-swa-窗口外-kv-强制过期--dsa-无效-token-kv-主动释放)
  - [6.4 细粒度 slot 回收 vs 粗粒度整会话回收](#64-细粒度-slot-回收-vs-粗粒度整会话回收)
  - [6.5 双层 RC 防误删：`TreeNode.lock_ref` + `host_ref_counter`](#65-双层-rc-防误删treenodelock_ref--host_ref_counter)
  - [6.6 淘汰后置：树分支修剪、`req_to_token` 索引刷新、slot 归还 `free_pages`、HiCache 下沉传输标记](#66-淘汰后置树分支修剪req_to_token-索引刷新slot-归还-free_pageshicache-下沉传输标记)

**第三部分：专项模块——KV Cache 跨设备传输机制**
- [第7章 KV Cache 跨设备传输体系设计](#第7章-kv-cache-跨设备传输体系设计)
  - [7.1 传输场景全景：SGLang 中实际存在的三类传输](#71-传输场景全景sglang-中实际存在的三类传输)
    - [7.1.1 GPU↔CPU 请求级 Offload：`Req.offload_kv_cache()` / `load_kv_cache()`](#711-gpucpu-请求级-offloadreqoffload_kv_cache--load_kv_cache)
    - [7.1.2 Prefill→Decode 分离式传输（Disaggregation PD）：NCCL / NIXL 跨节点](#712-prefilldecode-分离式传输disaggregation-pdnccl--nixl-跨节点)
    - [7.1.3 HiCache 层级传输：HostKVCache ↔ Storage Backend 的后台数据流转](#713-hicache-层级传输hostkvcache--storage-backend-的后台数据流转)
  - [7.2 GPU↔CPU Offload 详细链路](#72-gpucpu-offload-详细链路)
    - [7.2.1 `KVCache.get_cpu_copy()` / `load_cpu_copy()`：同步 D2H / H2D 拷贝](#721-kvcacheget_cpu_copy--load_cpu_copy同步-d2h--h2d-拷贝)
    - [7.2.2 `TorchMemorySaverAdapter`：显存压缩与 Memory Saver 机制](#722-torchmemorysaveradapter显存压缩与-memory-saver-机制)
    - [7.2.3 `LayerDoneCounter`：layer-wise 传输控制（`register_layer_transfer_counter`）](#723-layerdonecounterlayer-wise-传输控制register_layer_transfer_counter)
  - [7.3 Disaggregation PD 传输链路](#73-disaggregation-pd-传输链路)
    - [7.3.1 `DecodeReqToTokenPool`：预分配 slot + 传输 slot 的分离池设计](#731-decodereqtotokenpool预分配-slot--传输-slot-的分离池设计)
    - [7.3.2 NCCL 集合通信 vs NIXL 点对点传输 vs RDMA 零拷贝](#732-nccl-集合通信-vs-nixl-点对点传输-vs-rdma-零拷贝)
    - [7.3.3 `kv_cache_builder.py`：KV 数据序列化与反序列化](#733-kv_cache_builderpykv-数据序列化与反序列化)
    - [7.3.4 SWA allocator 的 `alloc_extend_swa_tail`：decode 端仅传输 SWA 尾部](#734-swa-allocator-的-alloc_extend_swa_taildecode-端仅传输-swa-尾部)
  - [7.4 HiCache 层级传输：HostKVCache ↔ Storage Backend](#74-hicache-层级传输hostkvcache--storage-backend)
    - [7.4.1 `PoolTransfer` / `PoolName`：多池类型的传输抽象](#741-pooltransfer--poolname多池类型的传输抽象)
    - [7.4.2 `GetPageContext` / `SetPageContext`：分页传输 API](#742-getpagecontext--setpagecontext分页传输-api)
    - [7.4.3 RDMA Batch 操作与 `STORAGE_BATCH_SIZE` 批量化](#743-rdma-batch-操作与-storage_batch_size-批量化)
    - [7.4.4 `PrefetchTimeoutConfig`：超时控制的线性策略](#744-prefetchtimeoutconfig超时控制的线性策略)
  - [7.5 [GLM-5.2 适配] 大 KV 量下的传输优化方向](#75-glm-52-适配-大-kv-量下的传输优化方向)
    - [7.5.1 MLA 低秩压缩 KV 减少传输字节量](#751-mla-低秩压缩-kv-减少传输字节量)
    - [7.5.2 DSA 稀疏 Mask 过滤仅传输有效 Token](#752-dsa-稀疏-mask-过滤仅传输有效-token)
  - [7.6 传输链路常见故障与排坑](#76-传输链路常见故障与排坑)

**第四部分：模型专属适配与深度交互（以 DeepSeek V4 / GLM 系列为例）**
- [第8章 非标准 Attention 架构对 KV Cache 的强约束](#第8章-非标准-attention-架构对-kv-cache-的强约束)
  - [8.1 SGLang 中已有的非标准 KV Cache 实现全景](#81-sglang-中已有的非标准-kv-cache-实现全景)
    - [8.1.1 `MLATokenToKVPool`：MLA 低秩压缩 KV 的专用物理池](#811-mlatokentokvpoolmla-低秩压缩-kv-的专用物理池)
    - [8.1.2 `DSATokenToKVPool`：DSA 稀疏注意力的专用物理池](#812-dsatokentokvpooldsa-稀疏注意力的专用物理池)
    - [8.1.3 `HiSparseDSATokenToKVPool` + `HiSparseTokenToKVPoolAllocator`：稀疏二级池](#813-hisparsedsatokentokvpool--hisparsetokentokvpoolallocator稀疏二级池)
    - [8.1.4 `DeepSeekV4TokenToKVPool`：c4/c128 多级压缩池体系](#814-deepseekv4tokentokvpoolc4c128-多级压缩池体系)
  - [8.2 MLA（Multi-Head Latent Attention）KV Cache 存储对比分析](#82-mlamulti-head-latent-attention-kv-cache-存储对比分析)
    - [8.2.1 `kv_lora_rank` 压缩 vs 全量 KV 的存储/传输差异](#821-kv_lora_rank-压缩-vs-全量-kv-的存储传输差异)
    - [8.2.2 flashinfer_mla_backend 中的 paged KV 转换适配](#822-flashinfer_mla_backend-中的-paged-kv-转换适配)
  - [8.3 DSA（Dense-Sparse Attention）稀疏窗口机制](#83-dsadense-sparse-attention稀疏窗口机制)
    - [8.3.1 Dense Layer + Sparse Layer 交替架构下的双缓存设计](#831-dense-layer--sparse-layer-交替架构下的双缓存设计)
    - [8.3.2 `sparsity/` 目录下的稀疏索引与压缩状态管理](#832-sparsity-目录下的稀疏索引与压缩状态管理)
  - [8.4 RoPE 位置编码偏移引发的索引修正原理](#84-rope-位置编码偏移引发的索引修正原理)
  - [8.5 Continuous Batch 动态批处理资源调度](#85-continuous-batch-动态批处理资源调度)
  - [8.6 FP8 量化 KV Cache：`store_dtype=torch.uint8` 的数值对齐与精度兼容](#86-fp8-量化-kv-cachestore_dtypetorchuint8-的数值对齐与精度兼容)
  - [8.7 MoE 专家并行 EP 下多卡 KV 分布与路由](#87-moe-专家并行-ep-下多卡-kv-分布与路由)
  - [8.8 [GLM-5.2 推演] 结合 MLA + DSA + MoE 的综合 KV Cache 架构设计方向](#88-glm-52-推演-结合-mla--dsa--moe-的综合-kv-cache-架构设计方向)

- [第9章 SGLang 端到端全推理链路时序闭环](#第9章-sglang-端到端全推理链路时序闭环)
  - [9.1 请求接入、分词、`Req` 对象元信息初始化](#91-请求接入分词req-对象元信息初始化)
  - [9.2 RadixCache `match_prefix()` 前缀匹配判定（命中/部分命中/未命中）](#92-radixcache-match_prefix-前缀匹配判定命中部分命中未命中)
  - [9.3 Prefill 全量 / Chunked Prefill 增量双分支执行流程](#93-prefill-全量--chunked-prefill-增量双分支执行流程)
  - [9.4 Decode 循环生成 + 增量 KV 持续挂载（`kv_committed_len` 递增）](#94-decode-循环生成--增量-kv-持续挂载kv_committed_len-递增)
  - [9.5 多轮对话缓存复用加速逻辑（`SessionRadixCache` 会话级缓存）](#95-多轮对话缓存复用加速逻辑sessionradixcache-会话级缓存)
  - [9.6 会话超时/结束资源回收链路：`release_kv_cache()` 完整流程](#96-会话超时结束资源回收链路release_kv_cache-完整流程)
  - [9.7 显存超限→触发淘汰→`evict()` → 可能触发 HiCache 下沉传输](#97-显存超限触发淘汰evict--可能触发-hicache-下沉传输)
  - [9.8 下级缓存命中→`load_back()` → `load_cpu_copy` 回灌 GPU](#98-下级缓存命中load_back--load_cpu_copy-回灌-gpu)
  - [9.9 GLM-5.2 场景推演：工具调用 / 长摘要 / 记忆裁剪特殊链路](#99-glm-52-场景推演工具调用--长摘要--记忆裁剪特殊链路)

**第五部分：HiCache 多级缓存工程优化**
- [第10章 SGLang HiCache 多级缓存架构原理](#第10章-sglang-hicache-多级缓存架构原理)
  - [10.1 三级存储层级定义](#101-三级存储层级定义)
    - [10.1.1 L1：GPU 显存（`KVCache` 物理池，原生读写，零拷贝）](#1011-l1gpu-显存kvcache-物理池原生读写零拷贝)
    - [10.1.2 L2：CPU 内存（`HostKVCache`，`pool_host/base.py`，pin_memory + DMA）](#1012-l2cpu-内存hostkvcachepool_hostbasepypin_memory--dma)
    - [10.1.3 L3：多后端存储层（`storage/`：file / mooncake_store / hf3fs / lmcache / nixl / eic / simm / aibrix_kvcache）](#1013-l3多后端存储层storagefile--mooncake_store--hf3fs--lmcache--nixl--eic--simm--aibrix_kvcache)
  - [10.2 多池类型管理：`PoolName` 枚举与 `PoolTransfer` 传输抽象](#102-多池类型管理poolname-枚举与-pooltransfer-传输抽象)
  - [10.3 `HiCacheController` + `HybridCacheController`：升降级调度与预取](#103-hicachecontroller--hybridcachecontroller升降级调度与预取)
    - [10.3.1 `PrefetchOperation`：预判后续访问路径，提前下发 H2D 传输](#1031-prefetchoperation预判后续访问路径提前下发-h2d-传输)
    - [10.3.2 `PoolHitPolicy`：命中策略与降级触发条件](#1032-poolhitpolicy命中策略与降级触发条件)
  - [10.4 与第 7 章传输链路的联动：`_cuda_host_unregister` / DMA / RDMA Batch](#104-与第-7-章传输链路的联动_cuda_host_unregister--dma--rdma-batch)
  - [10.5 HiCache 与 Disaggregation PD 的协同：`StorageMedium` 标记](#105-hicache-与-disaggregation-pd-的协同storagemedium-标记)
  - [10.6 工程稳定性方案：IO 限流、脏数据校验、过期会话自动清理](#106-工程稳定性方案io-限流脏数据校验过期会话自动清理)

**第六部分：源码导读、性能测评与技术展望**
- [第11章 核心源码路径导读](#第11章-核心源码路径导读)
  - [11.1 内存池与页表：`memory_pool.py`（`ReqToTokenPool` / `KVCache` 子类）→ `allocator/`（`TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` / `SWATokenToKVPoolAllocator`）](#111-内存池与页表memory_poolpyreqtotokenpool--kvcache-子类-allocatortokentokvpoolallocator--pagedtokentokvpoolallocator--swatokentokvpoolallocator)
  - [11.2 RadixCache 前缀匹配与淘汰：`radix_cache.py` → `hiradix_cache.py` → `unified_radix_cache.py` → `evict_policy.py`](#112-radixcache-前缀匹配与淘汰radix_cachepy--hiradix_cachepy--unified_radix_cachepy--evict_policypy)
  - [11.3 跨设备传输：`KVCache.get_cpu_copy/load_cpu_copy` → `disaggregation/decode.py` → `pool_host/` + `storage/`](#113-跨设备传输kvcacheget_cpu_copyload_cpu_copy--disaggregationdecodepy--pool_host--storage)
  - [11.4 Attention Backend 中的 paged KV 转换：`flashinfer_backend.py` → `triton_ops/kv_indices.py`](#114-attention-backend-中的-paged-kv-转换flashinfer_backendpy--triton_opskv_indicespy)
  - [11.5 调度入口与工具函数：`common.py`（`alloc_for_extend` / `release_kv_cache` / `write_cache_indices`）](#115-调度入口与工具函数commonpyalloc_for_extend--release_kv_cache--write_cache_indices)
- [第12章 多框架横向性能对比](#第12章-多框架横向性能对比)
  - [12.1 测试基线：vLLM (PagedAttention) / SGLang (RadixCache) / SGLang + HiCache](#121-测试基线vllm-pagedattention--sglang-radixcache--sglang--hicache)
  - [12.2 核心指标：TTFT、TPOT、QPS、显存占用、缓存命中率、传输时延](#122-核心指标ttfttpotqps显存占用缓存命中率传输时延)
- [第13章 架构局限与未来演进](#第13章-架构局限与未来演进)
  - [13.1 当前短板：树深度过高、`req_to_token` 对超长上下文的存储开销、PD offload 同步延迟](#131-当前短板)
  - [13.2 未来方向：Chunked Prefill 优化、分布式全局 KV Cache 集群、RDMA 零拷贝跨机传输、自适应多级缓存调度](#132-未来方向)

**附录**
- [A.1 传统原版 PagedAttention 原理回顾](#a1-传统原版-pagedattention-原理回顾)
- [A.2 `KVCache` 子类全景参考（`MHATokenToKVPool` / `MLATokenToKVPool` / `DSATokenToKVPool` / `HiSparseDSATokenToKVPool` / `DeepSeekV4TokenToKVPool`）](#a2-kvcache-子类全景参考)
- [A.3 `TokenToKVPoolAllocator` 子类全景参考（`TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` / `SWATokenToKVPoolAllocator` / `HiSparseTokenToKVPoolAllocator`）](#a3-tokentokvpoolallocator-子类全景参考)

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

#### `clear()`

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

[待补充]

#### 1.2.4 三层解耦优势：逻辑请求自由伸缩、物理显存统一池化

[待补充]

### 1.3 `req_to_token` 统一页表机制（per-token 直接映射）
- 1.3.1 `page_size=1`：per-token 粒度的 slot 映射
- 1.3.2 `page_size>1`：`PagedTokenToKVPoolAllocator` 隐式分页
- 1.3.3 与 vLLM BlockTable 的架构差异对比
- 1.3.4 flashinfer 后端中 `req_to_token` → paged KV 格式的转换

### 1.4 KV Cache 精细化元数据体系
- 1.4.1 请求维度：`kv_committed_len`、`kv_allocated_len`、`req_pool_idx`、`priority`、`time_stats`
- 1.4.2 前缀缓存维度：`prefix_indices`、`num_matched_prefix_tokens`、`host_hit_length`、`cache_protected_len`
- 1.4.3 SWA 维度：`swa_evicted_seqlen`、`sliding_window_size`
- 1.4.4 生命周期维度：`kv_committed_freed`、`kv_overallocated_freed`、`inflight_middle_chunks`

### 1.5 Chunked Prefill：长文本分段与 `req_pool_idx` 复用机制

### 1.6 工程踩坑与源码细节
- 1.6.1 `_alloc_size = size + 1`：索引 0 的 CUDA graph padding 约定
- 1.6.2 `need_sort` 与 `merge_and_sort_free`：碎片整理时机选择

## 第2章 SGLang RadixCache 基数树：全局前缀 KV 共享核心
### 2.1 RadixAttention 演进背景：PagedAttention 只能单请求复用
- 2.1.1 分页缓存只能「单请求复用」，无法「跨请求前缀复用」
- 2.1.2 企业级固定系统 Prompt 场景 70% 以上 KV 计算冗余

### 2.2 SGLang 中多种 Cache 类型的全景图
- 2.2.1 `RadixCache`（基础基数树缓存）
- 2.2.2 `HiRadixCache`（带 HiCache 多级存储的基数树）
- 2.2.3 `UnifiedRadixCache`（统一混合模型缓存，SSM + Attention）
- 2.2.4 `SWARadixCache` / `MambaRadixCache` / `ChunkCache` / `SessionRadixCache`

### 2.3 `TreeNode` 核心源码字段全解析
- 2.3.1 `key: RadixKey`（含 `token_ids` + `extra_key` + `is_bigram`）
- 2.3.2 `value` / `host_value`：设备侧与主机侧 KV 数据指针
- 2.3.3 `children` / `parent`：树拓扑结构
- 2.3.4 `lock_ref` / `host_ref_counter`：双层引用计数防误删
- 2.3.5 `last_access_time` / `hit_count` / `priority`：淘汰决策元数据
- 2.3.6 `hash_value` / `write_through_pending_id` / `creation_time`

### 2.4 最长公共前缀匹配算法流程
- 2.4.1 新请求从根节点逐层比对前缀 Token
- 2.4.2 完全匹配 / 部分匹配 / 零匹配三种分支处理逻辑

### 2.5 树节点分裂、新建、挂载、EvictionPolicy 多策略淘汰
- 2.5.1 前缀部分重合触发节点分裂
- 2.5.2 后缀增量生成叶子节点延伸与 Insert 挂载
- 2.5.3 `evict_policy.py`：可插拔淘汰策略（不限于 LRU+LFU）
- 2.5.4 `EvictParams` / `EvictResult`：逐出控制与后置处理

### 2.6 RadixCache 与 `req_to_token` + `token_to_kv_pool_allocator` 三者关系
- 2.6.1 树节点 value → 物理 slot 索引 → `allocator.free()` 释放链路
- 2.6.2 `PrefixCacheTrait` 协议：`req_to_token_pool` + `token_to_kv_pool_allocator` + `page_size`

### 2.7 引用计数 RC 联动机制：`lock_ref` + `host_ref_counter`
- 2.7.1 `IncLockRefResult` / `DecLockRefParams`：引用计数增减 API
- 2.7.2 计数归零触发分支销毁与 slot 回收

### 2.8 线上问题：前缀失效、树内存泄漏、共享缓存脏数据

# 第二部分：KV Cache 完整生命周期原理
## 第3章 KV Cache 生成机制：Prefill / Decode 双阶段
### 3.1 Prefill 预填充全量生成逻辑
- 3.1.1 张量并行批量 Prefill 计算流程图解
- 3.1.2 `alloc_for_extend()`：KV slot 分配 + `write_cache_indices()` 写入页表
- 3.1.3 前缀命中分支：`match_prefix()` → 跳过重复计算，复用历史树节点

### 3.2 Decode 增量单 Token 生成
- 3.2.1 单步 forward 产出 KV 切片
- 3.2.2 `alloc_decode()`：每 token 追加一个 slot，`kv_committed_len += 1`

### 3.3 惰性显存分配策略：不预占显存、随用随分
### 3.4 多并行架构下 KV 分片生成：TP 按 head 切分 / PP 按 layer 隔离
### 3.5 [GLM-5.2 适配] DSA 稀疏注意力的 token_mask 选择性 KV 生成

## 第4章 KV Cache 写入机制：显存固化与数据落地
### 4.1 GPU 原地零拷贝写入主路径：`KVCache.set_kv_buffer()`
- 4.1.1 `KVWriteLoc`：full pool + SWA pool 的二元写入目标
- 4.1.2 CUDA kernel 直接写入 Block 显存，张量视图复用，无 `clone/copy`

### 4.2 跨设备写入链路：`get_cpu_copy()` / `load_cpu_copy()` 同步 offload
### 4.3 追加写入（Decode 增量）vs 覆盖写入（上下文重置/窗口刷新）
### 4.4 Batch 并发写入锁竞争：RadixCache 全局读写锁与 Block 空闲池竞争规避
### 4.5 [GLM-5.2 适配] 超长文本分段写入与 SWA 滑动窗口截断写入

## 第5章 KV Cache 读取机制：Attention 计算核心链路
### 5.1 标准读取全流程：Query 计算 → `req_to_token` 查页表 → `create_flashinfer_kv_indices_triton` 转换为 paged KV → Attention kernel
### 5.2 `IndicesUpdater`：`req_to_token` → flashinfer `paged_kv_indices` 的 Triton 转换 kernel
### 5.3 零拷贝读取、in-place 视图复用、预取优化
### 5.4 多卡跨分片 KV 聚合读取：TP AllGather / EP 路由分发
### 5.5 多级缓存命中分支：L1 (GPU) 直接命中 / L2 (CPU HostKVCache) → `load_cpu_copy` / L3 (Storage Backend) → `PrefetchOperation`
### 5.6 [GLM-5.2 适配] RoPE 位置偏移修正与稀疏 Token 精准读取

## 第6章 KV Cache 淘汰与内存回收机制
### 6.1 显存水位线分级管控：软阈值降级 Swap / 硬阈值强制淘汰
### 6.2 `evict_policy.py`：可插拔淘汰策略（LRU / LFU / SLRU / FIFO）
### 6.3 [GLM-5.2 适配] SWA 窗口外 KV 强制过期 + DSA 无效 Token KV 主动释放
### 6.4 细粒度 slot 回收 vs 粗粒度整会话回收
### 6.5 双层 RC 防误删：`TreeNode.lock_ref` + `host_ref_counter`
### 6.6 淘汰后置：树分支修剪、`req_to_token` 索引刷新、slot 归还 `free_pages`、HiCache 下沉传输标记

# 第三部分：专项模块——KV Cache 跨设备传输机制
## 第7章 KV Cache 跨设备传输体系设计
### 7.1 传输场景全景：SGLang 中实际存在的三类传输
- 7.1.1 GPU↔CPU 请求级 Offload：`Req.offload_kv_cache()` / `load_kv_cache()`
- 7.1.2 Prefill→Decode 分离式传输（Disaggregation PD）：NCCL / NIXL 跨节点
- 7.1.3 HiCache 层级传输：HostKVCache ↔ Storage Backend 的后台数据流转

### 7.2 GPU↔CPU Offload 详细链路
- 7.2.1 `KVCache.get_cpu_copy()` / `load_cpu_copy()`：同步 D2H / H2D 拷贝
- 7.2.2 `TorchMemorySaverAdapter`：显存压缩与 Memory Saver 机制
- 7.2.3 `LayerDoneCounter`：layer-wise 传输控制（`register_layer_transfer_counter`）

### 7.3 Disaggregation PD 传输链路
- 7.3.1 `DecodeReqToTokenPool`：预分配 slot + 传输 slot 的分离池设计
- 7.3.2 NCCL 集合通信 vs NIXL 点对点传输 vs RDMA 零拷贝
- 7.3.3 `kv_cache_builder.py`：KV 数据序列化与反序列化
- 7.3.4 SWA allocator 的 `alloc_extend_swa_tail`：decode 端仅传输 SWA 尾部

### 7.4 HiCache 层级传输：HostKVCache ↔ Storage Backend
- 7.4.1 `PoolTransfer` / `PoolName`：多池类型的传输抽象
- 7.4.2 `GetPageContext` / `SetPageContext`：分页传输 API
- 7.4.3 RDMA Batch 操作与 `STORAGE_BATCH_SIZE` 批量化
- 7.4.4 `PrefetchTimeoutConfig`：超时控制的线性策略

### 7.5 [GLM-5.2 适配] 大 KV 量下的传输优化方向
- 7.5.1 MLA 低秩压缩 KV 减少传输字节量
- 7.5.2 DSA 稀疏 Mask 过滤仅传输有效 Token

### 7.6 传输链路常见故障与排坑

# 第四部分：模型专属适配与深度交互（以 DeepSeek V4 / GLM 系列为例）
## 第8章 非标准 Attention 架构对 KV Cache 的强约束
### 8.1 SGLang 中已有的非标准 KV Cache 实现全景
- 8.1.1 `MLATokenToKVPool`：MLA 低秩压缩 KV 的专用物理池
- 8.1.2 `DSATokenToKVPool`：DSA 稀疏注意力的专用物理池
- 8.1.3 `HiSparseDSATokenToKVPool` + `HiSparseTokenToKVPoolAllocator`：稀疏二级池
- 8.1.4 `DeepSeekV4TokenToKVPool`：c4/c128 多级压缩池体系

### 8.2 MLA（Multi-Head Latent Attention）KV Cache 存储对比分析
- 8.2.1 `kv_lora_rank` 压缩 vs 全量 KV 的存储/传输差异
- 8.2.2 flashinfer_mla_backend 中的 paged KV 转换适配

### 8.3 DSA（Dense-Sparse Attention）稀疏窗口机制
- 8.3.1 Dense Layer + Sparse Layer 交替架构下的双缓存设计
- 8.3.2 `sparsity/` 目录下的稀疏索引与压缩状态管理

### 8.4 RoPE 位置编码偏移引发的索引修正原理
### 8.5 Continuous Batch 动态批处理资源调度
### 8.6 FP8 量化 KV Cache：`store_dtype=torch.uint8` 的数值对齐与精度兼容
### 8.7 MoE 专家并行 EP 下多卡 KV 分布与路由
### 8.8 [GLM-5.2 推演] 结合 MLA + DSA + MoE 的综合 KV Cache 架构设计方向

## 第9章 SGLang 端到端全推理链路时序闭环
### 9.1 请求接入、分词、`Req` 对象元信息初始化
### 9.2 RadixCache `match_prefix()` 前缀匹配判定（命中/部分命中/未命中）
### 9.3 Prefill 全量 / Chunked Prefill 增量双分支执行流程
### 9.4 Decode 循环生成 + 增量 KV 持续挂载（`kv_committed_len` 递增）
### 9.5 多轮对话缓存复用加速逻辑（`SessionRadixCache` 会话级缓存）
### 9.6 会话超时/结束资源回收链路：`release_kv_cache()` 完整流程
### 9.7 显存超限→触发淘汰→`evict()` → 可能触发 HiCache 下沉传输
### 9.8 下级缓存命中→`load_back()` → `load_cpu_copy` 回灌 GPU
### 9.9 GLM-5.2 场景推演：工具调用 / 长摘要 / 记忆裁剪特殊链路

# 第五部分：HiCache 多级缓存工程优化
## 第10章 SGLang HiCache 多级缓存架构原理
### 10.1 三级存储层级定义
- 10.1.1 L1：GPU 显存（`KVCache` 物理池，原生读写，零拷贝）
- 10.1.2 L2：CPU 内存（`HostKVCache`，`pool_host/base.py`，pin_memory + DMA）
- 10.1.3 L3：多后端存储层（`storage/`：file / mooncake_store / hf3fs / lmcache / nixl / eic / simm / aibrix_kvcache）

### 10.2 多池类型管理：`PoolName` 枚举与 `PoolTransfer` 传输抽象
### 10.3 `HiCacheController` + `HybridCacheController`：升降级调度与预取
- 10.3.1 `PrefetchOperation`：预判后续访问路径，提前下发 H2D 传输
- 10.3.2 `PoolHitPolicy`：命中策略与降级触发条件

### 10.4 与第 7 章传输链路的联动：`_cuda_host_unregister` / DMA / RDMA Batch
### 10.5 HiCache 与 Disaggregation PD 的协同：`StorageMedium` 标记
### 10.6 工程稳定性方案：IO 限流、脏数据校验、过期会话自动清理

# 第六部分：源码导读、性能测评与技术展望
## 第11章 核心源码路径导读
### 11.1 内存池与页表：`memory_pool.py`（`ReqToTokenPool` / `KVCache` 子类）→ `allocator/`（`TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` / `SWATokenToKVPoolAllocator`）
### 11.2 RadixCache 前缀匹配与淘汰：`radix_cache.py` → `hiradix_cache.py` → `unified_radix_cache.py` → `evict_policy.py`
### 11.3 跨设备传输：`KVCache.get_cpu_copy/load_cpu_copy` → `disaggregation/decode.py` → `pool_host/` + `storage/`
### 11.4 Attention Backend 中的 paged KV 转换：`flashinfer_backend.py` → `triton_ops/kv_indices.py`
### 11.5 调度入口与工具函数：`common.py`（`alloc_for_extend` / `release_kv_cache` / `write_cache_indices`）

## 第12章 多框架横向性能对比
### 12.1 测试基线：vLLM (PagedAttention) / SGLang (RadixCache) / SGLang + HiCache
### 12.2 核心指标：TTFT、TPOT、QPS、显存占用、缓存命中率、传输时延

## 第13章 架构局限与未来演进
### 13.1 当前短板：树深度过高、`req_to_token` 对超长上下文的存储开销、PD offload 同步延迟
### 13.2 未来方向：Chunked Prefill 优化、分布式全局 KV Cache 集群、RDMA 零拷贝跨机传输、自适应多级缓存调度

## 附录
### A.1 传统原版 PagedAttention 原理回顾

论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（arXiv:2309.06180）

传统原版 PagedAttention = 把 LLM 推理的 KV 缓存做成操作系统虚拟内存分页系统，用离散固定大小显存块 + 页表映射替代整块连续内存预分配，根治 KV 缓存显存碎片化，大幅提升大模型在线服务并发吞吐量。

### A.2 `KVCache` 子类全景参考（`MHATokenToKVPool` / `MLATokenToKVPool` / `DSATokenToKVPool` / `HiSparseDSATokenToKVPool` / `DeepSeekV4TokenToKVPool`）

### A.3 `TokenToKVPoolAllocator` 子类全景参考（`TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` / `SWATokenToKVPoolAllocator` / `HiSparseTokenToKVPoolAllocator`）