# GLM 5.2 KV Cache 传输机制

## 目录

1. [概述](#1-概述)
2. [PD 分离架构总览](#2-pd-分离架构总览)
3. [传输协议与生命周期](#3-传输协议与生命周期)
4. [MLA/DSA 模型的传输特殊性](#4-mladsa-模型的传输特殊性)
5. [传输后端详解](#5-传输后端详解)
6. [同构 vs 异构 TP 传输](#6-同构-vs-异构-tp-传输)
7. [DeepSeek V4 (GLM 5.x 未来架构) 差异](#7-deepseek-v4-glm-5x-未来架构-差异)
8. [源码索引](#8-源码索引)

---

## 1. 概述

SGLang 支持 **Prefill-Decode (PD) 分离** 的 disaggregated serving 架构。预填充服务器计算提示词的 KV cache，然后通过网络（RDMA/InfiniBand 或 TCP）将其传输到解码服务器，解码服务器负责自回归生成。

GLM 5.2 采用 MLA + DSA 架构，其 KV cache 传输与标准 MHA 模型有本质区别：
- **MLA**: 每个 token 只有一个 `[1, 576]` 的紧凑 KV buffer，而非 MHA 的 K/V 分离格式
- **DSA**: 额外有一个 `index_k_with_scale_buffer`（Indexer KV Cache），需要随主 KV 一起传输

```
┌──────────────────┐         RDMA / TCP          ┌──────────────────┐
│   Prefill Node    │ ──────────────────────────> │   Decode Node     │
│                  │                              │                  │
│  1. 前向计算 KV   │   kv_buffer[layer]          │  3. 接收 KV      │
│  2. 写入内存池    │   index_k_with_scale[layer] │  4. 存入内存池    │
│     ↓            │   metadata (output_ids, ...) │     ↓            │
│  send_kv_chunk() │ ──────────────────────────> │  poll + commit   │
│                  │                              │     ↓            │
│                  │                              │  5. 自回归解码    │
└──────────────────┘                              └──────────────────┘
```

---

## 2. PD 分离架构总览

### 2.1 目录结构

```
python/sglang/srt/disaggregation/
├── base/conn.py              # 抽象基类: BaseKVSender, BaseKVReceiver, KVArgs, KVPoll
├── common/conn.py            # 共享实现: ZMQ 控制平面, PP/TP 映射, 心跳
├── common/utils.py           # TransferKVChunk, group_concurrent_contiguous, AuxDataCodec
├── common/staging_buffer.py  # GPU 暂存缓冲区 (Triton gather/scatter 内核)
├── common/staging_handler.py # 暂存生命周期 + 水位线协议
├── mooncake/conn.py          # Mooncake RDMA 后端
├── nixl/conn.py              # NIXL RDMA 后端 (UCX/OBJ/GDS_MT)
├── mori/conn.py              # MORI RDMA 后端
├── ascend/conn.py            # 昇腾 NPU 后端
├── fake/conn.py              # 测试用空操作后端
├── utils.py                  # TransferBackend 枚举, MetadataBuffers, poll_and_all_reduce
├── prefill.py                # 预填充端生命周期
├── decode.py                 # 解码端生命周期
├── decode_hicache_mixin.py   # 解码端 HiCache 集成
└── kv_events.py              # KV 事件发布 (ZMQ pub/sub)
```

### 2.2 核心抽象

```python
# base/conn.py
class KVPoll(enum.IntEnum):
    Failed = 0
    Bootstrapping = 1       # 正在握手
    WaitingForInput = 2     # 等待预填充端计算 KV
    Transferring = 3        # RDMA 传输进行中
    Success = 4             # 所有 chunk + aux 已到达

class StateType(enum.IntEnum):
    MAMBA = 1               # Mamba 状态
    SWA = 2                 # 滑动窗口注意力子池
    DSA = 3                 # DSA 索引器 KV 缓存 ★ GLM 5.2 关键类型
    SWA_RING = 4            # DeepSeek V4 SWA 环
```

### 2.3 预填充端队列

```
Client Request
     │
     ▼
PrefillBootstrapQueue   ← 创建 Sender, 向引导服务器注册
     │
     ▼
WaitingQueue            ← 等待解码端发来目标 KV 索引
     │
     ▼
Model Forward            ← 计算 KV cache
     │
     ▼
InflightQueue           ← send_kv_chunk() 分块发送
     │                    poll senders → 全部完成 → 归还请求
     ▼
Request Returned
```

### 2.4 解码端队列

```
预填充端通知
     │
     ▼
DecodePreallocQueue     ← 解析预填充信息, 预分配 KV slot
     │                    send_metadata → 发送目标索引给预填充
     ▼
DecodeTransferQueue     ← 轮询接收器 → 等待传输完成
     │                    poll_and_all_reduce (跨 TP 排名同步)
     ▼
WaitingQueue            ← _commit_transfer_to_req (读取 metadata)
     │                    构建 PrebuiltExtendBatch (跳过预填充 forward)
     ▼
RunningBatch            ← 合并到正在运行的解码批次
```

---

## 3. 传输协议与生命周期

### 3.1 阶段 1：引导注册

```
启动时:
  每个 Prefill Worker ──HTTP PUT──> BootstrapServer (aiohttp)
    注册信息: host, port, attn_tp_size, pp_size, dp_size,
             page_size, kv_cache_dtype, cp_size ...
```

```python
# common/conn.py :: CommonKVManager.register_to_bootstrap()
# POST /{host}:{port}/route
# body: json {
#   "rank": 0,
#   "attn_tp_size": 8,
#   "pp_size": 1,
#   "dp_size": 1,
#   "page_size": 1,
#   "kv_cache_dtype": "bfloat16",
#   "attn_cp_size": 1,
# }
```

BootstrapServer 等待所有预填充排名注册完毕 (`dp_size x cp_size x tp_size x pp_size` 个)，然后标记为 ready。

### 3.2 阶段 2：解码端握手

```
请求到达解码节点:
  1. DecodePreallocQueue.add(req)
     ├── 创建 KVReceiver
     │
  2. HTTP GET → BootstrapServer /route
     ├── 获取 PrefillServerInfo
     │   (预填充 ZMQ IP:Port, TP/PP/CP 信息)
     │
  3. _resolve_rank_mapping()
     ├── 计算预填充 TP rank → 解码 TP rank 映射
     ├── 处理 attn_tp_size 不同的情况
     └── 处理 PP/CP 映射
     │
  4. ZMQ PUSH → 预填充节点 (KVArgsRegisterInfo)
     ├── dst_kv_data_ptrs: 目标 GPU buffer 指针
     ├── dst_aux_data_ptrs: 元数据 buffer 指针
     ├── gpu_id, tp_rank, pp_rank, attn_cp_rank
     └── gpu_memory_pool_id
     │
  5. 预填充端存储到 decode_kv_args_table
     状态: Bootstrapping → WaitingForInput
```

### 3.3 阶段 3：预填充 KV 分配 + 索引发送

```
解码端:
  1. DecodePreallocQueue._pre_alloc()
     ├── allocator.alloc(seq_len) → kv_indices
     ├── req_to_token[req_pool_idx, :] = kv_indices
     └── 预分配给请求的 KV cache slot
     │
  2. ZMQ PUSH → 预填充节点 (TransferInfo)
     body: {
       room, endpoint, dst_port,
       dst_kv_indices: int32[],    ← 目标 slot 索引
       dst_aux_index: int,
       dst_state_indices: {         ← 状态缓冲区索引
         SWA: int32[],
         MAMBA: int32[],
         DSA:   int32[],            ← ★ GLM 5.2 的 indexer KV
         SWA_RING: int32[],
       },
       decode_prefix_len: int,
     }
     │
  3. 预填充端存储到 transfer_infos
     状态: WaitingForInput (等待被调度)
```

### 3.4 阶段 4：预填充前向 + KV 分块发送

```
预填充端:
  1. get_next_disagg_prefill_batch_to_run()
     └── 从 WaitingQueue 中选取请求, 组 batch

  2. Model forward
     └── KV cache 写入预填充端内存池

  3. process_batch_result_disagg_prefill()
     └── for each request:
           if chunked_prefill:
             send_kv_chunk(req, last_chunk=False)   ← 每完成一个 chunk 就发
           else:
             send_kv_chunk(req, last_chunk=True)

  4. send_kv_chunk(req, last_chunk)
     ├── 提取 prefill_kv_indices (page-level, int32)
     ├── 切片为 TransferKVChunk:
     │     room, prefill_kv_indices, index_slice, is_last_chunk,
     │     prefill_aux_index, state_indices, chunk_id
     └── sender.send(chunk) → 放入 FastQueue

  5. Background transfer_worker():
     ├── 从 FastQueue 获取 chunk
     ├── 选择传输路径 (generic/slice/staged)
     ├── 构建 RDMA 传输块
     └── 发布 RDMA WRITE 操作
```

### 3.5 阶段 5：RDMA 传输

传输引擎直接操作 GPU 内存。每层的源地址 = `kv_data_ptrs[layer_id] + src_index * item_len`，目标地址 = `dst_kv_data_ptrs[layer_id] + dst_index * item_len`：

```
GPU Memory (Prefill)                          GPU Memory (Decode)
┌─────────────────────┐                       ┌─────────────────────┐
│ kv_buffer[layer_0]  │                       │ kv_buffer[layer_0]  │
│  slot[17]: [1, 576] │ ──── RDMA WRITE ────> │  slot[42]: [1, 576] │
│  slot[18]: [1, 576] │ ──── RDMA WRITE ────> │  slot[43]: [1, 576] │
│  ...                │                       │  ...                │
├─────────────────────┤                       ├─────────────────────┤
│ kv_buffer[layer_1]  │                       │ kv_buffer[layer_1]  │
│  slot[17]: [1, 576] │ ──── RDMA WRITE ────> │  slot[42]: [1, 576] │
│  slot[18]: [1, 576] │ ──── RDMA WRITE ────> │  slot[43]: [1, 576] │
│  ...                │                       │  ...                │
├─────────────────────┤                       ├─────────────────────┤
│ index_k_with_scale  │                       │ index_k_with_scale  │
│  buffer[layer_X]    │ ──── RDMA WRITE ────> │  buffer[layer_X]    │
└─────────────────────┘                       └─────────────────────┘
         ↑                                             ↑
    直接 GPU 内存访问                           直接 GPU 内存访问
    (通过 RDMA 网卡)                           (通过 RDMA 网卡)
```

### 3.6 阶段 6：完成与提交

```
预填充端:
  process_disagg_prefill_inflight_queue()
    └── poll senders → KVPoll.Success → 归还请求

解码端:
  DecodeTransferQueue.pop_transferred()
    ├── poll receivers (每个 TP rank 独立)
    ├── poll_and_all_reduce (跨 TP rank 取 MIN)
    │     └── 所有 TP rank 都 Success 才算完成
    │
    └── _commit_transfer_to_req()
        ├── 从 MetadataBuffers 读取:
        │     output_ids, logprobs, cached_tokens
        │     output_hidden_states (EAGLE 推测解码用)
        │     bootstrap_room (损坏检测)
        ├── 填充 req 对象
        └── 放入 WaitingQueue → 构建 PrebuiltExtendBatch
```

### 3.7 故障处理

```python
# common/conn.py
class KVTransferError(Exception):
    bootstrap_room: int   # 发生故障的 room ID

# 心跳检测 (ZMQ 超时):
# - 预填充端定期发送心跳
# - 解码端超时未收到心跳 → 中止传输
# - failed_sessions 集合跟踪故障会话

# 中止流程:
# 预填充: Sender.abort(room)
# 解码:   Receiver.abort(room)
# 双方:   释放已分配的 KV slot, 归还请求
```

---

## 4. MLA/DSA 模型的传输特殊性

### 4.1 MLA 模型（GLM 5.2 主 KV Cache 传输）

标准 MLA 模型的 KV 布局与 MHA 完全不同：

**MHA 模型** (`MHATokenToKVPool`):
```
kv_data_ptrs = [
    layer_0_k_ptr, ..., layer_N_k_ptr,   ← K buffer per layer
    layer_0_v_ptr, ..., layer_N_v_ptr,   ← V buffer per layer
]
item_len = head_num x head_dim x dtype_size x page_size
```

**MLA 模型** (`MLATokenToKVPool`):
```
kv_data_ptrs = [
    layer_0_kv_ptr, ..., layer_N_kv_ptr,  ← 单一 KV buffer per layer
]
item_len = (kv_lora_rank + qk_rope_head_dim) x dtype_size x page_size
         = 576 x 2 x 1 = 1152 bytes (BF16, page_size=1)
```

```python
# memory_pool.py:2212 MLATokenToKVPool.get_contiguous_buf_infos()
def get_contiguous_buf_infos(self):
    # MLA 只有一个 kv_buffer，所以只返回此 buffer 的信息
    kv_data_ptrs = [kv_buffer[i].data_ptr() for i in range(layer_num)]
    kv_data_lens = [kv_buffer[i].nbytes for i in range(layer_num)]
    kv_item_lens = [kv_buffer[i][0].nbytes * page_size for i in range(layer_num)]
    return kv_data_ptrs, kv_data_lens, kv_item_lens
```

关键区别：MLA 返回的是**单一 buffer 指针列表**（每层一个），而 MHA 返回的是 **2x 层数的指针列表**（K + V 分离）。

### 4.2 DSA 模型的状态传输（Indexer KV Cache）

DSA（GLM 5.2）的 Indexer KV Cache 作为 `StateType.DSA` 单独传输：

```python
# GLM 5.2 需要传输两层数据:
#
# 1. 主 KV Cache (StateType=无, 自动传输):
#    kv_buffer[layer]: [size, 1, 576]  ← MLA 格式
#
# 2. DSA Indexer KV Cache (StateType=DSA):
#    index_k_with_scale_buffer[layer]: [pages, 8448] uint8  ← 页对齐 FP8
```

传输流程中的 DSA 特殊处理：

```python
# prefill.py :: send_kv_chunk() 中
def _dsa_payload(req, ...):
    # DSA 需要在最后一帧打包完整的 req_to_token 内容
    # 因为 Indexer KV Cache 的 page-level 索引与主 KV 可能不一致
    pass

# utils.py :: setup_state_kv_args()
# 将 DSA 索引器缓冲区指针注册为状态类型:
kv_args.state_types.append(StateType.DSA)
kv_args.state_data_ptrs.append(indexer_buffer_ptrs)
kv_args.state_data_lens.append(indexer_buffer_lens)
kv_args.state_item_lens.append(indexer_item_lens)

# common/conn.py :: CommonKVManager.get_state_buf_infos()
# 为 DSA 类型获取 Indexer buffer 信息:
if state_type == StateType.DSA:
    # 返回 index_k_with_scale_buffer 的指针/大小信息
```

### 4.3 MLA 的 TP 传输策略

MLA 的一个关键优势：由于 KV 是 `[1, kv_lora_rank + rope]`（1 个 KV head），**TP 排名之间没有头部切分的概念**。

```python
# common/conn.py :: get_mla_kv_ptrs_with_pp()
def get_mla_kv_ptrs_with_pp(kv_args, decode_tp_rank, ...):
    # MLA: 每层的单个 kv_buffer 指针
    # 不需要像 MHA 那样做 K/V 分离
    # TP 大小不同时: 整个 kv_buffer 直接传输 (无头部分片)
```

这意味着 MLA 模型在预填充和解码使用不同 `attn_tp_size` 时，走的是 **generic 全量传输路径**，不需要 per-token per-head 的 slice 路径。

---

## 5. 传输后端详解

SGLang 支持多种传输后端，通过 `--disaggregation-transfer-backend` 选择。

### 5.1 后端对比

| 后端 | 传输方式 | 协议 | 适用硬件 | 特点 |
|------|---------|------|---------|------|
| **Mooncake** | RDMA (GPU direct) | InfiniBand / RoCE / EFA | NVIDIA, 昇腾 | 默认后端, 最成熟 |
| **NIXL** | RDMA (GPU direct) | UCX / OBJ / GDS_MT / UCCL | NVIDIA | 插件化, 支持磁盘 offload |
| **MORI** | RDMA (GPU direct) | InfiniBand | NVIDIA | 高性能 |
| **Ascend** | NPU direct | HCCS | 昇腾 NPU | 继承 Mooncake, NPU 适配 |
| **Fake** | 无传输 | 无 | 任意 | 测试/预热 |

### 5.2 Mooncake 传输引擎

```python
# mooncake_transfer_engine.py
class MooncakeTransferEngine:
    def __init__(self, ...):
        # 协议选择: rdma, efa, tcp, ascend
        self.engine = mooncake.engine.TransferEngine(protocol)

    def initialize(self, local_hostname, metadata, ...):
        # 初始化 RDMA 连接

    def register(self, ptr, length):
        # 注册单个 GPU buffer

    def batch_register(self, ptrs, lengths):
        # 批量注册所有层的 buffer

    def transfer_sync(self, session_id, src_ptr, dst_ptr, length):
        # 同步 RDMA 写入

    def batch_transfer_sync(self, session_id, transfers):
        # 批量同步 RDMA 写入
```

### 5.3 传输工作线程（预填充端）

```python
# mooncake/conn.py :: MooncakeKVManager.transfer_worker()
def transfer_worker(self, transfer_queue, ...):
    while True:
        chunk = transfer_queue.get()  # TransferKVChunk

        # 1. 确定传输路径
        if is_mla or equal_tp:
            send_kvcache(chunk)         # 全量传输
        elif use_staging:
            _do_staging_transfer(chunk) # 暂存路径 (gather→bulk RDMA→scatter)
        else:
            send_kvcache_slice(chunk)   # 逐 token 逐 head 传输
```

### 5.4 传输路径详解

**路径 1: 全量传输 (MLA / 等 TP)**

```python
# mooncake/conn.py :: _send_kvcache_generic()
def _send_kvcache_generic(self, chunk, ...):
    # 1. 将 KV 索引按连续性分组
    src_indices = group_concurrent_contiguous(chunk.prefill_kv_indices)
    dst_indices = group_concurrent_contiguous(chunk.dst_kv_indices)

    # 2. 每层独立传输
    for layer_id in range(num_layers):
        for seg_src, seg_dst in zip(src_indices, dst_indices):
            src_addr = kv_data_ptrs[layer_id] + seg_src * item_len
            dst_addr = dst_kv_ptrs[layer_id] + seg_dst * item_len
            length = len(seg_src) * item_len

            engine.transfer_sync(session_id, src_addr, dst_addr, length)
```

**路径 2: 逐 Token 分片传输 (异构 TP)**

```python
# mooncake/conn.py :: send_kvcache_slice()
def send_kvcache_slice(self, chunk, ...):
    # 用于预填充解码 TP 大小不同时的 MHA 模型
    # 每个 token 每个 head 单独 RDMA
    for token_idx, (src_idx, dst_idx) in enumerate(zip(src_indices, dst_indices)):
        for head_slice in range(num_head_slices):
            src_addr = kv_ptrs[layer] + src_idx * full_item_len + head_offset
            dst_addr = dst_ptrs[layer] + dst_idx * full_item_len + head_offset
            engine.transfer_sync(session_id, src_addr, dst_addr, slice_len)
```

**路径 3: 暂存缓冲区路径 (异构 TP, 批量)**

```python
# staging_buffer.py :: StagingBuffer
# 预填充端: gather_all_layers_to_staging
#   将分散的 token slot 收集到连续 staging buffer
#   使用 Triton fused kernel: _fused_gather_to_staging_kernel

# 解码端: scatter_staging_to_kv
#   从 staging buffer 分散回 KV cache
#   使用 Triton fused kernel: _fused_scatter_from_staging_kernel
```

### 5.5 NIXL 后端的特殊特性

NIXL 最独特的特性是 **预构建描述符列表 (prep_xfer_dlist)**：

```python
# nixl/conn.py :: NixlKVManager._init_equal_tp_prep_handle()
def _init_equal_tp_prep_handle(self, ...):
    # 提前构建 NIXL 描述符列表
    # 每个 slot 位置对应一个 RDMA 描述符
    # 传输时直接 make_prepped_xfer() 而不需要重新枚举
    dlist = agent.prep_xfer_dlist()
    for layer_id in range(num_layers):
        for slot_id in range(num_slots):
            src_desc = nixlDesc(src_ptr + slot_id * item_len, item_len)
            dst_desc = nixlDesc(dst_ptr + slot_id * item_len, item_len)
            dlist.add_descriptor(src_desc, dst_desc)
    agent.store_prep_handle(name, dlist)
```

这使得 NIXL 在实际传输时可以跳过逐个描述的构建，直接发布预构建的传输句柄。

---

## 6. 同构 vs 异构 TP 传输

### 6.1 同构 TP (预填充 TP = 解码 TP)

预填充和解码使用相同的 `attn_tp_size`，每个 TP 排名的 KV 分片可直接对应：

```
Prefill TP Rank 0    ── RDMA ──>    Decode TP Rank 0
  kv[slot_i] [0:head_dim/tp]          kv[slot_j] [0:head_dim/tp]

Prefill TP Rank 1    ── RDMA ──>    Decode TP Rank 1
  kv[slot_i] [head_dim/tp:2*head_dim/tp]  kv[slot_j] [head_dim/tp:2*head_dim/tp]
```

对 MLA 模型而言，由于其 KV 结构是 `[1, 576]`（无头部分片），TP 排名之间传输的是**相同的完整 KV**（每个 rank 都有完整副本）。

### 6.2 异构 TP (预填充 TP ≠ 解码 TP)

当 TP 大小不同时，需要"重新分片"：

```
Prefill TP=8 (每 rank 1 个 head 的片段):
  Rank 0: head [0/8]
  Rank 1: head [1/8]
  ...

转换为:
Decode TP=4 (每 rank 2 个 head 的片段):
  Rank 0: head [0/8] + head [1/8]
  Rank 1: head [2/8] + head [3/8]
  ...
```

SGLang 支持三种路径：
1. **Slice 路径**：逐 token 逐 head 的细粒度 RDMA（低效）
2. **Generic 路径**（MLA 默认）：全量传输整个 buffer（高效）
3. **Staging 路径**：Gather → 批量 RDMA → Scatter（中等效率，适用于 MHA）

#### 暂存缓冲区工作流

```
Prefill Side                              Decode Side
┌──────────────────┐                     ┌──────────────────┐
│ KV Cache Pages   │                     │ KV Cache Pages   │
│ (scattered)      │                     │ (scattered)      │
│                  │                     │                  │
│ gather_to_staging│                     │ scatter_from     │
│      ↓           │                     │ _staging  ↑      │
│ Staging Buffer   │                     │ Staging Buffer   │
│ (contiguous)     │ ── bulk RDMA ──>   │ (contiguous)     │
└──────────────────┘                     └──────────────────┘

Watermark 协议 (通过 ZMQ):
  解码端: staging_allocator → 水位线状态 → ZMQ PUSH → 预填充端
  预填充端: PrefillStagingStrategy → 检查水位线 → 允许/阻止传输
```

---

## 7. DeepSeek V4 (GLM 5.x 未来架构) 差异

DeepSeek V4 更进一步使用了压缩注意力（c4/c128 状态环），其 KV 传输也更复杂：

| 特性 | MLA (GLM 5.2) | DeepSeek V4 |
|------|--------------|-------------|
| 主 buffer 类型 | 单一 `kv_buffer` per layer | 统一 buffer，按压缩比分桶 (c4, c128) |
| Buffer 布局 | 按层顺序 | 按 buffer 类型分组 (c4 层, c4 indexer, c128 层) |
| SWA 传输 | 无 | `StateType.SWA` + `StateType.SWA_RING` |
| PP 切片 | `get_mla_kv_ptrs_with_pp()` | `_mla_slice_ptrs_for_pp()` (按压缩比例重新排列) |
| 每 token 大小 | 576 维 | 多个 buffer 的加权和 |

```python
# DeepSeek V4 buffer 布局示例:
# kv_data_ptrs = [
#   # c4 buffer (每个 c4 层)
#   c4_layer_0_kv, c4_layer_1_kv, ..., c4_layer_L_kv,
#   # c4 indexer buffer
#   c4_idx_layer_0, ..., c4_idx_layer_L,
#   # c128 buffer (每个 c128 层)
#   c128_layer_0_kv, ..., c128_layer_L_kv,
# ]
```

---

## 8. 源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| | | **抽象基类** |
| `.../disaggregation/base/conn.py` | 1-83 | `KVPoll`, `StateType`, `KVArgs`, `BaseKVManager/Sender/Receiver/BootstrapServer` |
| | | **共享实现 (ZMQ 控制平面)** |
| `.../disaggregation/common/conn.py` | 108 | `CommonKVManager` 类 |
| `.../disaggregation/common/conn.py` | 280-357 | `_resolve_rank_mapping()` TP/PP/CP 映射 |
| `.../disaggregation/common/conn.py` | 389-449 | `register_to_bootstrap()` HTTP 注册 |
| `.../disaggregation/common/conn.py` | 490-521 | `get_mha_kv_ptrs_with_pp()` MHA PP 切片 |
| `.../disaggregation/common/conn.py` | 523-550 | `get_mla_kv_ptrs_with_pp()` **MLA PP 切片** |
| `.../disaggregation/common/conn.py` | 552-652 | `_mla_slice_ptrs_for_pp()` DSv4 PP 切片 |
| `.../disaggregation/common/conn.py` | 753 | `CommonKVSender` 类 |
| `.../disaggregation/common/conn.py` | 929 | `CommonKVReceiver` 类 |
| `.../disaggregation/common/conn.py` | 1202 | `CommonKVBootstrapServer` 类 |
| | | **MOONCAKE 传输后端** |
| `.../mooncake/conn.py` | 153 | `MooncakeKVManager` 类 |
| `.../mooncake/conn.py` | 474-569 | `send_kvcache_staged()` 暂存路径 |
| `.../mooncake/conn.py` | 580-686 | `_send_kvcache_generic()` **MLA 全量传输** |
| `.../mooncake/conn.py` | 706-827 | `send_kvcache_slice()` 异构 TP 分片 |
| `.../mooncake/conn.py` | 921 | `maybe_send_extra()` **DSA/MAMBA/SWA 状态传输** |
| `.../mooncake/conn.py` | 1158 | `transfer_worker()` 传输工作线程 |
| `.../mooncake/conn.py` | 1393 | `start_prefill_thread()` ZMQ 控制线程 |
| `.../mooncake/conn.py` | 1661 | `MooncakeKVSender` 类 |
| `.../mooncake/conn.py` | 1760 | `MooncakeKVReceiver` 类 |
| `.../distributed/.../mooncake_transfer_engine.py` | 99 | `MooncakeTransferEngine` RDMA 引擎 |
| | | **NIXL 传输后端** |
| `.../nixl/conn.py` | 239 | `NixlKVManager` 类 |
| `.../nixl/conn.py` | 555-677 | `_init_hetero_tp_prep_handle()` 异构 TP 预构建 |
| `.../nixl/conn.py` | 699 | `transfer_worker()` |
| `.../nixl/conn.py` | 960 | `_send_kvcache_generic()` 核心传输 |
| `.../nixl/conn.py` | 1125 | `send_kvcache_slice()` 分片路径 |
| `.../nixl/conn.py` | 1171 | `send_kvcache_staged()` 暂存路径 |
| `.../nixl/conn.py` | 1689 | `update_transfer_status()` 完成轮询 |
| `.../nixl/conn.py` | 1893 | `NixlKVSender` 类 |
| `.../nixl/conn.py` | 1991 | `NixlKVReceiver` 类 |
| | | **暂存系统** |
| `.../common/staging_buffer.py` | 117 | `StagingBuffer` GPU 暂存空间 |
| `.../common/staging_buffer.py` | 161 | `StagingAllocator` 环状分配器 |
| `.../common/staging_buffer.py` | 461 | `gather_all_layers_to_staging()` Triton 融合 gather |
| `.../common/staging_buffer.py` | 649 | `scatter_staging_to_kv()` Triton 融合 scatter |
| `.../common/staging_handler.py` | 56 | `DecodeStagingHandler` 分散管理 |
| `.../common/staging_handler.py` | 492 | `PrefillStagingStrategy` 水位线检查 |
| | | **传输工具** |
| `.../common/utils.py` | 17-30 | `TransferKVChunk` 数据结构 |
| `.../common/utils.py` | 54-63 | `pack_int_lists()` / `unpack_int_lists()` 状态索引序列化 |
| `.../common/utils.py` | 85-102 | `AuxDataCodec` 辅助数据编解码 |
| `.../common/utils.py` | 105-129 | `group_concurrent_contiguous()` 连续块分组 |
| | | **调度器集成** |
| `.../disaggregation/utils.py` | 197 | `MetadataBuffers` 元数据缓冲区 |
| `.../disaggregation/utils.py` | 384 | `TransferBackend` 枚举 |
| `.../disaggregation/utils.py` | 422 | `get_kv_class()` 后端工厂 |
| `.../disaggregation/utils.py` | 511 | `poll_and_all_reduce()` 跨 TP 同步 |
| `.../disaggregation/utils.py` | 537 | `setup_state_kv_args()` **DSA 状态注册** |
| `.../disaggregation/prefill.py` | - | `PrefillBootstrapQueue`, `send_kv_chunk()`, inflight 轮询 |
| `.../disaggregation/decode.py` | - | `DecodePreallocQueue`, `DecodeTransferQueue`, `_commit_transfer_to_req()` |
| | | **内存池传输接口** |
| `.../mem_cache/memory_pool.py` | 595-622 | `MHATokenToKVPool.get_contiguous_buf_infos()` MHA KV 指针 |
| `.../mem_cache/memory_pool.py` | 2212-2219 | `MLATokenToKVPool.get_contiguous_buf_infos()` **MLA KV 指针** |
| `.../mem_cache/memory_pool.py` | 2529 | `DSATokenToKVPool` **DSA 内存池** |
| `.../mem_cache/memory_pool.py` | 2704-2727 | `DSATokenToKVPool.get_cpu_copy()` 含 Indexer KV 的 CPU offload |
| `.../mem_cache/memory_pool.py` | 2729-2760 | `DSATokenToKVPool.get_state_buf_infos()` **DSA 状态 buffer 信息** |
| `.../mem_cache/memory_pool.py` | 1997 | `HybridLinearKVPool.get_state_buf_infos()` Mamba 状态 |
| `.../mem_cache/deepseek_v4_memory_pool.py` | 642 | `DeepSeekV4TokenToKVPool.get_contiguous_buf_infos()` DSv4 |
| | | **HiCache 集成** |
| `.../disaggregation/decode_hicache_mixin.py` | - | `HiCacheRestoreGatedKVReceiver` 恢复门控 |
| `.../disaggregation/decode_kvcache_offload_manager.py` | - | L1→L2 卸载生命周期 |
| `.../disaggregation/kv_events.py` | - | KV 事件 pub/sub (ZMQ) |

---

## 附录：GLM 5.2 传输关键要点

1. **MLA 单 buffer 传输**: 每层一个 `kv_buffer` 指针，不需要像 MHA 那样分离 K/V。
2. **DSA 状态单独传输**: `index_k_with_scale_buffer` 作为 `StateType.DSA` 随主 KV 一起传输。
3. **TP 全量传输**: MLA 只有 1 个 KV head，不同 TP 大小下走 generic 全量传输，不需要 head slice。
4. **Page 对齐**: 主 KV 的 `page_size=1`（token 级），Indexer KV 的 `page_size=64`（批量高效）。
5. **RDMA 直传**: KV 数据直接从 GPU 内存通过 RDMA 传输，无需 CPU 中转。
6. **ZMQ 控制平面**: 引导、握手、元数据交换通过 ZMQ + HTTP 完成，与 RDMA 数据平面分离。
7. **跨 TP 同步**: 解码端通过 `poll_and_all_reduce(MIN)` 确保所有 TP 排名传输完成。
8. **分块传输**: 长 prompt 的 chunked prefill 支持增量传输，传输与计算重叠。
9. **HiCache 集成**: 解码端支持 L1(GPU)/L2(Host)/L3(SSD) 分层缓存，可与 RDMA 传输协作。
