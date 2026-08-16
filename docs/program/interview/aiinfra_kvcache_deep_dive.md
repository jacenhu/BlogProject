# KV Cache 核心知识体系

> 首次编写：2026-08-12 | 最后更新：2026-08-12

> **AI Infra / KV Cache 面试系列** · 本文是技术主线
> - [导读](aiinfra_overview.md)
> - [招聘岗位全景](aiinfra_jobs.md)
> - **KV Cache 核心知识体系** ← 当前文章
> - [面经与高频考点](aiinfra_interview_questions.md)
> - [系统设计专题](aiinfra_system_design.md)
> - [2 个月面试复习计划](aiinfra_2month_plan.md)
> - [资源汇总](aiinfra_resources.md)

本文把 KV Cache 拆成六个递进的子问题：**为什么需要 -> 占多少显存 -> 怎么存 -> 怎么变小 -> 怎么复用 -> 怎么调度生命周期**。每一节都标注面试高频度（⭐越多越常考），并附"考法"与"答法要点"。

---

## 1. 为什么需要 KV Cache ⭐⭐⭐⭐⭐

### 1.1 自回归解码的重复计算问题

Transformer Decoder 是**自回归（autoregressive）**的：每生成一个新 token，都要对"整段历史"做 self-attention。若不缓存，生成第 $t$ 个 token 时需要重新计算前 $t-1$ 个 token 的 K、V，复杂度 $O(t)$；生成 $n$ 个 token 的总复杂度 $O(n^2)$。

更关键的是：**历史 token 的 K、V 不依赖新 token**（K、V 由历史 token 自身经线性投影得到）。所以可以把它们缓存下来，下一步直接复用——这就是 KV Cache。

> Q 不缓存：Q 由"当前 token"算出，每步只有一个新 Q，无需缓存历史 Q。

### 1.2 带/不带 Cache 的复杂度对比

| 模式 | 单步计算 | 生成 $n$ token 总计算 |
| --- | --- | --- |
| 无 Cache | $O(t)$（重算全部历史 K/V + attention） | $O(n^2)$ |
| 有 Cache | $O(1)$（只算新 token 的 K/V，attention 读 cache） | $O(n)$ |

**代价**：用空间换时间——把历史 K/V 存下来。下一个问题就是这个"空间"有多大。

### 1.3 面试考法

- "为什么 Transformer 推理要缓存 K 和 V，而不缓存 Q？" → Q 每步都是新的且只用于当前步；K/V 要被所有未来 step 的 attention 复用。
- "KV Cache 是用空间换时间，换来的是什么复杂度下降？" → 单步 $O(t)\to O(1)$，整体 $O(n^2)\to O(n)$。
- "训练时为什么不用 KV Cache？" → 训练时整个序列已知，并行计算所有位置，没有"逐步生成"的依赖链；且训练 forward 一次就出全部 logits，缓存无意义。

---

## 2. KV Cache 显存建模 ⭐⭐⭐⭐⭐

这是 KV Cache 方向**最高频的笔试/口算题**，务必熟练。

### 2.1 基本公式

对单条请求、单个 token，KV Cache 占用：

$$
\text{PerToken} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot p
$$

- `2`：K 和 V 两份。
- `L`：Transformer 层数。
- `n_kv`：KV 头数（MHA 时 = q 头数；GQA/MQA 时小于 q 头数）。
- `d_h`：每个头的维度（head_dim）。
- `p`：每个元素字节数（fp16/bf16 = 2，int8 = 1，fp8 = 1）。

对 batch 为 $b$、序列长度 $s$ 的请求组：

$$
\text{KVCache} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot s \cdot b \cdot p
$$

### 2.2 经典口算（必背）

**Llama-2-7B（MHA）**：$L=32,\ n_{kv}=32,\ d_h=128,\ p=2$

$$
\text{PerToken} = 2 \times 32 \times 32 \times 128 \times 2 = 524288\ \text{B} = 512\ \text{KB}
$$

单条 4096 上下文：$512\text{KB} \times 4096 \approx 2\ \text{GB}$。而 7B 权重 fp16 才 ~14GB——**一条长上下文请求的 KV Cache 就能吃掉权重的 1/7**。batch 一上去，KV Cache 远超权重。

**Llama-3-8B（GQA, n_kv=8）**：$L=32,\ n_{kv}=8,\ d_h=128,\ p=2$

$$
\text{PerToken} = 2 \times 32 \times 8 \times 128 \times 2 = 131072\ \text{B} = 128\ \text{KB}
$$

GQA 相比 MHA 把 KV Cache 压到 1/4（32→8 头）。

**DeepSeek-V3（MLA）**：缓存压缩 latent $d_c=512$ + 解耦 RoPE 部分 $d_{rope}=64$，$L=61,\ p=2$

$$
\text{PerToken} = (512 + 64) \times 2 \times 61 = 70272\ \text{B} \approx 69\ \text{KB}
$$

对比"等价 MHA"（128 头 × 128 维）：$2\times128\times128\times2\times61 \approx 3.9\ \text{MB/token}$。**MLA 把单 token KV Cache 压了约 57 倍**——这是 MLA 最直接的卖点。

### 2.3 为什么 KV Cache 是推理瓶颈

1. **权重是常驻、可摊销的**；KV Cache 随 batch × seq 线性增长，并发越高涨得越快。
2. **Decode 是 memory-bound**：每生成一个 token，要把整段 KV Cache 从显存搬到 SM 做点积。算力（FLOPs）几乎不增加，但**访存量随 seq 线性增加**。HBM 带宽打满前，算力大量闲置。
3. 因此推理优化的核心矛盾是：**如何让 KV Cache 更小（省显存、减访存）、更好复用（省算力）、更好调度（提吞吐、降延迟）**。后文全部围绕这三点。

### 2.4 面试考法

- "7B 模型在 A100-80G 上跑 8K 上下文、batch 32，KV Cache 占多少？还剩多少给权重和激活？" → 代公式，结论是 KV Cache 可能 50GB+，权重 14GB，已经很紧。
- "为什么 decode 阶段算力利用率（MFU）很低？" → memory-bound，瓶颈是 HBM 带宽不是算力；增大 batch 把 GEMV 拼成 GEMM 才能提高利用率。
- "增大 batch 能提升 decode 吞吐吗？极限在哪？" → 能，直到 HBM 带宽打满或 KV Cache 撑爆显存。后者先到。

---

## 3. KV Cache 的存储管理 ⭐⭐⭐⭐⭐

### 3.1 朴素连续分配的问题

最直观的方案：为每个请求预分配一段**连续的**、按 `max_seq_len` 大小的 KV 空间。问题：

- **内部碎片（internal fragmentation）**：实际生成长度远小于 max_seq_len 时，预分配空间大量浪费。实测利用率可低至 20%~40%。
- **外部碎片（external fragmentation）**：请求频繁进出，显存被切成碎片，新请求可能因"没有足够大的连续块"而无法接入，尽管总剩余空间够。
- **无法共享**：相同前缀的请求各存一份 KV，重复计算、重复占用。

### 3.2 PagedAttention（vLLM, SOSP'23）⭐⭐⭐⭐⭐

借鉴 OS **虚拟内存分页**思想，是 KV Cache 存储管理的里程碑。

**核心机制：**

- KV Cache 空间被划分为固定大小的**物理块（block）**，每块存 `block_size` 个 token（vLLM 默认 16）的 KV。
- 维护**块表（block table）**：每个请求一张表，第 $i$ 个**逻辑块** -> 物理块编号。逻辑上连续，物理上可分散。
- 请求按需申请物理块，生成到第 $k$ 个 token 才分配它所在的物理块（按需分配，不是预分配 max）。
- attention kernel 被改写：通过块表取 K/V，处理物理不连续。

**收益：**

- 内部碎片只剩"最后一块"的局部浪费（< block_size token），显存利用率从 ~20%~40% 提升到 ~96%+。
- 外部碎片消失（块粒度分配）。
- **天然支持共享**：相同前缀的请求可指向同一组物理块，引用计数管理；当某个请求要"分叉"（生成不同 token）时，copy-on-write 复制该块。

**面试考法：**

- "PagedAttention 解决了什么问题？借鉴了什么思想？" → 碎片化 + 无法共享；借鉴 OS 虚拟内存分页 + 页表 + 按需分配 + 写时复制。
- "block_size 设大设小各有什么影响？" → 大：块表更短、kernel 调度开销小，但内部碎片大、共享粒度粗；小：碎片小、共享粒度细，但块表长、kernel 间接寻址开销大。vLLM 默认 16 是经验折中。
- "PagedAttention 的 attention kernel 和普通 FlashAttention 有什么区别？" → 要按块表做 gather，处理物理不连续 + 变长 + 可能共享；实现更复杂，通常结合 FlashAttention 思想做 tiling。

### 3.3 连续批处理 Continuous Batching ⭐⭐⭐⭐⭐

又称 **iteration-level scheduling**（ORCA, OSDI'22）。

**对比静态批处理：**

- 静态批处理：凑齐一个 batch 后，所有序列一起跑直到**最长的那个**结束，期间短序列空转、还要 padding。
- 连续批处理：在**每个 decode step** 的边界上，调度器可以**插入新请求、踢出已完成请求**，无需等整批结束。配合 PagedAttention，无需 padding。

**收益：** 吞吐显著提升（短序列不拖累长序列，GPU 不空转），TTFT/Tail latency 更可控。**代价：** 调度器更复杂，需要每步重算 batch 组成、处理 KV Cache 的分配/释放。

**面试考法：**

- "连续批处理和静态批处理的区别？" → 调度粒度从"请求级"降到"迭代级"；动态进出；配合分页免 padding。
- "为什么连续批处理要配合 PagedAttention？" → 连续批处理下请求频繁进出、长度各异，连续分配会碎片化严重；分页才能灵活拼装。
- "iteration-level scheduling 的开销在哪？" → 每步调度决策、KV 块分配/释放、kernel launch 重组 batch。SGLang/vLLM 用 CUDA Graph 捕获固定形状来摊薄 launch 开销。

### 3.4 Prefill / Decode 两阶段 ⭐⭐⭐⭐⭐

- **Prefill（预填）**：处理整段 prompt，一次性算出所有 prompt token 的 KV。**计算密集（compute-bound）**：大 GEMM，GPU 利用率高。
- **Decode（解码）**：逐 token 生成，每步只算一个新 token 的 Q 去和全部 KV 做 attention。**访存密集（memory-bound）**：小 GEMV，GPU 利用率低。

两者的资源画像几乎相反：prefill 吃算力、decode 吃带宽。把它们混在一个 batch 里会互相拖累——prefill 的大 GEMM 把 decode 的小请求堵在后面（decode 延迟 spike），decode 的小请求又让 prefill 的 batch 拼不大（算力浪费）。

**衍生优化：**

- **Chunked Prefill（分块预填）⭐⭐⭐⭐**：把长 prompt 切成 chunk，分多个迭代处理，与 decode 交错，避免单个长 prompt 长时间独占 GPU 导致 decode 卡顿。vLLM、SGLang 均支持。
- **Prefill-Decode 分离（Disaggregation）⭐⭐⭐⭐⭐**：把 prefill 和 decode 调度到不同的 GPU/实例上，各自按自身画像优化。见 §6.4。

**面试考法：**

- "prefill 和 decode 各是什么瓶颈？" → prefill compute-bound，decode memory-bound。
- "为什么长 prompt 会让在线服务的 decode 延迟抖动？" → 长 prompt 的 prefill 是一次性大计算，若与 decode 混批，会独占 GPU 若干步，decode 请求排队等待。chunked prefill / PD 分离解决此问题。
- "chunked prefill 的 chunk size 怎么选？" → 太大退化成全量 prefill（卡 decode）；太小 prefill 效率低（GEMM 拼不大）、kernel launch 开销大。通常按"保证 decode 延迟 SLO + prefill 吞吐"折中，常为几百到上千 token。

---

## 4. 让 KV Cache 变小：压缩与结构改造 ⭐⭐⭐⭐⭐

### 4.1 GQA / MQA ⭐⭐⭐⭐⭐

减少 KV 头数 $n_{kv}$，直接按比例缩小 KV Cache。

| 方案 | KV 头数 | KV Cache 大小 | 质量 |
| --- | --- | --- | --- |
| MHA（标准多头） | $n_{heads}$ | 基准 | 最好 |
| GQA（分组查询） | $n_{groups}$（如 8） | $1/k$（$k=n_{heads}/n_{groups}$） | 接近 MHA |
| MQA（多查询） | 1 | $1/n_{heads}$ | 略有损失 |

- **GQA**（Llama-2-70B、Llama-3 全系、Qwen2 等）：多个 query 头共享一组 KV，质量几乎不降，是当前主流。
- **MQA**（PaLM、Falcon 等）：所有 query 头共享一组 KV，压缩最狠但质量略降。

**面试考法：**

- "GQA 相比 MHA 省了什么？为什么质量损失小？" → 省的是 KV Cache 显存与 decode 访存；不同 query 头关注不同子空间，但相邻头学的 K/V 相关性高，共享一组仍能覆盖主要信息。
- "MQA 为什么质量下降？" → 所有头共享同一组 KV，表达能力下降，尤其细粒度任务。

### 4.2 MLA：Multi-head Latent Attention（DeepSeek-V2/V3）⭐⭐⭐⭐⭐

> 这是 2024-2025 KV Cache 方向**最热**的结构创新，DeepSeek 方向面试高频。

**核心思想：** 不缓存完整 K、V，而是缓存一个**低秩压缩的 latent 向量** $c_{KV}$。

- 下投影：把 hidden 压到低维 $c_{KV}$（DeepSeek-V3 中 $d_c=512$）。
- 缓存：只存 $c_{KV}$（+ 一个小的解耦 RoPE 部分 $k_{pe}$，$d_{rope}=64$，因为 RoPE 对位置敏感、难压缩）。
- 上投影：decode 时把 $c_{KV}$ 通过矩阵升回 K、V。

**为什么省：** 等价 MHA 要缓存 $n_h \times d_h$ 维的 K 和 V（DeepSeek-V3 是 128×128=16384 维 ×2）；MLA 只缓存 576 维（512+64）。**单 token 压缩约 57 倍**。

**为什么 RoPE 要解耦：** RoPE 对 K 注入位置信息，是"旋转"操作；若先压再旋会破坏低秩结构。MLA 把 K 拆成 **不带位置的 nope 部分（可压缩进 $c_{KV}$）** 和 **带 RoPE 的 rope 部分（$k_{pe}$，单独缓存不压缩）**，query 侧对应拆分。这就是"decoupled RoPE"。

**面试考法：**

- "MLA 和 GQA 的本质区别？" → GQA 是"减少头数"（结构上共享），MLA 是"低秩压缩缓存内容"（表示上压缩）。MLA 压缩比远大于 GQA，且保持质量。
- "MLA 为什么要把 RoPE 解耦？" → RoPE 破坏低秩可压缩性；解耦后 nope 部分可压进 latent，rope 部分单独存。
- "MLA 的代价？" → decode 时要额外做上投影（升回 K/V）的矩阵乘，增加少量算力；权重多了 up/down 矩阵。训练实现也更复杂。
- "MLA 对推理引擎的影响？" → vLLM/SGLang 等需要专门支持 latent cache 的 attention kernel 和调度（不能直接当普通 KV 处理）；TP 切分方式也不同。

#### 4.2.1 形状级推导与吸收技巧（面试深挖版）⭐⭐⭐⭐⭐

> DeepSeek / 智谱等会追到"写出每一步的 shape 和矩阵"。以 DeepSeek-V3 配置为例。

**配置**：$h=7168$（hidden），$n_h=128$（q 头），$d_n=128$（每头 nope 维），$d_r=64$（rope 维），$d_v=128$（v 头维），$d_q=1536$（q 压缩），$d_c=512$（KV 压缩 latent）。

**Query 路径**：

$$
h_{(h)} \xrightarrow{W_{dq}} c_q{}_{(d_q=1536)} \xrightarrow{W_{uq}} q_{(n_h\cdot(d_n+d_r)=24576)}
$$

$q$ 拆成 $q_{nope}\in\mathbb{R}^{n_h\times d_n}$ 与 $q_{rope}\in\mathbb{R}^{n_h\times d_r}$，$q_{rope}$ 施 RoPE。

**Key/Value 路径（缓存点在此）**：

$$
h_{(h)} \xrightarrow{W_{dkv}} \underbrace{c_{KV}}_{(d_c=512)}\ \text{（缓存）}\qquad
h_{(h)} \xrightarrow{W_{kr}} \underbrace{k_{pe}}_{(d_r=64)}\ \text{（缓存，跨头共享）}
$$

- **每 token 每层只缓存** $c_{KV}$（512）+ $k_{pe}$（64）= **576 元素**。
- decode 时由 $c_{KV}$ 上投影回 $k_{nope}^{(h)}=c_{KV}W_{uk}^{(h)\top}\in\mathbb{R}^{d_n}$、$v^{(h)}=c_{KV}W_{uv}^{(h)\top}\in\mathbb{R}^{d_v}$（每头各一组上投影矩阵）。
- $k^{(h)}=\text{concat}(k_{nope}^{(h)},\ k_{pe})$（$k_{pe}$ 跨头共享并施 RoPE）。

**Attention**（每头，单步 decode，1 个 q 对 $t$ 个历史位）：

$$
\text{scores} = q_{nope}^{(h)} k_{nope}^{(h)\top} + q_{rope}^{(h)} k_{pe}^{\top}\quad(\in\mathbb{R}^{1\times t})
$$

**吸收技巧（Absorption，工程关键，常被深挖）⭐⭐⭐⭐⭐**

朴素做法要先把 $c_{KV}$ 上投影成完整的 $k_{nope}, v$（每头 $d_n$/$d_v$ 维，128 头 = 大），再算 attention--decode 时这部分 GEMM 很贵。MLA 用结合律把上投影"吸收"掉，全程在 $d_c=512$ 的 latent 上算：

- **K 侧吸收**：$k_{nope}^{(h)}=c_{KV}W_{uk}^{(h)\top}$，故

  $$
  q_{nope}^{(h)}k_{nope}^{(h)\top} = q_{nope}^{(h)}W_{uk}^{(h)}\,c_{KV}^{\top} = \underbrace{q_{nope}^{\prime(h)}}_{(d_c)}\ c_{KV}^{\top}
  $$

  即把 $W_{uk}^{(h)}$ 提前折进 query 投影，得到 $q'_{nope}\in\mathbb{R}^{d_c}$，scores 的 nope 部分直接用 $q'_{nope}\cdot c_{KV}^\top$，**无需显式上投影出 $k_{nope}$**。

- **V 侧吸收**：$v^{(h)}=c_{KV}W_{uv}^{(h)\top}$，输出 $o^{(h)}=\text{attn}\cdot v^{(h)}$。对 $t$ 个历史位：

  $$
  o^{(h)} = \sum_{i} \text{attn}_i\cdot v_i^{(h)} = \sum_i \text{attn}_i\cdot(c_{KV,i}W_{uv}^{(h)\top}) = \underbrace{\Big(\sum_i \text{attn}_i\,c_{KV,i}\Big)}_{(d_c)}\ W_{uv}^{(h)\top}
  $$

  即先对 $c_{KV}$ 做注意力加权求和（得 $d_c$ 维），再上投影回 $d_v$。**全程不物化完整的 $v$ 矩阵**。

**收益**：decode 的 attention 计算从"每头 $d_n/d_v$（128 头 × 128 维）"降到"latent $d_c=512$"，算力与访存大幅下降，同时 cache 还小几十倍--这是 MLA 能同时省显存又不增多少 decode 算力的根因。

**工程难点（追问）**：

- $W_{uk}/W_{uv}$ 的吸收要求 RoPE 不介入 nope 部分--正因如此才把 RoPE 解耦到 $k_{pe}$（rope 部分不吸收，单独算）。
- TP 切分：MLA 按 latent 维度切而非按头切，$W_{uk}/W_{uv}$ 的切分要与 $q$ 的头分布对齐，比 GQA 复杂。
- 分离式推理下传 $c_{KV}+k_{pe}$（576 维）而非完整 K/V，传输量天然小--PD 分离的隐藏优势。

**面试考法（追加）**：

- "写出 MLA decode 一步的矩阵 shape 与吸收技巧。" -> 上式。
- "为什么吸收技巧能省算力？" -> 把上投影折进 q / 延后到加权求和之后，全程在 $d_c$ latent 上算，避免物化每头完整 K/V。
- "吸收技巧对 RoPE 有什么前提？" -> RoPE 不能进入被吸收的 nope 部分，故必须解耦。

### 4.3 KV Cache 量化 ⭐⭐⭐⭐

把缓存的 K、V 从 fp16 量化到 int8/fp8/int4。

- **收益**：显存与 decode 访存 2x / 4x 下降，直接提吞吐。
- **难点**：K 中存在显著**离群值（outliers）**（尤其某些通道），朴素 per-tensor 量化精度崩。常用 **per-channel**（按通道缩放）或 **per-token** 量化缓解。
- **方案**：KIVI（per-channel K + per-token V）、KVQuant、Atom、QuaRot；Hopper 上 **FP8** 因硬件原生支持、精度损失小，成为工程甜点。
- **策略**：保留少量精度敏感的（如 attention sink、首尾）高精度，其余低精度。

**面试考法：**

- "KV Cache 量化为什么 K 和 V 用不同粒度？" → K 的离群值沿 channel 分布，per-channel 更稳；V 相对平滑，per-token 更利于在线更新。
- "FP8 量化 KV 相比 INT8 的优势？" → Hopper 原生 FP8 计算、动态范围大、精度损失小，工程上几乎是"免费午餐"。

---

## 5. 让 KV Cache 复用：共享与命中 ⭐⭐⭐⭐⭐

### 5.1 RadixAttention（SGLang）⭐⭐⭐⭐⭐

> 作者本博客已有 [SGLang × GLM-5.2 KV Cache](../kvcache/sglang_kvcache_glm5.2.md) 系列，此处给面试视角。

**核心思想：** 把 KV Cache 组织成一棵 **Radix Tree（基数树）**，以 token 序列为键。共享前缀的请求共享树上的 KV 节点。

- 新请求到来：沿树匹配前缀；命中部分直接复用 KV，不重算；分叉处新建节点。
- **天然适合**：多轮对话（每轮拼接历史）、共享 system prompt / few-shot、并行采样（同一 prompt 多条采样）、Tree-of-Thought、批量结构化生成（JSON schema 相同前缀）。
- **淘汰**：显存不足时按 LRU 淘汰叶节点，引用计数保证共享节点不被误删。

**对比 vLLM 早期：** vLLM 用块表 + copy-on-write 也能共享，但以"块"为粒度、需要显式前缀匹配；RadixAttention 以树为索引，自动、细粒度。后期 vLLM 也加了 Automatic Prefix Caching（APC），二者趋同。

**面试考法：**

- "RadixAttention 相比朴素 KV Cache 的收益场景？" → 任何有共享前缀的批量场景：多轮对话、并行采样、few-shot、结构化输出。命中率越高收益越大，极端情况 prefill 计算几乎归零。
- "Radix Tree 在高并发下的工程难点？" → 树的并发读写、引用计数与淘汰的锁竞争、内存碎片。SGLang 用锁 + 细粒度引用计数。
- "命中率不高时 RadixAttention 有开销吗？" → 有：树查找、节点维护、且占用的 KV 不能淘汰给新请求，可能反而降低吞吐。要配合好的淘汰策略。

### 5.2 Prefix Caching（前缀缓存）⭐⭐⭐⭐

- 缓存"公共前缀"（system prompt、长文档、few-shot）的 KV，新请求命中则跳过该段 prefill。
- vLLM APC、TGI、TRT-LLM 均有实现。
- 与 RadixAttention 思想一致，区别主要在数据结构（块表 hash vs 树）。

### 5.3 Cross-request / Session 复用

- 同一会话多轮：历史轮次的 KV 直接复用，只对新输入做 prefill。
- 多用户共享同一文档 RAG：文档 KV 池化复用。
- 这是 Mooncake 类"KV Cache as a Service"的出发点（见 §6.4）。

---

## 6. KV Cache 的生命周期与调度 ⭐⭐⭐⭐⭐

### 6.1 驱逐策略

显存不够时，谁走谁留？

- **LRU**（最近最少使用）：RadixAttention、多数缓存默认。
- **FIFO**：简单但不如 LRU 贴合访问局部性。
- **基于 attention score**：H2O（Heavy-Hitter Oracle）保留"重要"token 的 KV。
- **Sliding Window**：只留最近 $W$ 个 token，最激进。

### 6.2 Sliding Window 与 StreamingLLM ⭐⭐⭐⭐

- **Sliding Window Attention**（Mistral 等）：每个 token 只 attend 最近 $W$ 个，KV Cache $O(W)$。简单有效，但**完全丢弃早期 token**，长上下文召回能力受限。
- **Attention Sink 现象**：研究发现 attention 分数会异常集中在**序列开头的几个 token**（"sink"），即使它们语义无关——它们充当了"注意力垃圾桶"。若朴素滑窗把 sink 也丢掉，模型质量会**雪崩**。
- **StreamingLLM**：保留 **sink（前几个 token）+ 最近窗口**，即可在固定显存下流式生成超长序列（百万级 token）而不崩。

**面试考法：**

- "什么是 attention sink？为什么不能直接滑窗丢弃前几个 token？" → 前几个 token 吸收了过量 attention（softmax 需要分配概率质量的"垃圾桶"），丢弃会导致分布畸变、质量崩。
- "StreamingLLM 怎么实现近乎无限上下文？" → 固定保留 sink + 滑动窗口，显存恒定；代价是早期（窗口外）内容无法回忆，只适合"近因 + 全局语气"场景。

### 6.3 KV Cache 卸载与分层存储 ⭐⭐⭐⭐

KV Cache 不必只在 GPU HBM，可分级：

```
GPU HBM (快/小)  ⇄  CPU DRAM (中/中)  ⇄  SSD (慢/大)
```

- 显存吃紧时把"暂时不活跃"的 KV 换出（swap）到 CPU/SSD；需要时换入，与计算 overlap。
- vLLM 的 `swap_space` 把 KV 在 GPU↔CPU 间换页；长上下文/低 QPS 场景可显著提并发。
- 工程难点：换入换出的带宽与延迟、prefetch 时机、与调度的协同。

**面试考法：**

- "KV Cache 卸载到 CPU 的代价是什么？什么场景值得？" → 代价是 PCIe 传输延迟与带宽；适合显存受限、容忍较高单次延迟、但需高并发的场景。换入换出必须与计算 overlap 才不亏。

### 6.4 分离式推理与 KV Cache 池化 ⭐⭐⭐⭐⭐

> 这是 2024-2025 推理系统最热的架构方向，KV Cache 是其核心纽带。

**动机：** Prefill（compute-bound）和 Decode（memory-bound）资源画像相反，混部互相拖累（见 §3.4）。把它们**物理分离**到不同实例/池子，各自最优配置。

**代表工作：**

- **DistServe（OSDI'24）**：prefill 与 decode 分配到不同 GPU，整体 goodput 显著提升。
- **Splitwise（ISCA'24）**：按 phase 切分，复用资源池。
- **Mooncake（月之暗面，2024）**：**KVCache-centric 的分离式架构**。核心是**全局 KV Cache 池**——prefill 实例算出 KV 后，KV 通过网络传到 decode 实例（或全局池），decode 直接复用。KV Cache 被当作一等公民（"KVCache as a Service"），可跨请求、跨实例、跨节点复用。
- **DeepSeek 的 PD 分离**：线上服务实际采用 prefill/decode 分离部署。

**关键技术点（面试深挖）：**

- **KV 传输**：prefill->decode 之间要把可能几个 GB 的 KV 通过 RDMA/网络搬过去，传输本身是瓶颈。需压缩（MLA 这类天然小）、流式传输、与 decode 首步 overlap。
- **调度**：何时分离、KV 路由到哪个 decode 实例、负载均衡。
- **KV 复用**：全局池使"同一文档/会话"的 KV 跨请求复用成为可能，把 prefill 从"每请求必算"变成"命中即跳过"。

**面试考法：**

- "为什么要做 prefill-decode 分离？" → 二者瓶颈不同，混部互相拖累；分离后各自按画像配置（prefill 多算力、decode 多带宽/显存）。
- "分离式推理的核心难点？" → KV Cache 的跨实例传输（带宽/延迟）、全局调度与路由、KV 池的一致性与复用、故障容错。
- "Mooncake 为什么强调 KVCache-centric？" → 把 KV Cache 当作可复用的全局资源池，而不是每请求每实例的临时产物；KV 的复用与迁移成为系统一等公民，直接决定吞吐与成本。

---

## 7. KV Cache 与其他机制的交互

### 7.1 Speculative Decoding（投机解码）⭐⭐⭐⭐

- Draft 模型一次猜 $k$ 个 token，Target 模型并行验证（tree attention 一次 forward）。
- **KV 管理**：被接受的 token 的 KV 要保留、被拒绝的要回滚丢弃；draft 与 target 的 KV 各自维护。EAGLE/Medusa/lookahead 实现细节不同。
- **与 KV Cache 关系**：投机解码增加单步计算但减少"串行 decode 步数"，本质是用多余算力换访存次数下降；在 decode memory-bound 场景收益大。

### 7.2 Tensor Parallel 下的 KV 切分 ⭐⭐⭐⭐

- TP 按**头**切分：每个 GPU 只持有自己负责的 KV 头。所以 **KV Cache 随 TP 等比例缩小**，不复制。
- 推论：增大 TP 既能放下更大模型，也能等比降低单卡 KV 显存压力。但 TP 增加通信（每层 all-reduce）。
- PP（流水并行）下：KV Cache 随层切分到不同 stage，每个 stage 只存自己的层。

**面试考法：**

- "TP=2 时每张卡的 KV Cache 是原来的多少？" → 1/2（按头切分，不复制）。
- "为什么 TP 能缓解 KV Cache 显存压力而 DP 不能（单卡视角）？" → TP 切头缩小单卡 KV；DP 是复制整模型到各组，单卡 KV 不变（只是各组处理不同请求）。

### 7.3 MoE 推理的 KV ⭐⭐⭐

- MoE 把 FFN 换成专家，但 **attention 仍是 dense 的**，KV Cache 计算与非 MoE 一致。
- MoE 的显存大头是**专家权重**（激活稀疏但权重全驻留），KV Cache 相对没那么突出，但长上下文下仍是瓶颈。
- Expert Parallelism（EP）下专家分布到多卡，路由通信是关键，与 KV 关系不大但常一起考。

---

## 8. 高频计算题速查

| 题目 | 公式/思路 |
| --- | --- |
| 单 token KV Cache | $2 L n_{kv} d_h p$ |
| batch×seq KV | $\times\ s\ b$ |
| A100-80G 能跑多大 batch | $(80\text{G} - \text{权重} - \text{激活}) / \text{单条 KV}$ |
| GQA 相对 MHA 压缩比 | $n_{kv}/n_h$ |
| MLA 相对 MHA 压缩比 | $(d_c+d_{rope}) / (2 n_h d_h)$ 量级（注意 MLA 缓存不含显式 V） |
| PagedAttention 碎片浪费 | $<\text{block\_size}\times b$ token |
| Sliding Window 显存 | $O(W)$ 而非 $O(s)$ |

---

## 9. 易错点与反思

- ❌ 把 Q 也算进 KV Cache。→ 只缓存 K、V。
- ❌ 算 KV Cache 忘乘 `2`（K 和 V 两份）。
- ❌ GQA/MQA 下仍用 q 头数算。→ 必须用 $n_{kv}$。
- ❌ 说"PagedAttention 减少计算量"。→ 它优化的是**显存利用与共享**，计算量基本不变（甚至 kernel 略有开销）。
- ❌ 把 MLA 的 latent 当成普通 KV 直接喂给标准 attention。→ 需要上投影 + 解耦 RoPE 专用 kernel。
- ❌ 认为连续批处理 = 减少计算。→ 它提升的是**利用率/吞吐**，单 token 计算量不变。
- ❌ 混淆 chunked prefill（切 prompt）与 PD 分离（切实例）。→ 前者是单实例内时间分片，后者是跨实例物理分离。
- ❌ 认为 KV 卸载到 CPU 一定更快。→ PCIe 传输有成本，必须与计算 overlap 且命中复用才划算。

---

## 10. 一页纸总结（面试前默写版）

```
KV Cache = 用空间换时间，单步 O(t)->O(1)
显存 = 2·L·n_kv·d_h·s·b·p   (GQA减头, MLA压latent)
瓶颈 = 显存占用 + decode访存(memory-bound)
存储 = PagedAttention(分页+块表+COW共享) + 连续批处理(iteration级)
阶段 = prefill(compute-bound) vs decode(memory-bound)
        -> chunked prefill / PD分离
变小 = GQA/MQA(减头) MLA(压缩latent) 量化(int8/fp8)
复用 = RadixAttention(树) / Prefix Caching / 全局池
生命周期 = LRU/H2O淘汰 + SlidingWindow/StreamingLLM(sink) + 卸载分层
架构 = DistServe/Splitwise/Mooncake(KVCache-centric分离式)
交互 = SpeculativeDecoding / TP切头减KV / MoE(attention仍dense)
```

下一篇：[面经与高频考点](aiinfra_interview_questions.md) 把本文知识点还原成"题-答"形式。
