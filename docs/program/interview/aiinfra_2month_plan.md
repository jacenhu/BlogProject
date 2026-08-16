# 2 个月面试复习计划

> 首次编写：2026-08-12 | 最后更新：2026-08-12

> **AI Infra / KV Cache 面试系列**
> - [导读](aiinfra_overview.md)
> - [招聘岗位全景](aiinfra_jobs.md)
> - [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)
> - [面经与高频考点](aiinfra_interview_questions.md)
> - [系统设计专题](aiinfra_system_design.md)
> - **2 个月面试复习计划** ← 当前文章
> - [资源汇总](aiinfra_resources.md)

一份 8 周、以 **KV Cache 为主线** 的 AI Infra 面试复习计划。每周有目标、任务、交付物、自检题。配合 [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md) 与 [面经题库](aiinfra_interview_questions.md) 使用。

---

## 0. 计划原则

1. **主线优先**：KV Cache / 推理系统占 60% 时间，分布式/算子 30%，coding 10%（每天保持手感）。
2. **输出驱动**：每周末必须产出可看的东西--源码笔记、复现 benchmark、一篇博客、一个 PR。**只读不写等于没学**。
3. **按目标公司裁剪**：大模型公司重源码+架构，大厂重分布式+工程，芯片厂重 kernel。第 5 周起按你的目标侧重加码。
4. **模拟面试前置**：第 4 周就开始找人 mock，别等"准备好"再练--永远没"准备好"。
5. **投递与复习并行**：第 5 周开始投递，用真实面试校准方向，面经实时回填 [题库](aiinfra_interview_questions.md)。

## 时间预算（按在职复习，每天 ~3-4h 工作日 + 6-8h 周末）

- 若全职复习，可将每周压缩到 4-5 天，总周期缩到 5-6 周。
- 工作日：1h 源码/论文 + 1h 手写/复现 + 0.5h coding + 0.5h 面经/投递。
- 周末：集中大块时间做复现与系统设计练习。

---

## 第 1 周：地基（Transformer / Attention / KV Cache 建模）

**目标**：把"为什么有 KV Cache、占多少、为什么是瓶颈"讲得滴水不漏。

| 日 | 任务 |
| --- | --- |
| 周一 | 手推 self-attention：Q/K/V 投影、scaled dot-product、causal mask、softmax。写纯 numpy 版并验证。 |
| 周二 | 推导 KV Cache：为什么缓存 K/V 不缓存 Q；带/不带 cache 的复杂度对比。读 [KV Cache 核心知识体系 §1-§2](aiinfra_kvcache_deep_dive.md)。 |
| 周三 | 显存建模：背公式 $2Ln_{kv}d_hsp$，手算 Llama-2-7B / Llama-3-8B / DeepSeek-V3(MLA) 三种的单 token 与 batch 显存。 |
| 周四 | GQA/MQA/MLA 通读：理解三者如何缩小 KV Cache。MLA 的 latent + 解耦 RoPE 画图。 |
| 周五 | prefill vs decode：理解 compute-bound vs memory-bound、为什么 decode MFU 低。 |
| 周六 | 手写 numpy attention（带 KV cache 的 decode 版），测正确性。 |
| 周日 | 整理本周笔记成一篇博客；刷 [题库第 1 章 Q1.1-Q1.6](aiinfra_interview_questions.md)。 |

**交付物**：numpy attention + kv-cache decode 实现（git 仓库）、显存建模速查表、一篇博客。
**自检**：能 5 分钟讲清"KV Cache 为什么需要、占多少、为什么是瓶颈"并口算 7B/8K/batch16 的显存。

---

## 第 2 周：GPU/CUDA 基础 + 手撕 attention kernel

**目标**：理解 GPU 执行模型，能写/读简单 attention kernel。

| 日 | 任务 |
| --- | --- |
| 周一 | GPU 执行模型：SM/warp/lanes、thread/block/grid、occupancy、shared memory、registers。 |
| 周二 | memory coalescing、bank conflict、cp.async/TMA（Hopper）、异步拷贝与多 stream。 |
| 周三 | FlashAttention 原理：tiling + online softmax，IO 复杂度推导。读 [题库 Q3.1-Q3.2](aiinfra_interview_questions.md)。 |
| 周四 | 用 Triton 写一个简化 FlashAttention（forward），profile 对比朴素版。 |
| 周五 | FlashDecoding / FlashAttention-3（Hopper 特性）。 |
| 周六 | CUDA 基础编程练习：向量加法->矩阵乘->tiled GEMM。用 Nsight Compute 看 occupancy/带宽。 |
| 周日 | 整理 kernel 笔记；刷 coding：LRU 146 + 手写 attention。 |

**交付物**：Triton FlashAttention 简化实现 + profile 截图、CUDA tiled GEMM。
**自检**：能讲清 FlashAttention 为什么快（不减 FLOPs、减访存、SRAM tiling、online softmax），occupancy 与 coalescing 是什么。

---

## 第 3 周：vLLM 源码 + PagedAttention + 连续批处理

**目标**：吃透 vLLM 的 KV Cache 管理与调度。

| 日 | 任务 |
| --- | --- |
| 周一 | vLLM 整体架构：engine/scheduler/worker/model_runner 分层。读 [KV Cache §3](aiinfra_kvcache_deep_dive.md)。 |
| 周二 | PagedAttention：物理块、块表、按需分配、COW 共享。读 vLLM PagedAttention CUDA kernel。 |
| 周三 | Scheduler：连续批处理逻辑、每步调度决策、抢占与 swap。 |
| 周四 | prefill/decode 调度、chunked prefill 实现。 |
| 周五 | 前缀缓存（APC）实现细节。 |
| 周六 | 跑 vLLM benchmark，调整 batch/block_size 观察吞吐与显存变化。 |
| 周日 | 给 vLLM 提一个小 PR（文档/边界 case/小优化均可）；整理源码笔记。 |

**交付物**：vLLM 调度器与 PagedAttention 源码笔记、benchmark 对比表、（可选）一个 PR。
**自检**：能画出 vLLM 调度器每步状态机，讲清抢占 swap 流程，PagedAttention 块表如何工作。

---

## 第 4 周：SGLang 源码 + RadixAttention + 高性能调度

**目标**：吃透 SGLang 的前缀复用与工程优化。（作者本博客已有 SGLang×GLM 系列可对照）

| 日 | 任务 |
| --- | --- |
| 周一 | SGLang 架构：scheduler/runtime 分离、RadixAttention 数据结构。 |
| 周二 | RadixAttention：树结构、前缀匹配、引用计数、LRU 淘汰、并发。 |
| 周三 | CUDA Graph 捕获、overlap、tensor reuse 等工程优化。 |
| 周四 | 结构化生成（compressed FSM / jump-forward）如何复用前缀。 |
| 周五 | 对比 vLLM APC vs SGLang RadixAttention 的设计差异。 |
| 周六 | 跑 SGLang benchmark，对比多轮/并行采样场景下的 prefix 命中收益。 |
| 周日 | **第一次模拟面试**（找朋友/校友，40min 技术 + 20min 反馈）；整理面经。 |

**交付物**：SGLang RadixAttention 源码笔记、vLLM vs SGLang 对比表、第一次 mock 反馈记录。
**自检**：能讲清 RadixAttention 收益场景与开销、SGLang 相对 vLLM 的工程优势与代价。

---

## 第 5 周：KV Cache 深水区（MLA / 量化 / 前缀复用）

**目标**：掌握 2024-2025 KV Cache 前沿，能应付深挖追问。

| 日 | 任务 |
| --- | --- |
| 周一 | MLA 深挖：latent 维度、上/下投影、解耦 RoPE、压缩比推导、对 kernel/调度/TP 的影响。对照作者 [SGLang×GLM-5.2 KV Cache](../kvcache/sglang_kvcache_glm5.2.md)。 |
| 周二 | KV 量化：KIVI/KVQuant/Atom/QuaRot、FP8、per-channel vs per-token。 |
| 周三 | 前缀复用进阶：RadixAttention vs Prefix Caching vs cross-request/session 复用。 |
| 周四 | 淘汰与长上下文：H2O、sliding window、StreamingLLM（attention sink）。 |
| 周五 | MoE 推理的 KV（attention 仍 dense）、专家并行路由。 |
| 周六 | 复现一个 MLA 或 KV 量化的小实验，记录精度/显存数据。 |
| 周日 | **开始投递**目标公司；整理本周为博客；第二次 mock。 |

**交付物**：MLA 深度笔记（含推导）、量化/长上下文对比表、投递清单（3-5 家）、第二次 mock 反馈。
**自检**：能推导 MLA 压缩比、解释 attention sink、对比各类 KV 减小方案。

---

## 第 6 周：分离式推理 + KV 池化 + 关键论文

**目标**：掌握分离式架构与 KV 池化，这是最高阶也是区分度最大的方向。

| 日 | 任务 |
| --- | --- |
| 周一 | PD 分离动机与设计：DistServe（OSDI'24）。 |
| 周二 | Splitwise（ISCA'24）、资源画像与配比。 |
| 周三 | Mooncake：KVCache-centric 架构、全局 KV 池、传输与复用。 |
| 周四 | DeepSeek 的 PD 分离实践 + MLA 在分离式下的传输优势。 |
| 周五 | KV 卸载与分层存储（GPU-CPU-SSD）、vLLM swap。 |
| 周六 | 精读 1 篇代表作（推荐 Mooncake 或 DistServe），写笔记。 |
| 周日 | 系统设计练习：设计分离式推理服务（画图讲 20min）；第三次 mock。 |

**交付物**：分离式架构对比表（DistServe/Splitwise/Mooncake/DeepSeek）、1 篇论文精读笔记、分离式系统设计图。
**自检**：能讲清 PD 分离的核心难点（KV 传输/调度/池化）、Mooncake 为什么 KVCache-centric。

---

## 第 7 周：分布式 + 通信 + 算子强化

**目标**：补齐分布式与通信短板（大厂面试重点）。

| 日 | 任务 |
| --- | --- |
| 周一 | DP/TP/PP/EP/CP 全梳理，各切什么、通信原语。读 [题库第 4 章](aiinfra_interview_questions.md)。 |
| 周二 | TP 下 KV Cache 切分、PP 下层切分；TP 通信开销分析。 |
| 周三 | NCCL：Ring all-reduce 原理、tree+ring 混合、拓扑感知。 |
| 周四 | 通信计算 overlap：梯度分桶、独立 stream、PP 零气泡。 |
| 周五 | Ring Attention / CP 超长上下文。 |
| 周六 | 算子强化：CUTLASS、给低效 kernel 做优化的练习题。 |
| 周日 | 系统设计练习：设计高并发 LLM serving（端到端）；第四次 mock。 |

**交付物**：5 种并行对比表、NCCL Ring 推导、端到端 serving 系统设计图（迭代版）。
**自检**：能讲清 5 种并行各切什么、TP 为什么减单卡 KV、Ring all-reduce 步数与通信量、overlap 手段。

---

## 第 8 周：系统设计 + 面经冲刺 + 模拟面试

**目标**：把所有知识收敛成"能面试输出"的状态。

| 日 | 任务 |
| --- | --- |
| 周一 | 系统设计专题练 3 题：高并发 serving、KV 池化、新模型接入引擎。每题画图讲 20min。读 [系统设计专题](aiinfra_system_design.md) + [题库第 5 章](aiinfra_interview_questions.md)。 |
| 周二 | 面经冲刺：过 [题库](aiinfra_interview_questions.md) 全部题，每题自答一遍，标记卡壳的。 |
| 周三 | 行为面准备：STAR 故事 3 个（项目挑战/协作冲突/技术决策）。读 [题库第 8 章](aiinfra_interview_questions.md)。 |
| 周四 | 手撕强化：attention / 简化 PagedAttention / 调度模拟 / LRU，限时默写。 |
| 周五 | 模拟面试 ×2（技术 + 系统设计），全程录像复盘。 |
| 周六 | 查漏补缺：回看 8 周笔记，补薄弱点。整理 [题库](aiinfra_interview_questions.md) 中标 ❓ 的开放题。 |
| 周日 | 复盘总账：哪些公司已面/在面/待面，各自反馈，下一步。 |

**交付物**：3 套系统设计图、行为面 STAR 故事、手撕代码默写集、模拟面试复盘。
**自检**：能 30 秒内对任何题库题目开口作答、系统设计能讲满 20min 不卡、手撕 attention 无误。

---

## 自检总表（每周勾选）

| 周 | 核心能力 | 能否脱口而出 |
| --- | --- | --- |
| 1 | KV Cache 为什么/占多少/为什么瓶颈 | ☐ |
| 2 | FlashAttention 为什么快、GPU 执行模型 | ☐ |
| 3 | vLLM 调度器 + PagedAttention 源码 | ☐ |
| 4 | SGLang RadixAttention + 工程优化 | ☐ |
| 5 | MLA 推导 + 量化 + 长上下文 | ☐ |
| 6 | PD 分离 + Mooncake + KV 池化 | ☐ |
| 7 | 5 种并行 + NCCL + overlap | ☐ |
| 8 | 系统设计端到端 + 面经全覆盖 | ☐ |

## 投递节奏建议

- **第 5 周**：投 1-2 家"练手"公司（非最心仪），用真实面试校准。
- **第 6-7 周**：投 3-5 家目标公司，错开时间避免撞期。
- **第 8 周**：冲刺最心仪公司，此时状态最佳。
- 每场面完**当天**回填面经到 [题库](aiinfra_interview_questions.md)，标注公司/日期/原题/我的回答/正确答案。

## 如果时间不够（压缩版 5 周计划）

- 第 1 周 = 原 1+2（地基 + GPU 基础，精简 kernel）。
- 第 2 周 = 原 3（vLLM 为主，SGLang 略读）。
- 第 3 周 = 原 5+6（KV Cache 深水区 + 分离式，这是区分度核心，不能省）。
- 第 4 周 = 原 7（分布式，按目标公司取舍深度）。
- 第 5 周 = 原 8（系统设计 + 面经 + mock）。
- MLA、PD 分离、Mooncake、PagedAttention、FlashAttention、连续批处理--**这 6 个无论如何要会**。

下一篇：[资源汇总](aiinfra_resources.md)。
