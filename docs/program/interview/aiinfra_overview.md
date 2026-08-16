# AI Infra 面试 Wiki · 导读

> 首次编写：2026-08-12 | 最后更新：2026-08-12

> **AI Infra / KV Cache 面试系列**
> - **导读（本页）** ← 当前文章
> - [招聘岗位全景](aiinfra_jobs.md)
> - [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)
> - [面经与高频考点](aiinfra_interview_questions.md)
> - [系统设计专题](aiinfra_system_design.md)
> - [2 个月面试复习计划](aiinfra_2month_plan.md)
> - [资源汇总](aiinfra_resources.md)

---

## 这是什么

这是一份面向 **AI Infra（AI 基础设施）方向，尤其聚焦 KV Cache 与大模型推理系统** 的面试准备 Wiki。覆盖三件事：

1. **招聘岗位全景**：哪些公司在招、岗位类型划分、JD 拆解、职级与能力要求。
2. **面经与高频考点**：按主题归类的题库 + 参考答题思路，覆盖 KV Cache、推理引擎、分布式与通信、系统设计、Coding。
3. **2 个月复习计划**：周粒度的可执行计划，把上面两块串成一条复习主线。

## 为什么以 KV Cache 为锚点

KV Cache 是大模型推理系统里**最集中体现"系统 + 算法 + 硬件"交叉**的子系统：

- 它是**显存瓶颈**：推理时显存占用的大头往往不是权重而是 KV Cache，长上下文场景下尤为突出。
- 它是**吞吐瓶颈**：Decode 阶段是 memory-bound，KV Cache 的访存效率直接决定 token/s。
- 它是**优化富矿**：PagedAttention、RadixAttention、MLA、前缀复用、量化、卸载、分离式推理……近两年几乎所有有影响力的推理工作都围绕它展开。
- 它是**面试富矿**：因为它既考"为什么"（自回归、attention 计算）、又考"怎么算"（显存公式）、又考"怎么工程"（分页、调度、池化），一面能区分候选人的系统功底。

因此本 Wiki 把 KV Cache 作为主线，向外辐射到推理引擎、注意力内核、批处理调度、分布式通信与系统设计。

## 关于信息来源的说明（重要）

> 本 Wiki 撰写于 2026-08-12。环境中的 **WebSearch / WebFetch 工具被策略阻断**，但通过 **Bash curl 实测**，arxiv.org、github.com、nowcoder.com（牛客）、vllm.ai、sglang 文档等域名**可达**，google/huggingface 不可达。据此做了真实抓取：
>
> - **论文链接**：[资源汇总](aiinfra_resources.md) 中 20+ 篇 arXiv 链接均经 curl 逐一核实标题与可达性。
> - **面经**：[题库第 7.0 节](aiinfra_interview_questions.md) 为实时抓取自牛客的真实面经帖（含 2026 校招快手 AI Infra、美团北斗、华为海思等），链接已核实。
> - **招聘岗位的"具体 JD/薪资"**：Boss 直聘/拉勾等需登录、JS 渲染，curl 抓不到有效内容，仍为**结构化盘点**，需你用实时渠道补全（见各页"待补全"清单）。
> - **技术知识体系、复习计划、题库答题思路**：基于该领域公开稳定知识整理，可作复习骨架。
>
> 简言之：**论文与面经已是真链接，JD/薪资仍需你补**。本 Wiki 给地图和训练方法，实时赛况部分已替你刷过一轮。

## 怎么用这份 Wiki

| 你的状态 | 建议路径 |
| --- | --- |
| 刚转方向 / 基础薄弱 | 导读 → [招聘全景](aiinfra_jobs.md)（知道要学什么）→ [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)（打地基）→ [2 个月计划](aiinfra_2month_plan.md) |
| 有一定基础，临近面试 | [面经与高频考点](aiinfra_interview_questions.md)（查漏补缺）→ [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)（深挖重点）→ 计划里的"冲刺周" |
| 已有 offer，想横向比较 | [招聘全景](aiinfra_jobs.md) 的职级/薪资/团队盘点 |

## 岗位方向速览（详见 [招聘全景](aiinfra_jobs.md)）

AI Infra 是个筐，不同公司拆法不同，但大致可归为五类：

1. **推理系统 / Serving**：推理引擎、调度、KV Cache 管理、批处理。**KV Cache 方向的主战场。**
2. **训练框架 / 分布式**：Megatron / DeepSpeed 类，TP/PP/EP/CP，通信优化。
3. **算子 / Kernel**：CUDA、Triton、FlashAttention 类融合算子、量化算子。
4. **硬件 / 编译**：算子库（cuBLAS/cutlass）、图编译（TensorRT, XLA, TorchInductor）、芯片侧。
5. **平台 / 工程化**：大集群调度、容错、监控、多租户、成本。

KV Cache 最常出现在 **1 和 3**，其次是 **5**（分离式推理池化、多租户 KV 隔离）。

## 核心知识地图（详见 [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)）

```
                        KV Cache
                           │
   ┌───────────┬───────────┼───────────┬──────────────┬──────────────┐
   ▼           ▼           ▼           ▼              ▼              ▼
 为什么需要    显存建模     存储管理     压缩/减少       复用/共享       生命周期
 自回归解码    占用公式     PagedAttention  GQA/MQA/MLA   RadixAttention  驱逐/卸载/池化
              显存瓶颈     连续批处理     量化           前缀缓存        sliding window
                          chunked prefill                cross-request   StreamingLLM
                          分离式推理                     复用            Mooncake 池化
```

## 复习计划速览（详见 [2 个月计划](aiinfra_2month_plan.md)）

- **第 1–2 周**：地基。Transformer/Attention 推导 + KV Cache 显存建模 + CUDA 基础。
- **第 3–4 周**：推理引擎。vLLM / SGLang / TRT-LLM 源码与设计，PagedAttention、连续批处理。
- **第 5–6 周**：KV Cache 深水区。MLA、RadixAttention、量化、卸载、分离式推理（DistServe/Splitwise/Mooncake）。
- **第 7 周**：分布式 + 算子。TP/PP/EP/CP、NCCL、FlashAttention、量化算子。
- **第 8 周**：系统设计 + 面经冲刺 + 模拟面试。

## 阅读约定

- 公式中 `L`=层数，`n_kv`=KV 头数，`d_h`=每头维度，`s`=序列长度，`b`=batch，`p`=每元素字节数。
- "高频⭐"标注表示该知识点在面经中反复出现。
- 标注 ❓ 的为开放题 / 有争议题，需结合最新进展判断。
