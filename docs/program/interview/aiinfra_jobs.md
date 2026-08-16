# AI Infra 招聘岗位全景

> 首次编写：2026-08-12 | 最后更新：2026-08-12

> **AI Infra / KV Cache 面试系列**
> - [导读](aiinfra_overview.md)
> - **招聘岗位全景** ← 当前文章
> - [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)
> - [面经与高频考点](aiinfra_interview_questions.md)
> - [系统设计专题](aiinfra_system_design.md)
> - [2 个月面试复习计划](aiinfra_2month_plan.md)
> - [资源汇总](aiinfra_resources.md)

---

## ⚠️ 信息来源说明（务必先读）

本页是**结构化盘点**，基于该领域公开、稳定的招聘格局整理。WebSearch/WebFetch 工具被沙箱阻断，但 **Bash curl 实测**：arxiv/github/nowcoder/vllm/sglang 等可达，google/huggingface/Boss 直聘不可达（后者需登录+JS）。

- ✅ **已核实**：[资源汇总](aiinfra_resources.md) 的 arXiv 论文链接、[题库 7.0 节](aiinfra_interview_questions.md) 的牛客真实面经帖。
- ❓ **仍需 [实时核对]**：本页的具体团队、薪资区间、最新 HC--请用以下渠道核对：

薪资区间、具体团队、最新岗位请用以下实时渠道核对（[资源汇总](aiinfra_resources.md) 有清单）：

- **国内**：Boss 直聘、牛客网（面经+内推）、脉脉（薪资+团队风评）、公司招聘官网、知乎/小红书面经帖。
- **海外**：LinkedIn、levels.fyi、一亩三分地、各家 careers 页。

> 文中凡标 **[实时核对]** 的字段，请以最新招聘信息为准。

---

## 1. 岗位类型划分

AI Infra 是个统称，不同公司组织拆法不同。按"离模型/离硬件"的光谱，可分五类。**KV Cache 方向主要集中在第 1、3 类，其次第 5 类。**

### 1.1 推理系统 / Serving（KV Cache 主战场）⭐⭐⭐⭐⭐

- **做什么**：LLM 推理引擎设计与实现--调度器、批处理、KV Cache 管理、显存规划、分离式架构、多租户。
- **技能栈**：C++/Python、CUDA 基础、熟悉 vLLM/SGLang/TensorRT-LLM/TGI/LMDeploy 至少其一的源码、理解 PagedAttention/连续批处理/PD 分离、分布式（TP/PP/EP）。
- **KV Cache 关联**：直接负责。本 Wiki 的技术主线就是为这个岗准备的。
- **典型团队**：字节火山方舟/豆包、阿里 PAI/通义、腾讯机智/混元、华为诺亚/昇腾、百度文心、DeepSeek、Moonshot、智谱、MiniMax、商汤、阶跃星辰、美团/小红书大模型平台组、海外 NVIDIA TensorRT-LLM、Meta (vLLM/PyTorch)、Microsoft (DeepSpeed/vLLM)、Google (Pathways/JAX)、Together/Anyscale 等。

### 1.2 训练框架 / 分布式 ⭐⭐⭐⭐

- **做什么**：大模型预训练/微调框架--Megatron/DeepSpeed 类 3D 并行、通信优化、checkpoint、容错。
- **技能栈**：NCCL/分布式通信、PyTorch 内部、TP/PP/DP/EP/CP/SP、ZeRO、流水线气泡优化、显存复用。
- **KV Cache 关联**：弱（训练无 KV Cache）。但 attention/通信/并行的底层知识高度复用，常与推理岗互转。
- **典型团队**：同上大厂训练团队，尤其头部大模型公司的 pretrain infra。

### 1.3 算子 / Kernel ⭐⭐⭐⭐⭐

- **做什么**：高性能算子开发--FlashAttention 类融合 attention、量化算子、MoE 路由/组合算子、自定义 CUDA/Triton kernel。
- **技能栈**：CUDA（warp/shared memory/occupancy/async copy/TMA）、CUTLASS、Triton、GPU 微架构（Hopper/Blackwell）、性能 profiling（Nsight）。
- **KV Cache 关联**：强。PagedAttention kernel、MLA kernel、FP8 KV 量化算子都属此类。
- **典型团队**：NVIDIA（cuBLAS/cutlass/TRT-LLM）、Meta、各芯片厂、大厂 kernel 组、DeepSeek/Moonshot/智谱的 infra-kernel 组。

### 1.4 硬件 / 编译 ⭐⭐⭐

- **做什么**：图编译（TensorRT/XLA/TorchInductor/MLIR）、算子库、芯片侧算子适配、硬件协同设计。
- **技能栈**：编译器后端、MLIR、硬件架构、CUTLASS、芯片 ISA。
- **KV Cache 关联**：中。编译器要支持 paged/MLA 等 memory 访问 pattern。
- **典型团队**：NVIDIA、AMD、Intel、华为昇腾、寒武纪、燧原、壁仞、摩尔线程、各家 AI 编译组。

### 1.5 平台 / 工程化 ⭐⭐⭐

- **做什么**：大规模训练/推理集群调度、容错、监控、多租户、成本优化、CI/CD。
- **技能栈**：K8s/调度系统、分布式系统、SRE、Go/Java、监控体系。
- **KV Cache 关联**：中。分离式推理的 KV 池化、多租户 KV 隔离、全局调度属此范畴（Mooncake 类）。
- **典型团队**：大厂云平台/AI 平台组。

---

## 2. 公司盘点（国内为主）

> 仅列出有 AI Infra 招聘需求的代表性公司/团队，**非完整列表**，排序无意义。薪资、HC 以 [实时核对] 为准。

### 2.1 大厂 / 云

| 公司 | 代表团队/产品 | 方向侧重 | 备注 |
| --- | --- | --- | --- |
| 字节跳动 | 火山方舟、豆包大模型、AML | 推理 serving、训练、kernel 全栈 | 国内 AI Infra HC 最多的公司之一，SGLang/vLLM 都有深度参与 |
| 阿里云 | PAI、通义、弹性计算 | 推理、训练、平台 | 有自研推理框架与硬件协同 |
| 腾讯 | 机智、混元、腾讯云 | 训练框架、推理、平台 | 机智是大规模训练平台代表 |
| 华为 | 诺亚方舟、昇腾、MindSpore | 全栈含硬件/编译 | 软硬协同，昇腾生态 |
| 百度 | 文心、飞桨 | 训练框架、推理 | 飞桨 + 文心 |
| 美团/小红书/快手 | 大模型平台组 | 推理 serving 为主 | 应用驱动，工程导向 |

> 📌 **实时招聘信号（2026-08-12 抓取自牛客，✅ 已核实）**：[百度基础架构部急招 异构计算/CUDA 研发/C++/HPC 优化](https://www.nowcoder.com/discuss/353158230501171200)--提前批、部门直面。印证大厂基础架构部对 CUDA/HPC 算子人才的需求。快手/美团/华为海思的 AI Infra 校招面经见 [题库 §7.0](aiinfra_interview_questions.md)。

### 2.2 大模型创业公司（明星团队，KV Cache 方向高密度）

| 公司 | 团队亮点 | 与 KV Cache 关系 |
| --- | --- | --- |
| DeepSeek（深度求索） | MLA、PD 分离、MoE 训推一体 | **MLA 的提出者**，KV Cache 方向面试必考其架构 |
| Moonshot（月之暗面） | 长上下文、Mooncake KVCache-centric 分离式架构 | **KV Cache 池化的代表**，面试高频 |
| 智谱（Zhipu/GLM） | GLM 系列、SGLang 协同 | 作者本博客就在做 SGLang×GLM-5.2 KV Cache |
| MiniMax | 大规模推理、线性/稀疏 attention 探索 | 推理优化有特色 |
| 阶跃星辰（StepFun） | 多模态、大模型训练 | infra 团队扩张中 |
| 百川/零一/面壁 等 | 各有侧重 | 训推 infra |

### 2.3 芯片厂

NVIDIA（TRT-LLM、cutlass、cuBLAS）、AMD（ROCm/MIOpen）、Intel（oneAPI/SYCL）、华为昇腾、寒武纪、燧原、壁仞、摩尔线程、海光等。**硬件厂 KV Cache 工作偏 kernel/编译侧**。

### 2.4 海外（供参考）

- **NVIDIA**：TensorRT-LLM、cutlass、vLLM 社区核心。
- **Meta**：vLLM、PyTorch、xformers、Megatron 的发源地，开源贡献巨大。
- **Microsoft**：DeepSpeed、vLLM 协同、Azure 推理。
- **Google**：Pathways、JAX、TPU 软硬一体。
- **OpenAI / Anthropic**：自研推理栈，HC 少而精。
- **Together / Anyscale / Replicate / Fireworks**：纯推理服务公司，技术驱动。

---

## 3. 典型 JD 拆解（推理系统岗为例）

下面是一个**合成版**典型 JD（综合多家推理系统岗的共性要求），用于反推复习重点。**非任何一家公司的原文 JD**，[实时核对] 真实 JD。

> **岗位：大模型推理系统工程师 / Infra Engineer - LLM Serving**
>
> 职责：
> - 设计与实现大规模 LLM 推理引擎，优化吞吐与延迟（TTFT/TPOT/Goodput）。
> - 负责 KV Cache 管理、批处理调度、显存规划、分离式推理架构。
> - 参与高性能算子（attention/量化）的开发与调优。
> - 支持多模态、长上下文、MoE 模型的推理优化。
> - 线上集群的稳定性、成本与多租户隔离。
>
> 要求：
> - 熟悉 C++/Python，扎实的系统编程与数据结构功底。
> - 深入理解 Transformer、attention、KV Cache、GQA/MQA/MLA。
> - 熟悉 vLLM/SGLang/TensorRT-LLM 至少其一的设计与源码。
> - 熟悉 GPU 架构与 CUDA 编程，有算子优化经验加分。
> - 熟悉分布式训练/推理（TP/PP/EP/CP）、NCCL 通信。
> - 有大规模线上系统经验、熟悉容器/调度加分。
> - 有顶会论文（OSDI/SOSP/MLSys/ASPLOS/ISCA/SC）或开源贡献加分。

**反推复习重点（映射到本 Wiki）：**

| JD 关键词 | 复习位置 |
| --- | --- |
| KV Cache 管理 | [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md) 全文 |
| 批处理调度 | 同上 §3.2/§3.3 连续批处理 |
| 显存规划 | 同上 §2 显存建模 |
| 分离式推理 | 同上 §6.4 |
| GQA/MQA/MLA | 同上 §4.1/§4.2 |
| 推理引擎源码 | [资源汇总](aiinfra_resources.md) + [2 个月计划](aiinfra_2month_plan.md) 第 3-4 周 |
| GPU/CUDA/算子 | 计划第 1、7 周 + 资源汇总 |
| 分布式/通信 | 计划第 7 周 |
| 系统设计 | [面经](aiinfra_interview_questions.md) 系统设计章 + [系统设计专题](aiinfra_system_design.md) |

### 3.1 技能自评矩阵

用下表量化自己的准备度，**短板项优先排进 [2 个月计划](aiinfra_2month_plan.md)**。每项自评 4 档：①没听过 ②能讲概念 ③能讲清原理/推导 ④能改源码/手撕/优化。

| 技能 | 自评①-④ | 复习位置 | 目标档 |
| --- | --- | --- | --- |
| KV Cache 显存建模与口算 | ☐ | [KV Cache §2](aiinfra_kvcache_deep_dive.md) | ④ |
| PagedAttention 原理与源码 | ☐ | 同 §3.2 + vLLM 源码 | ④ |
| 连续批处理 / 调度 | ☐ | 同 §3.3 + vLLM/SGLang 源码 | ④ |
| GQA/MQA/MLA（含推导） | ☐ | 同 §4.1/§4.2/§4.2.1 | ④ |
| KV 量化（FP8/per-channel） | ☐ | 同 §4.3 | ③ |
| RadixAttention / 前缀复用 | ☐ | 同 §5 + SGLang 源码 | ④ |
| 长上下文（sink/streaming/window） | ☐ | 同 §6.2 | ③ |
| 分离式推理 / KV 池化（Mooncake 等） | ☐ | 同 §6.4 | ④ |
| FlashAttention 原理 | ☐ | [面经 Q3.1-3.2](aiinfra_interview_questions.md) | ③ |
| CUDA/Triton 算子 | ☐ | [计划第 2、7 周](aiinfra_2month_plan.md) | ③ |
| 分布式（TP/PP/EP/CP/NCCL） | ☐ | [面经第 4 章](aiinfra_interview_questions.md) | ③ |
| 系统设计（LLM serving） | ☐ | [系统设计专题](aiinfra_system_design.md) | ④ |
| 手撕 attention/LRU/调度 | ☐ | [面经附录 A](aiinfra_interview_questions.md) | ④ |

> 经验法则：目标岗位是**推理系统/serving** -> 表中前 8 项必须到 ④；**算子/kernel** 岗 -> FlashAttention/CUDA/手撕到 ④；**大厂训练框架**岗 -> 分布式到 ④、KV 项到 ③。

---

## 4. 职级与薪资（结构化参考，[实时核对]）

> 薪资受年份、职级、股票、城市、个人背景影响极大，下表仅为**量级感知**，绝非报价。务必用 levels.fyi / 脉脉 / 牛客核对最新。

| 档位 | 典型对应 | 大致年包（国内，人民币，量级） | 大致年包（海外，美元，量级） |
| --- | --- | --- | --- |
| 应届/初级 | 校招 SP/SSP、社招 1-3 年 | 40w - 80w | 150k - 250k 美元 |
| 中级 | 社招 3-6 年、大厂 P6/T5 左右 | 80w - 150w | 250k - 450k 美元 |
| 高级/资深 | P7/T6+、技术骨干 | 150w - 300w+ | 400k - 700k+ 美元 |
| 专家/架构 | P8/T7+、团队负责人 | 300w+ 上不封顶 | 600k - 1M+ 美元 |

**趋势观察（截至知识截止，需 [实时核对]）：**

- 2024-2025 AI Infra 是**卖方市场**，头部大模型公司为 infra 人才开出的溢价高于一般后端。
- **KV Cache / 推理方向**因直接关系到大模型服务成本，溢价尤其明显。
- 大模型创业公司**期权占比大、现金可能低于大厂**，需评估退出预期。
- 海外 remote 岗位（NVIDIA/Meta/Together 等）对国内候选人也是重要选项。

---

## 5. 简历匹配建议

把本 Wiki 的知识点映射到简历"能写出来"的经历：

1. **有推理引擎源码贡献**：给 vLLM/SGLang 提过 PR 是强信号。哪怕一个小优化（KV 调度、prefix caching 边界 case、某算子）。
2. **做过 KV Cache 相关实验**：复现 PagedAttention/MLA/RadixAttention 的某项指标对比，写成 benchmark 报告。
3. **有算子优化经历**：写一个 FlashAttention 变体或 FP8 KV 量化 kernel，贴 Nsight profile 数据。
4. **系统设计能力**：能画出"高并发 LLM serving 系统"架构并讲清调度/KV/分离式。
5. **论文/开源**：有 OSDI/SOSP/MLSys/ASPLOS/ISCA/SC 任一作，或 vLLM/SGLang/Megatron 显著贡献。

> 若以上都没有，**本 Wiki 的 2 个月计划**就是用来"制造这些经历"的--第 3-6 周的源码阅读 + 复现 + PR 任务即为此设计。

---

## 6. 待你补全的实时信息清单

请用实时渠道补全并回填本页（建议在评论/补充区记录）：

- [ ] 目标公司当前在招的 AI Infra 岗位 JD 原文（3-5 家）
- [ ] 目标公司最新薪资区间与签字费/股票
- [ ] 目标团队的技术栈与近期技术博客/分享
- [ ] 最近 3 个月目标公司的面经原题（牛客/一亩三分地/小红书）
- [ ] 内推联系人（脉脉/校友）

下一篇：[KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)。
