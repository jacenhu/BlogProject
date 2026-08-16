# 资源汇总

> 首次编写：2026-08-12 | 最后更新：2026-08-12

> **AI Infra / KV Cache 面试系列**
> - [导读](aiinfra_overview.md)
> - [招聘岗位全景](aiinfra_jobs.md)
> - [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)
> - [面经与高频考点](aiinfra_interview_questions.md)
> - [系统设计专题](aiinfra_system_design.md)
> - [2 个月面试复习计划](aiinfra_2month_plan.md)
> - **资源汇总** ← 当前文章

按主题归类的学习资源。**链接为知识库中的名称与出处，部分 URL 需你自行搜索确认**（本环境无法联网验证）。建议建立 Zotero/Notion 库管理。

---

## 1. 必读论文（按主题）

> ✅ 以下 arXiv 链接均于 **2026-08-12 经 curl 核实可达**（标题已逐一校对）。ORCA 为 OSDI'22 未上 arXiv，按标题检索即可。

### 1.1 KV Cache 存储与调度

| 论文 | 会议/年份 | 核心贡献 | arXiv |
| --- | --- | --- | --- |
| Efficient Memory Management for LLM Serving with PagedAttention（vLLM） | SOSP'23 | PagedAttention 分页 KV 管理 | [2309.06180](https://arxiv.org/abs/2309.06180) |
| Orca: A Distributed Serving System for Transformer-Based Generative Models | OSDI'22 | iteration-level 连续批处理 | （OSDI'22，按标题检索） |
| SGLang: Efficient Execution of Structured Language Model Programs | 2024 | RadixAttention 前缀复用 | [2312.07104](https://arxiv.org/abs/2312.07104) |

### 1.2 Attention 与 KV 结构

| 论文 | 会议/年份 | 核心贡献 | arXiv |
| --- | --- | --- | --- |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS'22 | tiling + online softmax | [2205.14135](https://arxiv.org/abs/2205.14135) |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | 占用率/并行优化 | [2307.08691](https://arxiv.org/abs/2307.08691) |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | 2024 | Hopper TMA/FP8/warp-spec | [2407.08608](https://arxiv.org/abs/2407.08608) |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | EMNLP'23 | 分组 KV 头 | [2305.13245](https://arxiv.org/abs/2305.13245) |
| DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model | 2024 | MLA 低秩压缩 KV | [2405.04434](https://arxiv.org/abs/2405.04434) |
| DeepSeek-V3 Technical Report | 2024 | MLA + MoE 训推实践 | [2412.19437](https://arxiv.org/abs/2412.19437) |

> MQA（Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need"）见作者博客原文，未单独上 arXiv。

### 1.3 长上下文与淘汰

| 论文 | 会议/年份 | 核心贡献 | arXiv |
| --- | --- | --- | --- |
| Efficient Streaming Language Models with Attention Sinks（StreamingLLM） | ICLR'24 | attention sink + 滑窗流式 | [2309.17453](https://arxiv.org/abs/2309.17453) |
| H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs | NeurIPS'23 | 按 attention score 淘汰 | [2306.14048](https://arxiv.org/abs/2306.14048) |
| Ring Attention with Blockwise Transformers for Near-Infinite Context | 2023 | 环形上下文并行 | [2310.01889](https://arxiv.org/abs/2310.01889) |

### 1.4 分离式推理与池化（高阶）

| 论文 | 会议/年份 | 核心贡献 | arXiv |
| --- | --- | --- | --- |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving | OSDI'24 | PD 分离 | [2401.09670](https://arxiv.org/abs/2401.09670) |
| Splitwise: Efficient Generative LLM Inference Using Phase Splitting | ISCA'24 | 按 phase 切分资源池 | [2311.18677](https://arxiv.org/abs/2311.18677) |
| Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | 2024 | KVCache-centric 全局池 | [2407.00079](https://arxiv.org/abs/2407.00079) |

### 1.5 KV 量化

| 论文 | 核心贡献 | arXiv |
| --- | --- | --- |
| KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | K per-channel / V per-token | [2402.02750](https://arxiv.org/abs/2402.02750) |
| KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | KV 量化方案 | [2401.18079](https://arxiv.org/abs/2401.18079) |
| QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs | 旋转消除离群值 + 低比特 | [2404.00456](https://arxiv.org/abs/2404.00456) |

### 1.6 投机解码与并行

| 论文 | 核心贡献 | arXiv |
| --- | --- | --- |
| Fast Inference from Transformers via Speculative Decoding | 投机解码框架 | [2211.17192](https://arxiv.org/abs/2211.17192) |
| Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | 多头并行 draft | [2401.10774](https://arxiv.org/abs/2401.10774) |
| EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | 自回归特征 draft | [2401.15077](https://arxiv.org/abs/2401.15077) |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | 优化器/梯度/参数分片 | [1910.02054](https://arxiv.org/abs/1910.02054) |

> Megatron-LM 系列论文（TP/PP/DP 3D 并行）有多篇，按 "Megatron-LM NVIDIA" 检索官方技术博客与论文。

---

## 2. 开源项目（源码必读）

### 2.1 推理引擎

| 项目 | 重点阅读 | 仓库 |
| --- | --- | --- |
| **vLLM** | `vllm/core/scheduler.py`、`vllm/attention/backends/`、PagedAttention kernel、block_manager | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) · [docs.vllm.ai](https://docs.vllm.ai/en/latest/) |
| **SGLang** | `python/sglang/srt/managers/`、RadixAttention、scheduler、CUDA Graph | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) · [docs.sglang.ai](https://docs.sglang.ai/) |
| **TensorRT-LLM** | plugin、attention plugin、runtime | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| **LMDeploy / TGI / DeepSpeed-MII** | 各自调度与 KV 管理对比 | [LMDeploy](https://github.com/InternLM/lmdeploy) · [TGI](https://github.com/huggingface/text-generation-inference) · [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) |

### 2.2 算子与并行

| 项目 | 重点 | 仓库 |
| --- | --- | --- |
| **FlashAttention** | forward/backward kernel | [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| **CUTLASS** | GEMM 模板、Hopper TMA 范例 | [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) |
| **Triton** | attention/layernorm 教程 | [github.com/triton-lang/triton](https://github.com/triton-lang/triton) |
| **Megatron-LM / torchtitan** | TP/PP/EP/CP 实现 | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) · [torchtitan](https://github.com/pytorch/torchtitan) |

### 2.3 工具

- **Nsight Compute / Nsight Systems**：kernel 与系统级 profiling。
- **PyTorch Profiler**：算子级耗时与显存。
- **torch.cuda.memory._record_memory_history**：显存追踪。

---

## 3. 博客与文档

> ✅ 官方文档链接于 2026-08-12 经 curl 核实可达（docs.vllm.ai / docs.sglang.ai 均 HTTP 200）。

- **vLLM 官方文档**（[docs.vllm.ai](https://docs.vllm.ai/en/latest/)）：PagedAttention 设计、连续批处理、prefix caching 文档与博客。
- **SGLang 官方文档**（[docs.sglang.ai](https://docs.sglang.ai/)）：RadixAttention、结构化生成、性能优化、benchmark。
- **NVIDIA Developer Blog**：TensorRT-LLM、FP8、Hopper/Blackwell 特性。
- **HuggingFace Blog**：TGI、量化、attention 变体科普（站点偶有不可达，需重试）。
- **Lilian Weng (Lil'Log)**：attention、long context 综述。
- **作者本博客 KV Cache 系列**（直接对照，强相关）：
  - [KV Cache 基础知识](../kvcache/sglang_kvcache_knowhow.md)
  - [深度解析 SGLang × GLM-5.2 KV Cache](../kvcache/sglang_kvcache_glm5.2.md)
  - [GLM 5.2 KV Cache 对接原理](../kvcache/sglang_glm52_kv_cache_integration.md)
  - [GLM 5.2 Attention 机制](../kvcache/sglang_glm52_attention_mechanism.md)
  - [GLM 5.2 KV Cache 传输](../kvcache/sglang_glm52_kv_cache_transfer.md)
  - [DeepSeek V4 KV Cache](../kvcache/sglang_kvcache_deeepseekv4.md)
  - [sglang llama 模型实现分析](../models/sglang_llama_model.md)
  - [sglang deepseek v4 模型实现分析](../models/sglang_deepseekv4_model.md)
  - [sglang glm5.2 模型实现分析](../models/sglang_glm5.2_model.md)

---

## 4. 课程与基础

| 资源 | 方向 |
| --- | --- |
| **CUDA C++ Programming Guide / CUDA C++ Best Practices** | GPU 编程基础 |
| **UIUC ECE408 (Applications of Parallel Computers)** | CUDA 系统课 |
| **CMU 15-418 / Stanford CS149** | 并行计算体系 |
| **CMU 10-414 (Deep Learning Systems, dlsyscourse)** | 深度学习系统实现（手写 autograd/nn） |
| **NVIDIA cuDLF / GTC talks** | Hopper/Blackwell kernel 实战 |
| **《深度学习系统》相关公开课** | 训推框架全栈 |

---

## 5. 面经与招聘实时渠道（[实时核对]）

- **牛客网**：面经 + 内推 + 笔试，AI 方向面经密度高。
- **一亩三分地**：海外岗面经与薪资。
- **脉脉**：国内薪资/团队风评/真实 HC。
- **Boss 直聘 / LinkedIn**：JD 原文与投递。
- **levels.fyi**：海外薪资对标。
- **知乎 / 小红书**：面经帖与公司点评（需甄别）。
- **公司招聘官网**：最权威 JD 与团队介绍。
- **GitHub Discussion / 项目 Discord**：vLLM/SGLang 社区常有人分享面经与招聘。

---

## 6. 待你补全的资源清单

把以下补齐后，本 Wiki 就是一个完整的、有实时引用的复习中枢（✅ 为 2026-08-12 已完成）：

- [x] 必读论文的 arXiv 链接（§1，已核实可达）--精读笔记链接待补
- [x] vLLM/SGLang 官方文档与仓库链接（§2/§3，已核实）--具体 commit / 文件行号锚点待补
- [x] 牛客真实面经原帖链接（[题库 §7.0](aiinfra_interview_questions.md)，已抓取 5 篇）--更多面经与一亩三分地海外面经待补
- [ ] 目标公司最新 JD 链接（5 家，Boss/官网，需登录抓取）
- [ ] 内推联系人
- [ ] 自己的复现仓库链接（numpy attention / Triton FA / benchmark）
- [ ] 模拟面试录像与复盘文档链接

---

## 7. 推荐阅读顺序（给时间紧的人）

1. 本 Wiki [导读](aiinfra_overview.md) + [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md)（建立地图）
2. vLLM（SOSP'23）+ FlashAttention（NeurIPS'22）两篇论文（地基）
3. DeepSeek-V2/V3 技术报告的 MLA 章节（前沿核心）
4. Mooncake / DistServe 任一（分离式架构）
5. SGLang 论文 + 源码（前缀复用 + 工程优化）
6. 本 Wiki [面经题库](aiinfra_interview_questions.md) + [2 个月计划](aiinfra_2month_plan.md)（输出转化）

---

> 本 Wiki 到此完成主体。建议每周末回顾一次 [2 个月计划](aiinfra_2month_plan.md) 的自检表，并把新学到的点回填到 [KV Cache 核心知识体系](aiinfra_kvcache_deep_dive.md) 与 [题库](aiinfra_interview_questions.md)。祝面试顺利。
