---
title: AI Infra 图文
description: 配合技术示意图，分析 GPU、推理引擎、KV Cache 与分布式推理的机制和工程取舍。
---

# AI Infra 图文

用文章和技术示意图，解释 GPU、推理引擎、KV Cache 与分布式推理中的机制和工程取舍。每篇文章保留来源、版本与适用条件，便于阅读和复查。

## 推理系统

### [SGLang 并行策略分析：一组 GPU，究竟该怎么拆？](./sglang-parallelism.md)

2026-09-18 · 4 张技术配图

同样是 8 张 GPU，怎样选择 TP、普通 DP、DP Attention 和 EP？从切分对象、通信代价与 KV 布局出发，分析各类并行策略的组合关系，以及 CP/DCP、PP 和 PD 分离的适用边界。

[阅读全文 →](./sglang-parallelism.md)

## 相关阅读

- [KV Cache 核心知识体系](/program/interview/aiinfra_kvcache_deep_dive.md)
- [SGLang 的 Llama 模型实现](/program/models/sglang_llama_model.md)
- [AI Infra 系统设计专题](/program/interview/aiinfra_system_design.md)
