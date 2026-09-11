---
layout: home
markdownStyles: false
topics:
  - title: C++ & 系统
    icon: 🖥️
    link: /program/#c
    description: 数据结构、网络协议、分布式存储、设计模式
  - title: LLM & 模型
    icon: 🤖
    link: /program/#llm
    description: SGLang 模型分析，Llama、DeepSeek、GLM 部署与推理
  - title: KV Cache
    icon: ⚡
    link: /program/#kv-cache
    description: GLM 5.2 KV Cache 机制、Attention、传输与集成
  - title: 后端 & 架构
    icon: 🏗️
    link: /program/#java后端
    description: SpringBoot、数据湖、分片技术、低代码实践
  - title: 论文 & 工具
    icon: 📖
    link: /program/#论文
    description: GFS 经典论文阅读、性能压测、开发环境配置
featured:
  - category: 推理系统
    title: KV Cache 核心知识体系
    description: 从显存估算、分页管理到前缀复用，梳理大模型推理的关键机制。
    link: /program/interview/aiinfra_kvcache_deep_dive.html
  - category: 源码阅读
    title: SGLang 的 Llama 模型实现
    description: 沿着模型结构与执行路径，理解推理引擎中的 Llama 实现。
    link: /program/models/sglang_llama_model.html
  - category: 分布式存储
    title: 重读 Google File System
    description: 从架构、租约与副本管理，理解经典分布式文件系统的设计取舍。
    link: /program/paper/gfs.html
---

<BlogHome :featured="$frontmatter.featured">
  <TopicCarousel :items="$frontmatter.topics" />
</BlogHome>
