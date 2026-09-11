---
layout: home
hero:
  tagline: 系统编程、大模型推理与分布式架构
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
---

<TopicCarousel :items="$frontmatter.topics" />
