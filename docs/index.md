---
layout: home
hero:
  name: Jacen's Blog
  tagline: 记录技术学习与实践，探索系统编程、大模型推理与分布式架构
  actions:
    - theme: brand
      text: 浏览文章
      link: /program/
    - theme: alt
      text: GitHub
      link: https://github.com/jacenhu
---

<div class="home-intro">

欢迎。这里是我在工作与学习中积累的技术笔记，涵盖系统编程、大模型推理、分布式存储等方向。

</div>

<div class="home-categories">

<div class="category-grid">

<a href="/program/#c" class="category-card">
  <span class="category-icon">🖥️</span>
  <span class="category-title">C++ & 系统</span>
  <span class="category-desc">数据结构、网络协议、分布式存储</span>
</a>

<a href="/program/#llm" class="category-card">
  <span class="category-icon">🤖</span>
  <span class="category-title">LLM & 模型</span>
  <span class="category-desc">SGLang、Llama、DeepSeek、GLM 模型分析</span>
</a>

<a href="/program/#kv-cache" class="category-card">
  <span class="category-icon">⚡</span>
  <span class="category-title">KV Cache</span>
  <span class="category-desc">GLM 5.2 KV Cache 机制与传输</span>
</a>

<a href="/program/#java后端" class="category-card">
  <span class="category-icon">🏗️</span>
  <span class="category-title">后端 & 实践</span>
  <span class="category-desc">SpringBoot、数据湖、开发环境</span>
</a>

<a href="/program/#论文" class="category-card">
  <span class="category-icon">📖</span>
  <span class="category-title">论文 & 工具</span>
  <span class="category-desc">GFS 论文阅读、性能压测实践</span>
</a>

</div>

</div>

<style>
.home-intro {
  max-width: 600px;
  margin: 0 auto;
  text-align: center;
  color: var(--vp-c-text-2);
  font-size: 1.05rem;
  line-height: 1.7;
  padding: 0 1rem;
}

.home-categories {
  max-width: 800px;
  margin: 2.5rem auto 0;
  padding: 0 1rem;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.category-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 1.5rem 1rem;
  border-radius: 12px;
  border: 1px solid var(--vp-c-bg-soft);
  background: var(--vp-c-bg-soft);
  transition: border-color 0.25s, background 0.25s;
  text-decoration: none !important;
}

.category-card:hover {
  border-color: var(--vp-c-brand);
  background: var(--vp-c-bg);
}

.category-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.category-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.25rem;
}

.category-desc {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}
</style>
