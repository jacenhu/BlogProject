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
/* ── 压缩 Hero 区域，让卡片尽量不滚动即见 ── */
:deep(.VPHero) {
  padding: 2rem 0 1.5rem !important;
}
:deep(.VPHero .container) {
  gap: 0.5rem !important;
}
:deep(.VPHero .name) {
  display: none !important;
}
:deep(.VPHero .tagline) {
  font-size: 1.05rem !important;
  margin-top: 0.25rem !important;
}
:deep(.VPHero .actions) {
  margin-top: 1rem !important;
}
:deep(.VPHero .actions .VPButton) {
  padding: 0.4rem 1.2rem !important;
  font-size: 0.85rem !important;
  border-radius: 20px !important;
}

.home-intro {
  max-width: 600px;
  margin: 0 auto;
  text-align: center;
  color: var(--vp-c-text-2);
  font-size: 1rem;
  line-height: 1.6;
  padding: 0 1rem;
}

.home-categories {
  max-width: 800px;
  margin: 1.5rem auto 0;
  padding: 0 1rem;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.75rem;
}

@media (max-width: 768px) {
  .category-grid {
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  }
}

.category-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 1rem 0.75rem;
  border-radius: 10px;
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
  font-size: 1.5rem;
  margin-bottom: 0.3rem;
}

.category-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.15rem;
}

.category-desc {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
  line-height: 1.4;
}
</style>
