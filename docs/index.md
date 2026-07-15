---
layout: home
hero:
  tagline: 记录技术学习与实践，探索系统编程、大模型推理与分布式架构
---

<div class="home-nav">

<a href="/program/#c" class="nav-chip">🖥️ C++ & 系统</a>
<a href="/program/#llm" class="nav-chip">🤖 LLM & 模型</a>
<a href="/program/#kv-cache" class="nav-chip">⚡ KV Cache</a>
<a href="/program/#java后端" class="nav-chip">🏗️ 后端 & 实践</a>
<a href="/program/#论文" class="nav-chip">📖 论文 & 工具</a>

</div>

<style>
/* ── Hero ── */
:deep(.VPHero) {
  padding: 3rem 0 2rem !important;
}
:deep(.VPHero .container) {
  gap: 0.25rem !important;
}
:deep(.VPHero .tagline) {
  max-width: 480px !important;
  margin: 0 auto !important;
  font-size: 1.15rem !important;
  line-height: 1.7 !important;
  color: var(--vp-c-text-2) !important;
}
/* ── 分类导航 ── */
.home-nav {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.5rem;
  max-width: 640px;
  margin: 0 auto;
  padding: 0 1rem 2rem;
}

.nav-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.9rem;
  border-radius: 20px;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
  border: 1px solid transparent;
  transition: border-color 0.2s, background 0.2s;
  text-decoration: none !important;
  white-space: nowrap;
}

.nav-chip:hover {
  border-color: var(--vp-c-brand);
  background: var(--vp-c-bg);
}
</style>
