---
layout: home
hero:
  tagline: 系统编程、大模型推理与分布式架构
---

<div class="carousel-wrapper">

<div class="carousel-track" id="carousel-track">

<a href="/program/#c" class="carousel-card">
  <span class="card-icon">🖥️</span>
  <span class="card-title">C++ &amp; 系统</span>
  <span class="card-desc">数据结构、网络协议、分布式存储、设计模式</span>
</a>

<a href="/program/#llm" class="carousel-card">
  <span class="card-icon">🤖</span>
  <span class="card-title">LLM &amp; 模型</span>
  <span class="card-desc">SGLang 模型分析，Llama、DeepSeek、GLM 部署与推理</span>
</a>

<a href="/program/#kv-cache" class="carousel-card">
  <span class="card-icon">⚡</span>
  <span class="card-title">KV Cache</span>
  <span class="card-desc">GLM 5.2 KV Cache 机制、Attention、传输与集成</span>
</a>

<a href="/program/#java后端" class="carousel-card">
  <span class="card-icon">🏗️</span>
  <span class="card-title">后端 &amp; 架构</span>
  <span class="card-desc">SpringBoot、数据湖、分片技术、低代码实践</span>
</a>

<a href="/program/#论文" class="carousel-card">
  <span class="card-icon">📖</span>
  <span class="card-title">论文 &amp; 工具</span>
  <span class="card-desc">GFS 经典论文阅读、性能压测、开发环境配置</span>
</a>

</div>

<div class="carousel-dots" id="carousel-dots"></div>

</div>

<style>
/* ── Carousel ── */
.carousel-wrapper {
  max-width: 720px;
  margin: 0 auto;
  padding: 0 1rem 2rem;
  overflow: hidden;
}

.carousel-track {
  display: flex;
  transition: transform 0.5s ease;
}

.carousel-card {
  flex: 0 0 calc(100% / 3);
  min-width: calc(100% / 3);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 1.5rem 1rem;
  border-radius: 12px;
  border: 1px solid var(--vp-c-bg-soft);
  background: var(--vp-c-bg-soft);
  text-decoration: none !important;
  transition: border-color 0.25s, background 0.25s;
  box-sizing: border-box;
}

.carousel-card:hover {
  border-color: var(--vp-c-brand);
  background: var(--vp-c-bg);
}

.card-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.card-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.3rem;
}

.card-desc {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}

/* ── Dots ── */
.carousel-dots {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
}

.carousel-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 1px solid var(--vp-c-text-3);
  background: transparent;
  cursor: pointer;
  padding: 0;
  transition: background 0.3s, border-color 0.3s;
}

.carousel-dot.active {
  background: var(--vp-c-brand);
  border-color: var(--vp-c-brand);
}

/* ── Mobile ── */
@media (max-width: 640px) {
  .carousel-card {
    flex: 0 0 100%;
    min-width: 100%;
  }
}
</style>

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const track = document.getElementById('carousel-track');
  const dotsContainer = document.getElementById('carousel-dots');
  if (!track || !dotsContainer) return;
  const cards = track.querySelectorAll('.carousel-card');
  const total = cards.length;

  // Determine visible count based on screen width
  function visibleCount() {
    return window.innerWidth <= 640 ? 1 : 3;
  }

  let current = 0;
  let timer = null;

  function renderDots() {
    const vis = visibleCount();
    const dotCount = total - vis + 1;
    dotsContainer.innerHTML = '';
    for (let i = 0; i < dotCount; i++) {
      const dot = document.createElement('button');
      dot.className = 'carousel-dot' + (i === current ? ' active' : '');
      dot.setAttribute('aria-label', '第' + (i + 1) + '组');
      dot.addEventListener('click', () => goTo(i));
      dotsContainer.appendChild(dot);
    }
  }

  function goTo(index) {
    const vis = visibleCount();
    const max = total - vis;
    if (index < 0) index = max;
    if (index > max) index = 0;
    current = index;
    const percent = -(current * (100 / vis));
    track.style.transform = 'translateX(' + percent + '%)';
    renderDots();
  }

  function next() {
    const vis = visibleCount();
    const max = total - vis;
    if (current >= max) {
      goTo(0);
    } else {
      goTo(current + 1);
    }
  }

  function startAuto() {
    stopAuto();
    timer = setInterval(next, 3500);
  }

  function stopAuto() {
    if (timer) { clearInterval(timer); timer = null; }
  }

  renderDots();
  startAuto();

  track.parentElement.addEventListener('mouseenter', stopAuto);
  track.parentElement.addEventListener('mouseleave', startAuto);
  window.addEventListener('resize', () => { renderDots(); goTo(0); });
});
</script>
