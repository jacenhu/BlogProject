<script setup>
defineProps({ featured: { type: Array, required: true } })
</script>

<template>
  <div class="blog-home">
    <section class="home-intro" aria-labelledby="home-title">
      <div class="intro-copy">
        <p class="eyebrow">JACEN / ENGINEERING NOTES</p>
        <h1 id="home-title">系统与智能的<br /><span>工程笔记</span></h1>
        <p class="intro-description">记录系统编程、分布式架构与大模型推理的学习和实践。<br class="desktop-break" />从原理到代码，把问题理解得更深一点。</p>
        <a class="browse-link" href="/program/">浏览技术笔记 <span aria-hidden="true">↗</span></a>
      </div>
      <a class="reading-feature" href="/program/interview/aiinfra_overview.html">
        <div class="feature-top"><span>专题阅读</span><span>01 / AI INFRA</span></div>
        <div>
          <p class="feature-kicker">从 KV Cache 出发</p>
          <h2>走进大模型<br />推理系统</h2>
          <p class="feature-description">核心知识、系统设计与面试准备。</p>
        </div>
        <div class="feature-bottom"><span>阅读系列导读</span><span aria-hidden="true">↗</span></div>
      </a>
    </section>

    <section class="home-topics" aria-labelledby="topics-title">
      <div class="section-heading">
        <div><p class="eyebrow">EXPLORE</p><h2 id="topics-title">按专题探索</h2></div>
        <span class="section-caption">从感兴趣的方向开始</span>
      </div>
      <slot />
    </section>

    <section class="home-reading" aria-labelledby="reading-title">
      <div class="section-heading">
        <div><p class="eyebrow">SELECTED NOTES</p><h2 id="reading-title">精选笔记</h2></div>
        <a href="/program/">全部笔记 <span aria-hidden="true">↗</span></a>
      </div>
      <div class="reading-list">
        <a v-for="(article, index) in featured" :key="article.link" :href="article.link" class="reading-row">
          <span class="reading-number" aria-hidden="true">{{ String(index + 1).padStart(2, '0') }}</span>
          <div class="reading-copy"><span class="reading-category">{{ article.category }}</span><h3>{{ article.title }}</h3><p>{{ article.description }}</p></div>
          <span class="reading-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
    </section>
  </div>
</template>

<style scoped>
.blog-home { max-width: 1120px; padding: 0 32px; margin: 0 auto; }
.home-intro { display: grid; grid-template-columns: 1.4fr 1fr; gap: 72px; align-items: center; padding: 76px 0 64px; }
.blog-home .eyebrow { margin: 0 0 18px; font: 500 12px/1.5 var(--blog-font-mono); letter-spacing: .12em; color: var(--vp-c-text-2); }
.blog-home h1 { margin: 0; font-size: clamp(38px, 4.4vw, 58px); line-height: 1.23; letter-spacing: -.045em; font-weight: 750; }
.blog-home h1 span { color: var(--vp-c-brand-1); }
.blog-home .intro-description { margin: 26px 0 30px; font-size: 16px; line-height: 1.9; color: var(--vp-c-text-2); }
.browse-link { display: inline-flex; gap: 32px; align-items: center; background: var(--vp-c-brand-1); color: var(--vp-c-white); padding: 12px 20px; border-radius: 8px; font-size: 15px; font-weight: 600; transition: background .2s; }
.browse-link:hover { background: var(--vp-c-brand-2); }
.reading-feature { min-height: 328px; padding: 28px 30px 22px; display: flex; flex-direction: column; justify-content: space-between; gap: 30px; background: #132e52; color: #fff; border: 1px solid #294361; border-radius: 14px; transition: border-color .2s, transform .2s; }
.reading-feature:hover { border-color: #629bea; transform: translateY(-3px); }
.feature-top, .feature-bottom { display: flex; justify-content: space-between; gap: 12px; align-items: center; font-size: 13px; }
.feature-top { color: #b9cbe1; }
.feature-top span:last-child { font-family: var(--blog-font-mono); font-size: 12px; }
.blog-home .feature-kicker { font-size: 14px; color: #b9cbe1; margin: 0 0 10px; }
.blog-home .reading-feature h2 { font-size: 32px; line-height: 1.4; letter-spacing: -.025em; margin: 0; }
.blog-home .feature-description { margin: 12px 0 0; font-size: 14px; color: #c3d1e4; }
.feature-bottom { padding-top: 18px; border-top: 1px solid #3b5270; font-size: 14px; }
.feature-bottom span:last-child { font-size: 21px; }
.home-topics { padding: 30px 0 44px; border-top: 1px solid var(--vp-c-divider); }
.section-heading { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 24px; }
.section-heading .blog-home .eyebrow { margin-bottom: 7px; }
.blog-home .section-heading h2 { margin: 0; font-size: 25px; font-weight: 650; letter-spacing: -.03em; }
.section-caption, .section-heading > a { font-size: 14px; color: var(--vp-c-text-2); }
.section-heading > a:hover { color: var(--vp-c-brand-1); }
.home-reading { padding-bottom: 72px; }
.reading-list { border-top: 1px solid var(--vp-c-divider); }
.reading-row { display: grid; grid-template-columns: 48px 1fr 32px; align-items: center; gap: 16px; padding: 26px 10px; border-bottom: 1px solid var(--vp-c-divider); transition: background .2s; }
.reading-row:hover { background: var(--vp-c-bg-soft); }
.reading-number { font: 14px var(--blog-font-mono); color: var(--vp-c-text-3); align-self: start; padding-top: 5px; }
.reading-category { font-size: 12px; color: var(--vp-c-brand-1); }
.blog-home .reading-copy h3 { font-size: 20px; line-height: 1.5; margin: 5px 0 6px; font-weight: 600; }
.blog-home .reading-copy p { margin: 0; font-size: 14px; line-height: 1.8; color: var(--vp-c-text-2); }
.reading-arrow { font-size: 22px; color: var(--vp-c-text-2); }
@media (max-width: 959px) {
  .home-intro { gap: 32px; padding-top: 48px; }
  .desktop-break { display: none; }
  .reading-feature { padding: 24px; }
}
@media (max-width: 640px) {
  .blog-home { padding: 0 22px; }
  .home-intro { grid-template-columns: 1fr; gap: 32px; padding: 40px 0; }
  .blog-home .eyebrow { font-size: 12px; margin-bottom: 16px; }
  .blog-home .intro-description { margin: 20px 0 24px; }
  .reading-feature { min-height: 284px; gap: 24px; }
  .blog-home .reading-feature h2 { font-size: 28px; }
  .section-caption { display: none; }
  .blog-home .section-heading h2 { font-size: 23px; }
  .home-topics { padding-bottom: 30px; }
  .reading-row { grid-template-columns: 26px 1fr 18px; gap: 8px; padding: 22px 0; }
  .blog-home .reading-copy h3 { font-size: 18px; }
  .home-reading { padding-bottom: 40px; }
}
@media (prefers-reduced-motion: reduce) {
  .reading-feature, .reading-row, .browse-link { transition: none; }
  .reading-feature:hover { transform: none; }
}
</style>
