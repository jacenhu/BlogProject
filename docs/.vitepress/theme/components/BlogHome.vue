<script setup>
defineProps({ featured: { type: Array, required: true }, topics: { type: Array, required: true } })
</script>

<template>
  <div class="blog-home">
    <section class="home-intro" aria-labelledby="home-title">
      <div class="intro-copy">
        <p class="hello"><span aria-hidden="true">✳</span> Jacen / 技术笔记</p>
        <h1 id="home-title">系统工程与<br /><span>模型推理</span></h1>
        <p class="intro-description">记录系统编程、分布式架构与大模型推理的学习和实践。</p>
        <a class="browse-link" href="/program/">浏览技术笔记 <span aria-hidden="true">↗</span></a>
      </div>
      <a class="reading-feature" href="/program/interview/aiinfra_overview.html">
        <div class="feature-top"><span>AI INFRA / 01</span><span class="feature-icon" aria-hidden="true">↗</span></div>
        <div><p class="feature-kicker">知识梳理与面试准备</p><h2>KV Cache<br />与推理系统</h2></div>
        <p class="feature-description">KV Cache · 推理引擎 · 系统设计</p>
        <div class="feature-bottom">阅读专题导读 <span aria-hidden="true">→</span></div>
      </a>
    </section>

    <section class="home-topics" aria-labelledby="topics-title">
      <div class="section-heading"><h2 id="topics-title">技术方向</h2><span class="section-caption">TOPICS</span></div>
      <div class="topic-grid">
        <a v-for="topic in topics" :key="topic.link" :href="topic.link" class="topic-card">
          <span class="topic-icon" aria-hidden="true">{{ topic.icon }}</span>
          <h3>{{ topic.title }}</h3>
          <p>{{ topic.description }}</p>
        </a>
      </div>
    </section>

    <section class="home-reading" aria-labelledby="reading-title">
      <div class="section-heading"><h2 id="reading-title">精选文章</h2><a href="/program/">全部笔记 <span aria-hidden="true">↗</span></a></div>
      <div class="reading-grid">
        <a v-for="(article, index) in featured" :key="article.link" :href="article.link" class="reading-card">
          <div class="reading-top"><span class="reading-category">{{ article.category }}</span><span class="reading-number">0{{ index + 1 }}</span></div>
          <h3>{{ article.title }}</h3>
          <p>{{ article.description }}</p>
          <span class="reading-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
    </section>
  </div>
</template>

<style scoped>
.blog-home { max-width: 1180px; padding: 0 40px; margin: 0 auto; }
.home-intro { display: grid; grid-template-columns: 1.6fr 1fr; align-items: center; gap: 64px; padding: 76px 0 80px; }
.hello { display: flex; align-items: center; gap: 10px; margin: 0 0 24px; font-size: 16px; font-weight: 550; }
.hello span { color: var(--vp-c-brand-1); font-size: 32px; line-height: 1; }
h1 { font-size: clamp(48px, 6.6vw, 80px); line-height: 1.2; letter-spacing: -.065em; font-weight: 800; margin: 0; }
h1 span { color: var(--vp-c-brand-1); }
.intro-description { font-size: 17px; line-height: 1.8; color: var(--vp-c-text-2); margin: 26px 0; }
.browse-link { display: inline-flex; align-items: center; gap: 24px; background: var(--vp-c-text-1); color: var(--vp-c-bg); border-radius: 99px; padding: 14px 22px; font-size: 15px; font-weight: 600; transition: transform .2s; }
.browse-link:hover { transform: translateY(-3px); }
.reading-feature { max-width: 348px; width: 100%; justify-self: end; align-self: end; min-height: 338px; margin: 0 8px 8px 0; padding: 28px; display: flex; flex-direction: column; gap: 24px; background: #e1f2d7; color: #304834; border: 1px solid #cce2c1; border-radius: 22px; transform: rotate(3deg); box-shadow: 7px 8px 0 var(--blog-feature-shadow); transition: transform .25s, box-shadow .25s; }
.reading-feature:hover, .reading-feature:focus-visible { transform: rotate(0deg) translateY(-4px); box-shadow: 3px 8px 0 var(--blog-feature-shadow); }
.feature-top { display: flex; justify-content: space-between; align-items: center; font: 12px/1.5 var(--blog-font-mono); letter-spacing: .04em; }
.feature-icon { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid #8aa47d; border-radius: 50%; font-size: 22px; }
.feature-kicker { font-size: 14px; margin: 0 0 8px; }
.reading-feature h2 { font-size: 38px; font-weight: 750; line-height: 1.3; letter-spacing: -.045em; margin: 0; }
.feature-description { font-size: 13px; margin: 0; }
.feature-bottom { display: flex; justify-content: space-between; border-top: 1px solid #bfd8b4; padding-top: 16px; font-size: 14px; font-weight: 550; margin-top: auto; }
.home-topics { padding: 0 0 58px; }
.section-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 22px; }
.section-heading h2 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -.04em; line-height: 1.5; }
.section-caption { font: 12px var(--blog-font-mono); color: var(--vp-c-text-3); }
.section-heading > a { font-size: 14px; color: var(--vp-c-text-2); white-space: nowrap; }
.section-heading > a:hover { color: var(--vp-c-brand-1); }
.topic-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
.topic-card { padding: 22px 18px; background: var(--vp-c-bg-elv); border: 1px solid var(--vp-c-divider); border-radius: 16px; transition: border-color .2s, transform .2s; }
.topic-card:hover { transform: translateY(-5px); border-color: var(--vp-c-brand-1); }
.topic-icon { display: inline-flex; align-items: center; justify-content: center; width: 42px; height: 42px; background: var(--vp-c-bg-soft); border-radius: 13px; font-size: 24px; }
.topic-card h3 { font-size: 16px; font-weight: 650; margin: 16px 0 8px; line-height: 1.5; }
.topic-card p { margin: 0; font-size: 14px; color: var(--vp-c-text-2); line-height: 1.75; }
.home-reading { padding-bottom: 80px; }
.reading-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
.reading-card { position: relative; display: flex; flex-direction: column; align-items: start; padding: 26px; border-radius: 20px; border: 1px solid var(--vp-c-divider); background: var(--vp-c-bg-elv); transition: transform .2s, border-color .2s; }
.reading-card:first-child { background: var(--blog-card-tint); border-color: transparent; }
.reading-card:nth-child(2) { background: var(--blog-card-warm); border-color: transparent; }
.reading-card:nth-child(3) { background: var(--blog-card-cool); border-color: transparent; }
.reading-card:hover { transform: translateY(-5px); border-color: var(--vp-c-brand-1); }
.reading-top { display: flex; justify-content: space-between; align-items: center; width: 100%; gap: 12px; margin-bottom: 30px; }
.reading-category { font-size: 12px; border: 1px solid var(--vp-c-border); padding: 3px 9px; border-radius: 99px; }
.reading-number { font: 13px var(--blog-font-mono); color: var(--vp-c-text-3); }
.reading-card h3 { font-size: 22px; line-height: 1.5; margin: 0 0 14px; font-weight: 650; letter-spacing: -.02em; }
.reading-card p { margin: 0 0 26px; color: var(--vp-c-text-2); font-size: 14px; line-height: 1.8; }
.reading-arrow { align-self: end; margin-top: auto; font-size: 22px; }
@media (max-width: 959px) {
  .blog-home { padding: 0 28px; }
  .home-intro { gap: 26px; padding-top: 54px; }
  .reading-feature { padding: 22px; min-height: 310px; }
  .reading-feature h2 { font-size: 32px; }
  .topic-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .reading-grid { gap: 12px; }
  .reading-card { padding: 20px; }
}
@media (max-width: 640px) {
  .blog-home { padding: 0 22px; }
  .home-intro { grid-template-columns: 1fr; gap: 34px; padding: 38px 0 52px; }
  h1 { font-size: clamp(44px, 12vw, 64px); }
  .hello { margin-bottom: 20px; font-size: 15px; }
  .intro-description { font-size: 16px; margin: 22px 0; }
  .reading-feature { justify-self: center; max-width: 100%; width: calc(100% - 18px); min-height: 290px; margin: 0; transform: rotate(2deg); gap: 18px; }
  .reading-feature h2 { font-size: 32px; }
  .section-caption { display: none; }
  .section-heading h2 { font-size: 23px; }
  .topic-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .topic-card { padding: 18px 14px; }
  .topic-card:last-child { grid-column: 1 / -1; }
  .home-topics { padding-bottom: 40px; }
  .reading-grid { grid-template-columns: 1fr; }
  .reading-top { margin-bottom: 20px; }
  .reading-card { padding: 24px; }
  .home-reading { padding-bottom: 48px; }
}
@media (prefers-reduced-motion: reduce) {
  .reading-feature, .reading-card, .topic-card, .browse-link { transition: none; }
  .reading-feature, .reading-feature:hover { transform: none; }
  .reading-card:hover, .topic-card:hover, .browse-link:hover { transform: none; }
}
</style>
