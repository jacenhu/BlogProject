<script setup>
import BlogIcon from './BlogIcon.vue'

defineProps({ featured: { type: Array, required: true }, topics: { type: Array, required: true } })
</script>

<template>
  <div class="blog-home">
    <section class="home-intro" aria-labelledby="home-title">
      <div class="intro-copy">
        <p class="eyebrow">Jacen / 技术笔记</p>
        <h1 id="home-title">系统工程与<br /><span>模型推理</span></h1>
        <p class="intro-description">记录系统编程、分布式架构与大模型推理的学习和实践。</p>
        <a class="browse-link" href="/program/">浏览技术笔记 <BlogIcon name="right" /></a>
      </div>
      <a class="reading-feature" href="/program/interview/aiinfra_overview.html">
        <div class="feature-top"><span>AI INFRA</span><span>01</span></div>
        <div><p class="feature-kicker">原理解析与工程实践</p><h2>KV Cache<br />与推理系统</h2></div>
        <p class="feature-description">KV Cache · 推理引擎 · 系统设计</p>
        <div class="feature-bottom">阅读专题导读 <BlogIcon name="arrow" /></div>
      </a>
    </section>

    <section class="home-topics" aria-labelledby="topics-title">
      <div class="section-heading"><h2 id="topics-title">技术方向</h2><span class="section-caption">TOPICS</span></div>
      <div class="topic-grid">
        <a v-for="topic in topics" :key="topic.link" :href="topic.link" class="topic-link">
          <BlogIcon :name="topic.icon" class="topic-icon" />
          <h3>{{ topic.title }}</h3>
          <p>{{ topic.description }}</p>
        </a>
      </div>
    </section>

    <section class="home-reading" aria-labelledby="reading-title">
      <div class="section-heading"><h2 id="reading-title">精选文章</h2><a href="/program/">全部笔记 <BlogIcon name="arrow" /></a></div>
      <div class="reading-list">
        <a v-for="(article, index) in featured" :key="article.link" :href="article.link" class="reading-row">
          <span class="reading-number" aria-hidden="true">{{ String(index + 1).padStart(2, '0') }}</span>
          <div class="reading-copy">
            <span class="reading-category">{{ article.category }}</span>
            <h3>{{ article.title }}</h3>
            <p>{{ article.description }}</p>
          </div>
          <BlogIcon name="arrow" class="reading-arrow" />
        </a>
      </div>
    </section>
  </div>
</template>

<style scoped>
.blog-home { max-width: 1180px; padding: 0 40px; margin: 0 auto; }
.home-intro { display: grid; grid-template-columns: 1.4fr 1fr; align-items: center; gap: 88px; padding: 88px 0 80px; }
.eyebrow { margin: 0 0 24px; font-size: 13px; line-height: 1.6; letter-spacing: .06em; color: var(--vp-c-text-2); }
h1 { font-size: clamp(44px, 5.2vw, 64px); line-height: 1.25; letter-spacing: -.035em; font-weight: 650; margin: 0; }
h1 span { color: var(--vp-c-brand-1); }
.intro-description { max-width: 440px; font-size: 16px; line-height: 1.9; color: var(--vp-c-text-2); margin: 24px 0 28px; }
.browse-link { display: inline-flex; align-items: center; gap: 24px; color: var(--vp-c-brand-1); padding: 0 0 8px; border-bottom: 1px solid var(--vp-c-border); font-size: 14px; font-weight: 500; transition: color .2s, border-color .2s; }
.browse-link:hover { color: var(--vp-c-brand-2); border-color: currentColor; }
.browse-link svg, .feature-bottom svg, .section-heading svg { width: 18px; height: 18px; }
.reading-feature { max-width: 348px; width: 100%; justify-self: end; padding: 28px 32px; display: flex; flex-direction: column; gap: 26px; background: #e1f2d7; color: #304834; border: 1px solid transparent; border-radius: 12px; transition: border-color .2s, background .2s; }
.reading-feature:hover { background: #dceecf; border-color: #9ab78d; }
.feature-top { display: flex; justify-content: space-between; align-items: center; font: 11px/1.5 var(--blog-font-mono); letter-spacing: .08em; }
.feature-kicker { font-size: 13px; margin: 0 0 12px; }
.reading-feature h2 { font-size: 32px; font-weight: 600; line-height: 1.4; letter-spacing: -.025em; margin: 0; }
.feature-description { font-size: 12px; line-height: 1.8; margin: 0; }
.feature-bottom { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #bfd8b4; padding-top: 18px; font-size: 13px; margin-top: auto; }
.home-topics { padding-bottom: 64px; }
.section-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 24px; }
.section-heading h2 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: -.02em; line-height: 1.5; }
.section-caption { font: 11px var(--blog-font-mono); letter-spacing: .08em; color: var(--vp-c-text-3); }
.section-heading > a { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--vp-c-text-2); white-space: nowrap; }
.section-heading > a:hover { color: var(--vp-c-brand-1); }
.topic-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 32px; }
.topic-link { padding: 24px 0 0; border-top: 1px solid var(--vp-c-divider); transition: border-color .2s; }
.topic-link:hover { border-color: var(--vp-c-brand-1); }
.topic-icon { display: block; width: 24px; height: 24px; color: var(--vp-c-brand-1); }
.topic-link h3 { font-size: 15px; font-weight: 600; margin: 20px 0 10px; line-height: 1.5; transition: color .2s; }
.topic-link:hover h3 { color: var(--vp-c-brand-1); }
.topic-link p { margin: 0; font-size: 13px; color: var(--vp-c-text-2); line-height: 1.85; }
.home-reading { padding-bottom: 80px; }
.reading-list { border-top: 1px solid var(--vp-c-divider); }
.reading-row { display: grid; grid-template-columns: 48px 1fr 24px; align-items: center; gap: 24px; padding: 28px 0; border-bottom: 1px solid var(--vp-c-divider); }
.reading-number { font: 12px/1.7 var(--blog-font-mono); color: var(--vp-c-text-3); align-self: start; }
.reading-category { font-size: 12px; line-height: 1.7; color: var(--vp-c-brand-1); }
.reading-copy h3 { font-size: 22px; line-height: 1.5; margin: 8px 0; font-weight: 550; letter-spacing: -.015em; transition: color .2s; }
.reading-copy p { margin: 0; color: var(--vp-c-text-2); font-size: 14px; line-height: 1.85; }
.reading-arrow { width: 20px; height: 20px; color: var(--vp-c-text-3); transition: color .2s; }
.reading-row:hover h3, .reading-row:hover .reading-arrow { color: var(--vp-c-brand-1); }
@media (max-width: 959px) {
  .blog-home { padding: 0 28px; }
  .home-intro { gap: 40px; padding-top: 56px; }
  .reading-feature { padding: 24px; }
  .topic-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 28px; }
}
@media (max-width: 640px) {
  .blog-home { padding: 0 24px; }
  .home-intro { grid-template-columns: 1fr; gap: 36px; padding: 40px 0 48px; }
  h1 { font-size: clamp(38px, 10.5vw, 48px); }
  .eyebrow { margin-bottom: 20px; font-size: 12px; }
  .intro-description { font-size: 15px; margin: 20px 0 24px; }
  .reading-feature { justify-self: stretch; max-width: none; gap: 20px; }
  .reading-feature h2 { font-size: 28px; }
  .section-heading h2 { font-size: 20px; }
  .topic-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px 24px; }
  .topic-link:last-child { grid-column: 1 / -1; }
  .home-topics { padding-bottom: 48px; }
  .reading-row { grid-template-columns: 22px 1fr 18px; gap: 12px; padding: 24px 0; }
  .reading-copy h3 { font-size: 18px; }
  .reading-copy p { font-size: 13px; }
  .reading-arrow { width: 18px; height: 18px; }
  .home-reading { padding-bottom: 48px; }
}
@media (prefers-reduced-motion: reduce) {
  .reading-feature, .topic-link, .topic-link h3, .browse-link, .reading-copy h3, .reading-arrow { transition: none; }
}
</style>
