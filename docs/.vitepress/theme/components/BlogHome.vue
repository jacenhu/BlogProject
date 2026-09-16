<script setup>
import BlogIcon from './BlogIcon.vue'
import InferenceMap from './InferenceMap.vue'

defineProps({ featured: { type: Array, required: true }, topics: { type: Array, required: true } })
</script>

<template>
  <div class="blog-home">
    <section class="home-intro" aria-labelledby="home-title">
      <div class="intro-copy">
        <h1 id="home-title">系统工程与<br />模型推理</h1>
        <p class="intro-description">从数据结构到分布式系统，从模型实现到推理引擎。这里记录原理、源码与工程实践。</p>
        <a class="browse-link" href="/program/">浏览全部笔记</a>
        <p class="intro-author">Jacen Hu 的技术博客</p>
      </div>
      <InferenceMap />
    </section>

    <div class="home-library">
      <section class="home-reading" aria-labelledby="reading-title">
        <div class="section-heading"><h2 id="reading-title">精选文章</h2><a href="/program/">查看目录</a></div>
        <div class="reading-list">
          <article v-for="article in featured" :key="article.link" class="reading-entry">
            <span class="reading-category">{{ article.category }}</span>
            <h3><a :href="article.link">{{ article.title }}</a></h3>
            <p>{{ article.description }}</p>
          </article>
        </div>
      </section>

      <nav class="home-topics" aria-labelledby="topics-title">
        <h2 id="topics-title">按技术方向阅读</h2>
        <a v-for="topic in topics" :key="topic.link" :href="topic.link" class="topic-link">
          <BlogIcon :name="topic.icon" class="topic-icon" />
          <div><h3>{{ topic.title }}</h3><p>{{ topic.description }}</p></div>
        </a>
      </nav>
    </div>
  </div>
</template>

<style scoped>
.blog-home { max-width: 1200px; margin: 0 auto; padding: 0 48px; }
.home-intro { display: grid; grid-template-columns: 1fr 1.1fr; gap: 72px; align-items: center; padding: 64px 0 72px; }
h1 { margin: 0; font-size: clamp(40px, 4.7vw, 58px); line-height: 1.3; font-weight: 600; letter-spacing: -.04em; }
.intro-description { max-width: 390px; margin: 24px 0 28px; font-size: 16px; line-height: 1.95; color: var(--vp-c-text-2); }
.browse-link { display: inline-block; border-radius: 8px; padding: 11px 20px; background: var(--vp-c-brand-1); color: var(--vp-c-bg); font-size: 14px; font-weight: 500; }
.browse-link:hover { background: var(--vp-c-brand-2); }
.intro-author { margin: 22px 0 0; font-size: 12px; color: var(--vp-c-text-3); }
.home-library { display: grid; grid-template-columns: minmax(0, 1fr) 290px; gap: 76px; padding: 0 0 80px; }
.section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 8px; }
.section-heading h2, .home-topics > h2 { font-size: 21px; font-weight: 600; line-height: 1.5; margin: 0; letter-spacing: -.025em; }
.section-heading > a { color: var(--vp-c-brand-1); font-size: 13px; text-underline-offset: 4px; white-space: nowrap; }
.section-heading > a:hover { text-decoration: underline; }
.reading-entry { padding: 30px 0; }
.reading-entry + .reading-entry { border-top: 1px solid var(--vp-c-divider); }
.reading-category { display: inline-block; font-size: 12px; color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); border-radius: 4px; padding: 2px 8px; }
.reading-entry h3 { margin: 12px 0 10px; font-size: 25px; line-height: 1.5; font-weight: 550; letter-spacing: -.02em; }
.reading-entry h3 a:hover { color: var(--vp-c-brand-1); text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 5px; }
.reading-entry > p { max-width: 60ch; margin: 0; font-size: 15px; color: var(--vp-c-text-2); line-height: 1.9; }
.home-topics { align-self: start; padding: 26px; background: var(--vp-c-bg-alt); border-radius: 16px; }
.home-topics > h2 { font-size: 17px; margin-bottom: 24px; }
.topic-link { display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 14px; align-items: start; }
.topic-link + .topic-link { margin-top: 25px; }
.topic-icon { width: 21px; height: 21px; margin-top: 2px; color: var(--vp-c-brand-1); }
.topic-link h3 { margin: 0 0 5px; font-size: 14px; line-height: 1.6; font-weight: 600; }
.topic-link p { margin: 0; font-size: 12px; color: var(--vp-c-text-2); line-height: 1.8; }
.topic-link:hover h3 { color: var(--vp-c-brand-1); text-decoration: underline; text-underline-offset: 4px; }
@media (max-width: 959px) {
  .blog-home { padding: 0 32px; }
  .home-intro { gap: 32px; grid-template-columns: .9fr 1.1fr; padding-top: 44px; }
  .home-library { grid-template-columns: minmax(0, 1fr) 250px; gap: 40px; }
  .home-topics { padding: 22px; }
}
@media (max-width: 740px) {
  .home-intro { grid-template-columns: 1fr; gap: 32px; }
  .intro-description { max-width: 520px; }
  .home-library { grid-template-columns: 1fr; gap: 32px; }
  .home-topics { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .home-topics > h2 { grid-column: 1 / -1; margin: 0; }
  .topic-link + .topic-link { margin-top: 0; }
}
@media (max-width: 480px) {
  .blog-home { padding: 0 22px; }
  .home-intro { padding: 36px 0 44px; }
  h1 { font-size: 40px; }
  .intro-description { font-size: 15px; margin: 20px 0 24px; }
  .intro-author { margin-top: 16px; }
  .reading-entry { padding: 24px 0; }
  .reading-entry h3 { font-size: 21px; }
  .reading-entry > p { font-size: 14px; }
  .home-library { padding-bottom: 48px; }
  .home-topics { grid-template-columns: 1fr; }
}
</style>
