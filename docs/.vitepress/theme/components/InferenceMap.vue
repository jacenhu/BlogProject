<template>
  <figure class="inference-map" aria-labelledby="inference-title">
    <figcaption id="inference-title">一次生成，两个阶段</figcaption>
    <div class="stage-pair">
      <a href="/program/interview/aiinfra_kvcache_deep_dive.html#_3-4-prefill-decode-两阶段-⭐⭐⭐⭐⭐" class="stage">
        <span>Prefill</span><small>处理输入的上下文</small>
      </a>
      <a href="/program/models/sglang_llama_model.html" class="stage decode">
        <span>Decode</span><small>逐步生成新 token</small>
      </a>
    </div>
    <div class="cache-flow" aria-hidden="true">
      <span>写入</span><span>读取并追加</span>
      <svg viewBox="0 0 400 64" preserveAspectRatio="none" fill="none">
        <defs><marker id="cache-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1l5 3-5 3" /></marker></defs>
        <path d="M88 0v24q0 12 12 12h72q12 0 12 12v12" marker-end="url(#cache-arrow)" />
        <path d="M312 4v20q0 12-12 12h-72q-12 0-12 12v12" marker-start="url(#cache-arrow)" marker-end="url(#cache-arrow)" />
      </svg>
    </div>
    <a class="cache-store" href="/program/interview/aiinfra_kvcache_deep_dive.html">
      <div class="cache-heading"><strong>KV Cache</strong><span>保存已计算的状态</span></div>
      <div class="cache-pages" aria-hidden="true">
        <div class="cache-row"><span>K</span><i v-for="n in 6" :key="n"></i></div>
        <div class="cache-row"><span>V</span><i v-for="n in 6" :key="n"></i></div>
      </div>
    </a>
    <p>Prefill 写入缓存；Decode 读取并追加。<a href="/program/interview/aiinfra_kvcache_deep_dive.html">阅读原理解析</a></p>
  </figure>
</template>

<style scoped>
.inference-map { margin: 0; padding: 28px 32px 24px; border-radius: 28px 28px 28px 8px; background: var(--blog-map-bg); }
figcaption { margin-bottom: 24px; font-size: 16px; font-weight: 600; color: var(--vp-c-text-1); }
.stage-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; }
.stage { display: flex; flex-direction: column; gap: 3px; padding: 16px 18px; background: var(--vp-c-bg); border-radius: 12px; }
.stage span { font-size: 22px; font-weight: 600; letter-spacing: -.025em; }
.stage small { font-size: 12px; color: var(--vp-c-text-2); }
.decode { background: var(--blog-sand); }
.stage:hover span, .cache-store:hover strong { text-decoration: underline; text-underline-offset: 5px; }
.cache-flow { position: relative; height: 76px; font-size: 11px; color: var(--vp-c-text-2); }
.cache-flow span { position: absolute; top: 7px; left: calc(22% - 36px); }
.cache-flow span:last-of-type { top: 44px; left: auto; right: 0; }
.cache-flow svg { position: absolute; width: 100%; height: 64px; inset: 0; overflow: visible; }
.cache-flow path { stroke: var(--blog-connector); stroke-width: 1.4; }
.cache-store { display: block; padding: 18px 22px; background: var(--blog-mint); border-radius: 14px; }
.cache-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.cache-heading strong { font-size: 22px; font-weight: 600; letter-spacing: -.025em; }
.cache-heading > span { font-size: 12px; color: var(--vp-c-text-2); }
.cache-pages { margin-top: 14px; display: grid; gap: 6px; }
.cache-row { display: grid; grid-template-columns: 18px repeat(6, 1fr); gap: 6px; align-items: center; }
.cache-row > span { font-size: 12px; color: var(--vp-c-text-2); }
.cache-row i { height: 17px; background: var(--blog-page); border-radius: 3px; }
.cache-row:last-child i { opacity: .6; }
.inference-map > p { margin: 18px 0 0; font-size: 12px; line-height: 1.9; color: var(--vp-c-text-2); }
.inference-map > p a { color: var(--vp-c-brand-1); text-decoration: underline; text-underline-offset: 3px; white-space: nowrap; }
@media (max-width: 640px) {
  .inference-map { padding: 24px 20px; border-radius: 20px; }
  .stage-pair { gap: 24px; }
  .stage { padding: 14px 12px; }
  .stage span { font-size: 20px; }
  .cache-store { padding: 16px; }
  .cache-heading { flex-wrap: wrap; }
}
</style>
