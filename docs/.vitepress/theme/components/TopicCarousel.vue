<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'

const props = defineProps({ items: { type: Array, required: true } })
const current = ref(0)
const visible = ref(3)
const hovered = ref(false)
const focused = ref(false)
const pageCount = computed(() => Math.max(1, props.items.length - visible.value + 1))
let timer
let mobile

function goTo(index) {
  current.value = (index + pageCount.value) % pageCount.value
}

function resize() {
  visible.value = mobile.matches ? 1 : 3
  current.value = Math.min(current.value, pageCount.value - 1)
}

onMounted(() => {
  mobile = window.matchMedia('(max-width: 640px)')
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  resize()
  mobile.addEventListener('change', resize)
  timer = window.setInterval(() => {
    if (!hovered.value && !focused.value && !reducedMotion.matches && !document.hidden) {
      goTo(current.value + 1)
    }
  }, 3500)
})

onUnmounted(() => {
  window.clearInterval(timer)
  mobile?.removeEventListener('change', resize)
})
</script>

<template>
  <section
    class="carousel-wrapper"
    aria-label="技术专题"
    aria-roledescription="轮播"
    @mouseenter="hovered = true"
    @mouseleave="hovered = false"
    @focusin="focused = true"
    @focusout="focused = $event.currentTarget.contains($event.relatedTarget)"
  >
    <div class="carousel-row">
      <button class="carousel-arrow" aria-label="上一组" @click="goTo(current - 1)">‹</button>
      <div class="carousel-viewport">
        <div class="carousel-track" :style="{ transform: `translateX(-${current * 100 / visible}%)` }">
          <a
            v-for="(item, index) in items"
            :key="item.link"
            :href="item.link"
            class="carousel-card"
            :tabindex="index >= current && index < current + visible ? 0 : -1"
            :aria-hidden="index < current || index >= current + visible"
          >
            <span class="card-icon" aria-hidden="true">{{ item.icon }}</span>
            <span class="card-title">{{ item.title }}</span>
            <span class="card-desc">{{ item.description }}</span>
          </a>
        </div>
      </div>
      <button class="carousel-arrow" aria-label="下一组" @click="goTo(current + 1)">›</button>
    </div>
    <div class="carousel-dots" aria-label="选择专题组">
      <button
        v-for="page in pageCount"
        :key="page"
        class="carousel-dot"
        :class="{ active: current === page - 1 }"
        :aria-label="`第${page}组`"
        :aria-current="current === page - 1 ? 'true' : undefined"
        @click="goTo(page - 1)"
      />
    </div>
  </section>
</template>

<style scoped>
.carousel-wrapper { width: 100%; margin: 0 auto; }
.carousel-row { display: flex; align-items: center; gap: 12px; }
.carousel-viewport { flex: 1; min-width: 0; overflow: hidden; }
.carousel-track { display: flex; transition: transform .4s ease; }
.carousel-card { flex: 0 0 calc(100% / 3); min-width: 0; display: flex; flex-direction: column; align-items: flex-start; padding: 22px 24px; border-right: 1px solid var(--vp-c-divider); background: var(--vp-c-bg-soft); text-decoration: none !important; box-sizing: border-box; transition: background .2s; }
.carousel-card:hover { background: var(--vp-c-brand-soft); }
.card-icon { font-size: 23px; margin-bottom: 18px; }
.card-title { font-size: 17px; font-weight: 600; color: var(--vp-c-text-1); margin-bottom: 8px; }
.card-desc { font-size: 14px; color: var(--vp-c-text-2); line-height: 1.8; }
.carousel-arrow { flex-shrink: 0; width: 34px; height: 44px; border: 1px solid var(--vp-c-border); background: var(--vp-c-bg); border-radius: 7px; color: var(--vp-c-text-2); font-size: 22px; display: flex; align-items: center; justify-content: center; cursor: pointer; }
.carousel-arrow:hover { color: var(--vp-c-brand-1); border-color: var(--vp-c-brand-1); }
.carousel-dots { display: flex; justify-content: center; margin-top: 12px; }
.carousel-dot { display: grid; place-items: center; width: 28px; height: 28px; cursor: pointer; }
.carousel-dot::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--vp-c-border); transition: width .2s, background .2s; }
.carousel-dot.active::before { width: 18px; border-radius: 4px; background: var(--vp-c-brand-1); }
@media (max-width: 640px) {
  .carousel-row { gap: 8px; }
  .carousel-card { flex-basis: 100%; padding: 22px; border: 0; border-radius: 8px; }
  .carousel-arrow { width: 30px; }
}
@media (prefers-reduced-motion: reduce) {
  .carousel-track, .carousel-card, .carousel-dot::before { transition: none; }
}
</style>
