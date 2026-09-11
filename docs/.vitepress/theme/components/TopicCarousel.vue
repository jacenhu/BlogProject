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
/* ── Carousel ── */
.carousel-wrapper {
  max-width: 780px;
  margin: 0 auto;
  padding: 0 1rem 2rem;
}

.carousel-row {
  display: flex;
  align-items: center;
}

.carousel-viewport {
  flex: 1;
  overflow: hidden;
}

/* ── Arrows ── */
.carousel-arrow {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 1px solid var(--vp-c-text-3);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 1.2rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color 0.2s, background 0.2s;
  line-height: 1;
  padding: 0;
  margin: 0 0.15rem;
}

.carousel-arrow:hover {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-soft);
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
  border-color: var(--vp-c-brand-1);
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
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

/* ── Mobile ── */
@media (max-width: 640px) {
  .carousel-card {
    flex: 0 0 100%;
    min-width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .carousel-track { transition: none; }
}
</style>
