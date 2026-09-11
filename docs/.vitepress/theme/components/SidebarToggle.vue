<script setup>
import { computed, onMounted, onUnmounted, ref, watchEffect } from 'vue'
import { useSidebar } from 'vitepress/theme'

const STORAGE_KEY = 'vp-sidebar-collapsed'
const { isSidebarEnabled } = useSidebar()
const collapsed = ref(false)
const label = computed(() => collapsed.value ? '展开侧边栏' : '收起侧边栏')

function toggle() {
  collapsed.value = !collapsed.value
  try {
    localStorage.setItem(STORAGE_KEY, String(collapsed.value))
  } catch {
    // Storage may be disabled; the toggle still works for this visit.
  }
}

onMounted(() => {
  try {
    collapsed.value = localStorage.getItem(STORAGE_KEY) === 'true'
  } catch {
    // Fall back to the expanded sidebar when storage is unavailable.
  }
  watchEffect(() => {
    document.documentElement.classList.toggle('sidebar-collapsed', isSidebarEnabled.value && collapsed.value)
  })
})

onUnmounted(() => document.documentElement.classList.remove('sidebar-collapsed'))
</script>

<template>
  <template v-if="isSidebarEnabled">
    <div v-if="collapsed" class="sidebar-hover-zone" aria-hidden="true" />
    <button
      class="sidebar-toggle-btn"
      :title="label"
      :aria-label="label"
      :aria-expanded="!collapsed"
      @click="toggle"
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path :d="collapsed ? 'M6 4L10 8L6 12' : 'M10 4L6 8L10 12'" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </button>
  </template>
</template>

<style>
/* ========== 侧边栏折叠按钮 ========== */
.sidebar-toggle-btn {
  position: fixed;
  left: calc(var(--vp-sidebar-width, 272px) - 1px);
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 100;
  width: 26px;
  height: 48px;
  border-radius: 0 12px 12px 0;
  border: 1px solid var(--vp-c-divider);
  border-left: none;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  box-shadow: 1px 0 4px rgba(0, 0, 0, 0.06);
  transition: left 0.3s cubic-bezier(0.25, 0.8, 0.25, 1.2),
              color 0.2s ease,
              background 0.2s ease,
              box-shadow 0.2s ease,
              opacity 0.2s ease;
  opacity: 0.25;
}

/* 悬停时完全显示 */
.sidebar-toggle-btn:hover {
  opacity: 1;
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-divider);
  box-shadow: 1px 0 8px rgba(0, 113, 227, 0.15);
}

/* 左侧边缘悬浮热区：鼠标靠近左边缘时显示按钮（折叠后使用） */
.sidebar-hover-zone {
  position: fixed;
  left: 0;
  top: var(--vp-nav-height, 64px);
  bottom: 0;
  width: 20px;
  z-index: 99;
}

.sidebar-hover-zone:hover ~ .sidebar-toggle-btn {
  opacity: 1;
}

/* 移动端隐藏折叠按钮（VitePress 自带 hamburger） */
@media (max-width: 959px) {
  .sidebar-toggle-btn,
  .sidebar-hover-zone {
    display: none !important;
  }
}

/* 折叠状态下按钮贴在左边缘 */
.sidebar-collapsed .sidebar-toggle-btn {
  left: 0;
  opacity: 1;
  transform: translate(0, -50%);
  border-radius: 0 12px 12px 0;
  border-left: none;
}

/* ========== 侧边栏折叠状态 ========== */
.sidebar-collapsed .VPSidebar {
  width: 0 !important;
  min-width: 0 !important;
  padding: 0 !important;
  overflow: hidden;
  border-right: none;
}

.sidebar-collapsed .VPSidebar > * {
  opacity: 0;
  pointer-events: none;
}

@media (min-width: 960px) {
  .sidebar-collapsed .VPContent {
    padding-left: 0 !important;
  }
}

</style>
