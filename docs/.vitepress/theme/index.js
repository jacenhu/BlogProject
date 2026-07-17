import DefaultTheme from 'vitepress/theme'
import { h, ref, onMounted, watch, Fragment } from 'vue'
import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  Layout: () => {
    const DefaultLayout = DefaultTheme.Layout
    return {
      setup() {
        const sidebarCollapsed = ref(false)

        // Apply / sync state to document root
        function applyState(val) {
          if (val) {
            document.documentElement.classList.add('sidebar-collapsed')
          } else {
            document.documentElement.classList.remove('sidebar-collapsed')
          }
        }

        onMounted(() => {
          const saved = localStorage.getItem('vp-sidebar-collapsed')
          if (saved !== null) {
            sidebarCollapsed.value = saved === 'true'
          }
          applyState(sidebarCollapsed.value)
        })

        watch(sidebarCollapsed, (val) => {
          localStorage.setItem('vp-sidebar-collapsed', String(val))
          applyState(val)
        })

        function toggleSidebar() {
          sidebarCollapsed.value = !sidebarCollapsed.value
        }

        return () => {
          const collapsed = sidebarCollapsed.value
          return h(Fragment, null, [
            h(DefaultLayout),
            h('div', { class: 'sidebar-hover-zone' }),
            h('button', {
              class: 'sidebar-toggle-btn' + (collapsed ? ' collapsed' : ''),
              onClick: toggleSidebar,
              title: collapsed ? '展开侧边栏' : '收起侧边栏',
              innerHTML: collapsed
                ? '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M6 4L10 8L6 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                : '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 4L6 8L10 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            })
          ])
        }
      }
    }
  }
}
