import DefaultTheme from 'vitepress/theme'
import { onMounted } from 'vue'
import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  setup() {
    onMounted(() => {
      const STORAGE_KEY = 'vp-sidebar-collapsed'

      // --- Create toggle button ---
      const btn = document.createElement('button')
      btn.className = 'sidebar-toggle-btn'
      btn.title = '收起侧边栏'

      const svgCollapse =
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 4L6 8L10 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
      const svgExpand =
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M6 4L10 8L6 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'

      // --- Create hover zone ---
      const hoverZone = document.createElement('div')
      hoverZone.className = 'sidebar-hover-zone'

      // --- State ---
      let collapsed = false

      function applyState(val) {
        collapsed = val
        if (val) {
          document.documentElement.classList.add('sidebar-collapsed')
          btn.classList.add('collapsed')
          btn.title = '展开侧边栏'
          btn.innerHTML = svgExpand
        } else {
          document.documentElement.classList.remove('sidebar-collapsed')
          btn.classList.remove('collapsed')
          btn.title = '收起侧边栏'
          btn.innerHTML = svgCollapse
        }
      }

      // Restore saved state
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved === 'true') {
        applyState(true)
      } else {
        // Ensure fresh render has correct icon
        btn.innerHTML = svgCollapse
      }

      // --- Toggle on click ---
      btn.addEventListener('click', () => {
        applyState(!collapsed)
        localStorage.setItem(STORAGE_KEY, String(collapsed))
      })

      // --- Inject into DOM ---
      document.body.appendChild(hoverZone)
      document.body.appendChild(btn)
    })
  }
}
