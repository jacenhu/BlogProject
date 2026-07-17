import DefaultTheme from 'vitepress/theme'
import { onMounted } from 'vue'
import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,

  // Theme.setup() is called inside VitePressApp's setup() — composition API works here
  setup() {
    onMounted(() => {
      const STORAGE_KEY = 'vp-sidebar-collapsed'

      const btn = document.createElement('button')
      btn.className = 'sidebar-toggle-btn'

      const svgCollapse =
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 4L6 8L10 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
      const svgExpand =
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M6 4L10 8L6 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'

      const hoverZone = document.createElement('div')
      hoverZone.className = 'sidebar-hover-zone'

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

      const saved = localStorage.getItem(STORAGE_KEY)
      applyState(saved === 'true')

      btn.addEventListener('click', () => {
        applyState(!collapsed)
        localStorage.setItem(STORAGE_KEY, String(collapsed))
      })

      document.body.appendChild(hoverZone)
      document.body.appendChild(btn)
    })
  }
}
