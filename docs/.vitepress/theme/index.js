import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import SidebarToggle from './components/SidebarToggle.vue'
import BlogHome from './components/BlogHome.vue'
import TopicCarousel from './components/TopicCarousel.vue'
import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  Layout: () => h(DefaultTheme.Layout, null, {
    'layout-bottom': () => h(SidebarToggle)
  }),
  enhanceApp({ app }) {
    app.component('TopicCarousel', TopicCarousel)
    app.component('BlogHome', BlogHome)
  }
}
