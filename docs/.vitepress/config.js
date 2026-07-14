import { defineConfig } from 'vitepress'
import { createRequire } from 'module'

const require = createRequire(import.meta.url)
const sidebar = require('./sidebar.json')
const nav = require('./nav.json')

export default defineConfig({
  title: "Jacen's Blog",
  description: 'Jacen 的技术博客，记录计算机相关的实践笔记',
  base: '/',
  lang: 'zh-CN',
  lastUpdated: true,
  ignoreDeadLinks: 'localhostLinks',
  markdown: {
    lineNumbers: true
  },
  head: [
    ['meta', { charset: 'utf-8' }],
    ['meta', { name: 'viewport', content: 'width=device-width,initial-scale=1' }],
    ['meta', { name: 'author', content: 'Jacen Hu' }],
    ['link', { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
    ['link', { rel: 'apple-touch-icon', href: '/favicon.ico' }],
    ['link', { rel: 'dns-prefetch', href: 'https://github.com' }],
    ['link', { rel: 'preconnect', href: 'https://github.com' }]
  ],
  themeConfig: {
    nav,
    sidebar,
    outlineTitle: '本页目录',
    lastUpdatedText: '更新时间',
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/jacenhu' }
    ],
    footer: {
      copyright: '© 2026 Jacen Hu'
    }
  }
})
