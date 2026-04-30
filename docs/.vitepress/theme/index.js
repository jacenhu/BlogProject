import DefaultTheme from 'vitepress/theme'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    if (typeof window === 'undefined') return

    const initReveal = () => {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add('revealed')
              observer.unobserve(entry.target)
            }
          })
        },
        { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
      )

      const selectors = [
        '.vp-doc h2',
        '.vp-doc h3',
        '.vp-doc p',
        '.vp-doc ul',
        '.vp-doc ol',
        '.vp-doc blockquote',
        '.vp-doc table',
        '.VPFeatures .box',
        '.vp-doc .custom-block',
        '.vp-doc div[class*="language-"]',
      ]

      selectors.forEach((selector) => {
        document.querySelectorAll(selector).forEach((el) => {
          el.classList.add('reveal')
          observer.observe(el)
        })
      })
    }

    const setupReveal = () => {
      document.querySelectorAll('.revealed').forEach((el) => {
        el.classList.remove('revealed')
      })
      setTimeout(initReveal, 50)
    }

    const originalOnAfterRouteChanged = router.onAfterRouteChanged
    router.onAfterRouteChanged = (to) => {
      if (originalOnAfterRouteChanged) originalOnAfterRouteChanged(to)
      setupReveal()
    }

    setTimeout(initReveal, 100)
  },
}
