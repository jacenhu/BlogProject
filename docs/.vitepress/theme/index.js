import DefaultTheme from 'vitepress/theme'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    if (typeof window === 'undefined') return

    let spotlightEl = null
    let rafId = null
    let mouseHandler = null
    let scrollHandler = null
    let progressEl = null
    let cardHandlers = []
    let revealObserver = null

    const cleanup = () => {
      if (spotlightEl) {
        spotlightEl.remove()
        spotlightEl = null
      }
      if (rafId) {
        cancelAnimationFrame(rafId)
        rafId = null
      }
      if (mouseHandler) {
        document.removeEventListener('mousemove', mouseHandler)
        mouseHandler = null
      }
      if (scrollHandler) {
        window.removeEventListener('scroll', scrollHandler)
        scrollHandler = null
      }
      if (progressEl) {
        progressEl.remove()
        progressEl = null
      }
      cardHandlers.forEach(({ el, move }) => {
        el.removeEventListener('mousemove', move)
      })
      cardHandlers = []
      if (revealObserver) {
        revealObserver.disconnect()
        revealObserver = null
      }
    }

    const init = () => {
      cleanup()

      // Scroll Reveal
      revealObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add('revealed')
              revealObserver.unobserve(entry.target)
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
          revealObserver.observe(el)
        })
      })

      // Global mouse spotlight
      spotlightEl = document.createElement('div')
      spotlightEl.id = 'spotlight'
      spotlightEl.style.cssText = `
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        z-index: 9999;
        opacity: 0;
        transition: opacity 0.5s;
      `
      document.body.appendChild(spotlightEl)

      let mouseX = -1000
      let mouseY = -1000
      let currentX = -1000
      let currentY = -1000

      mouseHandler = (e) => {
        mouseX = e.clientX
        mouseY = e.clientY
        if (spotlightEl) spotlightEl.style.opacity = '1'
      }
      document.addEventListener('mousemove', mouseHandler)

      const animate = () => {
        currentX += (mouseX - currentX) * 0.08
        currentY += (mouseY - currentY) * 0.08
        if (spotlightEl) {
          spotlightEl.style.background = `radial-gradient(600px circle at ${currentX}px ${currentY}px, rgba(99,102,241,0.07), transparent 40%)`
        }
        rafId = requestAnimationFrame(animate)
      }
      rafId = requestAnimationFrame(animate)

      // Card inner spotlight
      document.querySelectorAll('.VPFeatures .box').forEach((card) => {
        const handleMove = (e) => {
          const rect = card.getBoundingClientRect()
          card.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`)
          card.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`)
        }
        card.addEventListener('mousemove', handleMove)
        cardHandlers.push({ el: card, move: handleMove })
      })

      // Reading progress bar
      progressEl = document.createElement('div')
      progressEl.className = 'reading-progress'
      document.body.appendChild(progressEl)

      scrollHandler = () => {
        const scrollTop = window.scrollY
        const docHeight = document.documentElement.scrollHeight - window.innerHeight
        const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0
        if (progressEl) progressEl.style.width = `${progress}%`
      }
      window.addEventListener('scroll', scrollHandler)
      scrollHandler()
    }

    const originalOnAfterRouteChanged = router.onAfterRouteChanged
    router.onAfterRouteChanged = (to) => {
      if (originalOnAfterRouteChanged) originalOnAfterRouteChanged(to)
      setTimeout(init, 50)
    }

    setTimeout(init, 100)
  },
}
