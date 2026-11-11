import { useEffect, useRef } from 'react'

const PARTICLE_COUNT = 54
const CONNECTION_DISTANCE = 128

export default function LandingParticles() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const page = canvas?.parentElement
    const context = canvas?.getContext('2d')
    if (!canvas || !page || !context) return undefined

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const particles = []
    let width = 0
    let height = 0
    let animationFrame = 0

    const resizeCanvas = () => {
      const bounds = page.getBoundingClientRect()
      width = window.innerWidth
      height = window.innerHeight
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.round(width * pixelRatio)
      canvas.height = Math.round(height * pixelRatio)
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
      particles.length = 0

      const count = Math.min(PARTICLE_COUNT, Math.max(24, Math.round(width / 24)))
      for (let index = 0; index < count; index += 1) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          radius: Math.random() * 1.4 + 0.45,
          opacity: Math.random() * 0.34 + 0.12,
          speedX: (Math.random() - 0.5) * 0.16,
          speedY: (Math.random() - 0.5) * 0.16,
        })
      }

      if (bounds.height <= height) drawParticles(false)
    }

    const drawParticles = (animate) => {
      context.clearRect(0, 0, width, height)
      for (let index = 0; index < particles.length; index += 1) {
        const particle = particles[index]
        if (animate) {
          particle.x += particle.speedX
          particle.y += particle.speedY
          if (particle.x < -5) particle.x = width + 5
          if (particle.x > width + 5) particle.x = -5
          if (particle.y < -5) particle.y = height + 5
          if (particle.y > height + 5) particle.y = -5
        }

        for (let otherIndex = index + 1; otherIndex < particles.length; otherIndex += 1) {
          const other = particles[otherIndex]
          const dx = particle.x - other.x
          const dy = particle.y - other.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          if (distance < CONNECTION_DISTANCE) {
            context.beginPath()
            context.moveTo(particle.x, particle.y)
            context.lineTo(other.x, other.y)
            context.strokeStyle = `rgba(166, 192, 170, ${(1 - distance / CONNECTION_DISTANCE) * 0.09})`
            context.lineWidth = 0.65
            context.stroke()
          }
        }

        context.beginPath()
        context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
        context.fillStyle = `rgba(195, 216, 198, ${particle.opacity})`
        context.fill()
      }

      if (animate && !reducedMotion.matches) {
        animationFrame = window.requestAnimationFrame(() => drawParticles(true))
      }
    }

    const startAnimation = () => {
      window.cancelAnimationFrame(animationFrame)
      if (reducedMotion.matches) {
        drawParticles(false)
      } else {
        animationFrame = window.requestAnimationFrame(() => drawParticles(true))
      }
    }

    resizeCanvas()
    startAnimation()
    window.addEventListener('resize', resizeCanvas)
    reducedMotion.addEventListener('change', startAnimation)

    return () => {
      window.cancelAnimationFrame(animationFrame)
      window.removeEventListener('resize', resizeCanvas)
      reducedMotion.removeEventListener('change', startAnimation)
    }
  }, [])

  return <canvas ref={canvasRef} className="landing-particle-canvas" aria-hidden="true" />
}
