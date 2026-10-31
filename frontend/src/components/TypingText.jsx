import { useEffect, useRef, useState } from 'react'

export default function TypingText({ text, speed = 700, onComplete, completionValue, onProgress }) {
  const [visible, setVisible] = useState('')
  const lastProgressStep = useRef(-1)

  useEffect(() => {
    lastProgressStep.current = -1
    if (!text) {
      onProgress?.(1)
      onComplete?.(completionValue)
      return
    }

    let frameId
    let startedAt
    let visibleLength = 0
    const revealFrame = (timestamp) => {
      if (startedAt === undefined) startedAt = timestamp
      const nextLength = Math.min(text.length, Math.floor(((timestamp - startedAt) * speed) / 1000) + 1)
      if (nextLength !== visibleLength) {
        visibleLength = nextLength
        setVisible(text.slice(0, visibleLength))
        const progress = visibleLength / text.length
        const progressStep = Math.floor(progress * 12)
        if (progressStep !== lastProgressStep.current) {
          lastProgressStep.current = progressStep
          onProgress?.(progress)
        }
      }
      if (visibleLength >= text.length) {
        onProgress?.(1)
        onComplete?.(completionValue)
        return
      }
      frameId = requestAnimationFrame(revealFrame)
    }

    frameId = requestAnimationFrame(revealFrame)

    return () => cancelAnimationFrame(frameId)
  }, [text, speed, onComplete, completionValue, onProgress])

  return <>{visible}</>
}
