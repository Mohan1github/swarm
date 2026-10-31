import { useEffect, useRef, useState } from 'react'
import serializeAttachment from '../utils/serializeAttachment'

const PROACTIVE_IDLE_DELAY_MS = 45_000
const MAX_PROACTIVE_TURNS_PER_SESSION = 2

async function consumeAssistantStream(response, onEvent) {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail || 'The assistant could not complete that response.')
  }
  if (!response.body) throw new Error('The server did not return a response stream.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = false

  const processFrame = (frame) => {
    const data = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
    if (!data) return

    const event = JSON.parse(data)
    if (event.type === 'error') throw new Error(event.message || 'The assistant could not complete that response.')
    if (event.type === 'complete') completed = true
    onEvent(event)
  }

  while (true) {
    const { value, done } = await reader.read()
    if (value) buffer += decoder.decode(value, { stream: !done })

    let boundary = buffer.match(/\r?\n\r?\n/)
    while (boundary) {
      processFrame(buffer.slice(0, boundary.index))
      buffer = buffer.slice(boundary.index + boundary[0].length)
      boundary = buffer.match(/\r?\n\r?\n/)
    }

    if (done) {
      if (buffer.trim()) processFrame(buffer)
      break
    }
  }

  if (!completed) throw new Error('The assistant stream ended before the response was complete.')
}

function appendStreamText(setMessages, text) {
  setMessages((previous) => {
    const last = previous[previous.length - 1]
    if (last?.role !== 'assistant' || !last.streaming) return previous
    return [...previous.slice(0, -1), { ...last, content: last.content + text }]
  })
}

function completeStreamedMessage(setMessages) {
  setMessages((previous) => {
    const last = previous[previous.length - 1]
    if (last?.role !== 'assistant') return previous
    const { streamingType: _streamingType, ...message } = last
    return [...previous.slice(0, -1), { ...message, streaming: false, instant: true }]
  })
}

function removeStreamingMessage(setMessages, streamingType) {
  setMessages((previous) => previous.filter((message) => message.streamingType !== streamingType))
}

export default function useAssistantConversation({ token, active }) {
  const [messages, setMessages] = useState([])
  const [sessionLoaded, setSessionLoaded] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const [memoryTurnCount, setMemoryTurnCount] = useState(0)
  const [isProactive, setIsProactive] = useState(false)
  const [proactiveEnabled, setProactiveEnabled] = useState(true)
  const [proactiveCount, setProactiveCount] = useState(0)
  const proactiveController = useRef(null)
  const isLoading = active && !sessionLoaded && !error

  useEffect(() => {
    if (!active || sessionLoaded) return undefined

    const controller = new AbortController()

    fetch('/api/assistant/session/stream', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    })
      .then((response) => consumeAssistantStream(response, (event) => {
        if (event.type === 'session') {
          setError('')
          setMessages([
            ...(event.messages || []),
            { role: 'assistant', content: '', streaming: true, streamingType: 'opening' },
          ])
          setMemoryTurnCount(event.memory_turn_count || 0)
        } else if (event.type === 'delta') {
          appendStreamText(setMessages, event.text || '')
        } else if (event.type === 'complete') {
          completeStreamedMessage(setMessages)
          setMemoryTurnCount(event.memory_turn_count || 0)
        }
      }))
      .catch((sessionError) => {
        if (sessionError.name !== 'AbortError') {
          setError(sessionError.message)
          removeStreamingMessage(setMessages, 'opening')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setSessionLoaded(true)
      })

    return () => controller.abort()
  }, [token, active, sessionLoaded])

  useEffect(() => {
    if (
      !active
      || !sessionLoaded
      || isLoading
      || isSending
      || error
      || !proactiveEnabled
      || proactiveCount >= MAX_PROACTIVE_TURNS_PER_SESSION
    ) return undefined

    let controller
    const timeoutId = window.setTimeout(async () => {
      controller = new AbortController()
      proactiveController.current = controller
      setIsProactive(true)
      setMessages((previous) => [
        ...previous,
        { role: 'assistant', content: '', streaming: true, streamingType: 'proactive' },
      ])

      try {
        const response = await fetch('/api/assistant/proactive/stream', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        })
        await consumeAssistantStream(response, (event) => {
          if (event.type === 'delta') {
            appendStreamText(setMessages, event.text || '')
          } else if (event.type === 'complete') {
            completeStreamedMessage(setMessages)
            setMemoryTurnCount(event.memory_turn_count || 0)
            setProactiveCount((count) => count + 1)
          }
        })
      } catch (proactiveError) {
        removeStreamingMessage(setMessages, 'proactive')
        if (proactiveError.name !== 'AbortError') {
          setError(proactiveError.message || 'The assistant could not generate a proactive message.')
        }
      } finally {
        if (proactiveController.current === controller) proactiveController.current = null
        setIsProactive(false)
      }
    }, PROACTIVE_IDLE_DELAY_MS)

    return () => {
      window.clearTimeout(timeoutId)
      if (controller && !controller.signal.aborted) controller.abort()
    }
  }, [token, active, sessionLoaded, isLoading, isSending, error, proactiveEnabled, proactiveCount])

  const reopenSession = () => {
    proactiveController.current?.abort()
    setError('')
    setProactiveCount(0)
    setSessionLoaded(false)
  }

  const sendMessage = async (message, attachments = []) => {
    const trimmed = message.trim()
    if (!trimmed || isSending) return

    proactiveController.current?.abort()
    setIsSending(true)
    setError('')
    setMessages((previous) => [
      ...previous,
      { role: 'user', content: trimmed },
      { role: 'assistant', content: '', streaming: true, streamingType: 'reply' },
    ])

    try {
      const serializedAttachments = await Promise.all(
        attachments.map(({ file }) => serializeAttachment(file)),
      )
      const response = await fetch('/api/assistant/turn/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: trimmed,
          attachments: serializedAttachments.map(({ name, media_type, content }) => ({
            name,
            media_type,
            content,
          })),
        }),
      })
      await consumeAssistantStream(response, (event) => {
        if (event.type === 'delta') {
          appendStreamText(setMessages, event.text || '')
        } else if (event.type === 'complete') {
          completeStreamedMessage(setMessages)
          setMemoryTurnCount(event.memory_turn_count || 0)
        }
      })
    } catch (sendError) {
      removeStreamingMessage(setMessages, 'reply')
      setError(sendError.message || 'The assistant could not complete that reply.')
    } finally {
      setIsSending(false)
    }
  }

  const clearMemory = async () => {
    proactiveController.current?.abort()
    setError('')
    try {
      const response = await fetch('/api/assistant/session', {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.detail || 'Could not clear assistant memory.')
      setMessages([{ role: 'assistant', content: 'Memory cleared. What would you like to work on?' }])
      setMemoryTurnCount(0)
      setProactiveCount(0)
    } catch (clearError) {
      setError(clearError.message || 'Could not clear assistant memory.')
    }
  }

  const clearLocalConversation = () => {
    proactiveController.current?.abort()
    setMessages([{ role: 'assistant', content: 'History cleared. What would you like to work on?' }])
    setSessionLoaded(true)
    setIsSending(false)
    setError('')
    setMemoryTurnCount(0)
    setProactiveCount(0)
  }

  const toggleProactive = () => {
    setProactiveEnabled((enabled) => !enabled)
  }

  return {
    messages,
    isLoading,
    isSending,
    isProactive,
    proactiveEnabled,
    toggleProactive,
    error,
    memoryTurnCount,
    sendMessage,
    clearMemory,
    clearLocalConversation,
    reopenSession,
  }
}