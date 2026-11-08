import { useRef, useState } from 'react'

const OBSERVER_SESSION_KEY_PREFIX = 'swarm-observer-session-id'

function createSessionId(storageKey) {
  const cryptoProvider = globalThis.crypto
  let sessionId = cryptoProvider?.randomUUID?.()
  if (!sessionId && cryptoProvider?.getRandomValues) {
    const bytes = cryptoProvider.getRandomValues(new Uint8Array(16))
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
    sessionId = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
  }
  if (!sessionId) throw new Error('This browser cannot create a secure Observer session.')
  if (typeof window !== 'undefined') {
    try {
      window.sessionStorage.setItem(storageKey, sessionId)
    } catch {
      // The session remains usable in memory when browser storage is unavailable.
    }
  }
  return sessionId
}

function getSessionId(token) {
  if (typeof window === 'undefined') return null
  const storageKey = `${OBSERVER_SESSION_KEY_PREFIX}:${token || 'anonymous'}`
  try {
    const existing = window.sessionStorage.getItem(storageKey)
    if (existing) return existing
  } catch {
    return createSessionId(storageKey)
  }
  return createSessionId(storageKey)
}

async function observerRequest(path, token, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.detail || `Observer request failed with status ${response.status}.`)
  }
  return payload
}

export default function useObserver({ token }) {
  const [enabled, setEnabled] = useState(false)
  const [isPending, setIsPending] = useState(false)
  const [isObserving, setIsObserving] = useState(false)
  const storageKey = `${OBSERVER_SESSION_KEY_PREFIX}:${token || 'anonymous'}`
  const [sessionId, setSessionId] = useState(() => {
    try {
      return getSessionId(token)
    } catch {
      return null
    }
  })
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const requestGeneration = useRef(0)

  const toggle = async () => {
    if (isPending) return
    setIsPending(true)
    setError('')
    try {
      if (enabled && sessionId) {
        await observerRequest('/api/observer/session/stop', token, {
          session_id: sessionId,
        })
        requestGeneration.current += 1
        setEnabled(false)
        setResult(null)
      } else {
        const activeSessionId = sessionId || createSessionId(storageKey)
        const payload = await observerRequest('/api/observer/session', token, {
          session_id: activeSessionId,
        })
        if (activeSessionId !== sessionId) {
          setSessionId(activeSessionId)
        }
        setResult({
          user_profile: payload.user_profile || '',
          context_summary: payload.context_summary || '',
        })
        setEnabled(true)
      }
    } catch (requestError) {
      setError(requestError.message || 'Observer mode could not be changed.')
    } finally {
      setIsPending(false)
    }
  }

  const observeTurn = async (query, answer) => {
    if (!enabled || !sessionId) return
    const generation = requestGeneration.current
    setIsObserving(true)
    try {
      const payload = await observerRequest('/api/observer/observe', token, {
        session_id: sessionId,
        query,
        answer,
      })
      if (requestGeneration.current === generation) {
        setResult(payload)
        setError('')
      }
    } catch (requestError) {
      if (requestGeneration.current === generation) {
        setError(requestError.message || 'Observer could not update this conversation.')
      }
    } finally {
      if (requestGeneration.current === generation) setIsObserving(false)
    }
  }

  const reset = () => {
    requestGeneration.current += 1
    if (enabled && sessionId) {
      observerRequest('/api/observer/session/stop', token, { session_id: sessionId })
        .catch((requestError) => {
          setError(requestError.message || 'Observer session could not be stopped.')
        })
    }
    let newSessionId = null
    let resetError = ''
    try {
      newSessionId = createSessionId(storageKey)
    } catch (requestError) {
      resetError = requestError.message || 'Observer session could not be reset.'
    }
    setSessionId(newSessionId)
    setEnabled(false)
    setIsPending(false)
    setIsObserving(false)
    setResult(null)
    setError(resetError)
  }

  return {
    enabled,
    isPending,
    isObserving,
    sessionId,
    result,
    error,
    toggle,
    observeTurn,
    reset,
  }
}
