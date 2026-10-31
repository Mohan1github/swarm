import { useEffect, useState } from 'react'
import AuthContext from './context'
const TOKEN_KEY = 'swarm-ai-access-token'

async function readResponse(response) {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = payload.detail
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(' ')
      : detail || 'The request could not be completed.'
    throw new Error(message)
  }
  return payload
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY))
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(() => Boolean(sessionStorage.getItem(TOKEN_KEY)))

  useEffect(() => {
    if (!token) return undefined
    const controller = new AbortController()

    fetch('/api/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    })
      .then(readResponse)
      .then((currentUser) => setUser(currentUser))
      .catch((error) => {
        if (error.name === 'AbortError') return
        sessionStorage.removeItem(TOKEN_KEY)
        setToken(null)
        setUser(null)
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false)
      })

    return () => controller.abort()
  }, [token])

  const authenticate = async (endpoint, email, password) => {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    const payload = await readResponse(response)
    sessionStorage.setItem(TOKEN_KEY, payload.access_token)
    setToken(payload.access_token)
    setUser(payload.user)
    setIsLoading(false)
  }

  const login = (email, password) => authenticate('/api/auth/login', email, password)
  const register = (email, password) => authenticate('/api/auth/register', email, password)

  const logout = async () => {
    const currentToken = token
    try {
      if (currentToken) {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: { Authorization: `Bearer ${currentToken}` },
        })
      }
    } catch {
      // Clear local credentials even when the API is unreachable.
    } finally {
      sessionStorage.removeItem(TOKEN_KEY)
      setToken(null)
      setUser(null)
      setIsLoading(false)
    }
  }

  return (
    <AuthContext.Provider value={{ token, user, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}
