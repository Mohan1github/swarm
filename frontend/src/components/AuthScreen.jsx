import { useEffect, useRef, useState } from 'react'

let googleIdentityServicesPromise

function loadGoogleIdentityServices() {
  if (globalThis.google?.accounts?.id) return Promise.resolve(globalThis.google)
  if (googleIdentityServicesPromise) return googleIdentityServicesPromise

  googleIdentityServicesPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.defer = true
    script.onload = () => {
      if (globalThis.google?.accounts?.id) resolve(globalThis.google)
      else reject(new Error('Google sign-in could not be initialized.'))
    }
    script.onerror = () => reject(new Error('Google sign-in could not be loaded.'))
    document.head.appendChild(script)
  }).catch((error) => {
    googleIdentityServicesPromise = null
    throw error
  })

  return googleIdentityServicesPromise
}

function GoogleSignInButton({ onSignIn, onError, isRegistering }) {
  const buttonRef = useRef(null)
  const signInRef = useRef(onSignIn)
  const errorRef = useRef(onError)
  const [view, setView] = useState({
    mode: null,
    status: 'Loading Google sign-in…',
    configured: false,
  })

  useEffect(() => {
    signInRef.current = onSignIn
    errorRef.current = onError
  }, [onError, onSignIn])

  useEffect(() => {
    let cancelled = false

    const initialize = async () => {
      const configResponse = await fetch('/api/auth/google/config')
      const config = await configResponse.json().catch(() => ({}))
      if (!configResponse.ok) {
        throw new Error(config.detail || 'Google sign-in configuration could not be loaded.')
      }
      if (!config.enabled) {
        if (!cancelled) {
          setView({
            mode: isRegistering,
            status: 'Google sign-in is not configured yet.',
            configured: false,
          })
        }
        return
      }

      const google = await loadGoogleIdentityServices()
      if (cancelled || !buttonRef.current) return

      buttonRef.current.replaceChildren()
      google.accounts.id.initialize({
        client_id: config.client_id,
        nonce: config.nonce,
        auto_select: false,
        callback: (response) => {
          signInRef.current(response.credential, config.nonce)
            .catch((error) => errorRef.current(error.message || 'Google sign-in failed.'))
        },
      })
      google.accounts.id.renderButton(buttonRef.current, {
        theme: 'outline',
        size: 'large',
        shape: 'pill',
        text: isRegistering ? 'signup_with' : 'continue_with',
        width: Math.min(360, Math.max(240, Math.floor(window.innerWidth - 80))),
      })
      setView({ mode: isRegistering, status: '', configured: true })
    }

    initialize().catch((error) => {
      if (!cancelled) {
        setView({
          mode: isRegistering,
          status: error.message || 'Google sign-in is unavailable.',
          configured: false,
        })
      }
    })

    return () => {
      cancelled = true
    }
  }, [isRegistering])

  const isCurrentMode = view.mode === isRegistering
  const status = isCurrentMode ? view.status : 'Loading Google sign-in…'
  const isConfigured = isCurrentMode && view.configured

  return (
    <div className="google-auth">
      <div
        ref={buttonRef}
        className="google-auth-button"
        aria-label={isRegistering ? 'Sign up with Google' : 'Continue with Google'}
        hidden={!isConfigured}
      />
      {!isConfigured ? (
        <button className="google-auth-fallback" type="button" disabled>
          {isRegistering ? 'Sign up with Google' : 'Continue with Google'}
        </button>
      ) : null}
      {status ? <p className="google-auth-status" role="status">{status}</p> : null}
    </div>
  )
}

export default function AuthScreen({ onLogin, onRegister, onGoogleSignIn }) {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const isRegistering = mode === 'register'

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setIsSubmitting(true)
    try {
      await (isRegistering ? onRegister(email.trim(), password) : onLogin(email.trim(), password))
    } catch (requestError) {
      setError(requestError.message || 'Authentication failed. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="auth-screen">
      <section className="auth-panel" aria-labelledby="auth-title">
        <div className="auth-brand">
          <div className="brand-mark">AI</div>
          <div>
            <span className="eyebrow">Research engine</span>
            <strong>Swarm AI</strong>
          </div>
        </div>

        <div className="auth-heading">
          <span className="auth-kicker">Secure workspace</span>
          <h1 id="auth-title">{isRegistering ? 'Create your account' : 'Welcome back'}</h1>
          <p>{isRegistering ? 'Create an account to start your workspace.' : 'Sign in to continue to your workspace.'}</p>
        </div>

        <div className="auth-tabs" role="tablist" aria-label="Account access">
          <button
            className={mode === 'login' ? 'selected' : ''}
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            onClick={() => { setMode('login'); setError('') }}
          >
            Sign in
          </button>
          <button
            className={mode === 'register' ? 'selected' : ''}
            type="button"
            role="tab"
            aria-selected={mode === 'register'}
            onClick={() => { setMode('register'); setError('') }}
          >
            Create account
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <GoogleSignInButton
            onSignIn={onGoogleSignIn}
            onError={setError}
            isRegistering={isRegistering}
          />

          <div className="auth-divider">
            <span>{isRegistering ? 'or sign up with email' : 'or continue with email'}</span>
          </div>

          <label htmlFor="auth-email">Email address</label>
          <input
            id="auth-email"
            type="email"
            autoComplete="email"
            required
            maxLength={254}
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@company.com"
          />

          <label htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            type="password"
            autoComplete={isRegistering ? 'new-password' : 'current-password'}
            required
            minLength={isRegistering ? 10 : 1}
            maxLength={128}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder={isRegistering ? 'At least 10 characters' : 'Enter your password'}
          />

          {error ? <div className="auth-error" role="alert">{error}</div> : null}

          <button
            className={`auth-submit ${isRegistering ? 'auth-submit-register' : ''}`}
            type="submit"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Please wait...' : isRegistering ? 'Create account' : 'Sign in'}
          </button>
        </form>

        <p className="auth-footnote">Your account secures access to your AI workspace.</p>
      </section>
    </main>
  )
}
