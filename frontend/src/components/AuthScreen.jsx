import { useState } from 'react'

export default function AuthScreen({ onLogin, onRegister }) {
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
