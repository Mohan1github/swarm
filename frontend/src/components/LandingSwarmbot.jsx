import { useState } from 'react'

export default function LandingSwarmbot() {
  const [showInstallStatus, setShowInstallStatus] = useState(false)

  return (
    <section className="landing-swarmbot" id="swarmbot" aria-labelledby="swarmbot-title" data-landing-reveal>
      <div className="swarmbot-visual" aria-hidden="true">
        <div className="swarmbot-orbit swarmbot-orbit-outer" />
        <div className="swarmbot-orbit swarmbot-orbit-inner" />
        <div className="swarmbot-orbit-dot swarmbot-orbit-dot-one" />
        <div className="swarmbot-orbit-dot swarmbot-orbit-dot-two" />
        <div className="swarmbot-core">
          <svg viewBox="0 0 96 96">
            <path d="M48 13v8M40 14h16" />
            <rect x="20" y="25" width="56" height="48" rx="18" />
            <path d="M33 44c3-4 8-4 11 0m8 0c3-4 8-4 11 0" />
            <path d="M37 57c6 6 16 6 22 0" />
            <path d="M20 43h-5v12h5m56-12h5v12h-5" />
            <circle cx="48" cy="14" r="3" />
          </svg>
        </div>
        <span className="swarmbot-visual-caption">YOUR PERSONAL COMPANION</span>
      </div>

      <div className="swarmbot-copy">
        <span className="landing-section-kicker">Introducing Swarmbot</span>
        <h2 id="swarmbot-title">Intelligence that feels<br /><em>closer to home.</em></h2>
        <p className="swarmbot-lead">
          Swarmbot is our vision for a personal AI companion: emotionally aware, helpful between tasks,
          and designed to run locally so your day-to-day relationship with AI feels more personal.
        </p>
        <div className="swarmbot-qualities">
          <div>
            <span className="swarmbot-quality-icon" aria-hidden="true">
              <svg viewBox="0 0 20 20"><path d="M10 17s-6.5-3.8-6.5-8.3A3.7 3.7 0 0 1 10 6.8a3.7 3.7 0 0 1 6.5 1.9C16.5 13.2 10 17 10 17Z" /></svg>
            </span>
            <span><strong>Emotionally aware</strong><small>Notices conversational cues and responds with care—without pretending to replace human support.</small></span>
          </div>
          <div>
            <span className="swarmbot-quality-icon" aria-hidden="true">
              <svg viewBox="0 0 20 20"><path d="M10 2.5 16 5v4.4c0 3.8-2.5 6.6-6 8.1-3.5-1.5-6-4.3-6-8.1V5z" /><path d="m7.2 9.8 1.8 1.8 3.8-4" /></svg>
            </span>
            <span><strong>Local-first vision</strong><small>Designed around a personal, device-centered experience and greater control over your information.</small></span>
          </div>
          <div>
            <span className="swarmbot-quality-icon" aria-hidden="true">
              <svg viewBox="0 0 20 20"><path d="M10 2.5v3m0 9v3M2.5 10h3m9 0h3M4.7 4.7l2.1 2.1m6.4 6.4 2.1 2.1m0-10.6-2.1 2.1m-6.4 6.4-2.1 2.1" /><circle cx="10" cy="10" r="3" /></svg>
            </span>
            <span><strong>Proactive, with your say</strong><small>Could help organize goals and suggest useful next steps, with meaningful user control.</small></span>
          </div>
        </div>
        <div className="swarmbot-detail-grid" aria-label="Swarmbot design principles">
          <article>
            <span className="swarmbot-detail-index">01 / CONTEXT</span>
            <h3>Remember what matters</h3>
            <p>Keep goals and preferences in view, so you can continue a thought without starting from scratch.</p>
          </article>
          <article>
            <span className="swarmbot-detail-index">02 / INITIATIVE</span>
            <h3>Offer a timely nudge</h3>
            <p>Surface a relevant reminder or option when it helps—without taking consequential actions silently.</p>
          </article>
          <article>
            <span className="swarmbot-detail-index">03 / EMPATHY</span>
            <h3>Meet the moment</h3>
            <p>Respond with patience and warmth, adapting its tone to the conversation rather than forcing a script.</p>
          </article>
        </div>
        <div className="swarmbot-conversation" aria-label="Illustrative Swarmbot interaction">
          <div className="swarmbot-conversation-heading">
            <span className="swarmbot-conversation-indicator" />
            <span>AN EXAMPLE OF THE VISION</span>
          </div>
          <p className="swarmbot-example-user">I have a lot on my mind and don’t know where to start.</p>
          <div className="swarmbot-example-reply">
            <span className="swarmbot-example-avatar" aria-hidden="true">S</span>
            <p>That sounds like a lot to carry. Want to talk it through, or would it help to choose one small thing to focus on first?</p>
          </div>
          <span className="swarmbot-example-note">Illustrative example. Swarmbot is not currently available to install.</span>
        </div>
        <div className="swarmbot-readiness-note" role="note">
          <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5" /><path d="M10 9v4m0-7h.01" /></svg>
          <p><strong>Still in development.</strong> The local Swarmbot runtime and installer are not available yet. These details describe the intended direction, not current product capabilities.</p>
        </div>
        <div className="swarmbot-actions">
          <button
            type="button"
            className="landing-primary-button"
            aria-expanded={showInstallStatus}
            aria-controls="swarmbot-install-status"
            onClick={() => setShowInstallStatus((visible) => !visible)}
          >
            Install Swarmbot
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3v9m-4-4 4 4 4-4M4 14v3h12v-3" /></svg>
          </button>
          <span className="swarmbot-platform-note">Desktop installation</span>
        </div>
        {showInstallStatus ? (
          <p className="swarmbot-install-status" id="swarmbot-install-status" role="status" aria-live="polite">
            The Swarmbot installer is coming soon. There isn’t a downloadable installer available yet.
          </p>
        ) : null}
      </div>
    </section>
  )
}
