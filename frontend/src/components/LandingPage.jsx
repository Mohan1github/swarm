import { useEffect } from 'react'
import LandingCapabilities from './LandingCapabilities'
import LandingParticles from './LandingParticles'
import LandingSwarmbot from './LandingSwarmbot'
import LandingWorkflow from './LandingWorkflow'

export default function LandingPage({ onGetStarted, onSignIn }) {
  useEffect(() => {
    const page = document.querySelector('.landing-page')
    if (!page) return undefined

    page.classList.add('landing-motion-ready')
    const revealItems = page.querySelectorAll('[data-landing-reveal]')
    if (!('IntersectionObserver' in window)) {
      revealItems.forEach((item) => item.classList.add('is-visible'))
      return () => page.classList.remove('landing-motion-ready')
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible')
          observer.unobserve(entry.target)
        }
      })
    }, { threshold: 0.12, rootMargin: '0px 0px -32px 0px' })

    revealItems.forEach((item) => observer.observe(item))
    return () => {
      observer.disconnect()
      page.classList.remove('landing-motion-ready')
    }
  }, [])

  const scrollTo = (sectionId) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <main className="landing-page">
      <LandingParticles />
      <div className="landing-content">
        <nav className="landing-nav" aria-label="Primary navigation">
        <button type="button" className="landing-brand" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
          <span className="brand-mark">AI</span>
          <span>Swarm AI</span>
        </button>
        <div className="landing-nav-links">
          <button type="button" onClick={() => scrollTo('workflow')}>Platform</button>
          <button type="button" onClick={() => scrollTo('capabilities')}>Capabilities</button>
          <button type="button" onClick={() => scrollTo('swarmbot')}>Swarmbot</button>
          <button type="button" onClick={onSignIn}>Sign in</button>
          <button type="button" className="landing-nav-cta" onClick={onGetStarted}>Open workspace</button>
        </div>
        </nav>

        <section className="landing-hero" aria-labelledby="landing-title">
        <div className="landing-hero-copy" data-landing-reveal>
          <div className="landing-kicker"><span /> Intelligence, orchestrated</div>
          <h1 id="landing-title">Research deeply.<br /><em>Decide clearly.</em></h1>
          <p>
            Swarm AI coordinates specialized agents to explore the web, validate evidence,
            understand context, and turn complex questions into clear decisions.
          </p>
          <div className="landing-actions">
            <button type="button" className="landing-primary-button" onClick={onGetStarted}>
              Start researching
              <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h12M11 5l5 5-5 5" /></svg>
            </button>
            <button type="button" className="landing-secondary-button" onClick={() => scrollTo('workflow')}>
              Explore the platform
            </button>
          </div>
          <div className="landing-trust-row">
            <span>Live public research</span>
            <span>Source validation</span>
            <span>Private workspace</span>
          </div>
        </div>

        <div className="landing-product-stage" aria-label="Swarm AI research workspace preview">
          <svg className="landing-agent-network" viewBox="0 0 620 520" aria-hidden="true">
            <defs>
              <linearGradient id="landingNetworkLine" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="#8ca99a" stopOpacity=".08" />
                <stop offset=".5" stopColor="#bdd1c4" stopOpacity=".7" />
                <stop offset="1" stopColor="#8ca99a" stopOpacity=".08" />
              </linearGradient>
              <filter id="landingNetworkGlow">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
            <path className="landing-network-path" pathLength="1" d="M82 258 C180 258 186 118 310 118 S440 258 538 258" />
            <path className="landing-network-path landing-network-path-secondary" pathLength="1" d="M82 258 C180 258 186 402 310 402 S440 258 538 258" />
            <circle className="landing-network-pulse" r="4" fill="#b9cbbb" filter="url(#landingNetworkGlow)">
              <animateMotion dur="3.6s" repeatCount="indefinite" path="M82 258 C180 258 186 118 310 118 S440 258 538 258" />
            </circle>
            <circle className="landing-network-pulse landing-network-pulse-delayed" r="3" fill="#8ca99a">
              <animateMotion dur="4.2s" begin="-1.7s" repeatCount="indefinite" path="M82 258 C180 258 186 402 310 402 S440 258 538 258" />
            </circle>
            {[ [82,258], [310,118], [310,402], [538,258] ].map(([cx, cy]) => (
              <g className="landing-network-node" key={`${cx}-${cy}`}>
                <circle cx={cx} cy={cy} r="8" />
                <circle cx={cx} cy={cy} r="3" />
              </g>
            ))}
          </svg>
          <div className="landing-orbit orbit-one" />
          <div className="landing-orbit orbit-two" />
          <div className="landing-preview">
            <div className="landing-preview-bar">
              <div className="landing-preview-dots"><span /><span /><span /></div>
              <span>Research workspace</span>
              <div className="landing-preview-status"><i /> Live</div>
            </div>
            <div className="landing-preview-body">
              <aside>
                <span className="preview-sidebar-mark">AI</span>
                <span className="preview-side-active" />
                <span />
                <span />
              </aside>
              <div className="preview-workspace">
                <div className="preview-query">
                  <span>Research mode</span>
                  <strong>Assess the opportunities for a sustainable city relocation</strong>
                </div>
                <div className="preview-agent-row">
                  <article>
                    <div className="preview-agent-icon">01</div>
                    <div><span>Query planning</span><strong>Intent mapped</strong></div>
                    <b>✓</b>
                  </article>
                  <article>
                    <div className="preview-agent-icon">02</div>
                    <div><span>Live retrieval</span><strong>12 sources found</strong></div>
                    <b>✓</b>
                  </article>
                  <article className="preview-agent-active">
                    <div className="preview-agent-icon">03</div>
                    <div><span>Evidence validator</span><strong>Checking sources</strong></div>
                    <i />
                  </article>
                </div>
                <div className="preview-answer">
                  <span>Most important</span>
                  <strong>Prioritize neighborhoods with resilient infrastructure and verified long-term housing costs.</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
        </section>

        <section className="landing-proof" aria-label="Platform principles" data-landing-reveal>
        <div><strong>Research</strong><span>Adaptive live search</span></div>
        <div><strong>Reason</strong><span>Specialist collaboration</span></div>
        <div><strong>Verify</strong><span>Citations and evidence</span></div>
        <div><strong>Understand</strong><span>Context-aware answers</span></div>
        </section>

        <LandingWorkflow />

        <section className="landing-features" id="features" aria-labelledby="features-title" data-landing-reveal>
        <div className="landing-section-heading">
          <div>
            <span className="landing-section-kicker">Built for complex work</span>
            <h2 id="features-title">One intelligence layer.<br />Every mode you need.</h2>
          </div>
          <p>Move from a single question to a fully investigated, evidence-led answer without changing tools.</p>
        </div>

        <div className="landing-feature-grid">
          <article className="landing-feature-card landing-feature-large">
            <span className="landing-feature-number">01</span>
            <div className="landing-feature-visual research-visual">
              <svg className="research-flow-svg" viewBox="0 0 620 145" role="img" aria-label="Animated research workflow">
                <defs>
                  <linearGradient id="researchFlowGradient" x1="0" x2="1">
                    <stop offset="0" stopColor="#718477" />
                    <stop offset=".55" stopColor="#d2ddd4" />
                    <stop offset="1" stopColor="#718477" />
                  </linearGradient>
                </defs>
                <path className="research-flow-track" d="M92 72 H528" />
                <circle className="research-flow-packet" r="4">
                  <animateMotion dur="4s" repeatCount="indefinite" path="M92 72 H528" />
                </circle>
                {[92, 310, 528].map((x, index) => (
                  <g className="research-flow-node" key={x} style={{ animationDelay: `${index * -0.8}s` }}>
                    <circle cx={x} cy="72" r="22" />
                    <circle cx={x} cy="72" r="7" />
                  </g>
                ))}
                <text x="92" y="116" textAnchor="middle">Clarify intent</text>
                <text x="310" y="116" textAnchor="middle">Explore sources</text>
                <text x="528" y="116" textAnchor="middle">Validate findings</text>
              </svg>
            </div>
            <h3>Agentic research</h3>
            <p>Generate focused queries, decide when more evidence is needed, crawl public sources, and synthesize verified findings.</p>
          </article>

          <article className="landing-feature-card">
            <span className="landing-feature-number">02</span>
            <div className="landing-feature-symbol">↗</div>
            <h3>Proactive Assistant</h3>
            <p>A natural, context-aware companion that follows your conversation and anticipates the next useful action.</p>
          </article>

          <article className="landing-feature-card">
            <span className="landing-feature-number">03</span>
            <div className="landing-feature-symbol">⌘</div>
            <h3>Context graph</h3>
            <p>Keep people, decisions, documents, and ideas connected across the work you do in Swarm AI.</p>
          </article>

          <article className="landing-feature-card landing-feature-wide">
            <span className="landing-feature-number">04</span>
            <div className="landing-feature-symbol">✓</div>
            <h3>Evidence, not guesswork</h3>
            <p>Consequential claims are checked against live sources, with uncertainty and limitations made explicit.</p>
          </article>
        </div>
        </section>

        <LandingCapabilities />

        <LandingSwarmbot />

        <section className="landing-final-cta" data-landing-reveal>
        <span className="landing-section-kicker">Your next question</span>
        <h2>Think further.<br /><em>Move faster.</em></h2>
        <p>Bring depth, context, and specialist intelligence to every decision.</p>
        <button type="button" className="landing-primary-button" onClick={onGetStarted}>
          Open Swarm AI
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h12M11 5l5 5-5 5" /></svg>
        </button>
        </section>

        <footer className="landing-footer">
        <div className="landing-brand-static"><span className="brand-mark">AI</span><span>Swarm AI</span></div>
        <span>Intelligence, orchestrated.</span>
        </footer>
      </div>
    </main>
  )
}
