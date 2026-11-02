import { stageLabels, stageOrder } from '../constants/workflow'

export default function ChatSidebar({
  user,
  selectedAgents,
  latency,
  validationScore,
  isRunning,
  researchMode,
  steps,
  activeStage,
  onShowHistory,
}) {
  const stepLookup = Object.fromEntries(steps.map((step) => [step.role, step]))

  return (
    <aside className="chat-sidebar">
      <div className="brand-block">
        <div className="brand-row">
          <div className="brand-mark">AI</div>
          <div>
            <span className="eyebrow">Research engine</span>
              <h1>Swarm AI</h1>
          </div>
        </div>
      </div>

      <button type="button" className="sidebar-history-button" onClick={onShowHistory}>
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <path d="M3 5.5h14M3 10h14M3 14.5h9" />
          <circle cx="16" cy="14.5" r="2.5" />
        </svg>
        <span>Chat history</span>
      </button>

      {researchMode ? (
        <div className="sidebar-card">
          <div className="card-header">
            <span>Live workflow</span>
            <span className={`status-dot ${isRunning ? 'live' : ''}`} />
          </div>
          <div className="stage-stack">
            {stageOrder.map((stage) => {
              const isComplete = Boolean(stepLookup[stage])
              const isActive = isRunning && activeStage === stage
              return (
                <div
                  key={stage}
                  className={`stage-pill ${isComplete ? 'complete' : ''} ${isActive ? 'active' : ''}`}
                  aria-current={isActive ? 'step' : undefined}
                  aria-label={`${stageLabels[stage]}${isActive ? ', working' : isComplete ? ', completed' : ', waiting'}`}
                >
                  <span>{stageLabels[stage]}</span>
                  <span className="stage-pill-state" aria-hidden="true">
                    {isActive ? 'Working' : isComplete ? 'Done' : ''}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      ) : null}

      <div className="sidebar-card compact">
        <div className="metric-row">
          <span>Workers</span>
          <strong>{selectedAgents}</strong>
        </div>
        <div className="metric-row">
          <span>Latency</span>
          <strong>{latency !== null ? `${latency} ms` : '--'}</strong>
        </div>
        <div className="metric-row">
          <span>Validator</span>
          <strong>{validationScore !== null ? validationScore.toFixed(2) : '--'}</strong>
        </div>
      </div>

      <div
        className="sidebar-profile"
        tabIndex={0}
        aria-label={`Current user: ${user.email}`}
        data-email={user.email}
      >
        <span className="sidebar-profile-icon" aria-hidden="true">
          {user.email.charAt(0).toUpperCase()}
        </span>
        <span className="sidebar-profile-label">User</span>
      </div>
    </aside>
  )
}
