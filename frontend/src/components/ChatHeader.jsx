export default function ChatHeader({
  activePage,
  assistantMode,
  onAssistantModeChange,
  researchMode,
  onResearchModeChange,
  observerEnabled,
  observerPending,
  onObserverToggle,
  isRunning,
  onLogout,
}) {
  return (
    <header className="chat-header">
      <div className="header-actions">
        <div className="mode-switch" role="group" aria-label="Conversation mode">
          <button
            className={!assistantMode ? 'selected' : ''}
            type="button"
            aria-pressed={!assistantMode}
            onClick={() => onAssistantModeChange(false)}
          >
            Chat
          </button>
          <button
            className={assistantMode ? 'selected' : ''}
            type="button"
            aria-pressed={assistantMode}
            onClick={() => onAssistantModeChange(true)}
          >
            Assistant
          </button>
        </div>
        {activePage === 'chat' ? (
          <>
          <button
            className={`research-mode-button ${researchMode ? 'active' : ''}`}
            type="button"
            aria-pressed={researchMode}
            aria-label={`Research mode ${researchMode ? 'on' : 'off'}`}
            title="Force multi-step agent research"
            disabled={isRunning}
            onClick={() => onResearchModeChange(!researchMode)}
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M10 2.5 11.7 7l4.8 1.4-4.8 1.5L10 14.5l-1.7-4.6L3.5 8.4 8.3 7 10 2.5Z" />
              <path d="m15.6 12.8.8 2.1 2.1.7-2.1.7-.8 2.1-.7-2.1-2.1-.7 2.1-.7.7-2.1Z" />
            </svg>
            <span>Research</span>
          </button>
          </>
        ) : null}
        {activePage === 'chat' ? (
          <button
            className={`observer-mode-button ${observerEnabled ? 'active' : ''}`}
            type="button"
            aria-pressed={observerEnabled}
            disabled={observerPending || isRunning}
            onClick={onObserverToggle}
            title="Build private context for this chat session"
          >
            <span className="observer-button-signal" aria-hidden="true"><i /></span>
            <span>{observerPending ? 'Starting…' : observerEnabled ? 'Observer on' : 'Observer'}</span>
          </button>
        ) : null}
        <nav className="header-nav-tools" aria-label="Workspace actions">
          <button className="ghost-button chat-header-signout" type="button" onClick={onLogout}>Sign out</button>
        </nav>
      </div>
    </header>
  )
}
