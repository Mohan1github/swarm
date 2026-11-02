export default function ChatHeader({
  activePage,
  assistantMode,
  onAssistantModeChange,
  researchMode,
  onResearchModeChange,
  onOpenDocumentReviewer,
  isRunning,
  onShowContextGraph,
  onBackToChat,
  onLogout,
  onReset,
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
          {researchMode ? (
            <button
              className="document-reviewer-trigger"
              type="button"
              disabled={isRunning}
              onClick={onOpenDocumentReviewer}
            >
              Documents
            </button>
          ) : null}
          </>
        ) : null}
        <nav className="header-nav-tools" aria-label="Workspace actions">
          {activePage === 'contextGraph' ? (
            <button className="ghost-button" type="button" onClick={onBackToChat}>Back to chat</button>
          ) : (
            <button className="ghost-button" type="button" onClick={onShowContextGraph}>Context graph</button>
          )}
          <button className="ghost-button" type="button">Share</button>
          {activePage === 'chat' ? (
            <button className="ghost-button" type="button" onClick={onReset}>New task</button>
          ) : null}
          <button className="ghost-button" type="button" onClick={onLogout}>Sign out</button>
        </nav>
      </div>
    </header>
  )
}
