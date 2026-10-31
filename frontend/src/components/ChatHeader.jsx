export default function ChatHeader({
  activePage,
  assistantMode,
  onAssistantModeChange,
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
