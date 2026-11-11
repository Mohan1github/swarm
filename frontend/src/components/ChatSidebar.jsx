export default function ChatSidebar({
  user,
  activePage,
  onShowHistory,
  onShowContextGraph,
  onBackToChat,
  onNewTask,
  onShowLanding,
}) {
  return (
    <aside className="chat-sidebar">
      <div className="brand-block">
        <div className="brand-row">
          <button
            type="button"
            className="sidebar-brand-button"
            onClick={onShowLanding}
            aria-label="Open Swarm AI landing page"
          >
            <div className="brand-mark">AI</div>
          </button>
          <div>
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

      <nav className="sidebar-workspace-actions" aria-label="Workspace actions">
        <button
          type="button"
          className="sidebar-action-button"
          onClick={activePage === 'contextGraph' ? onBackToChat : onShowContextGraph}
        >
          <span aria-hidden="true">{activePage === 'contextGraph' ? '←' : '⌘'}</span>
          {activePage === 'contextGraph' ? 'Back to chat' : 'Context graph'}
        </button>
        <button type="button" className="sidebar-action-button">
          <span aria-hidden="true">↗</span>
          Share
        </button>
        {activePage === 'chat' ? (
          <button type="button" className="sidebar-action-button" onClick={onNewTask}>
            <span aria-hidden="true">＋</span>
            New task
          </button>
        ) : null}
      </nav>

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
