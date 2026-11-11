import { useCallback, useEffect, useRef, useState } from 'react'
import AssistantPresence from './AssistantPresence'
import ChatComposer from './ChatComposer'
import ChatMessages from './ChatMessages'

export default function AssistantWorkspace({
  user,
  isLoading,
  isRunning,
  isProactive,
  proactiveEnabled,
  onToggleProactive,
  messages,
  error,
  memoryTurnCount,
  query,
  setQuery,
  attachments,
  attachmentError,
  voiceState,
  setVoiceState,
  onShowContextGraph,
  onShowHistory,
  onLogout,
  onSubmit,
  onClearMemory,
  onAddFiles,
  onRemoveFile,
  onVoiceSubmit,
  assistantName,
  emotionalAssistantEnabled,
  onSavePreferences,
  embedded = false,
}) {
  const [speechProgress, setSpeechProgress] = useState(0)
  const [nameDraft, setNameDraft] = useState(assistantName)
  const [emotionalDraft, setEmotionalDraft] = useState(emotionalAssistantEnabled)
  const [preferencesOpen, setPreferencesOpen] = useState(false)
  const [preferenceError, setPreferenceError] = useState('')
  const [actionsOpen, setActionsOpen] = useState(false)
  const [proactiveDraft, setProactiveDraft] = useState(proactiveEnabled)
  const [isSavingPreferences, setIsSavingPreferences] = useState(false)
  const [preferencesSaved, setPreferencesSaved] = useState(false)
  const actionsMenuRef = useRef(null)
  const isSpeaking = speechProgress > 0 && speechProgress < 1
  const handleAssistantProgress = useCallback((progress) => {
    setSpeechProgress(progress >= 1 ? 0 : progress)
  }, [])

  useEffect(() => {
    setNameDraft(assistantName)
    setEmotionalDraft(emotionalAssistantEnabled)
    setProactiveDraft(proactiveEnabled)
  }, [assistantName, emotionalAssistantEnabled, proactiveEnabled])

  useEffect(() => {
    if (!actionsOpen) return undefined
    const closeMenu = (event) => {
      if (!actionsMenuRef.current?.contains(event.target)) setActionsOpen(false)
    }
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setActionsOpen(false)
    }
    document.addEventListener('pointerdown', closeMenu)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeMenu)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [actionsOpen])

  const runMenuAction = (action) => {
    setActionsOpen(false)
    action?.()
  }

  const savePreferences = async (event) => {
    event.preventDefault()
    setPreferenceError('')
    setPreferencesSaved(false)
    setIsSavingPreferences(true)
    try {
      await onSavePreferences(nameDraft, emotionalDraft, proactiveDraft)
      setPreferencesSaved(true)
    } catch (saveError) {
      setPreferenceError(saveError.message || 'Could not save assistant preferences.')
    } finally {
      setIsSavingPreferences(false)
    }

  }


  const WorkspaceRoot = embedded ? 'section' : 'main'

  return (
    <WorkspaceRoot className={`assistant-page ${embedded ? 'assistant-page-embedded' : ''}`}>
      <header className="assistant-page-header">
        <div className="assistant-page-brand">
          <span className="assistant-brand-mark" aria-hidden="true">S</span>
          <div>
            <strong>{assistantName}</strong>
            <span>Workspace companion</span>
          </div>
        </div>

        <div className="assistant-page-status" aria-live="polite">
          <span className={`assistant-status-dot ${isRunning || isSpeaking ? 'working' : voiceState === 'listening' ? 'listening' : ''}`} />
          <span>
            {voiceState === 'listening'
              ? 'Listening'
              : isRunning
                ? isProactive ? 'Thinking of a next step' : 'Thinking'
                : isLoading
                  ? 'Reconnecting to your conversation'
                  : isSpeaking
                    ? 'Responding'
                  : `Ready · ${memoryTurnCount} remembered turns`}
          </span>
        </div>

        <nav className="assistant-page-actions" aria-label="Assistant workspace navigation">
          <div className="assistant-actions-menu" ref={actionsMenuRef}>
            <button
              type="button"
              className="assistant-more-button"
              onClick={() => setActionsOpen((open) => !open)}
              aria-label="More assistant actions"
              aria-haspopup="menu"
              aria-expanded={actionsOpen}
              title="More assistant actions"
            >
              <i aria-hidden="true" /><i aria-hidden="true" /><i aria-hidden="true" />
            </button>
            {actionsOpen ? (
              <div className="assistant-actions-dropdown" role="menu">
                <button type="button" role="menuitem" onClick={() => runMenuAction(() => setPreferencesOpen(true))}>Assistant settings</button>
                <button type="button" role="menuitem" onClick={() => runMenuAction(onToggleProactive)} aria-pressed={proactiveEnabled}>
                  Proactive: {proactiveEnabled ? 'On · Turn off' : 'Off · Turn on'}
                </button>
                <button type="button" role="menuitem" onClick={() => runMenuAction(onShowHistory)}>History</button>
                <button type="button" role="menuitem" onClick={() => runMenuAction(onShowContextGraph)}>Context graph</button>
                <button type="button" role="menuitem" onClick={() => runMenuAction(onClearMemory)}>Forget memory</button>
                <button type="button" role="menuitem" onClick={() => runMenuAction(onLogout)} title={user.email}>Sign out</button>
              </div>
            ) : null}
          </div>
        </nav>
      </header>

      {preferencesOpen ? (
        <form className="assistant-preferences-panel" onSubmit={savePreferences}>
          <label>
            Personal assistant name
            <input
              value={nameDraft}
              onChange={(event) => setNameDraft(event.target.value)}
              placeholder="What should I call myself?"
            />
          </label>
          <label className="assistant-preferences-toggle">
            <input
              type="checkbox"
              checked={proactiveDraft}
              onChange={(event) => setProactiveDraft(event.target.checked)}
            />
            <span>
              <strong>Proactive follow-ups</strong>
              <small>{proactiveDraft ? 'On' : 'Off'}</small>
            </span>
          </label>
          <label className="assistant-preferences-toggle">
            <input
              type="checkbox"
              checked={emotionalDraft}
              onChange={(event) => setEmotionalDraft(event.target.checked)}
            />
            <span>
              <strong>Allow emotional support check</strong>
              <small>Only after strong emotional cues, ask for one camera frame with your permission.</small>
            </span>
          </label>
          {preferenceError ? <p role="alert">{preferenceError}</p> : null}
          {preferencesSaved ? <p className="assistant-preferences-success" role="status">Settings saved.</p> : null}
          <div>
            <button type="submit" className="assistant-nav-button primary" disabled={isSavingPreferences}>
              {isSavingPreferences ? 'Saving…' : 'Save'}
            </button>
            <button type="button" className="assistant-nav-button" onClick={() => setPreferencesOpen(false)}>Cancel</button>
          </div>
        </form>
      ) : null}


      <div className={`assistant-page-body ${embedded ? 'assistant-live-space' : ''}`}>
        <section className="assistant-presence-stage" aria-label="Live assistant">
          <span className="assistant-stage-kicker">A little space to think</span>
          <AssistantPresence
            isListening={voiceState === 'listening'}
            isThinking={isRunning}
            isSpeaking={isSpeaking}
            speechProgress={speechProgress}
          />
          <p className="assistant-stage-caption">
            {voiceState === 'listening'
              ? 'Take your time. I’m listening.'
              : isRunning
                ? 'Let me think that through.'
                : isSpeaking
                  ? 'Putting the response together for you.'
                  : 'Pick up where you left off, or start with whatever is on your mind.'}
          </p>
        </section>

        <section className="assistant-conversation assistant-live-thread" aria-label="Assistant conversation">
          <header className="assistant-conversation-header">
            <div>
              <span>LIVE THREAD</span>
              <h1>Conversation</h1>
            </div>
            <div className="assistant-conversation-header-actions">
              <span className="assistant-context-indicator">
                <i aria-hidden="true" />
                Context-aware
              </span>
            </div>
          </header>
          <div className="assistant-messages">
            <ChatMessages
              messages={messages}
              error={error}
              onAssistantProgress={handleAssistantProgress}
            />
          </div>
          <ChatComposer
            query={query}
            setQuery={setQuery}
            attachments={attachments}
            attachmentError={attachmentError}
            isRunning={isRunning || isLoading}
            voiceState={voiceState}
            onVoiceStateChange={setVoiceState}
            onVoiceSubmit={onVoiceSubmit}
            onAddFiles={onAddFiles}
            onRemoveFile={onRemoveFile}
            onSubmit={onSubmit}
          />
        </section>
      </div>
    </WorkspaceRoot>
  )
}