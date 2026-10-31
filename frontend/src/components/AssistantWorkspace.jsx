import { useCallback, useState } from 'react'
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
  onBackToChat,
  onShowContextGraph,
  onShowHistory,
  onLogout,
  onSubmit,
  onClearMemory,
  onAddFiles,
  onRemoveFile,
  onVoiceSubmit,
}) {
  const [speechProgress, setSpeechProgress] = useState(0)
  const isSpeaking = speechProgress > 0 && speechProgress < 1
  const handleAssistantProgress = useCallback((progress) => {
    setSpeechProgress(progress >= 1 ? 0 : progress)
  }, [])

  return (
    <main className="assistant-page">
      <header className="assistant-page-header">
        <div className="assistant-page-brand">
          <span className="assistant-brand-mark" aria-hidden="true">S</span>
          <div>
            <strong>Swarm Assistant</strong>
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
          <button
            type="button"
            className="assistant-nav-button"
            onClick={onToggleProactive}
            aria-pressed={proactiveEnabled}
            title={proactiveEnabled ? 'Pause automatic follow-ups' : 'Resume automatic follow-ups'}
          >
            {proactiveEnabled ? 'Pause proactive' : 'Resume proactive'}
          </button>
          <button type="button" className="assistant-nav-button" onClick={onShowHistory}>History</button>
          <button type="button" className="assistant-nav-button" onClick={onShowContextGraph}>Context graph</button>
          <button type="button" className="assistant-nav-button" onClick={onClearMemory}>Forget memory</button>
          <button type="button" className="assistant-nav-button primary" onClick={onBackToChat}>Back to chat</button>
          <button type="button" className="assistant-signout" onClick={onLogout} aria-label={`Sign out ${user.email}`} title={user.email}>Sign out</button>
        </nav>
      </header>

      <div className="assistant-page-body">
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

        <section className="assistant-conversation" aria-label="Assistant conversation">
          <header className="assistant-conversation-header">
            <div>
              <span>PRIVATE WORKSPACE</span>
              <h1>Your conversation</h1>
            </div>
            <span className="assistant-context-indicator">
              <i aria-hidden="true" />
              Context-aware
            </span>
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
    </main>
  )
}