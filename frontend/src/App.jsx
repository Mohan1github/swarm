import { useEffect, useRef, useState } from 'react'
import './App.css'
import ChatSidebar from './components/ChatSidebar'
import ChatHeader from './components/ChatHeader'
import ChatMessages from './components/ChatMessages'
import ChatComposer from './components/ChatComposer'
import DocumentReviewer from './components/DocumentReviewer'
import AuthScreen from './components/AuthScreen'
import ContextGraphPage from './components/ContextGraphPage'
import AssistantWorkspace from './components/AssistantWorkspace'
import ChatHistoryDrawer from './components/ChatHistoryDrawer'
import TypingText from './components/TypingText'
import useAuth from './auth/useAuth'
import useAssistantConversation from './hooks/useAssistantConversation'
import useSwarmStream from './hooks/useSwarmStream'
import useObserver from './hooks/useObserver'
import ObserverAssistant from './components/ObserverAssistant'
import LandingPage from './components/LandingPage'
import { stageLabels } from './constants/workflow'

const CHAT_STARTERS = [
  {
    title: 'Untangle an idea',
    prompt: 'Help me think through an idea and turn it into a clear plan.',
    icon: '✳',
  },
  {
    title: 'Make a plan',
    prompt: 'Help me break my next task into practical steps.',
    icon: '↗',
  },
  {
    title: 'Explore a topic',
    prompt: 'Explain a topic I am curious about, with useful examples.',
    icon: '⌕',
  },
]

function ModeTransition({ transition }) {
  if (!transition) return null

  const isAssistant = transition.target === 'assistant'
  return (
    <div
      className="mode-transition-layer"
      key={transition.id}
      role="status"
      aria-live="polite"
      aria-label={isAssistant ? 'Opening Assistant' : 'Returning to Chat'}
    >
      <div className="mode-transition-card">
        <div className={`mode-transition-orbit ${isAssistant ? 'assistant' : 'chat'}`} aria-hidden="true">
          <span />
          <span />
          <i />
        </div>
        <span className="mode-transition-kicker">Changing workspace</span>
        <strong>{isAssistant ? 'Opening Assistant' : 'Returning to Chat'}</strong>
        <span className="mode-transition-caption">
          {isAssistant ? 'Bringing your conversation into focus' : 'Returning to your live workflow'}
        </span>
      </div>
    </div>
  )
}

function AuthenticatedWorkspace({ token, user, onLogout, onShowLanding }) {
  const [activePage, setActivePage] = useState('chat')
  const [query, setQuery] = useState('')
  const [attachments, setAttachments] = useState([])
  const [attachmentError, setAttachmentError] = useState('')
  const [assistantMode, setAssistantMode] = useState(false)
  const [researchMode, setResearchMode] = useState(false)
  const [documentReviewerOpen, setDocumentReviewerOpen] = useState(false)
  const [documentReviewFiles, setDocumentReviewFiles] = useState([])
  const [documentReviewQuery, setDocumentReviewQuery] = useState('')
  const [historyOpen, setHistoryOpen] = useState(false)
  const [voiceState, setVoiceState] = useState('idle')
  const [modeTransition, setModeTransition] = useState(null)
  const transitionTimeout = useRef(null)
  const transitionId = useRef(0)
  const observer = useObserver({ token })
  const {
    isRunning,
    activeStage,
    stageTotal,
    isResearchRun,
    messages,
    error,
    runWorkflow,
    clearLocalHistory,
  } = useSwarmStream({
    token,
    observerEnabled: observer.enabled,
    observerSessionId: observer.sessionId,
    onObserverResult: observer.observeTurn,
  })
  const assistantConversation = useAssistantConversation({
    token,
    active: assistantMode && activePage === 'chat',
  })
  const hasUserMessages = messages.some((message) => message.role === 'user')

  const handleHistoryCleared = () => {
    clearLocalHistory()
    observer.reset()
    assistantConversation.clearLocalConversation()
    setQuery('')
    setAttachmentError('')
    attachments.forEach(({ previewUrl }) => previewUrl && URL.revokeObjectURL(previewUrl))
    setAttachments([])
  }

  useEffect(() => () => {
    window.clearTimeout(transitionTimeout.current)
  }, [])

  const showModeTransition = (target) => {
    window.clearTimeout(transitionTimeout.current)
    const id = transitionId.current + 1
    transitionId.current = id
    setModeTransition({ target, id })
    const duration = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 60 : 820
    transitionTimeout.current = window.setTimeout(() => {
      setModeTransition((current) => (current?.id === id ? null : current))
    }, duration)
  }

  const handleAddFiles = (files) => {
    const allowedExtensions = new Set([
      'png', 'jpg', 'jpeg', 'webp', 'pdf', 'txt', 'md', 'csv', 'json', 'py',
      'docx', 'js', 'jsx', 'ts', 'tsx', 'html', 'css', 'xml', 'yaml', 'yml', 'log', 'sql', 'sh',
    ])
    const availableSlots = Math.max(0, 5 - attachments.length)
    const accepted = []
    const rejected = []
    let totalPayloadSize = attachments.reduce((total, { file }) => {
      const extension = file.name.split('.').pop()?.toLowerCase()
      const isEncoded = ['png', 'jpg', 'jpeg', 'webp', 'pdf', 'docx'].includes(extension)
      return total + (isEncoded ? Math.ceil(file.size * 4 / 3) + 64 : file.size)
    }, 0)

    files.forEach((file) => {
      const extension = file.name.split('.').pop()?.toLowerCase()
      const isEncoded = ['png', 'jpg', 'jpeg', 'webp', 'pdf', 'docx'].includes(extension)
      const payloadSize = isEncoded ? Math.ceil(file.size * 4 / 3) + 64 : file.size
      if (!allowedExtensions.has(extension)) {
        rejected.push(`${file.name}: unsupported file type`)
      } else if (file.size > 5_000_000) {
        rejected.push(`${file.name}: exceeds the 5 MB limit`)
      } else if (totalPayloadSize + payloadSize > 15_000_000) {
        rejected.push(`${file.name}: combined attachment limit is 15 MB`)
      } else if (accepted.length >= availableSlots) {
        rejected.push('You can attach up to 5 files per message')
      } else {
        const isImage = file.type.startsWith('image/') || /\.(png|jpe?g|webp)$/i.test(file.name)
        accepted.push({
          id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
          file,
          previewUrl: isImage ? URL.createObjectURL(file) : '',
        })
        totalPayloadSize += payloadSize
      }
    })

    if (accepted.length) setAttachments((current) => [...current, ...accepted])
    setAttachmentError(rejected.join('. '))
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    const task = query.trim() || (attachments.length ? 'Analyze the attached files and photos.' : '')
    if (researchMode) {
      const documents = attachments
        .filter(({ file }) => /\.(pdf|docx)$/i.test(file.name))
        .map(({ file }) => file)
      if (documents.length) {
        setDocumentReviewFiles(documents)
        setDocumentReviewQuery(query.trim())
        setDocumentReviewerOpen(true)
      }
    }
    runWorkflow(task, attachments, undefined, researchMode)
    if (task) {
      attachments.forEach(({ previewUrl }) => previewUrl && URL.revokeObjectURL(previewUrl))
      setQuery('')
      setAttachments([])
      setAttachmentError('')
    }
  }

  const handleVoiceSubmit = (transcript) => {
    if (transcript.trim()) runWorkflow(transcript.trim(), [], undefined, researchMode)
  }

  const handleResearchModeChange = (enabled) => {
    if (enabled && !researchMode) clearLocalHistory()
    setResearchMode(enabled)
    if (!enabled) setDocumentReviewerOpen(false)
  }

  const handleModeChange = (enabled) => {
    if (enabled !== assistantMode || activePage === 'contextGraph') {
      showModeTransition(enabled ? 'assistant' : 'chat')
    }
    setAssistantMode(enabled)
    setVoiceState('idle')
    if (enabled) assistantConversation.reopenSession()
    if (activePage === 'contextGraph') setActivePage('chat')
  }

  const handleBackToChat = () => {
    if (assistantMode) showModeTransition('chat')
    setAssistantMode(false)
    setVoiceState('idle')
  }

  const handleShowContextGraph = () => {
    setAssistantMode(false)
    setActivePage('contextGraph')
  }

  const handleNewTask = () => {
    clearLocalHistory()
    attachments.forEach(({ previewUrl }) => previewUrl && URL.revokeObjectURL(previewUrl))
    setQuery('')
    setAttachments([])
    setAttachmentError('')
  }

  return (
    <div className="chat-shell">
      <ModeTransition transition={modeTransition} />
      <ChatSidebar
        user={user}
        isRunning={isRunning}
        researchMode={researchMode}
        activeStage={activeStage}
        activePage={activePage}
        onShowHistory={() => setHistoryOpen(true)}
        onShowContextGraph={handleShowContextGraph}
        onBackToChat={handleBackToChat}
        onNewTask={handleNewTask}
        onShowLanding={onShowLanding}
      />

      <main className={`chat-main ${assistantMode ? 'assistant-mode-shell' : ''} ${activePage === 'contextGraph' ? 'context-graph-open' : ''} ${hasUserMessages ? 'has-conversation' : 'empty-chat'}`}>
        <ChatHeader
          activePage={assistantMode ? 'assistant' : activePage}
          assistantMode={assistantMode}
          onAssistantModeChange={handleModeChange}
          researchMode={researchMode}
          observerEnabled={observer.enabled}
          observerPending={observer.isPending}
          onObserverToggle={observer.toggle}
          onResearchModeChange={handleResearchModeChange}
          isRunning={isRunning}
          onLogout={onLogout}
        />

        <div className={`chat-workspace ${assistantMode ? 'assistant-mode' : ''}`} hidden={activePage === 'contextGraph'}>
          {assistantMode ? (
            <div key="assistant-mode" className="workspace-mode-transition">
              <AssistantWorkspace
                embedded
                user={user}
                isLoading={assistantConversation.isLoading}
                isRunning={assistantConversation.isSending || assistantConversation.isProactive}
                isProactive={assistantConversation.isProactive}
                proactiveEnabled={assistantConversation.proactiveEnabled}
                onToggleProactive={assistantConversation.toggleProactive}
                assistantName={assistantConversation.assistantName}
                emotionalAssistantEnabled={assistantConversation.emotionalAssistantEnabled}
                onSavePreferences={assistantConversation.savePreferences}
                messages={assistantConversation.messages}
                error={assistantConversation.error}
                memoryTurnCount={assistantConversation.memoryTurnCount}
                query={query}
                setQuery={setQuery}
                attachments={attachments}
                attachmentError={attachmentError}
                voiceState={voiceState}
                setVoiceState={setVoiceState}
                onShowContextGraph={handleShowContextGraph}
                onShowHistory={() => setHistoryOpen(true)}
                onLogout={onLogout}
                onSubmit={(event) => {
                  event.preventDefault()
                  assistantConversation.sendMessage(query, attachments)
                  setQuery('')
                  attachments.forEach(({ previewUrl }) => previewUrl && URL.revokeObjectURL(previewUrl))
                  setAttachments([])
                  setAttachmentError('')
                }}
                onClearMemory={assistantConversation.clearMemory}
                onAddFiles={handleAddFiles}
                onRemoveFile={(id) => {
                  const attachment = attachments.find((item) => item.id === id)
                  if (attachment?.previewUrl) URL.revokeObjectURL(attachment.previewUrl)
                  setAttachments((current) => current.filter((item) => item.id !== id))
                }}
                onVoiceSubmit={(transcript) => assistantConversation.sendMessage(transcript)}
              />
            </div>
          ) : (
            <div key="chat-mode" className="workspace-mode-transition">
            {hasUserMessages ? null : (
              <div className="empty-chat-greeting" aria-label="Welcome">
                <span className="empty-chat-eyebrow"><i aria-hidden="true" /> YOUR WORKSPACE</span>
                <h1><TypingText text="What’s on your mind?" speed={38} /></h1>
                <p>Bring a question, a half-formed idea, or a task. We’ll work it through together.</p>
                <div className="chat-starter-grid" role="group" aria-label="Conversation starters">
                  {CHAT_STARTERS.map(({ title, prompt, icon }) => (
                    <button
                      className="chat-starter-card"
                      key={title}
                      type="button"
                      onClick={() => setQuery(prompt)}
                    >
                      <span className="chat-starter-icon" aria-hidden="true">{icon}</span>
                      <span className="chat-starter-copy">
                        <strong>{title}</strong>
                        <small>{prompt}</small>
                      </span>
                      <span className="chat-starter-arrow" aria-hidden="true">↗</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className={`chat-conversation-layout ${researchMode && stageTotal ? 'workflow-running' : ''}`}>
              {researchMode && isResearchRun && isRunning && activeStage ? (
                <div className="research-active-step" role="status" aria-live="polite">
                  <span className="research-active-step-dot" aria-hidden="true" />
                  <span>{stageLabels[activeStage] ?? activeStage.replaceAll('_', ' ')}</span>
                </div>
              ) : null}
              <ChatMessages messages={hasUserMessages ? messages : []} error={error} />
            </div>

            <ChatComposer
              query={query}
              setQuery={setQuery}
              attachments={attachments}
              attachmentError={attachmentError}
              isRunning={isRunning}
              researchMode={researchMode}
              onOpenDocumentReviewer={() => setDocumentReviewerOpen(true)}
              isActive={activePage === 'chat'}
              voiceState={voiceState}
              onVoiceStateChange={setVoiceState}
              onVoiceSubmit={handleVoiceSubmit}
              onAddFiles={handleAddFiles}
              onRemoveFile={(id) => {
                const attachment = attachments.find((item) => item.id === id)
                if (attachment?.previewUrl) URL.revokeObjectURL(attachment.previewUrl)
                setAttachments((current) => current.filter((item) => item.id !== id))
              }}
              onSubmit={handleSubmit}
            />
            </div>
          )}
        </div>

        {activePage === 'contextGraph' ? (
          <ContextGraphPage
            token={token}
            isRunning={isRunning}
            onUsePrompt={(prompt) => {
              setQuery(prompt)
              setActivePage('chat')
            }}
          />
        ) : null}
      </main>
      {documentReviewerOpen && researchMode && activePage === 'chat' ? (
        <DocumentReviewer
          token={token}
          initialDocuments={messages.flatMap((message) => message.supportingDocuments || [])}
          initialFiles={documentReviewFiles}
          initialReviewQuery={documentReviewQuery}
          autoReviewOnOpen={documentReviewFiles.length > 0}
          onClose={() => setDocumentReviewerOpen(false)}
        />
      ) : null}
      {(observer.enabled || observer.error) && activePage === 'chat' && !assistantMode ? (
        <ObserverAssistant
          result={observer.result}
          error={observer.error}
          isObserving={observer.isObserving}
          contextCards={observer.contextCards}
          onUpdateContextCard={observer.updateContextCard}
          onDeleteContextCard={observer.deleteContextCard}
        />
      ) : null}
      <ChatHistoryDrawer
        token={token}
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        onCleared={handleHistoryCleared}
        isBusy={isRunning || assistantConversation.isSending}
        onSelectPrompt={(prompt) => {
          setQuery(prompt)
          setHistoryOpen(false)
          if (activePage === 'contextGraph') setActivePage('chat')
        }}
      />
    </div>
  )
}

function App() {
  const {
    token,
    user,
    isLoading,
    login,
    register,
    authenticateWithGoogle,
    logout,
  } = useAuth()
  const [showLanding, setShowLanding] = useState(true)
  const [showAuthenticatedLanding, setShowAuthenticatedLanding] = useState(false)

  if (isLoading) {
    return <main className="auth-loading" aria-label="Loading account">Checking your session...</main>
  }
  if (!user) {
    if (showLanding) {
      return (
        <LandingPage
          onGetStarted={() => setShowLanding(false)}
          onSignIn={() => setShowLanding(false)}
        />
      )
    }
    return (
      <AuthScreen
        onLogin={login}
        onRegister={register}
        onGoogleSignIn={authenticateWithGoogle}
      />
    )
  }

  if (showAuthenticatedLanding) {
    return (
      <LandingPage
        onGetStarted={() => setShowAuthenticatedLanding(false)}
        onSignIn={() => setShowAuthenticatedLanding(false)}
      />
    )
  }

  return (
    <AuthenticatedWorkspace
      token={token}
      user={user}
      onLogout={logout}
      onShowLanding={() => setShowAuthenticatedLanding(true)}
    />
  )
}

export default App
