import { useCallback, useState } from 'react'
import TypingText from './TypingText'
import AgentOutput from './AgentOutput'
import StreamLoader from './StreamLoader'
import sanitizeStreamText from '../utils/sanitizeStreamText'

function formatFileSize(size) {
  if (!Number.isFinite(size) || size <= 0) return ''
  if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function EvidenceLedger({ report, error }) {
  if (!report && !error) return null

  if (error) {
    return (
      <section className="evidence-ledger evidence-ledger-warning" aria-label="Evidence ledger status">
        <strong>Evidence ledger unavailable</strong>
        <p>{error}</p>
      </section>
    )
  }

  const counts = report.claims.reduce((summary, item) => {
    summary[item.verdict] = (summary[item.verdict] || 0) + 1
    return summary
  }, {})

  return (
    <section className="evidence-ledger" aria-label="Evidence ledger">
      <header className="evidence-ledger-header">
        <div>
          <strong>Evidence ledger</strong>
          <span>Source-based assessment, not a guarantee of truth</span>
        </div>
        <div className="evidence-ledger-counts" aria-label="Claim assessment counts">
          {Object.entries(counts).map(([verdict, count]) => (
            <span className={`evidence-verdict-count ${verdict}`} key={verdict}>
              {count} {verdict.replaceAll('_', ' ')}
            </span>
          ))}
        </div>
      </header>
      <ol className="evidence-claims">
        {report.claims.map((item, index) => (
          <li className="evidence-claim" key={`${index}-${item.claim}`}>
            <div className="evidence-claim-heading">
              <span className={`evidence-verdict ${item.verdict}`}>
                {item.verdict.replaceAll('_', ' ')}
              </span>
              <span>{item.claim}</span>
            </div>
            {item.evidence.length ? (
              <ul className="evidence-citations">
                {item.evidence.map((citation, citationIndex) => (
                  <li key={`${citation.url}-${citationIndex}`}>
                    <a href={citation.url} target="_blank" rel="noreferrer">
                      {citation.title}
                    </a>
                    <blockquote>{citation.quote}</blockquote>
                    <span className="evidence-source-date">
                      {citation.published_date
                        ? `Published ${citation.published_date}`
                        : 'Publication date not provided'}
                      {citation.fetched_at ? ` · Retrieved ${citation.fetched_at}` : ''}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="evidence-no-citation">No matching source passage was verified.</p>
            )}
          </li>
        ))}
      </ol>
    </section>
  )
}

function ImageSourceGallery({ sources, error }) {
  if ((!Array.isArray(sources) || !sources.length) && !error) return null

  return (
    <section className="image-source-gallery" aria-label="Image search results">
      <header className="image-source-gallery-heading">
        <strong>Images from the web</strong>
        <span>Open each source for attribution and usage details.</span>
      </header>
      {error ? <p className="image-source-error" role="status">{error}</p> : null}
      {sources?.length ? (
        <div className="image-source-grid">
          {sources.map((source, index) => (
            <a
              className="image-source-card"
              href={source.source_url}
              key={`${source.image_url}-${index}`}
              target="_blank"
              rel="noreferrer"
              aria-label={`Open image source ${index + 1}${source.description ? `: ${source.description}` : ''}`}
            >
              <img
                src={source.image_url}
                alt={source.description || `Image result ${index + 1}`}
                loading="lazy"
                referrerPolicy="no-referrer"
                onError={(event) => {
                  event.currentTarget.hidden = true
                }}
              />
              <span>{source.description || `Image source ${index + 1}`}</span>
            </a>
          ))}
        </div>
      ) : null}
    </section>
  )
}

function SupportingDocumentList({ documents, error }) {
  if ((!Array.isArray(documents) || !documents.length) && !error) return null

  return (
    <section className="agent-source-output" aria-label="Supporting documents">
      {error ? <p className="image-source-error" role="status">{error}</p> : null}
      {documents?.map((document, index) => (
        <article className="agent-source-card" key={`${document.url}-${index}`}>
          <div className="agent-source-index">
            {document.file_type || 'Document'} {index + 1}
          </div>
          <a href={document.url} target="_blank" rel="noreferrer">
            {document.title || document.url}
          </a>
          <div className="agent-source-meta">
            {document.pages ? `${document.pages} pages · ` : ''}
            Source page: <a href={document.source_url} target="_blank" rel="noreferrer">
              {document.source_url}
            </a>
          </div>
          {document.excerpt ? <p>{document.excerpt}</p> : null}
        </article>
      ))}
    </section>
  )
}

async function copyTextToClipboard(text) {
  if (!text) return false
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text)
    return true
  }

  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.setAttribute('readonly', '')
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()
  try {
    document.execCommand('copy')
    return true
  } catch {
    return false
  } finally {
    document.body.removeChild(textarea)
  }
}

export default function ChatMessages({ messages, error, onAssistantProgress }) {
  const [nextAssistantIndex, setNextAssistantIndex] = useState(0)
  const [copiedMessageId, setCopiedMessageId] = useState(null)
  const [highlightedMessages, setHighlightedMessages] = useState(() => new Set())
  const completeTyping = useCallback((index) => {
    setNextAssistantIndex(index + 1)
  }, [])
  const activeAssistantIndex = messages.findIndex(
    (message, index) => index >= nextAssistantIndex && message.role === 'assistant' && !message.instant,
  )
  const visibleMessages = messages
    .map((message, index) => ({ message, index }))
    .filter(({ message, index }) => (
      message.role === 'user' || message.instant || index < nextAssistantIndex || index === activeAssistantIndex
    ))

  const handleCopy = async (message, index) => {
    const rawContent = typeof message.content === 'string' ? message.content : ''
    const content = message.role === 'assistant' ? sanitizeStreamText(rawContent) : rawContent
    const didCopy = await copyTextToClipboard(content)
    if (didCopy) {
      setCopiedMessageId(`${message.role}-${index}`)
      window.setTimeout(() => setCopiedMessageId((current) => (current === `${message.role}-${index}` ? null : current)), 1200)
    }
  }

  const handleShare = async (message, index) => {
    const rawContent = typeof message.content === 'string' ? message.content : ''
    const content = message.role === 'assistant' ? sanitizeStreamText(rawContent) : rawContent
    if (!content) return

    if (navigator.share) {
      try {
        await navigator.share({
          title: message.role === 'assistant' ? 'Assistant response' : 'Chat message',
          text: content,
        })
        return
      } catch {
        // Fall through to clipboard copy when share is cancelled or unsupported.
      }
    }

    const didCopy = await copyTextToClipboard(content)
    if (didCopy) {
      setCopiedMessageId(`${message.role}-${index}`)
      window.setTimeout(() => setCopiedMessageId((current) => (current === `${message.role}-${index}` ? null : current)), 1200)
    }
  }

  return (
    <div className="messages-panel">
      {visibleMessages.map(({ message, index }) => {
        const messageKey = `${message.role}-${index}`
        const isCopied = copiedMessageId === messageKey
        const isAssistantStreaming = message.role === 'assistant'
          && !message.instant
          && !message.loading
          && index === activeAssistantIndex
        const contentKey = `${messageKey}-${String(message.content ?? '').length}-${String(message.content ?? '').slice(0, 24)}-${String(message.content ?? '').slice(-24)}`
        const isHighlighted = highlightedMessages.has(contentKey)
        const isAnswerReady = message.role === 'assistant'
          && !message.loading
          && !message.streaming
          && (message.instant || index < nextAssistantIndex)
        const content = typeof message.content === 'string' && message.role === 'assistant'
          ? sanitizeStreamText(message.content)
          : message.content
        return (
          <div key={messageKey} className={`message-row ${message.role}`}>
            <div className="bubble-wrap">
              <div className="bubble">
                {message.agent ? <div className="agent-name">{message.agent}</div> : null}
                {message.mostImportant ? (
                  <section className="most-important-takeaway" aria-label="Most important takeaway">
                    <div className="most-important-label">Most important</div>
                    <p>{message.mostImportant}</p>
                  </section>
                ) : null}
                {message.attachments?.length ? (
                  <div className="message-attachments">
                    {message.attachments.map((attachment) => (
                      <div className="message-attachment" key={`${attachment.name}-${attachment.size}`}>
                        {attachment.preview_url ? (
                          <a href={attachment.preview_url} target="_blank" rel="noreferrer">
                            <img src={attachment.preview_url} alt={attachment.name} className="message-attachment-image" />
                          </a>
                        ) : (
                          <span className="message-attachment-icon" aria-hidden="true">FILE</span>
                        )}
                        <span className="message-attachment-name" title={attachment.name}>{attachment.name}</span>
                        {attachment.camera_snapshot ? (
                          <span className="message-camera-captured">Camera frame captured</span>
                        ) : null}
                        <span className="message-attachment-size">{formatFileSize(attachment.size)}</span>
                      </div>
                    ))}
                  </div>
                ) : null}
                {message.streaming ? (
                  <AgentOutput content={content} role={message.agentRole} uiData={message.uiData} />
                ) : message.loading ? (
                  <StreamLoader />
                ) : isAssistantStreaming ? (
                  <TypingText
                    text={content}
                    speed={700}
                    onComplete={completeTyping}
                    completionValue={index}
                    onProgress={onAssistantProgress}
                  />
                ) : (
                  <AgentOutput
                    content={message.content}
                    role={message.agentRole}
                    uiData={message.uiData}
                    highlightCore={isHighlighted}
                  />
                )}
                {message.role === 'assistant' ? (
                  <ImageSourceGallery
                    sources={message.imageSources}
                    error={message.imageSearchError}
                  />
                ) : null}
                {message.role === 'assistant' ? (
                  <SupportingDocumentList
                    documents={message.supportingDocuments}
                    error={message.supportingDocumentError}
                  />
                ) : null}
              </div>

              {message.role === 'assistant' ? (
                <EvidenceLedger report={message.evidenceLedger} error={message.evidenceLedgerError} />
              ) : null}

              {!message.instant || isAnswerReady ? (
                <div className="message-actions" aria-label={`Actions for ${message.role} message`}>
                  {!message.instant ? (
                    <>
                      <button
                        type="button"
                        className="message-action"
                        onClick={() => handleCopy(message, index)}
                        aria-label={isCopied ? 'Copied to clipboard' : 'Copy message'}
                        title={isCopied ? 'Copied' : 'Copy'}
                      >
                        {isCopied ? (
                          <svg viewBox="0 0 20 20" aria-hidden="true">
                            <path d="m4 10 4 4 8-8" />
                          </svg>
                        ) : (
                          <svg viewBox="0 0 20 20" aria-hidden="true">
                            <rect x="7" y="6" width="9" height="11" rx="1.5" />
                            <path d="M12 6V4.5A1.5 1.5 0 0 0 10.5 3h-6A1.5 1.5 0 0 0 3 4.5v8A1.5 1.5 0 0 0 4.5 14H7" />
                          </svg>
                        )}
                      </button>
                      <button
                        type="button"
                        className="message-action"
                        onClick={() => handleShare(message, index)}
                        aria-label="Share message"
                        title="Share"
                      >
                        <svg viewBox="0 0 20 20" aria-hidden="true">
                          <circle cx="15" cy="4" r="2" />
                          <circle cx="5" cy="10" r="2" />
                          <circle cx="15" cy="16" r="2" />
                          <path d="m6.8 9 6.4-4M6.8 11l6.4 4" />
                        </svg>
                      </button>
                    </>
                  ) : null}
                  {isAnswerReady ? (
                    <button
                      type="button"
                      className={`message-action message-highlight-action${isHighlighted ? ' active' : ''}`}
                      onClick={() => setHighlightedMessages((previous) => {
                        const next = new Set(previous)
                        if (next.has(contentKey)) next.delete(contentKey)
                        else next.add(contentKey)
                        return next
                      })}
                      aria-pressed={isHighlighted}
                      aria-label={isHighlighted ? 'Hide key point highlights' : 'Highlight key points'}
                      title={isHighlighted ? 'Hide key point highlights' : 'Highlight key points'}
                    >
                      <svg viewBox="0 0 20 20" aria-hidden="true">
                        <path d="m10 2 2.2 5.1 5.5.5-4.2 3.6 1.3 5.4-4.8-2.9-4.8 2.9 1.3-5.4L2.3 7.6l5.5-.5L10 2Z" />
                      </svg>
                      <span>{isHighlighted ? 'Hide highlights' : 'Highlight'}</span>
                    </button>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        )
      })}

      {error ? <div className="error-box">{error}</div> : null}
    </div>
  )
}
