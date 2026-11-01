import { useCallback, useState } from 'react'
import TypingText from './TypingText'
import AgentOutput from './AgentOutput'
import StreamLoader from './StreamLoader'
import sanitizeStreamText from '../utils/sanitizeStreamText'

function formatFileSize(size) {
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
        const content = typeof message.content === 'string' && message.role === 'assistant'
          ? sanitizeStreamText(message.content)
          : message.content
        return (
          <div key={messageKey} className={`message-row ${message.role}`}>
            <div className="bubble-wrap">
              <div className="bubble">
                {message.agent ? <div className="agent-name">{message.agent}</div> : null}
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
                  />
                )}
              </div>

              {message.role === 'assistant' ? (
                <EvidenceLedger report={message.evidenceLedger} error={message.evidenceLedgerError} />
              ) : null}

              {!message.instant ? (
                <div className="message-actions" aria-label={`Actions for ${message.role} message`}>
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
