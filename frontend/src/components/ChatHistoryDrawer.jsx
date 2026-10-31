import { useEffect, useMemo, useState } from 'react'

function formatTimestamp(value) {
  if (!value) return 'Time unavailable'
  const normalized = value.includes('T') ? value : value.replace(' ', 'T')
  const date = new Date(/[zZ]|[+-]\d{2}:\d{2}$/.test(normalized) ? normalized : `${normalized}Z`)
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
}

function getRunDetails(item) {
  const details = []
  const stageCount = item.insights?.steps?.length || 0
  if (stageCount) details.push(`${stageCount} ${stageCount === 1 ? 'stage' : 'stages'}`)
  if (item.attachment_count) {
    details.push(`${item.attachment_count} ${item.attachment_count === 1 ? 'file' : 'files'}`)
  }
  if (typeof item.insights?.validation_score === 'number') {
    details.push(`validation ${Math.round(item.insights.validation_score * 100)}%`)
  }
  details.push(item.cache_hit ? 'cached' : 'live run')
  return details
}

export default function ChatHistoryDrawer({
  token,
  open,
  onClose,
  onSelectPrompt,
  onCleared,
  isBusy = false,
}) {
  const [history, setHistory] = useState({
    items: [],
    swarmTotal: 0,
    graphLoaded: false,
    assistantLoaded: false,
    error: '',
  })
  const historyTotal = history.swarmTotal + history.items.filter((item) => item.kind === 'assistant').length
  const [search, setSearch] = useState('')
  const [isClearing, setIsClearing] = useState(false)
  const isLoading = open && (!history.graphLoaded || !history.assistantLoaded)

  const clearHistory = async () => {
    if (isClearing || isBusy) return
    const confirmed = window.confirm(
      'Clear all Swarm and Assistant history, including saved responses and memory? This cannot be undone.',
    )
    if (!confirmed) return

    setIsClearing(true)
    setHistory((current) => ({ ...current, error: '' }))
    try {
      const response = await fetch('/api/history', {
        method: 'DELETE',
        headers: { Authorization: 'Bearer ' + token },
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.detail || 'Could not clear history.')
      setHistory({
        items: [],
        swarmTotal: 0,
        graphLoaded: true,
        assistantLoaded: true,
        error: '',
      })
      setSearch('')
      onCleared?.()
    } catch (clearError) {
      setHistory((current) => ({
        ...current,
        error: clearError.message || 'Could not clear history.',
      }))
    } finally {
      setIsClearing(false)
    }
  }

  useEffect(() => {
    if (!open) return undefined

    const controller = new AbortController()
    const requestOptions = {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    }

    fetch('/api/activity/graph', requestOptions)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(payload.detail || 'Could not load Swarm prompt history.')
        return payload
      })
      .then((graph) => {
        const swarmItems = (graph.nodes || [])
          .filter((node) => node.kind === 'query')
          .map((node) => ({
            id: `swarm-${node.id}`,
            kind: 'swarm',
            query: node.query || '',
            created_at: node.created_at,
            cache_hit: Boolean(node.cache_hit),
            attachment_count: node.attachment_count || 0,
            insights: node.insights || {},
          }))
        setHistory((current) => ({
          ...current,
          items: [...current.items.filter((item) => item.kind !== 'swarm'), ...swarmItems],
          swarmTotal: graph.total || 0,
          graphLoaded: true,
        }))
      })
      .catch((historyError) => {
        if (historyError.name !== 'AbortError') {
          setHistory((current) => ({
            ...current,
            graphLoaded: true,
            error: current.error || historyError.message || 'Could not load Swarm prompt history.',
          }))
        }
      })

    fetch('/api/assistant/history', requestOptions)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(payload.detail || 'Could not load assistant conversation history.')
        return payload
      })
      .then((payload) => {
        const turns = payload.turns || []
        const assistantItems = []
        for (let index = 0; index + 1 < turns.length; index += 2) {
          const userTurn = turns[index]
          const assistantTurn = turns[index + 1]
          if (userTurn.role !== 'user' || assistantTurn.role !== 'assistant') continue
          assistantItems.push({
            id: `assistant-${index}-${userTurn.created_at || ''}`,
            kind: 'assistant',
            query: userTurn.content || '',
            answer: assistantTurn.content || '',
            created_at: userTurn.created_at,
            cache_hit: false,
            attachment_count: 0,
            insights: {},
          })
        }
        setHistory((current) => ({
          ...current,
          items: [...current.items.filter((item) => item.kind !== 'assistant'), ...assistantItems],
          assistantLoaded: true,
        }))
      })
      .catch((historyError) => {
        if (historyError.name !== 'AbortError') {
          setHistory((current) => ({
            ...current,
            assistantLoaded: true,
            error: current.error || historyError.message || 'Could not load assistant conversation history.',
          }))
        }
      })

    return () => controller.abort()
  }, [token, open])

  useEffect(() => {
    if (!open) return undefined
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open, onClose])

  const filteredItems = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase()
    const items = [...history.items].sort((left, right) => (
      new Date(right.created_at || 0).getTime() - new Date(left.created_at || 0).getTime()
    ))
    if (!normalizedSearch) return items
    return items.filter((item) => (
      item.query.toLowerCase().includes(normalizedSearch)
      || (item.answer || '').toLowerCase().includes(normalizedSearch)
    ))
  }, [history.items, search])

  if (!open) return null

  return (
    <div className="history-overlay" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose()
    }}>
      <aside className="history-drawer" role="dialog" aria-modal="true" aria-labelledby="history-title">
        <header className="history-header">
          <div>
            <span className="history-eyebrow">YOUR WORKSPACE</span>
            <h2 id="history-title">Prompt history</h2>
            <p>Recent prompts from your Swarm activity.</p>
          </div>
          <button type="button" className="history-close" onClick={onClose} aria-label="Close prompt history">
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="m5 5 10 10M15 5 5 15" />
            </svg>
          </button>
        </header>

        <div className="history-count-row">
          <span>{historyTotal} saved {historyTotal === 1 ? 'conversation' : 'conversations'}</span>
          <div className="history-count-actions">
            <span>Swarm + Assistant</span>
            <button
              type="button"
              className="history-clear-button"
              onClick={clearHistory}
              disabled={isLoading || isClearing || isBusy}
            >
              {isClearing ? 'Clearing...' : 'Clear history'}
            </button>
          </div>
        </div>

        <label className="history-search">
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <circle cx="8.5" cy="8.5" r="5.5" />
            <path d="m13 13 4 4" />
          </svg>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search your prompts"
            aria-label="Search prompt history"
          />
        </label>

        <div className="history-list" aria-live="polite">
          {history.error && history.items.length > 0 ? (
            <div className="history-partial-error" role="status">{history.error}</div>
          ) : null}
          {isLoading ? (
            <div className="history-state"><span className="activity-loading" />Loading your history...</div>
          ) : history.error && history.items.length === 0 ? (
            <div className="history-state history-error" role="alert">{history.error}</div>
          ) : filteredItems.length === 0 ? (
            <div className="history-state">
              {history.items.length ? 'No conversations match your search.' : 'Your conversations will appear here as you use the assistant and Swarm.'}
            </div>
          ) : filteredItems.map((item) => {
            const details = getRunDetails(item)
            const answer = item.insights?.answer
            return (
              <article className="history-entry" key={item.id}>
                <span className={`history-kind ${item.kind}`}>
                  {item.kind === 'assistant' ? 'Assistant' : 'Swarm'}
                </span>
                <time className="history-time" dateTime={item.created_at}>{formatTimestamp(item.created_at)}</time>
                <p className="history-prompt">{item.query || 'Conversation prompt unavailable.'}</p>
                <div className="history-meta">
                  {item.kind === 'swarm'
                    ? details.map((detail) => <span key={detail}>{detail}</span>)
                    : <span>Conversation</span>}
                </div>
                {item.kind === 'assistant' && item.answer ? (
                  <p className="history-answer"><strong>Response</strong>{item.answer}</p>
                ) : answer ? (
                  <p className="history-answer"><strong>Response</strong>{answer}</p>
                ) : (
                  <p className="history-answer unavailable">Response preview unavailable for this older prompt.</p>
                )}
                <button
                  type="button"
                  className="history-use-button"
                  onClick={() => onSelectPrompt(item.query)}
                  title="Place this prompt in the composer"
                >
                  Use prompt again
                  <svg viewBox="0 0 20 20" aria-hidden="true">
                    <path d="M4 10h11m-4-4 4 4-4 4" />
                  </svg>
                </button>
              </article>
            )
          })}
        </div>
      </aside>
    </div>
  )
}
