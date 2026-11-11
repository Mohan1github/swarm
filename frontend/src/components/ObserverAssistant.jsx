import { useState } from 'react'

function ObserverContextCard({ card, onUpdate, onDelete }) {
  const [isEditing, setIsEditing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [draft, setDraft] = useState(() => ({
    topic: card.topic,
    goal: card.goal,
    decisions: card.decisions.join('\n'),
    open_questions: card.open_questions.join('\n'),
    next_step: card.next_step,
  }))

  const save = async (event) => {
    event.preventDefault()
    setIsSaving(true)
    setSaveError('')
    try {
      await onUpdate(card.id, {
        topic: draft.topic,
        goal: draft.goal,
        decisions: draft.decisions.split('\n').map((item) => item.trim()).filter(Boolean).slice(0, 5),
        open_questions: draft.open_questions.split('\n').map((item) => item.trim()).filter(Boolean).slice(0, 5),
        next_step: draft.next_step,
      })
      setIsEditing(false)
    } catch (error) {
      setSaveError(error.message || 'This context could not be saved.')
    } finally {
      setIsSaving(false)
    }
  }

  const togglePinned = async () => {
    setSaveError('')
    try {
      await onUpdate(card.id, { pinned: !card.pinned })
    } catch (error) {
      setSaveError(error.message || 'Pin status could not be changed.')
    }
  }

  const deleteCard = async () => {
    setIsSaving(true)
    setSaveError('')
    try {
      await onDelete(card.id)
    } catch (error) {
      setSaveError(error.message || 'This context could not be deleted.')
      setIsSaving(false)
    }
  }

  const updateDraft = (field) => (event) => {
    setDraft((current) => ({ ...current, [field]: event.target.value }))
  }

  return (
    <article className="observer-thread-card">
      <header className="observer-thread-card-header">
        <div>
          <span className="observer-thread-card-kicker">{card.pinned ? 'PINNED CONTEXT' : 'CONTEXT THREAD'}</span>
          <h3>{card.topic}</h3>
        </div>
        <div className="observer-thread-actions">
          <button type="button" onClick={togglePinned} aria-label={card.pinned ? 'Unpin context' : 'Pin context'}>
            {card.pinned ? 'Unpin' : 'Pin'}
          </button>
          <button type="button" onClick={() => setIsEditing((editing) => !editing)}>
            {isEditing ? 'Cancel' : 'Edit'}
          </button>
          {confirmDelete ? (
            <>
              <button type="button" onClick={deleteCard} disabled={isSaving}>Confirm</button>
              <button type="button" onClick={() => setConfirmDelete(false)} disabled={isSaving}>Keep</button>
            </>
          ) : (
            <button type="button" onClick={() => setConfirmDelete(true)}>Delete</button>
          )}
        </div>
      </header>

      {isEditing ? (
        <form className="observer-thread-edit" onSubmit={save}>
          <label>
            Topic
            <input maxLength={180} value={draft.topic} onChange={updateDraft('topic')} required />
          </label>
          <label>
            Goal
            <textarea maxLength={600} rows={2} value={draft.goal} onChange={updateDraft('goal')} />
          </label>
          <label>
            Decisions (one per line)
            <textarea maxLength={1500} rows={3} value={draft.decisions} onChange={updateDraft('decisions')} />
          </label>
          <label>
            Open questions (one per line)
            <textarea maxLength={1500} rows={3} value={draft.open_questions} onChange={updateDraft('open_questions')} />
          </label>
          <label>
            Next step
            <textarea maxLength={500} rows={2} value={draft.next_step} onChange={updateDraft('next_step')} />
          </label>
          <button type="submit" disabled={isSaving}>{isSaving ? 'Saving…' : 'Save context'}</button>
        </form>
      ) : (
        <>
          {card.goal ? <p className="observer-thread-goal">{card.goal}</p> : null}
          {card.decisions.length ? (
            <div className="observer-thread-list">
              <strong>Decisions</strong>
              <ul>{card.decisions.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
            </div>
          ) : null}
          {card.open_questions.length ? (
            <div className="observer-thread-list">
              <strong>Open questions</strong>
              <ul>{card.open_questions.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
            </div>
          ) : null}
          {card.next_step ? (
            <div className="observer-thread-next"><strong>Possible next step</strong><p>{card.next_step}</p></div>
          ) : null}
        </>
      )}

      {card.source_excerpt ? (
        <details className="observer-thread-source">
          <summary>Why this was saved</summary>
          <p>{card.source_excerpt}</p>
        </details>
      ) : null}
      <footer className="observer-thread-meta">
        <span>{Math.round(card.confidence * 100)}% extraction confidence · Review and correct this summary</span>
        <time dateTime={card.updated_at} title={`Expires ${new Date(`${card.expires_at.replace(' ', 'T')}Z`).toLocaleDateString()} unless pinned`}>
          {new Date(`${card.updated_at.replace(' ', 'T')}Z`).toLocaleDateString()}
        </time>
      </footer>
      {saveError ? <p className="observer-error" role="alert">{saveError}</p> : null}
    </article>
  )
}

export default function ObserverAssistant({
  result,
  error,
  isObserving,
  contextCards = [],
  onUpdateContextCard,
  onDeleteContextCard,
}) {
  return (
    <aside className="observer-assistant" aria-label="Observer session insights">
      <header className="observer-assistant-header">
        <span className={`observer-signal ${isObserving ? 'active' : ''}`} aria-hidden="true">
          <i />
        </span>
        <div>
          <span className="observer-kicker">SESSION OBSERVER</span>
          <strong>{isObserving ? 'Connecting the dots' : 'Here with your context'}</strong>
        </div>
      </header>

      {error ? (
        <p className="observer-error" role="status">{error}</p>
      ) : (
        <>
          {result?.insight ? (
            <section className="observer-insight">
              <span>Worth noticing</span>
              <p>{result.insight}</p>
            </section>
          ) : (
            <p className="observer-empty">
              {isObserving
                ? 'I’m finding the useful thread in this conversation.'
                : 'I’ll quietly track the useful context as we talk.'}
            </p>
          )}
          {result?.next_idea ? (
            <section className="observer-next-idea">
              <span>One next thought</span>
              <p>{result.next_idea}</p>
            </section>
          ) : null}
          {result?.context_summary ? (
            <details className="observer-context">
              <summary>Session context</summary>
              <p>{result.context_summary}</p>
            </details>
          ) : null}
          <section className="observer-threadkeeper" aria-labelledby="observer-threadkeeper-title">
            <header>
              <div>
                <span>REVIEWABLE MEMORY</span>
                <h2 id="observer-threadkeeper-title">Context threads</h2>
              </div>
              <small>Memory stored locally · Chat text is processed by your configured AI provider</small>
            </header>
            {contextCards.length ? (
              <div className="observer-thread-list-cards">
                {contextCards.map((card) => (
                  <ObserverContextCard
                    key={card.id}
                    card={card}
                    onUpdate={onUpdateContextCard}
                    onDelete={onDeleteContextCard}
                  />
                ))}
              </div>
            ) : (
              <p className="observer-thread-empty">
                As the conversation develops, Observer can collect topics, goals, decisions, open questions, and possible next steps here. You can edit, pin, or delete each thread.
              </p>
            )}
          </section>
          {result?.user_profile ? (
            <details className="observer-context">
              <summary>How you like to work</summary>
              <p>{result.user_profile}</p>
            </details>
          ) : null}
          {result?.research_sources?.length ? (
            <nav className="observer-sources" aria-label="Observer research sources">
              <span>Related reading</span>
              {result.research_sources.map((source) => (
                <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
                  {source.title}
                </a>
              ))}
            </nav>
          ) : null}
          {result?.research_error ? (
            <p className="observer-research-note">{result.research_error}</p>
          ) : null}
        </>
      )}
    </aside>
  )
}
