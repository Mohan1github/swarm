export default function ObserverAssistant({ result, error, isObserving }) {
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
