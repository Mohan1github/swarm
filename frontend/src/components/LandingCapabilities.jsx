const capabilities = [
  {
    id: 'research',
    category: 'Research',
    title: 'Go beyond a quick search.',
    description: 'Explore public information with focused queries, iterative retrieval, and evidence-aware synthesis.',
    icon: <><circle cx="10" cy="10" r="6.5" /><path d="m14.7 14.7 3 3M7 10h6M10 7v6" /></>,
    className: 'landing-capability-research',
  },
  {
    id: 'assistant',
    category: 'Assistant',
    title: 'Keep the conversation moving.',
    description: 'Work through follow-ups naturally with an assistant that uses relevant conversation context.',
    icon: <><path d="M3 4.5h14v10H9l-4.5 3v-3H3z" /><path d="M6.5 8h7M6.5 11h4" /></>,
    className: '',
  },
  {
    id: 'documents',
    category: 'Documents',
    title: 'Bring your own material.',
    description: 'Review supporting documents and connect their contents to the question you are investigating.',
    icon: <><path d="M5 2.5h7l4 4v11H5z" /><path d="M12 2.5v4h4M8 10h5M8 13h5" /></>,
    className: '',
  },
  {
    id: 'context',
    category: 'Context graph',
    title: 'See the bigger picture.',
    description: 'Explore relationships between people, ideas, decisions, and the information in your workspace.',
    icon: <><circle cx="5" cy="10" r="2" /><circle cx="15" cy="5" r="2" /><circle cx="15" cy="15" r="2" /><path d="m7 9 6-3M7 11l6 3M15 7v6" /></>,
    className: '',
  },
  {
    id: 'code',
    category: 'Code',
    title: 'Read technical answers clearly.',
    description: 'Code responses are presented in a structured, syntax-highlighted editor-style view.',
    icon: <><path d="m7 6-4 4 4 4M13 6l4 4-4 4M11 4l-2 12" /></>,
    className: '',
  },
]

export default function LandingCapabilities() {
  return (
    <section className="landing-capabilities" id="capabilities" aria-labelledby="landing-capabilities-title" data-landing-reveal>
      <div className="landing-section-heading">
        <div>
          <span className="landing-section-kicker">One connected workspace</span>
          <h2 id="landing-capabilities-title">Built for how real work unfolds.</h2>
        </div>
        <p>Move between research, conversation, documents, and technical work without losing your focus.</p>
      </div>
      <div className="landing-capability-grid">
        {capabilities.map((capability) => (
          <article
            className={`landing-capability-card ${capability.className}`}
            key={capability.id}
          >
            <div className="landing-capability-top">
              <span>{capability.category}</span>
              <svg viewBox="0 0 20 20" aria-hidden="true">{capability.icon}</svg>
            </div>
            <h3>{capability.title}</h3>
            <p>{capability.description}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
