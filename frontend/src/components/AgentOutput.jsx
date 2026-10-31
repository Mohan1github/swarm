import sanitizeStreamText from '../utils/sanitizeStreamText'

function parseStructuredOutput(content) {
  if (typeof content !== 'string') return null
  const text = content.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')
  if (!text.startsWith('{') || !text.endsWith('}')) return null

  try {
    const parsed = JSON.parse(text)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null
  } catch {
    return null
  }
}

function renderTextBlocks(content) {
  const lines = sanitizeStreamText(content).split('\n')
  const blocks = []
  let paragraph = []
  let list = []

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: 'paragraph', text: paragraph.join(' ') })
      paragraph = []
    }
  }
  const flushList = () => {
    if (list.length) {
      blocks.push({ type: 'list', items: list })
      list = []
    }
  }

  for (const line of lines) {
    const text = line.trim()
    if (!text) {
      flushParagraph()
      flushList()
      continue
    }

    const bullet = text.match(/^(?:•|[-*])\s+(.+)$/)
    const numbered = text.match(/^\d+[.)]\s+(.+)$/)
    if (bullet || numbered) {
      flushParagraph()
      list.push((bullet || numbered)[1])
      continue
    }

    flushList()
    paragraph.push(text)
  }
  flushParagraph()
  flushList()
  return blocks
}

function SourceCards({ sources }) {
  if (!Array.isArray(sources) || !sources.length) return null

  return (
    <section className="agent-source-output" aria-label="Retrieved web sources">
      {sources.map((source, index) => (
        <article className="agent-source-card" key={`${source.url}-${index}`}>
          <div className="agent-source-index">Source {index + 1}</div>
          <a href={source.url} target="_blank" rel="noreferrer">{source.title || source.url}</a>
          <div className="agent-source-meta">
            {source.published_date ? `Published ${source.published_date}` : 'Publication date not provided'}
            {source.similarity !== null && source.similarity !== undefined
              ? ` · Match ${Math.round(source.similarity * 100)}%`
              : ''}
          </div>
        </article>
      ))}
    </section>
  )
}

function ValidationOutput({ data, parsed }) {
  const score = data?.validation_score ?? parsed?.validation_score
  const issues = data?.issues ?? parsed?.issues
  if (score === undefined && !Array.isArray(issues)) return null

  const numericScore = Number(score)
  const hasScore = Number.isFinite(numericScore)
  return (
    <section className="agent-validation-output" aria-label="Validation assessment">
      {hasScore ? (
        <div className="agent-validation-score">
          <span>Model review</span>
          <strong>{Math.round(numericScore * 100)}%</strong>
        </div>
      ) : null}
      {Array.isArray(issues) && issues.length ? (
        <ul>{issues.map((issue, index) => <li key={`${index}-${issue}`}>{String(issue)}</li>)}</ul>
      ) : (
        <p>No issues were reported by the model reviewer.</p>
      )}
      <small>This is a model assessment, not a guarantee of factual correctness.</small>
    </section>
  )
}

function PlannerOutput({ parsed }) {
  const plan = parsed?.plan
  if (typeof plan !== 'string' || !plan.trim()) return null

  const probabilities = parsed.action_probabilities
  return (
    <section className="agent-planner-output" aria-label="Workflow plan">
      <p>{sanitizeStreamText(plan)}</p>
      {probabilities && typeof probabilities === 'object' ? (
        <div className="agent-plan-signals">
          {Object.entries(probabilities)
            .filter(([, value]) => Number.isFinite(Number(value)) && Number(value) > 0)
            .map(([action, value]) => (
              <span key={action}>{action.replaceAll('_', ' ')} {Math.round(Number(value) * 100)}%</span>
            ))}
        </div>
      ) : null}
    </section>
  )
}

export default function AgentOutput({ content, role, uiData }) {
  const parsed = parseStructuredOutput(content)
  if (role === 'web_retriever' && uiData?.sources?.length) {
    return <SourceCards sources={uiData.sources} />
  }
  if (role === 'validator') {
    const validation = <ValidationOutput data={uiData} parsed={parsed} />
    if (validation) return validation
  }
  if (role === 'planner' && parsed?.plan) {
    return <PlannerOutput parsed={parsed} />
  }

  const displayText = parsed
    ? parsed.answer ?? parsed.output ?? parsed.response ?? parsed.summary ?? parsed.content ?? parsed.result
    : content
  const blocks = renderTextBlocks(typeof displayText === 'string' ? displayText : content)

  return (
    <div className={`agent-output-content ${role ? `agent-output-${role}` : ''}`}>
      {blocks.map((block, index) => block.type === 'list' ? (
        <ul key={`list-${index}`}>
          {block.items.map((item, itemIndex) => <li key={`${itemIndex}-${item}`}>{item}</li>)}
        </ul>
      ) : (
        <p key={`paragraph-${index}`}>{block.text}</p>
      ))}
    </div>
  )
}
