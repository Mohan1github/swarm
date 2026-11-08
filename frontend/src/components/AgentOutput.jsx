import { useState } from 'react'
import sanitizeStreamText from '../utils/sanitizeStreamText'

const CODE_KEYWORDS = new Set([
  'abstract', 'async', 'await', 'break', 'case', 'catch', 'class', 'const', 'continue',
  'def', 'default', 'delete', 'do', 'else', 'enum', 'export', 'extends', 'false', 'final',
  'finally', 'for', 'from', 'function', 'if', 'implements', 'import', 'in', 'interface',
  'let', 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'static',
  'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof', 'undefined', 'using', 'var',
  'while', 'yield', 'None', 'True', 'False', 'and', 'as', 'assert', 'elif', 'except', 'is',
  'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'with',
])

const LANGUAGE_NAMES = {
  bash: 'Terminal', sh: 'Shell', css: 'CSS', html: 'HTML', java: 'Java',
  javascript: 'JavaScript', js: 'JavaScript', json: 'JSON', jsx: 'React',
  python: 'Python', py: 'Python', sql: 'SQL', ts: 'TypeScript', tsx: 'React TSX',
  yaml: 'YAML', yml: 'YAML',
}

function highlightCodeLine(line) {
  const tokenPattern = /(\/\/.*$|#(?!\[).*?$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b(?:0x[\da-f]+|\d+(?:\.\d+)?)\b|\b[A-Za-z_$][\w$]*(?=\s*\()|\b[A-Za-z_$][\w$]*\b)/gi
  const tokens = []
  let lastIndex = 0

  for (const match of line.matchAll(tokenPattern)) {
    if (match.index > lastIndex) tokens.push(['plain', line.slice(lastIndex, match.index)])
    const token = match[0]
    let type = 'plain'
    if (/^(\/\/|#)/.test(token)) type = 'comment'
    else if (/^["'`]/.test(token)) type = 'string'
    else if (/^\d/.test(token)) type = 'number'
    else if (CODE_KEYWORDS.has(token)) type = 'keyword'
    else if (line.slice(match.index + token.length).trimStart().startsWith('(')) type = 'function'
    else if (/^[A-Z][\w$]*$/.test(token)) type = 'type'
    tokens.push([type, token])
    lastIndex = match.index + token.length
  }
  if (lastIndex < line.length) tokens.push(['plain', line.slice(lastIndex)])
  return tokens.map(([type, text], index) => (
    type === 'plain' ? text : <span className={`code-token-${type}`} key={`${type}-${index}`}>{text}</span>
  ))
}

function CodeBlock({ code, language = '' }) {
  const normalizedLanguage = language.trim().toLowerCase()
  const displayLanguage = LANGUAGE_NAMES[normalizedLanguage] || normalizedLanguage || 'Code'
  const [copied, setCopied] = useState(false)

  const copyCode = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }

  return (
    <section className="chat-code-editor" aria-label={`${displayLanguage} code`}>
      <header className="chat-code-editor-header">
        <div className="chat-code-editor-dots" aria-hidden="true"><i /><i /><i /></div>
        <span>{displayLanguage}</span>
        <button type="button" onClick={copyCode}>{copied ? 'Copied' : 'Copy code'}</button>
      </header>
      <div className="chat-code-editor-body">
        <div className="chat-code-line-numbers" aria-hidden="true">
          {code.split('\n').map((_, index) => <span key={index}>{index + 1}</span>)}
        </div>
        <pre><code>{code.split('\n').map((line, index) => (
          <span className="chat-code-line" key={`${index}-${line}`}>
            {highlightCodeLine(line)}{'\n'}
          </span>
        ))}</code></pre>
      </div>
    </section>
  )
}

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
  let code = null

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
    const fence = line.trim().match(/^```([\w+-]*)\s*$/)
    if (fence) {
      if (!code) {
        flushParagraph()
        flushList()
        code = { type: 'code', language: fence[1], lines: [] }
      } else {
        blocks.push({ ...code, code: code.lines.join('\n') })
        code = null
      }
      continue
    }
    if (code) {
      code.lines.push(line)
      continue
    }
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
  if (code) blocks.push({ ...code, code: code.lines.join('\n') })
  return blocks
}

function splitSentences(text) {
  return text.match(/[^.!?]+(?:[.!?]+(?=\s|$|["')\]])|$)/g) || [text]
}

function importanceScore(sentence) {
  const text = sentence.trim()
  if (!text || /^(?:hi|hello|sure|certainly|of course|great|absolutely)[!,. ]*$/i.test(text)) return 0

  let score = 0
  if (/\b(key point|important|critical|main takeaway|in summary|in conclusion|bottom line|remember that|note that)\b/i.test(text)) score += 4
  if (/\b(should|must|need to|recommend|best approach|avoid|ensure|make sure|required|only|never)\b/i.test(text)) score += 3
  if (/\b(therefore|because|as a result|which means|leads to|results in|depends on|risk|benefit|trade-?off)\b/i.test(text)) score += 2
  if (/\b\d+(?:[.,]\d+)?%?\b/.test(text)) score += 1
  return score
}

function renderCoreHighlight(text, enabled, limit = 2) {
  if (!enabled || !text) return text

  const sentences = splitSentences(text)
  const candidates = sentences
    .map((sentence, index) => ({ index, score: importanceScore(sentence) }))
    .filter(({ score }) => score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
  const selected = new Set(candidates.slice(0, limit).map(({ index }) => index))

  if (!selected.size) {
    const firstSubstantive = sentences.findIndex((sentence) => importanceScore(sentence) >= 0
      && !/^(?:hi|hello|sure|certainly|of course|great|absolutely)[!,. ]*$/i.test(sentence.trim()))
    if (firstSubstantive >= 0) selected.add(firstSubstantive)
  }

  return sentences.map((sentence, index) => (
    selected.has(index)
      ? <mark className="agent-output-highlight" key={`core-${index}`}>{sentence}</mark>
      : <span key={`text-${index}`}>{sentence}</span>
  ))
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

function DocumentCards({ documents }) {
  if (!Array.isArray(documents) || !documents.length) return null

  return (
    <section className="agent-source-output" aria-label="Supporting documents">
      {documents.map((document, index) => (
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

export default function AgentOutput({ content, role, uiData, highlightCore = false }) {
  const parsed = parseStructuredOutput(content)
  if (role === 'web_retriever' && uiData?.sources?.length) {
    return <SourceCards sources={uiData.sources} />
  }
  if (role === 'document_retriever' && uiData?.documents?.length) {
    return <DocumentCards documents={uiData.documents} />
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
      {blocks.map((block, index) => (
        block.type === 'code' ? (
          <CodeBlock code={block.code} language={block.language} key={`code-${index}`} />
        ) : block.type === 'list' ? (
          <ul key={`list-${index}`}>
            {block.items.map((item, itemIndex) => (
              <li key={`${itemIndex}-${item}`}>
                {renderCoreHighlight(item, highlightCore, 1)}
              </li>
            ))}
          </ul>
        ) : (
          <p key={`paragraph-${index}`}>
            {renderCoreHighlight(block.text, highlightCore)}
          </p>
        )
      ))}
    </div>
  )
}
