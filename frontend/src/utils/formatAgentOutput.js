import sanitizeStreamText from './sanitizeStreamText'

function humanizeKey(key) {
  return key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .replace(/^\w/, (letter) => letter.toUpperCase())
}

function formatValue(value) {
  if (typeof value === 'string') return value.trim()
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) {
    return value.map((item) => `- ${formatValue(item)}`).filter((item) => item !== '- ').join('\n')
  }
  if (typeof value === 'object') {
    return Object.entries(value)
      .map(([key, item]) => {
        const formatted = formatValue(item)
        return formatted ? `${humanizeKey(key)}: ${formatted}` : ''
      })
      .filter(Boolean)
      .join('\n')
  }
  return String(value)
}

function parseJsonOutput(output) {
  const candidate = output.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim()
  try {
    return JSON.parse(candidate)
  } catch {
    const start = candidate.indexOf('{')
    const end = candidate.lastIndexOf('}')
    if (start < 0 || end <= start) return null

    try {
      return JSON.parse(candidate.slice(start, end + 1))
    } catch {
      return null
    }
  }
}

export default function formatAgentOutput(output, role) {
  if (typeof output !== 'string') return ''
  const text = output.trim()
  if (!text) return ''

  const parsed = parseJsonOutput(text)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return sanitizeStreamText(text)
  }

  if (role === 'planner' && typeof parsed.plan === 'string') {
    return sanitizeStreamText(parsed.plan)
  }

  if (role === 'validator' && 'validation_score' in parsed) {
    const issues = Array.isArray(parsed.issues) ? parsed.issues : []
    const summary = `Validation score: ${parsed.validation_score}.`
    return sanitizeStreamText(
      issues.length ? `${summary}\nPotential issues: ${issues.join('; ')}` : `${summary} No issues identified.`,
    )
  }

  const preferredFields = ['answer', 'output', 'response', 'summary', 'content', 'result', 'recommendation', 'analysis', 'message', 'plan']
  const preferredField = preferredFields.find((field) => typeof parsed[field] === 'string' && parsed[field].trim())
  return sanitizeStreamText(preferredField ? parsed[preferredField].trim() : formatValue(parsed))
}
