function isTableSeparator(line) {
  const cells = line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
  return cells.length > 1 && cells.every((cell) => /^\s*:?-{3,}:?\s*$/.test(cell))
}

function plainTextPipeRows(value) {
  const lines = value.split('\n')
  const output = []

  for (let index = 0; index < lines.length;) {
    if (!lines[index].includes('|')) {
      output.push(lines[index])
      index += 1
      continue
    }

    const rows = []
    while (index < lines.length && lines[index].includes('|')) {
      rows.push(lines[index]
        .trim()
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((cell) => cell.trim()))
      index += 1
    }

    if (rows.length >= 2 && isTableSeparator(lines[index - rows.length + 1])) {
      const headers = rows[0]
      const dataRows = rows.slice(2)
      for (const row of dataRows) {
        output.push(row
          .map((cell, cellIndex) => {
            const header = headers[cellIndex]
            return header ? `${header}: ${cell}` : cell
          })
          .filter(Boolean)
          .join('; '))
      }
    } else {
      output.push(...rows.map((row) => row.filter(Boolean).join(' ')))
    }
  }

  return output.join('\n')
}

function decodeTextEntities(value) {
  const namedEntities = {
    '&amp;': '&',
    '&apos;': "'",
    '&gt;': '>',
    '&lt;': '<',
    '&nbsp;': ' ',
    '&quot;': '"',
  }
  return value
    .replace(/&(amp|apos|gt|lt|nbsp|quot);/gi, (entity) => namedEntities[entity.toLowerCase()] ?? entity)
    .replace(/&#(x[\da-f]+|\d+);/gi, (entity, code) => {
      const radix = code[0].toLowerCase() === 'x' ? 16 : 10
      const numericCode = radix === 16 ? code.slice(1) : code
      const point = Number.parseInt(numericCode, radix)
      return Number.isFinite(point) && point > 0 && point <= 0x10ffff
        ? String.fromCodePoint(point)
        : entity
    })
}

function readableStructuredValue(value, label = '') {
    if (typeof value === 'string') {
      const text = value.trim()
      return text ? `${label ? `${label}: ` : ''}${text}` : ''
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return `${label ? `${label}: ` : ''}${value}`
    }
    if (Array.isArray(value)) {
      return value
        .map((item) => readableStructuredValue(item))
        .filter(Boolean)
        .map((item) => `• ${item}`)
        .join('\n')
    }
    if (value && typeof value === 'object') {
      const preferred = ['answer', 'output', 'response', 'summary', 'content', 'result', 'message']
      const preferredKey = preferred.find((key) => typeof value[key] === 'string' && value[key].trim())
      if (preferredKey) return readableStructuredValue(value[preferredKey], label)

      return Object.entries(value)
        .map(([key, item]) => {
          const readableKey = key
            .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
            .replace(/[_-]+/g, ' ')
            .replace(/^\w/, (letter) => letter.toUpperCase())
          return readableStructuredValue(item, readableKey)
        })
        .filter(Boolean)
        .join('\n')
    }
    return ''
}

function extractStructuredText(value) {
    const trimmed = value
      .trim()
      .replace(/^```(?:json)?\s*/i, '')
      .replace(/\s*```$/, '')
      .trim()
    if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return value
    try {
      const parsed = JSON.parse(trimmed)
      if (!parsed || typeof parsed !== 'object') return value
      return readableStructuredValue(parsed) || value
    } catch {
      return value
    }
}

export default function sanitizeStreamText(value) {
    if (typeof value !== 'string') return ''

    const structuredText = extractStructuredText(value)
    const normalized = structuredText
      .replace(/\r\n?/g, '\n')
      .replace(/^\s*```[^\n]*\n?/gm, '')
      .replace(/```/g, '')
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<\/(?:p|div|li|h[1-6])\s*>/gi, '\n')
      .replace(/<[^>]*>/g, ' ')
      .replace(/^\s{0,3}[-*_]{3,}\s*$/gm, '')
      .replace(/^\s{0,3}#{1,6}\s*(.+)$/gm, '\n$1\n')
      .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '• ')
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '$1 ($2)')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    .replace(/~~(.*?)~~/g, '$1')
    .replace(/\\([\\`*_{}[\]()#+.!>])/g, '$1')
  const plainText = decodeTextEntities(plainTextPipeRows(normalized))
  return plainText
    .replace(/\|+/g, ' ')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n[ \t]+/g, '\n')
    .replace(/^[ \t]+/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}
