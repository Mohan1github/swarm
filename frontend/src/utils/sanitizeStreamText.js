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

export default function sanitizeStreamText(value) {
  if (typeof value !== 'string') return ''

  return plainTextPipeRows(value
    .replace(/\r\n?/g, '\n')
    .replace(/^\s*```[^\n]*\n?/gm, '')
    .replace(/```/g, '')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '• ')
    .replace(/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/gm, '')
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '$1 ($2)')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    .replace(/~~(.*?)~~/g, '$1')
    .replace(/\\([\\`*_{}[\]()#+.!>])/g, '$1')
  )
    .replace(/\|/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}
