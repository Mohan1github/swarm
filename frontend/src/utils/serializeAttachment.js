const dataUrlTypes = new Set(['image/jpeg', 'image/png', 'image/webp', 'application/pdf'])

const mediaTypesByExtension = {
  csv: 'text/csv',
  css: 'text/css',
  html: 'text/html',
  jpeg: 'image/jpeg',
  jpg: 'image/jpeg',
  js: 'text/plain',
  jsx: 'text/plain',
  json: 'application/json',
  log: 'text/plain',
  md: 'text/markdown',
  pdf: 'application/pdf',
  png: 'image/png',
  py: 'text/plain',
  sh: 'text/plain',
  sql: 'text/plain',
  ts: 'text/plain',
  tsx: 'text/plain',
  txt: 'text/plain',
  xml: 'application/xml',
  webp: 'image/webp',
  yaml: 'text/yaml',
  yml: 'text/yaml',
}

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`))
    reader.onload = () => resolve(reader.result)
    reader.readAsDataURL(file)
  })
}

export default async function serializeAttachment(file) {
  const extension = file.name.split('.').pop()?.toLowerCase() || ''
  const mediaType = mediaTypesByExtension[extension] || file.type || 'text/plain'
  const isDataUrl = dataUrlTypes.has(mediaType)
  const content = isDataUrl ? await readAsDataUrl(file) : await file.text()

  return {
    name: file.name,
    media_type: mediaType,
    content,
    preview_url: mediaType.startsWith('image/') ? content : '',
    size: file.size,
  }
}
