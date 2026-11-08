import { useCallback, useEffect, useRef, useState } from 'react'
import serializeAttachment from '../utils/serializeAttachment'

const MAX_FILE_BYTES = 5_000_000
const MAX_TOTAL_BYTES = 15_000_000

function documentName(document, index) {
  return document.title || document.url || `Supporting document ${index + 1}`
}

export default function DocumentReviewer({
  token,
  initialDocuments = [],
  initialFiles = [],
  initialReviewQuery = '',
  autoReviewOnOpen = false,
  onClose,
}) {
  const fileInput = useRef(null)
  const [activeTab, setActiveTab] = useState('online')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(initialDocuments)
  const [isSearching, setIsSearching] = useState(false)
  const [reviewQuery, setReviewQuery] = useState(
    initialReviewQuery
      || 'Review for accuracy, completeness, evidence quality, and actionable improvements.',
  )
  const [files, setFiles] = useState(initialFiles)
  const [review, setReview] = useState('')
  const [isReviewing, setIsReviewing] = useState(false)
  const [onlineReview, setOnlineReview] = useState('')
  const [isReviewingOnline, setIsReviewingOnline] = useState(false)
  const [error, setError] = useState('')
  const [selectedDocument, setSelectedDocument] = useState(null)
  const autoReviewStarted = useRef(false)

  const searchDocuments = async (event) => {
    event.preventDefault()
    if (isSearching || !searchQuery.trim()) return
    setIsSearching(true)
    setError('')
    try {
      const response = await fetch('/api/research/documents/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          research_mode: true,
        }),
      })
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail || `Document search failed (${response.status}).`)
      }
      setSearchResults(payload.documents || [])
      setSelectedDocument(null)
      setOnlineReview('')
    } catch (searchError) {
      setError(searchError.message || 'Document search failed. Please retry.')
    } finally {
      setIsSearching(false)
    }
  }

  const reviewOnlineDocument = async (document) => {
    if (isReviewingOnline) return
    setSelectedDocument(document)
    setOnlineReview('')
    setIsReviewingOnline(true)
    setError('')
    try {
      const response = await fetch('/api/research/documents/review-online', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          url: document.url,
          query: 'Review key findings, relevance, evidence quality, limitations, and useful citations.',
          research_mode: true,
        }),
      })
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail || `Online document review failed (${response.status}).`)
      }
      setOnlineReview(payload.review || '')
      setSelectedDocument(payload.document || document)
    } catch (reviewError) {
      setError(reviewError.message || 'Online document review failed. Please retry.')
    } finally {
      setIsReviewingOnline(false)
    }
  }

  const addFiles = (selectedFiles) => {
    const accepted = []
    const rejected = []
    let totalBytes = files.reduce((total, file) => total + file.size, 0)
    for (const file of selectedFiles) {
      const extension = file.name.split('.').pop()?.toLowerCase()
      if (!['pdf', 'docx'].includes(extension)) {
        rejected.push(`${file.name}: select a PDF or DOCX file`)
      } else if (file.size > MAX_FILE_BYTES) {
        rejected.push(`${file.name}: exceeds the 5 MB limit`)
      } else if (files.length + accepted.length >= 5) {
        rejected.push('You can review up to 5 files at once')
      } else if (totalBytes + file.size > MAX_TOTAL_BYTES) {
        rejected.push('Combined document size must be 15 MB or less')
      } else {
        accepted.push(file)
        totalBytes += file.size
      }
    }
    setFiles((current) => [...current, ...accepted])
    setError(rejected.join('. '))
    setReview('')
  }

  const reviewDocuments = useCallback(async (event) => {
    event?.preventDefault()
    if (!files.length || isReviewing) return
    setIsReviewing(true)
    setError('')
    setReview('')
    try {
      const serialized = await Promise.all(files.map(serializeAttachment))
      const response = await fetch('/api/research/documents/review', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          query: reviewQuery.trim(),
          research_mode: true,
          attachments: serialized.map(({ name, media_type, content }) => ({
            name,
            media_type,
            content,
          })),
        }),
      })
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail || `Document review failed (${response.status}).`)
      }
      setReview(payload.review || '')
    } catch (reviewError) {
      setError(reviewError.message || 'Document review failed. Please retry.')
    } finally {
      setIsReviewing(false)
    }
  }, [files, isReviewing, reviewQuery, token])

  useEffect(() => {
    if (!autoReviewOnOpen || !files.length || autoReviewStarted.current) return
    autoReviewStarted.current = true
    reviewDocuments()
  }, [autoReviewOnOpen, files.length, reviewDocuments])

  return (
    <div
      className="document-reviewer-overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <section
        className="document-reviewer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="document-reviewer-title"
      >
        <header className="document-reviewer-header">
          <div>
            <span className="document-reviewer-eyebrow">RESEARCH WORKSPACE</span>
            <h2 id="document-reviewer-title">Document reviewer</h2>
            <p>Find source documents online or review your own PDF and DOCX files.</p>
          </div>
          <button type="button" className="document-reviewer-close" onClick={onClose} aria-label="Close document reviewer">
            ×
          </button>
        </header>

        <div className="document-reviewer-tabs" role="tablist" aria-label="Document review tools">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'online'}
            className={activeTab === 'online' ? 'active' : ''}
            onClick={() => setActiveTab('online')}
          >
            Find online documents
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'upload'}
            className={activeTab === 'upload' ? 'active' : ''}
            onClick={() => setActiveTab('upload')}
          >
            Review my files
          </button>
        </div>

        {error ? <p className="document-reviewer-error" role="alert">{error}</p> : null}

        {activeTab === 'online' ? (
          <div className="document-reviewer-content">
            <form className="document-reviewer-search" onSubmit={searchDocuments}>
              <label htmlFor="document-search-query">Describe the exact report or evidence you need</label>
              <div>
                <input
                  id="document-search-query"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  placeholder="e.g. 2025 WHO report on air pollution and childhood asthma"
                  maxLength={1500}
                />
                <button type="submit" disabled={isSearching || !searchQuery.trim()}>
                  {isSearching ? 'Searching...' : 'Search PDFs and DOCX'}
                </button>
              </div>
            </form>

            {searchResults.length ? (
              <div className="document-reviewer-results">
                {searchResults.map((document, index) => (
                  <article className="document-reviewer-result" key={`${document.url}-${index}`}>
                    <div className="document-reviewer-result-heading">
                      <span>{document.file_type || 'Document'}</span>
                      {document.pages ? <small>{document.pages} pages</small> : null}
                    </div>
                    <h3>{documentName(document, index)}</h3>
                    {document.excerpt ? <p>{document.excerpt}</p> : null}
                    <div className="document-reviewer-links">
                      <button
                        type="button"
                        disabled={isReviewingOnline}
                        onClick={() => reviewOnlineDocument(document)}
                      >
                        {isReviewingOnline && selectedDocument?.url === document.url
                          ? 'Reviewing...'
                          : 'AI review'}
                      </button>
                      <a href={document.url} target="_blank" rel="noreferrer">Open original</a>
                      {document.source_url ? (
                        <a href={document.source_url} target="_blank" rel="noreferrer">Source page</a>
                      ) : null}
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <p className="document-reviewer-empty">
                Search results include extracted text and links to the original document and discovery page.
              </p>
            )}

            {selectedDocument ? (
              <section className="document-reviewer-preview" aria-label="Selected document review">
                <div>
                  <strong>{selectedDocument.title || selectedDocument.file_type}</strong>
                  <button type="button" onClick={() => setSelectedDocument(null)}>Close preview</button>
                </div>
                <p>{selectedDocument.excerpt || 'No text could be extracted from this document.'}</p>
                <a href={selectedDocument.url} target="_blank" rel="noreferrer">Open full document</a>
                {onlineReview ? (
                  <pre className="document-reviewer-online-review">{onlineReview}</pre>
                ) : null}
              </section>
            ) : null}
          </div>
        ) : (
          <form className="document-reviewer-content document-reviewer-upload" onSubmit={reviewDocuments}>
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              multiple
              hidden
              onChange={(event) => {
                addFiles(Array.from(event.currentTarget.files || []))
                event.currentTarget.value = ''
              }}
            />
            <button type="button" className="document-reviewer-dropzone" onClick={() => fileInput.current?.click()}>
              <strong>Add PDF or DOCX files</strong>
              <span>Up to 5 files · 5 MB each · 15 MB combined</span>
            </button>
            {files.length ? (
              <ul className="document-reviewer-file-list">
                {files.map((file, index) => (
                  <li key={`${file.name}-${index}`}>
                    <span>{file.name}</span>
                    <button
                      type="button"
                      aria-label={`Remove ${file.name}`}
                      onClick={() => setFiles((current) => current.filter((_, fileIndex) => fileIndex !== index))}
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
            <label htmlFor="document-review-instructions">Review instructions</label>
            <textarea
              id="document-review-instructions"
              value={reviewQuery}
              onChange={(event) => setReviewQuery(event.target.value)}
              maxLength={2000}
              rows={3}
            />
            <button className="document-reviewer-submit" type="submit" disabled={!files.length || isReviewing}>
              {isReviewing ? 'Reviewing documents...' : 'Review documents'}
            </button>
            {review ? (
              <section className="document-reviewer-review" aria-label="Document review results">
                <h3>Review findings</h3>
                <pre>{review}</pre>
              </section>
            ) : null}
          </form>
        )}
      </section>
    </div>
  )
}
