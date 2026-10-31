import { useEffect, useRef, useState } from 'react'

export default function ChatComposer({
  query,
  setQuery,
  attachments,
  attachmentError,
  isRunning,
  isActive = true,
  voiceState,
  onVoiceStateChange,
  onVoiceSubmit,
  onAddFiles,
  onRemoveFile,
  onSubmit,
}) {
  const fileInput = useRef(null)
  const recognitionRef = useRef(null)
  const baseQueryRef = useRef('')
  const finalTranscriptRef = useRef('')
  const shouldSubmitRef = useRef(false)
  const [voiceError, setVoiceError] = useState('')

  useEffect(() => {
    if (isActive || !recognitionRef.current) return
    shouldSubmitRef.current = false
    recognitionRef.current.abort()
    recognitionRef.current = null
    onVoiceStateChange('idle')
  }, [isActive, onVoiceStateChange])

  useEffect(() => () => {
    shouldSubmitRef.current = false
    recognitionRef.current?.abort()
  }, [])

  const toggleVoice = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setVoiceError('Speech input is not supported in this browser.')
      return
    }

    setVoiceError('')
    baseQueryRef.current = query.trim()
    finalTranscriptRef.current = ''
    shouldSubmitRef.current = true

    const recognition = new SpeechRecognition()
    recognition.lang = navigator.language || 'en-US'
    recognition.continuous = false
    recognition.interimResults = true
    recognitionRef.current = recognition

    recognition.onresult = (event) => {
      let interimTranscript = ''
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index]
        if (result.isFinal) {
          finalTranscriptRef.current += `${result[0].transcript} `
        } else {
          interimTranscript += result[0].transcript
        }
      }
      const transcript = [baseQueryRef.current, finalTranscriptRef.current.trim(), interimTranscript]
        .filter(Boolean)
        .join(' ')
      setQuery(transcript)
    }

    recognition.onerror = (event) => {
      shouldSubmitRef.current = false
      if (event.error !== 'aborted') setVoiceError('Microphone input could not be started.')
      onVoiceStateChange('idle')
    }

    recognition.onend = () => {
      recognitionRef.current = null
      onVoiceStateChange('idle')
      const spokenText = finalTranscriptRef.current.trim()
      const transcript = [baseQueryRef.current, spokenText].filter(Boolean).join(' ')
      if (shouldSubmitRef.current && spokenText) {
        shouldSubmitRef.current = false
        setQuery('')
        onVoiceSubmit(transcript)
      }
    }

    try {
      recognition.start()
      onVoiceStateChange('listening')
    } catch {
      recognitionRef.current = null
      shouldSubmitRef.current = false
      setVoiceError('Microphone input could not be started.')
      onVoiceStateChange('idle')
    }
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if ((query.trim() || attachments.length) && !isRunning) onSubmit(event)
    }
  }

  const handleFileChange = (event) => {
    onAddFiles(Array.from(event.currentTarget.files || []))
    event.currentTarget.value = ''
  }

  return (
    <div className="composer-wrap">
      {attachmentError ? <div className="attachment-error">{attachmentError}</div> : null}
      {voiceError ? <div className="attachment-error" role="status">{voiceError}</div> : null}

      <form onSubmit={onSubmit} className="composer">
        <input
          ref={fileInput}
          className="attachment-input"
          type="file"
          multiple
          accept="image/png,image/jpeg,image/webp,application/pdf,.txt,.md,.csv,.json,.py,.js,.jsx,.ts,.tsx,.html,.css,.xml,.yaml,.yml,.log,.sql,.sh"
          onChange={handleFileChange}
        />
        {attachments.length > 0 ? (
          <div className="attachment-strip">
            {attachments.map(({ id, file, previewUrl }) => (
              <div className="attachment-chip" key={id}>
                {previewUrl ? (
                  <img src={previewUrl} alt="" className="attachment-thumbnail" />
                ) : (
                  <span className="attachment-file-icon" aria-hidden="true">FILE</span>
                )}
                <span className="attachment-name" title={file.name}>{file.name}</span>
                <button
                  className="attachment-remove"
                  type="button"
                  disabled={isRunning}
                  aria-label={`Remove ${file.name}`}
                  title="Remove attachment"
                  onClick={() => onRemoveFile(id)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        ) : null}
        <div className="composer-controls">
          <button
            className="attachment-trigger"
            type="button"
            disabled={isRunning || attachments.length >= 5}
            aria-label="Add files or photos"
            title="Add files or photos"
            onClick={() => fileInput.current?.click()}
          >
            +
          </button>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            aria-label="Message Swarm AI"
            placeholder="Ask anything, attach a file, or add a photo..."
          />
          <button
            className={`voice-trigger ${voiceState === 'listening' ? 'listening' : ''}`}
            type="button"
            disabled={isRunning}
            aria-label={voiceState === 'listening' ? 'Stop voice input' : 'Start voice input'}
            aria-pressed={voiceState === 'listening'}
            title={voiceState === 'listening' ? 'Stop voice input' : 'Start voice input'}
            onClick={toggleVoice}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z" />
              <path d="M19 11a7 7 0 0 1-14 0M12 18v4m-4 0h8" />
            </svg>
          </button>
          <button
            type="submit"
            disabled={isRunning || (!query.trim() && attachments.length === 0)}
            aria-label={isRunning ? 'Workflow running' : 'Send message'}
            title={isRunning ? 'Workflow running' : 'Send message'}
          >
            {isRunning ? '...' : '\u2191'}
          </button>
        </div>
      </form>
    </div>
  )
}
