import { useState } from 'react'
import { initialMessages, stageLabels } from '../constants/workflow'
import formatAgentOutput from '../utils/formatAgentOutput'
import serializeAttachment from '../utils/serializeAttachment'

export default function useSwarmStream({
  token,
  observerEnabled = false,
  observerSessionId = null,
  onObserverResult,
}) {
  const [status, setStatus] = useState('Ready')
  const [isRunning, setIsRunning] = useState(false)
  const [steps, setSteps] = useState([])
  const [activeStage, setActiveStage] = useState(null)
  const [activeStageIndex, setActiveStageIndex] = useState(-1)
  const [stageTotal, setStageTotal] = useState(0)
  const [stageRoles, setStageRoles] = useState([])
  const [isResearchRun, setIsResearchRun] = useState(false)
  const [logs, setLogs] = useState([])
  const [messages, setMessages] = useState(initialMessages)
  const [selectedAgents, setSelectedAgents] = useState(0)
  const [latency, setLatency] = useState(null)
  const [error, setError] = useState('')
  const [actionProbabilities, setActionProbabilities] = useState({})
  const [validationScore, setValidationScore] = useState(null)
  const [validationIssues, setValidationIssues] = useState([])

  const clearLocalHistory = () => {
    setStatus('Ready')
    setSteps([])
    setActiveStage(null)
    setActiveStageIndex(-1)
    setStageTotal(0)
    setStageRoles([])
    setIsResearchRun(false)
    setLogs([])
    setMessages(initialMessages)
    setSelectedAgents(0)
    setLatency(null)
    setError('')
    setActionProbabilities({})
    setValidationScore(null)
    setValidationIssues([])
  }

  const runWorkflow = async (query, attachments = [], event, researchMode = false) => {
    event?.preventDefault()
    const trimmed = query.trim()
    if (!trimmed || isRunning) return
    const conversationHistory = messages
      .filter((message) => (
        (message.role === 'user' || message.role === 'assistant')
        && typeof message.content === 'string'
        && !message.loading
      ))
      .slice(-8)
      .map(({ role, content }) => ({
        role,
        content: content.slice(-2000),
      }))

    setIsRunning(true)
    setStatus(attachments.length ? 'Preparing attachments...' : 'Connecting to live swarm...')
    setError('')
    setSteps([])
    setActiveStage(null)
    setActiveStageIndex(-1)
    setStageTotal(0)
    setStageRoles([])
    setIsResearchRun(researchMode)
    setLogs([])
    setActionProbabilities({})
    setValidationScore(null)
    setValidationIssues([])
    setLatency(null)
    setSelectedAgents(0)

    let streamClosed = false
    const stepMap = new Map()
    const trace = []
    const observeTurn = (answer) => {
      if (observerEnabled && observerSessionId && answer) {
        onObserverResult?.(trimmed, answer)
      }
    }
    const handlePayload = (payload) => {
      if (payload.type === 'clarification_required') {
        if (streamClosed) return
        streamClosed = true
        setStatus('Needs a little more context')
        setActiveStage(null)
        setIsRunning(false)
        setSteps([])
        setMessages((previous) => {
          const last = previous[previous.length - 1]
          const withoutPlaceholder = last?.role === 'assistant' && last.loading
            ? previous.slice(0, -1)
            : previous
          return [...withoutPlaceholder, {
            role: 'assistant',
            agentRole: 'clarification',
            content: payload.question || 'What matters most to you about this request?',
            missingInformation: payload.missing_information || [],
          }]
        })
        observeTurn(payload.question || '')
        return
      }

      if (payload.type === 'setup') {
        setSelectedAgents(payload.selected_agents ?? 0)
        setStatus('Selecting specialist agents')
        return
      }

      if (payload.type === 'cache_hit') {
        if (streamClosed) return
        streamClosed = true
        const cachedAnswer = formatAgentOutput(payload.answer, 'synthesizer') || 'No cached answer found.'
        observeTurn(cachedAnswer)
        const cachedSteps = payload.steps || []
        setStatus('Instant answer from vector cache')
        setSteps(cachedSteps)
        setActiveStage(null)
        setLogs(cachedSteps.map((step) => ({
          role: step.role,
          label: stageLabels[step.role] ?? step.role,
          detail: formatAgentOutput(step.output, step.role).replace(/\s+/g, ' ').slice(0, 220) || 'Finished',
        })))
        setSelectedAgents(payload.selected_agents ?? 0)
        setActionProbabilities(payload.action_probabilities || {})
        setValidationScore(payload.validation_score ?? null)
        setValidationIssues(payload.validation_issues || [])
        setLatency(0)
        setMessages((previous) => {
          const last = previous[previous.length - 1]
          const withoutPlaceholder = last?.role === 'assistant' && last.loading
            ? previous.slice(0, -1)
            : previous
          return [...withoutPlaceholder, {
            role: 'assistant',
            agent: 'Vector cache',
            agentRole: 'final',
            content: cachedAnswer,
            instant: true,
          }]
        })
        setIsRunning(false)
        return
      }

      if (payload.type === 'stage_start') {
        setActiveStage(payload.role)
        setActiveStageIndex(payload.stage_index ?? -1)
        setStageTotal(payload.stage_total ?? 0)
        if (Array.isArray(payload.workflow_roles)) setStageRoles(payload.workflow_roles)
        setStatus(`${stageLabels[payload.role] ?? payload.role} working`)
        return
      }

      if (payload.type === 'step') {
        stepMap.set(payload.role, payload)
        setSteps(Array.from(stepMap.values()))
        setActiveStage(null)
        setStatus(`${stageLabels[payload.role] ?? payload.role} completed`)

        const displayOutput = formatAgentOutput(payload.output, payload.role).replace(/\s+/g, ' ')
        trace.push({
          role: payload.role,
          label: stageLabels[payload.role] ?? payload.role,
          detail: displayOutput ? displayOutput.slice(0, 220) : 'Finished',
        })
        setLogs([...trace])
        return
      }

      if (payload.type === 'error') {
        if (streamClosed) return
        streamClosed = true
        setError(payload.message || 'The workflow failed.')
        setStatus(`Workflow error in ${payload.stage || 'execution'}`)
        setActiveStage(null)
        setIsRunning(false)
        setMessages((previous) => {
          const last = previous[previous.length - 1]
          return last?.role === 'assistant' && last.loading
            ? previous.slice(0, -1)
            : previous
        })
        return
      }

      if (payload.type === 'complete') {
        if (streamClosed) return
        streamClosed = true
        setStatus('Workflow complete')
        setActiveStage(null)
        const finalAnswer = formatAgentOutput(payload.answer, 'synthesizer') || 'No answer generated.'
        observeTurn(finalAnswer)
        setMessages((previous) => {
          let conversationStart = 0
          if (payload.route?.conversation_reset) {
            for (let index = previous.length - 1; index >= 0; index -= 1) {
              if (previous[index].role === 'user' && previous[index].content === trimmed) {
                conversationStart = index
                break
              }
            }
          }
          const conversation = previous.slice(conversationStart)
          const last = conversation[conversation.length - 1]
          const withoutPlaceholder = last?.role === 'assistant' && last.loading
            ? conversation.slice(0, -1)
            : conversation
          return [...withoutPlaceholder, {
            role: 'assistant',
            agentRole: 'final',
            content: finalAnswer,
            mostImportant: payload.most_important || '',
            evidenceLedger: payload.evidence_ledger ?? null,
            evidenceLedgerError: payload.evidence_ledger_error ?? '',
            imageSources: payload.image_sources ?? [],
            imageSearchError: payload.image_search_error ?? '',
            supportingDocuments: payload.supporting_documents ?? [],
            supportingDocumentError: payload.supporting_document_error ?? '',
          }]
        })
        setSelectedAgents(payload.selected_agents ?? selectedAgents)
        setActionProbabilities(payload.action_probabilities || {})
        setValidationScore(payload.validation_score ?? null)
        setValidationIssues(payload.validation_issues || [])
        setLatency(payload.latency_ms ?? null)
        setIsRunning(false)
      }
    }

    const parseFrame = (frame) => {
      const data = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n')
      if (data) handlePayload(JSON.parse(data))
    }

    try {
      const serializedAttachments = await Promise.all(attachments.map(({ file }) => serializeAttachment(file)))
      setMessages((previous) => [
        ...previous,
        {
          role: 'user',
          content: trimmed,
          attachments: serializedAttachments.map(({ name, media_type, preview_url, size }) => ({
            name,
            media_type,
            preview_url,
            size,
          })),
        },
        { role: 'assistant', content: '', loading: true },
      ])

      const response = await fetch('/api/swarm/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          query: trimmed,
          research_mode: researchMode,
          conversation_history: conversationHistory,
          attachments: serializedAttachments.map(({ name, media_type, content }) => ({
            name,
            media_type,
            content,
          })),
        }),
      })

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => null)
        throw new Error(errorPayload?.detail || `Request failed with status ${response.status}.`)
      }
      if (!response.body) throw new Error('The server did not return a stream.')

      setStatus('Workflow started')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (value) buffer += decoder.decode(value, { stream: !done })

        let boundary = buffer.match(/\r?\n\r?\n/)
        while (boundary) {
          parseFrame(buffer.slice(0, boundary.index))
          buffer = buffer.slice(boundary.index + boundary[0].length)
          if (streamClosed) break
          boundary = buffer.match(/\r?\n\r?\n/)
        }

        if (done) {
          if (buffer.trim()) parseFrame(buffer)
          break
        }
        if (streamClosed) {
          await reader.cancel().catch(() => {})
          break
        }
      }

      if (!streamClosed) {
        setStatus('Stream ended')
        setActiveStage(null)
        setIsRunning(false)
        setError('The stream closed before the workflow finished. Please retry.')
        setMessages((previous) => {
          const last = previous[previous.length - 1]
          return last?.role === 'assistant' && last.loading
            ? previous.slice(0, -1)
            : previous
        })
      }
    } catch (streamError) {
      setStatus('Workflow error')
      setActiveStage(null)
      setIsRunning(false)
      setError(streamError.message || 'The workflow could not be completed.')
      setMessages((previous) => {
        const last = previous[previous.length - 1]
        return last?.role === 'assistant' && last.loading
          ? previous.slice(0, -1)
          : previous
      })
    }
  }

  return {
    status,
    isRunning,
    steps,
    activeStage,
    activeStageIndex,
    stageTotal,
    stageRoles,
    isResearchRun,
    logs,
    messages,
    selectedAgents,
    latency,
    error,
    actionProbabilities,
    validationScore,
    validationIssues,
    runWorkflow,
    clearLocalHistory,
  }
}
