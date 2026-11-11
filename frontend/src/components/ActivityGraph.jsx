import { useEffect, useState } from 'react'
import { stageLabels } from '../constants/workflow'

const EMPTY_GRAPH = { nodes: [], edges: [], total: 0, shown: 0, stage_total: 0, stage_shown: 0, connections: 0 }
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))

function formatTimestamp(value) {
  const date = new Date(`${value.replace(' ', 'T')}Z`)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function createLayout(graph, expandedConversationId) {
  const queryNodes = graph.nodes.filter((node) => node.kind === 'query')
  const stageNodes = graph.nodes.filter(
    (node) => node.kind === 'stage' && node.parent_id === expandedConversationId,
  )
  const positionedQueries = queryNodes.map((node, index) => {
    const radius = queryNodes.length === 1 ? 0 : Math.sqrt((index + 0.5) / queryNodes.length)
    const angle = index * GOLDEN_ANGLE
    return {
      ...node,
      x: 500 + Math.cos(angle) * 260 * radius,
      y: 214 + Math.sin(angle) * 105 * radius,
    }
  })
  const queryPositions = Object.fromEntries(positionedQueries.map((node) => [node.id, node]))
  const stagesByParent = new Map()

  stageNodes.forEach((node) => {
    const siblings = stagesByParent.get(node.parent_id) || []
    siblings.push(node)
    stagesByParent.set(node.parent_id, siblings)
  })

  const positionedStages = []
  stagesByParent.forEach((siblings, parentId) => {
    const parent = queryPositions[parentId]
    if (!parent) return
    siblings.forEach((node, index) => {
      const angle = -Math.PI / 2 + ((index + 0.5) / siblings.length) * Math.PI * 2
      const radius = siblings.length > 6 ? 48 : 38
      positionedStages.push({
        ...node,
        x: parent.x + Math.cos(angle) * radius,
        y: parent.y + Math.sin(angle) * radius,
      })
    })
  })

  return [...positionedQueries, ...positionedStages]
}

function createNeuralConnections(nodes, edges) {
  const queryNodes = nodes.filter((node) => node.kind === 'query')
  const visiblePairs = new Set(
    edges.map((edge) => [edge.source, edge.target].sort().join(':')),
  )
  const connections = new Map()

  queryNodes.forEach((node) => {
    queryNodes
      .filter((candidate) => candidate.id !== node.id)
      .sort((left, right) => (
        Math.hypot(left.x - node.x, left.y - node.y)
        - Math.hypot(right.x - node.x, right.y - node.y)
      ))
      .slice(0, 2)
      .forEach((neighbor) => {
        const key = [node.id, neighbor.id].sort().join(':')
        if (!visiblePairs.has(key)) connections.set(key, [node, neighbor])
      })
  })

  return [...connections.entries()].map(([id, [source, target]]) => ({
    id: `neural:${id}`,
    path: `M ${source.x} ${source.y} L ${target.x} ${target.y}`,
  }))
}

function getNodeColor(node) {
  if (node.kind === 'stage') return '#a3a3a3'
  if (node.cache_hit) return '#f5f5f5'
  if (node.attachment_count > 0) return '#d4d4d4'
  return '#ffffff'
}

export default function ActivityGraph({ token, isRunning, onUsePrompt }) {
  const [graphState, setGraphState] = useState({ graph: EMPTY_GRAPH, loading: true, error: '' })
  const [selectedItem, setSelectedItem] = useState(null)
  const [selectedConversationId, setSelectedConversationId] = useState(null)
  const [searchText, setSearchText] = useState('')
  const [nodeFilter, setNodeFilter] = useState('all')
  const [showStages, setShowStages] = useState(true)
  const [refreshVersion, setRefreshVersion] = useState(0)
  const [refreshedAt, setRefreshedAt] = useState(null)

  useEffect(() => {
    const controller = new AbortController()
    let refreshTimer

    const loadGraph = () => {
      fetch('/api/activity/graph', {
        headers: { Authorization: `Bearer ${token}` },
        signal: controller.signal,
      })
        .then(async (response) => {
          const payload = await response.json().catch(() => ({}))
          if (!response.ok) throw new Error(payload.detail || 'Could not load activity graph.')
          return payload
        })
        .then((graph) => {
          if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
            throw new Error('The activity graph response is incomplete.')
          }
          const latestConversation = graph.nodes
            .filter((node) => node.kind === 'query')
            .at(-1)
          if (latestConversation) {
            setSelectedConversationId((current) => (
              isRunning || !current ? latestConversation.id : current
            ))
          }
          setGraphState({ graph, loading: false, error: '' })
          setRefreshedAt(Date.now())
        })
        .catch((error) => {
          if (error.name !== 'AbortError') {
            setGraphState((current) => ({ ...current, loading: false, error: error.message }))
          }
        })
        .finally(() => {
          if (isRunning && !controller.signal.aborted) {
            refreshTimer = setTimeout(loadGraph, 900)
          }
        })
    }

    loadGraph()

    return () => {
      controller.abort()
      clearTimeout(refreshTimer)
    }
  }, [token, isRunning, refreshVersion])

  const normalizedSearch = searchText.trim().toLowerCase()
  const visibleQueryNodes = graphState.graph.nodes.filter((node) => {
    if (node.kind !== 'query') return false
    if (nodeFilter === 'attachments' && node.attachment_count === 0) return false
    if (nodeFilter === 'cached' && !node.cache_hit) return false
    return !normalizedSearch || node.query.toLowerCase().includes(normalizedSearch)
  })
  const visibleQueryIds = new Set(visibleQueryNodes.map((node) => node.id))
  const activeQueryId = graphState.graph.nodes
    .filter((node) => node.kind === 'query')
    .at(-1)?.id
  const expandedConversationId = isRunning ? activeQueryId : selectedConversationId
  const visibleNodes = [
    ...visibleQueryNodes,
    ...(showStages
      ? graphState.graph.nodes.filter(
          (node) => node.kind === 'stage' && node.parent_id === expandedConversationId,
        )
      : []),
  ]
  const visibleEdges = graphState.graph.edges.filter((edge) => (
    edge.kind === 'related'
      ? visibleQueryIds.has(edge.source) && visibleQueryIds.has(edge.target)
      : showStages && edge.source === expandedConversationId && visibleQueryIds.has(edge.source)
  ))
  const nodes = createLayout({ nodes: visibleNodes }, expandedConversationId)
  const neuralConnections = createNeuralConnections(nodes, visibleEdges)
  const nodesById = Object.fromEntries(nodes.map((node) => [node.id, node]))
  const activeNode = selectedItem?.type === 'node' ? nodesById[selectedItem.id] : null
  const activeEdge = selectedItem?.type === 'edge'
    ? visibleEdges.find((edge) => edge.id === selectedItem.id)
    : null
  const sourceNode = activeEdge ? nodesById[activeEdge.source] : null
  const targetNode = activeEdge ? nodesById[activeEdge.target] : null
  const selectedPrompt = activeNode?.kind === 'query'
    ? activeNode
    : activeNode?.kind === 'stage'
      ? graphState.graph.nodes.find((node) => node.id === activeNode.parent_id)
      : sourceNode?.kind === 'query'
        ? sourceNode
        : targetNode?.kind === 'query'
          ? targetNode
          : null

  const selectConversation = (node) => {
    const conversation = node?.kind === 'query' ? node : nodesById[node?.parent_id]
    if (conversation) setSelectedConversationId(conversation.id)
  }

  return (
    <section className="insight-panel activity-panel" aria-labelledby="activity-title">
      <div className="activity-heading-row">
        <div>
          <div className="panel-heading" id="activity-title">Your activity graph</div>
          <p className="activity-summary">
            {graphState.loading
              ? 'Loading your prompt network...'
              : `Showing ${visibleQueryNodes.length} of ${graphState.graph.shown} prompts · ${visibleEdges.length} connections`}
            {refreshedAt ? ` · Updated ${new Date(refreshedAt).toLocaleTimeString()}` : ''}
          </p>
        </div>
        <div className="activity-legend" aria-label="Node types">
          <span><i className="legend-dot prompt" />Prompt</span>
          <span><i className="legend-dot attachment" />With files</span>
          <span><i className="legend-dot cached" />Cached</span>
          <span><i className="legend-dot stage" />Agent stage</span>
        </div>
      </div>

      <div className="activity-toolbar" aria-label="Graph controls">
        <label className="activity-search">
          <span className="activity-control-icon" aria-hidden="true">⌕</span>
          <input
            type="search"
            value={searchText}
            onChange={(event) => setSearchText(event.target.value)}
            placeholder="Search prompts"
            aria-label="Search prompts"
          />
          {searchText ? (
            <button type="button" onClick={() => setSearchText('')} aria-label="Clear prompt search">
              ×
            </button>
          ) : null}
        </label>
        <label className="activity-filter">
          <span>Show</span>
          <select
            value={nodeFilter}
            onChange={(event) => setNodeFilter(event.target.value)}
            aria-label="Filter prompt nodes"
          >
            <option value="all">All prompts</option>
            <option value="attachments">With files</option>
            <option value="cached">Cached</option>
          </select>
        </label>
        <label className="activity-stage-toggle">
          <input
            type="checkbox"
            checked={showStages}
            onChange={(event) => setShowStages(event.target.checked)}
          />
          <span>Agent stages</span>
        </label>
        <button
          className="activity-refresh"
          type="button"
          onClick={() => setRefreshVersion((version) => version + 1)}
          disabled={graphState.loading}
        >
          <span aria-hidden="true">↻</span>
          Refresh
        </button>
      </div>

      {graphState.error ? (
        <p className="activity-empty" role="alert">{graphState.error}</p>
      ) : graphState.loading ? (
        <div className="activity-loading" aria-label="Loading activity graph" />
      ) : graphState.graph.nodes.length === 0 ? (
        <div className="activity-empty">Your prompts will form connections here as you use Swarm AI.</div>
      ) : visibleQueryNodes.length === 0 ? (
        <div className="activity-empty">No prompts match these filters. Try another search or filter.</div>
      ) : (
        <>
          <div className={`activity-canvas ${expandedConversationId ? 'conversation-expanded' : ''}`}>
        <div className={`activity-signal-status ${isRunning ? 'live' : ''}`} aria-hidden="true">
          <i />
          <span>{isRunning ? 'Live workflow' : 'Signal flow'}</span>
        </div>
        <svg
              viewBox="0 0 1000 430"
              preserveAspectRatio="xMidYMid meet"
              role="img"
              aria-label="Graph of your recent prompts and their connections"
            >
              <defs>
                <radialGradient id="brain-neuron-glow">
                  <stop offset="0" stopColor="#ffffff" stopOpacity="0.13" />
                  <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
                </radialGradient>
              </defs>
              <g transform="translate(500 215) scale(1.22) translate(-500 -215)">
              <ellipse cx="390" cy="215" rx="270" ry="175" fill="url(#brain-neuron-glow)" />
              <ellipse cx="610" cy="215" rx="270" ry="175" fill="url(#brain-neuron-glow)" />
              <path className="brain-outline" d="M499 83 C453 44 378 50 351 91 C294 74 241 112 250 163 C211 197 230 253 268 273 C262 322 310 350 352 330 C385 369 445 350 470 320 C486 301 494 272 499 248 C504 272 512 301 530 320 C555 350 615 369 648 330 C690 350 738 322 732 273 C770 253 789 197 750 163 C759 112 706 74 649 91 C622 50 547 44 501 83 Z" />
              <path className="brain-fold" d="M500 84 C493 128 507 166 499 208 C493 243 504 278 500 319" />

              <g className="activity-neural-network" aria-hidden="true">
                {neuralConnections.map((connection, index) => (
                  <g key={connection.id}>
                    <path
                      id={connection.id}
                      className="activity-neural-connection"
                      d={connection.path}
                    />
                    <circle className="activity-neural-pulse" r="2">
                      <animateMotion
                        dur="4.8s"
                        begin={`-${(index % 9) * 0.53}s`}
                        repeatCount="indefinite"
                        calcMode="paced"
                      >
                        <mpath href={`#${connection.id}`} />
                      </animateMotion>
                    </circle>
                  </g>
                ))}
              </g>

              {visibleEdges.map((edge, index) => {
                const source = nodesById[edge.source]
                const target = nodesById[edge.target]
                if (!source || !target) return null
                const path = `M ${source.x} ${source.y} L ${target.x} ${target.y}`
                const isLiveSignal = isRunning && edge.kind === 'stage' && edge.source === activeQueryId
                const hasNeuralSignal = edge.kind === 'related' || isLiveSignal
                const pathId = `activity-edge-path-${index}`
                return (
                  <g key={edge.id}>
                    <path
                      className="activity-edge-hit"
                      d={path}
                      tabIndex={0}
                      role="button"
                      aria-label={`${edge.context}: ${source.query} and ${target.query}`}
                      onFocus={() => {
                        setSelectedItem({ type: 'edge', id: edge.id })
                        selectConversation(source)
                      }}
                      onClick={() => {
                        setSelectedItem({ type: 'edge', id: edge.id })
                        selectConversation(source)
                      }}
                    />
                    <path
                      id={pathId}
                      className={`activity-edge ${edge.kind === 'related' ? 'neural' : ''} ${isLiveSignal ? 'live' : ''} ${selectedItem?.type === 'edge' && selectedItem.id === edge.id ? 'selected' : ''}`}
                      d={path}
                      strokeWidth={1 + edge.similarity * 2}
                      pointerEvents="none"
                    />
                    {hasNeuralSignal && Array.from({ length: isLiveSignal ? 2 : 1 }, (_, signalIndex) => (
                      <circle className={`activity-signal ${isLiveSignal ? 'live' : 'ambient'}`} key={signalIndex} r={isLiveSignal ? 3.5 : 2.5}>
                        <animateMotion
                          dur={isLiveSignal ? '1.7s' : '5.2s'}
                          begin={`-${signalIndex * 0.85 + (index % 7) * (isLiveSignal ? 0.2 : 0.48)}s`}
                          repeatCount="indefinite"
                          calcMode="paced"
                        >
                          <mpath href={`#${pathId}`} />
                        </animateMotion>
                      </circle>
                    ))}
                  </g>
                )
              })}

              {nodes.map((node, index) => (
                <g
                  className={`activity-node ${node.kind === 'query' ? 'query-node' : 'stage-node'} ${selectedItem?.type === 'node' && selectedItem.id === node.id ? 'selected' : ''}`}
                  style={{ '--neural-delay': `${(index % 8) * 260}ms` }}
                  key={node.id}
                  tabIndex={0}
                  role="button"
                  aria-label={node.kind === 'stage'
                    ? `${stageLabels[node.stage] ?? node.stage} output: ${node.output}`
                    : `Prompt: ${node.query}`}
                  onFocus={() => {
                    setSelectedItem({ type: 'node', id: node.id })
                    selectConversation(node)
                  }}
                  onClick={() => {
                    setSelectedItem({ type: 'node', id: node.id })
                    selectConversation(node)
                  }}
                >
                  <circle className="activity-node-halo" cx={node.x} cy={node.y} r={12} fill={getNodeColor(node)} />
                  <circle className="activity-node-core" cx={node.x} cy={node.y} r={5} fill={getNodeColor(node)} />
                </g>
              ))}
              </g>
            </svg>
          </div>

          <div className="activity-context" aria-live="polite">
            {activeNode ? (
              <>
                <div className="activity-context-label">
                  {activeNode.kind === 'stage'
                    ? `LLM output · ${stageLabels[activeNode.stage] ?? activeNode.stage}`
                    : 'Prompt context'}
                </div>
                <p>{activeNode.kind === 'stage' ? activeNode.output : activeNode.query}</p>
                <span>{activeNode.kind === 'stage'
                  ? `For: ${activeNode.parent_query} · ${formatTimestamp(activeNode.created_at)}`
                  : `${formatTimestamp(activeNode.created_at)} · ${activeNode.attachment_count} attachments${activeNode.cache_hit ? ' · answered from cache' : ''}`}</span>
                {selectedPrompt ? (
                  <button
                    className="activity-use-prompt"
                    type="button"
                    onClick={() => onUsePrompt?.(selectedPrompt.query)}
                  >
                    Use in chat
                  </button>
                ) : null}
              </>
            ) : activeEdge && sourceNode && targetNode ? (
              <>
                <div className="activity-context-label">
                  {activeEdge.kind === 'stage'
                    ? `Prompt → ${stageLabels[targetNode.stage] ?? targetNode.stage}`
                    : `Connection · ${Math.round(activeEdge.similarity * 100)}% related`}
                </div>
                <p>{sourceNode.query}</p>
                <p>{activeEdge.kind === 'stage' ? targetNode.output : targetNode.query}</p>
                <span>{activeEdge.context}</span>
                {selectedPrompt ? (
                  <button
                    className="activity-use-prompt"
                    type="button"
                    onClick={() => onUsePrompt?.(selectedPrompt.query)}
                  >
                    Use in chat
                  </button>
                ) : null}
              </>
            ) : (
              <span>Select a prompt to expand its agent stages and inspect the conversation details below.</span>
            )}
          </div>
        </>
      )}
    </section>
  )
}
