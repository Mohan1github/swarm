const NEURAL_NODES = [
  [42, 75], [68, 34], [105, 25], [215, 37], [263, 75],
  [278, 123], [243, 169], [199, 193], [89, 190], [48, 151],
]

export default function AssistantPresence({ isListening, isThinking, isSpeaking, speechProgress = 0 }) {
  const state = isListening
    ? 'listening'
    : isThinking
      ? 'thinking'
      : isSpeaking
        ? 'speaking'
        : 'ready'
  const label = isListening
    ? 'Listening'
    : isThinking
      ? 'Thinking'
      : isSpeaking
        ? 'Responding'
        : 'Here with you'
  const visualMotion = {
    '--speech-speed': `${Math.max(0.58, 1.35 - speechProgress * 0.6)}s`,
    '--speech-lift': `${3 + speechProgress * 7}px`,
    '--speech-scale': 1 + speechProgress * 0.025,
  }

  return (
    <section className={`assistant-presence ${state}`} aria-label={`Assistant ${label.toLowerCase()}`}>
      <div className="assistant-visual" style={visualMotion} aria-hidden="true">
        <svg viewBox="0 0 320 220">
          <circle className="assistant-orbit orbit-wide" cx="160" cy="110" r="86" />
          <circle className="assistant-orbit orbit-tight" cx="160" cy="110" r="62" />
          {NEURAL_NODES.map(([x, y], index) => (
            <line
              className="assistant-link"
              key={`link-${index}`}
              x1="160"
              y1="110"
              x2={x}
              y2={y}
              style={{ '--link-delay': `${index * 70}ms` }}
            />
          ))}
          {NEURAL_NODES.map(([x, y], index) => (
            <circle
              className="assistant-node"
              key={`node-${index}`}
              cx={x}
              cy={y}
              r={index % 3 === 0 ? 3 : 2}
              style={{ '--node-delay': `${index * 110}ms` }}
            />
          ))}
          <path className="assistant-core" d="M160 75 188 92 188 128 160 145 132 128 132 92Z" />
          <path className="assistant-wave" d="M143 111h8l5-12 8 24 6-15 4 3h7" />
        </svg>
      </div>
      <div className="assistant-presence-copy">
        <span className="assistant-presence-label">Your assistant</span>
        <strong>{label}</strong>
        <div className="assistant-level-meter" aria-hidden="true">
          {Array.from({ length: 19 }, (_, index) => (
            <i key={index} style={{ '--bar-index': index }} />
          ))}
        </div>
      </div>
    </section>
  )
}