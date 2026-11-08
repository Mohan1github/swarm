import { stageLabels, stageOrder } from '../constants/workflow'

export default function WorkflowProgress({
  activeStage,
  stageTotal,
  stageRoles,
  steps,
  isExiting = false,
}) {
  let stages = stageRoles.length ? stageRoles : [...stageOrder]

  if (activeStage && !stages.includes(activeStage)) {
    stages = [...stages, activeStage]
  }

  const currentLabel = stageLabels[activeStage] ?? activeStage
  const completedStages = new Set(steps.map((step) => step.role))
  const activeIndex = stages.indexOf(activeStage)
  const visibleStages = stages.filter((stage, index) => (
    stage === activeStage ||
    (completedStages.has(stage) && index >= activeIndex - 1)
  ))
  const runningCount = activeIndex >= 0 ? activeIndex + 1 : steps.length

  return (
    <div className={`chat-workflow-progress ${isExiting ? 'exiting' : ''}`} aria-label="Live agent workflow" aria-live="polite">
      <div className="chat-workflow-heading">
        <div className="chat-workflow-pulse" aria-hidden="true">
          <span />
        </div>
        <span>{activeStage ? currentLabel : 'Agents collaborating'}</span>
        <span className="chat-workflow-count">
          {Math.min(runningCount, stageTotal)} / {stageTotal}
        </span>
      </div>
      <ol className="chat-workflow-steps" aria-label="Agent steps">
        {visibleStages.map((stage) => {
          const index = stages.indexOf(stage)
          const isComplete = completedStages.has(stage)
          const isActive = activeStage === stage
          const label = stageLabels[stage] ?? stage.replaceAll('_', ' ')
          return (
            <li
              key={stage}
              className={`chat-workflow-step ${isComplete ? 'complete' : ''} ${isActive ? 'active' : ''}`}
              aria-current={isActive ? 'step' : undefined}
              aria-label={`${label}${isActive ? ', working' : isComplete ? ', complete' : ', upcoming'}`}
              style={{ '--workflow-step-index': index }}
            >
              <span className="chat-workflow-marker" aria-hidden="true">
                {isComplete ? (
                  <svg viewBox="0 0 16 16">
                    <path d="m3.5 8 3 3 6-6" />
                  </svg>
                ) : (
                  <span>{index + 1}</span>
                )}
              </span>
              <span className="chat-workflow-label">{label}</span>
              <span className="chat-workflow-state">{isActive ? 'Running' : isComplete ? 'Complete' : 'Queued'}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
