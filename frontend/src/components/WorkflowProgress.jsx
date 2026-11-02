import { stageLabels, stageOrder } from '../constants/workflow'

export default function WorkflowProgress({
  activeStage,
  activeStageIndex,
  stageTotal,
  stageRoles,
  steps,
}) {
  let stages = stageRoles.length ? stageRoles : [...stageOrder]

  if (activeStage && !stages.includes(activeStage)) {
    stages = [...stages, activeStage]
  }

  const completedStages = new Set(steps.map((step) => step.role))
  const currentLabel = stageLabels[activeStage] ?? activeStage

  return (
    <aside className="workflow-progress" aria-label="Live agent workflow">
      <header className="workflow-progress-header">
        <div>
          <span className="workflow-progress-eyebrow">Intelligence flow</span>
          <strong>{activeStage ? currentLabel : 'Agents collaborating'}</strong>
        </div>
        <span className="workflow-progress-count">
          {activeStageIndex >= 0 ? `${activeStageIndex + 1} / ${stageTotal}` : `${steps.length} / ${stageTotal}`}
        </span>
      </header>

      <ol className="workflow-progress-steps" aria-label="Agent steps">
        {stages.map((stage, index) => {
          const isComplete = completedStages.has(stage)
          const isActive = activeStage === stage
          const label = stageLabels[stage] ?? stage.replaceAll('_', ' ')
          return (
            <li
              key={stage}
              className={`workflow-progress-step ${isComplete ? 'complete' : ''} ${isActive ? 'active' : ''}`}
              aria-current={isActive ? 'step' : undefined}
              aria-label={`${label}${isActive ? ', working' : isComplete ? ', complete' : ', upcoming'}`}
              style={{ '--workflow-step-index': index }}
            >
              <span className="workflow-progress-marker" aria-hidden="true">
                {isComplete ? (
                  <svg viewBox="0 0 16 16">
                    <path d="m3.5 8 3 3 6-6" />
                  </svg>
                ) : (
                  <span>{index + 1}</span>
                )}
              </span>
              <span className="workflow-progress-label">{label}</span>
            </li>
          )
        })}
      </ol>
    </aside>
  )
}
