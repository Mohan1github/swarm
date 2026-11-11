const workflowSteps = [
  {
    number: '01',
    title: 'Start with your question',
    description: 'Share a goal, a problem, or a decision. If key context is missing, Swarm can ask a focused follow-up.',
    icon: <path d="M4 5.5h12M4 9.5h8M4 13.5h6M15 12v5m-2.5-2.5h5" />,
  },
  {
    number: '02',
    title: 'Explore the right evidence',
    description: 'Research agents shape search queries and retrieve relevant public sources for the task.',
    icon: <><circle cx="8.5" cy="8.5" r="4.5" /><path d="m12 12 4 4M6.5 8.5h4M8.5 6.5v4" /></>,
  },
  {
    number: '03',
    title: 'Check what matters',
    description: 'Specialist stages analyze findings, surface gaps, and validate claims against available evidence.',
    icon: <><path d="m4 10 4 4 8-8" /><circle cx="10" cy="10" r="8" /></>,
  },
  {
    number: '04',
    title: 'Get the clear takeaway', 
    description: 'Receive a considered response with the most important conclusion brought forward.',
    icon: <><path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4m7.8 7.8 1.4 1.4m0-10.6-1.4 1.4m-7.8 7.8-1.4 1.4" /><circle cx="10" cy="10" r="3" /></>,
  },
]

export default function LandingWorkflow() {
  return (
    <section className="landing-workflow" id="workflow" aria-labelledby="landing-workflow-title" data-landing-reveal>
      <div className="landing-section-heading">
        <div>
          <span className="landing-section-kicker">From question to clarity</span>
          <h2 id="landing-workflow-title">A thoughtful process.<br />A useful answer.</h2>
        </div>
        <p>Swarm coordinates the work behind the scenes, then brings the important result back to you.</p>
      </div>
      <div className="landing-workflow-grid">
        {workflowSteps.map((step) => (
          <article className="landing-workflow-step" key={step.number}>
            <div className="landing-workflow-step-top">
              <span>{step.number}</span>
              <svg viewBox="0 0 20 20" aria-hidden="true">{step.icon}</svg>
            </div>
            <h3>{step.title}</h3>
            <p>{step.description}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
