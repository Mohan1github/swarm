import ActivityGraph from './ActivityGraph'

export default function ContextGraphPage({ token, isRunning }) {
  return (
    <section className="context-graph-page" aria-labelledby="context-graph-page-title">
      <div className="context-graph-page-heading">
        <span className="auth-kicker">Personal activity</span>
        <h1 id="context-graph-page-title">Context graph</h1>
        <p>Your prompts and the concepts connecting them.</p>
      </div>
      <ActivityGraph token={token} isRunning={isRunning} />
    </section>
  )
}
