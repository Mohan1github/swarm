from __future__ import annotations

from typing import Dict, List


ROLE_PIPELINE = [
    ("planner", "decompose"),
    ("researcher", "research"),
    ("analyst", "analyze"),
    ("critic", "challenge"),
    ("executor", "execute"),
    ("synthesizer_draft", "summarize"),
    ("corrector", "validate"),
    ("synthesizer", "summarize"),
    ("query_focus", "refine_query"),
    ("response_summarizer", "retrieve_and_summarize"),
    ("validator", "validate"),
]


def build_live_execution_report(
    action_probabilities: Dict[str, float],
    completed_roles: List[str] | None = None,
    validation_score: float | None = None,
) -> Dict[str, object]:
    """Build a graph model from live planner probabilities and stage events."""
    completed = set(completed_roles or [])
    pipeline = [
        {
            "role": role,
            "action": action,
            "probability": float(action_probabilities.get(action, 0.0)),
            "status": "completed" if role in completed else "pending",
        }
        for role, action in ROLE_PIPELINE
    ]
    return {"validation_score": validation_score, "pipeline": pipeline}


def render_live_execution_graph(
    action_probabilities: Dict[str, float],
    query: str,
    selected_agents: int,
    validation_score: float,
    latency_ms: float,
    steps: List[dict],
    route: Dict[str, object] | None = None,
) -> str:
    """Render execution facts returned by the live LLM stages."""
    query_label = query[:54] + ("..." if len(query) > 54 else "")
    if route and route.get("decision") == "direct":
        return "\n".join(
            [
                "DIRECT LLM ROUTE",
                "================",
                f"QUERY: '{query_label}'",
                f"IMPORTANCE: {route.get('importance', 'unknown')} / 5",
                f"LATENCY: {latency_ms:.0f} ms",
                f"REASON: {route.get('reason', 'Classified for a direct answer.')}",
                "",
                "[INPUT] --> [QUERY LEVEL AND IMPORTANCE FINDER] --> [DIRECT ANSWER]",
            ]
        )

    completed_roles = {step["role"] for step in steps if step.get("status") == "completed"}
    pipeline = list(ROLE_PIPELINE)
    if route and route.get("requires_web_search"):
        pipeline.insert(1, ("web_retriever", "retrieve_public_text"))
    pipeline_text = " --> ".join(f"[{role.replace('_', ' ').upper()}]" for role, _ in pipeline)
    lines = [
        "LIVE LLM WORKFLOW",
        "=================",
        f"QUERY: '{query_label}'",
        f"MODEL VALIDATION SCORE: {validation_score:.2f} | LATENCY: {latency_ms:.0f} ms",
        f"SELECTED LIGHTWEIGHT WORKERS: {selected_agents}",
        "",
        f"[INPUT] --> {pipeline_text} --> [OUTPUT]",
        "",
    ]

    for role, action in pipeline:
        probability = float(action_probabilities.get(action, 0.0))
        status = "completed" if role in completed_roles else "not_run"
        lines.append(f"[{role.upper()}] action={action:<10} p={probability:.3f} status={status}")

    lines.extend(["", "Validation score is a model self-assessment, not a guarantee of factual correctness."])
    return "\n".join(lines)
