from __future__ import annotations

from .shared import BaseAgent


class AnalystAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="analyst",
            specialty="pattern_analysis",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Analytical pattern: identify dependencies, structure, and likely causal relationships. "
            f"For '{query}', connect the concept to the larger system and leading indicators."
        )
