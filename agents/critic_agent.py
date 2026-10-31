from __future__ import annotations

from .shared import BaseAgent


class CriticAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="critic",
            specialty="risk_assessment",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Critical review: challenge assumptions, identify blind spots, and flag uncertainty. "
            f"For '{query}', evaluate what could be wrong or under-specified before finalizing the result."
        )
