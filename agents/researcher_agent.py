from __future__ import annotations

from .shared import BaseAgent


class ResearcherAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="researcher",
            specialty="online_retrieval",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Research signal: gather evidence, relevant context, and assumptions before choosing a path. "
            f"The likely relevant area for '{query}' is the factual basis and prior context."
        )
