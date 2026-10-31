from __future__ import annotations

from .shared import BaseAgent


class MemoryAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="memory_agent",
            specialty="knowledge_retention",
            task_budget=6,
            memory_capacity=1024,
        )

    def reason(self, query: str) -> str:
        return (
            "Memory trace: reuse prior patterns and successful workflows to stabilize reasoning. "
            f"For '{query}', align this task with known precedents and retained context."
        )
