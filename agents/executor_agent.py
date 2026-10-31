from __future__ import annotations

from .shared import BaseAgent


class ExecutorAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="executor",
            specialty="task_execution",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Execution path: turn the plan into measurable actions and concrete next steps. "
            f"For '{query}', prioritize the operational steps that drive the best outcome."
        )
