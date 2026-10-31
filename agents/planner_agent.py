from __future__ import annotations

from .shared import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="planner",
            specialty="goal_breakdown",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Planned decomposition: break the problem into objective, constraints, and execution stages. "
            f"Focus on the intent behind '{query}'."
        )
