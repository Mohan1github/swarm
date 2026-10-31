from __future__ import annotations

from .shared import BaseAgent


class ValidatorAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="validator",
            specialty="verification",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Verification step: check consistency, confidence, and evidence before finalizing. "
            f"For '{query}', validate assumptions and identify unsupported claims or gaps."
        )
