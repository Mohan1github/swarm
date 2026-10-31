from __future__ import annotations

from .shared import BaseAgent


class CorrectorAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="corrector",
            specialty="evidence_grounded_correction",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Correction pass: compare the draft with the request and supplied evidence, "
            "then identify concrete, supported fixes. "
            f"For '{query}', prioritize omissions, contradictions, and unsupported claims."
        )
