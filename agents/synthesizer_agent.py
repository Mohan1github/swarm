from __future__ import annotations

from .shared import BaseAgent


class SynthesizerAgent(BaseAgent):
    def __init__(self, agent_id: int):
        super().__init__(
            agent_id=agent_id,
            role="synthesizer",
            specialty="final_inference",
            task_budget=6,
            memory_capacity=512,
        )

    def reason(self, query: str) -> str:
        return (
            "Synthesis layer: integrate the different perspectives into a coherent final answer. "
            f"For '{query}', reconcile evidence, uncertainty, and priorities into a clean conclusion."
        )
