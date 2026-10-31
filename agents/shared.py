from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentResult:
    role: str
    specialty: str
    output: str
    confidence: float = 0.0
    action: str = "reason"
    action_probability: float = 0.0


@dataclass
class BaseAgent:
    agent_id: int
    role: str
    specialty: str
    task_budget: int = 6
    memory_capacity: int = 512
    active: bool = True
    task_count: int = 0
    memory: List[str] = field(default_factory=list)
    action_scores: Dict[str, float] = field(default_factory=dict)

    def _role_action_profile(self) -> Dict[str, float]:
        role_profiles = {
            "planner": {"decompose": 0.34, "plan": 0.28, "prioritize": 0.18, "validate": 0.12, "execute": 0.08},
            "researcher": {"research": 0.36, "gather": 0.27, "compare": 0.15, "summarize": 0.12, "verify": 0.10},
            "analyst": {"analyze": 0.38, "compare": 0.22, "detect": 0.17, "forecast": 0.13, "summarize": 0.10},
            "critic": {"challenge": 0.33, "validate": 0.25, "risk_check": 0.22, "question": 0.12, "refine": 0.08},
            "corrector": {"correct": 0.36, "repair": 0.26, "verify": 0.20, "align": 0.12, "refine": 0.06},
            "executor": {"execute": 0.42, "optimize": 0.23, "schedule": 0.18, "monitor": 0.10, "recover": 0.07},
            "validator": {"validate": 0.39, "verify": 0.23, "audit": 0.18, "check": 0.12, "mitigate": 0.08},
            "synthesizer": {"synthesize": 0.41, "summarize": 0.22, "align": 0.18, "conclude": 0.12, "restate": 0.07},
            "memory_agent": {"remember": 0.34, "reuse": 0.27, "retrieve": 0.18, "connect": 0.12, "stabilize": 0.09},
        }
        return role_profiles.get(self.role, {"reason": 1.0})

    def _update_action_scores(self, query: str | None = None) -> Dict[str, float]:
        base = self._role_action_profile().copy()
        query_text = (query or "").lower()
        keyword_keys = {
            "search": ["how", "what", "where", "who", "when"],
            "analyze": ["compare", "trend", "pattern", "reason", "why"],
            "validate": ["check", "verify", "test", "audit", "ensure"],
            "execute": ["act", "run", "build", "deploy", "perform"],
            "schedule": ["plan", "timeline", "roadmap", "steps"],
            "risk_check": ["risk", "uncertain", "fail", "threat", "problem"],
        }

        for action_name, trigger_words in keyword_keys.items():
            if any(word in query_text for word in trigger_words):
                for key, value in base.items():
                    if key.startswith(action_name) or action_name in key:
                        base[key] = value * 1.35

        total = sum(base.values())
        normalized = {key: value / total for key, value in base.items()}
        self.action_scores = normalized
        return normalized

    def decide_action(self, query: str | None = None) -> tuple[str, float]:
        if not self.action_scores:
            self._update_action_scores(query)
        action_name, probability = random.choices(
            population=list(self.action_scores.keys()),
            weights=list(self.action_scores.values()),
            k=1,
        )[0], max(self.action_scores.values())
        return action_name, probability

    def process(self, workflow: str, query: str | None = None) -> AgentResult:
        if not self.active:
            return AgentResult(self.role, self.specialty, "inactive", 0.0, "idle", 0.0)

        self.task_count += 1
        if self.task_count > self.task_budget:
            self.active = False
            return AgentResult(self.role, self.specialty, "deactivated", 0.0, "stop", 0.0)

        self.memory.append(workflow)
        if len(self.memory) > self.memory_capacity:
            self.memory = self.memory[-self.memory_capacity:]

        result = self.reason(query or workflow)
        chosen_action, chosen_probability = self.decide_action(query or workflow)
        return AgentResult(
            self.role,
            self.specialty,
            result,
            self._confidence(),
            chosen_action,
            round(chosen_probability, 4),
        )

    def reason(self, query: str) -> str:
        return f"{self.role}:{query}"

    def _confidence(self) -> float:
        return round(min(0.99, 0.45 + (self.task_count / (self.task_budget + 2))), 2)
