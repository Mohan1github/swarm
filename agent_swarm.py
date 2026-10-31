#!/usr/bin/env python3
"""Entry point for the swarm architecture.

This module exposes the same ReasoningSwarmEngine API used by the tests while
loading the newer role-separated agent implementation from the agents package.
"""

from __future__ import annotations

from agents.orchestrator import ReasoningSwarmEngine


def main() -> None:
    engine = ReasoningSwarmEngine(total_agents=10_000)
    print("Swarm initialized.")
    print(f"Total agents: {len(engine.agents)}")
    print(f"Role distribution: { {role: len(pool) for role, pool in engine.role_buckets.items()} }")

    question = "How do autonomous agents coordinate under load?"
    if not engine.llm.enabled:
        raise SystemExit("Live LLM is not configured. Set GROQ_API_KEY and restart the application.")

    workflow = None
    print("\nLive stage execution:")
    for event in engine.iter_workflow(question):
        if event["type"] == "error":
            raise SystemExit(f"Live workflow failed at {event['stage']}: {event['message']}")
        if event["type"] == "setup":
            print(f"Selected lightweight workers: {event['selected_agents']}")
        elif event["type"] == "step":
            print(f"[{event['role'].upper()}] {event['duration_ms']:.0f} ms")
            print(event["output"])
        elif event["type"] == "complete":
            workflow = event

    if workflow is None:
        raise SystemExit("Live workflow ended without a completed result.")

    print("\nUser query:", question)
    print("\nAnswer preview:")
    print(workflow["answer"][:1500])

    print("\nSummary:")
    print(f"Agents active after processing: {sum(1 for agent in engine.agents if agent.active)}")
    print(f"Model: {workflow['model']} | Total latency: {workflow['latency_ms']:.0f} ms")


if __name__ == "__main__":
    main()
