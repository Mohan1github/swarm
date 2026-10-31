#!/usr/bin/env python3
"""Live terminal swarm interface for the 10,000-agent reasoning prototype.

This file gives the swarm a real-time terminal UI feel. It shows each agent role
working on the task sequentially with a live status area and final synthesized answer.
"""

from __future__ import annotations

import os
import sys

from agents.orchestrator import ReasoningSwarmEngine, WorkflowExecutionError
from graph_visualizer import render_live_execution_graph


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def render_header() -> None:
    print("=" * 90)
    print("   SWARM INTELLIGENCE TERMINAL AGENT")
    print("   10,000 virtual specialist agents coordinating on your query")
    print("=" * 90)


def render_progress(index: int, total: int, role: str, duration_ms: float) -> None:
    print(f"[{index}/{total}] {role.upper()} completed in {duration_ms:.0f} ms")


def run_live_query(query: str) -> None:
    engine = ReasoningSwarmEngine(total_agents=10_000)

    clear_screen()
    render_header()
    print("\nUser query:")
    print(f"  {query}\n")
    print("Swarm coordination stream:\n")

    if not engine.llm.enabled:
        raise SystemExit("Live LLM is not configured. Set GROQ_API_KEY and restart the terminal app.")

    workflow = None
    selected_agents = 0
    for event in engine.iter_workflow(query):
        if event["type"] == "error":
            raise SystemExit(f"Live workflow failed at {event['stage']}: {event['message']}")
        if event["type"] == "setup":
            selected_agents = event["selected_agents"]
            print(f"Selected lightweight workers: {selected_agents}")
        elif event["type"] == "step":
            render_progress(len([step for step in (workflow or {}).get("steps", [])]) + 1, 7, event["role"], event["duration_ms"])
            print(event["output"])
            if workflow is None:
                workflow = {"steps": []}
            workflow["steps"].append(event)
        elif event["type"] == "complete":
            workflow = event

    print("\n" + "-" * 90)
    print(render_live_execution_graph(
        workflow["action_probabilities"],
        query,
        selected_agents=workflow["selected_agents"],
        validation_score=workflow["validation_score"],
        latency_ms=workflow["latency_ms"],
        steps=workflow["steps"],
    ))
    print("\nProbabilistic action plan:")
    for action, probability in workflow["action_probabilities"].items():
        print(f"  {action}: {probability:.4f}")
    top_action = max(workflow["action_probabilities"].items(), key=lambda item: item[1])[0]
    print(f"Top action: {top_action}")
    print("\nFinal synthesis:")
    print(workflow["answer"])
    print("\n" + "-" * 90)
    print(f"\nSelected lightweight workers: {selected_agents}")
    print(f"Model: {workflow['model']} | Validation self-assessment: {workflow['validation_score']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "How do autonomous agents coordinate under load?"

    run_live_query(user_query)
