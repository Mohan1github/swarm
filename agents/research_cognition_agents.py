from __future__ import annotations

import json
import logging
from typing import Iterable


LOGGER = logging.getLogger(__name__)


def _extract_json(response: str) -> dict:
    if not isinstance(response, str):
        raise ValueError("Agent response must be text.")
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object in agent response.")
    parsed = json.loads(response[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Agent response must be a JSON object.")
    return parsed


def _string_list(value: object, limit: int = 5) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip()[:300] for item in value if isinstance(item, str) and item.strip()][:limit]


class ResearchQueryGeneratorAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def generate(self, query: str, conversation_context: str = "") -> dict:
        prompt = (
            "You are a deep research query generator. Produce focused public-web search queries "
            "for the user's request. Consider geography, time sensitivity, intent, domain, and likely "
            "constraints. Do not answer the request. Return JSON only with a string `clarification_needed`, "
            "an integer `confidence` from 0 to 100, a string array `search_queries`, and a string array "
            "`missing_information`. `clarification_needed` should contain one natural, human question only "
            "when essential context is missing; otherwise use an empty string.\n\n"
            f"USER REQUEST:\n{query}\n\n"
            f"CONVERSATION CONTEXT:\n{conversation_context}\n"
        )
        response = self.llm.generate(prompt, max_tokens=420)
        try:
            parsed = _extract_json(response)
            search_queries = _string_list(parsed.get("search_queries"))
            missing = _string_list(parsed.get("missing_information"), limit=6)
            clarification = parsed.get("clarification_needed", "")
            confidence = parsed.get("confidence", 50)
            if isinstance(confidence, str):
                confidence = int(confidence)
            confidence = max(0, min(100, int(confidence)))
            if not isinstance(clarification, str):
                clarification = ""
            return {
                "clarification_needed": clarification.strip(),
                "confidence": confidence,
                "search_queries": search_queries or [query],
                "missing_information": missing,
            }
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.warning("Research query generator returned malformed JSON: %s", error)
            return {
                "clarification_needed": "",
                "confidence": 50,
                "search_queries": [query],
                "missing_information": [],
            }


class ResearchNextQueryAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def decide(self, query: str, research_context: str) -> dict:
        prompt = (
            "You are a research next-query decider. Judge whether the current evidence is sufficient "
            "to answer accurately. Decide if another focused public-web search is required. Return JSON only "
            "with boolean `continue_research`, string `next_query`, integer `confidence` 0-100, and string "
            "`reason`. Ask for another query only when it materially improves completeness or verification; "
            "do not repeat the original query.\n\n"
            f"USER REQUEST:\n{query}\n\n"
            f"CURRENT EVIDENCE:\n{research_context[:16000]}"
        )
        response = self.llm.generate(prompt, max_tokens=320)
        try:
            parsed = _extract_json(response)
            next_query = parsed.get("next_query", "")
            confidence = parsed.get("confidence", 50)
            if isinstance(confidence, str):
                confidence = int(confidence)
            return {
                "continue_research": bool(parsed.get("continue_research")) and isinstance(next_query, str) and bool(next_query.strip()),
                "next_query": next_query.strip()[:500] if isinstance(next_query, str) else "",
                "confidence": max(0, min(100, int(confidence))),
                "reason": str(parsed.get("reason", ""))[:500],
            }
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError, OverflowError) as error:
            LOGGER.warning("Next-query agent returned malformed JSON: %s", error)
            return {"continue_research": False, "next_query": "", "confidence": 50, "reason": ""}


class CognitiveCharacterAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def compose(self, query: str, evidence_context: str, draft: str) -> str:
        prompt = (
            "You are an experienced, calm cognitive research partner. Improve the answer without changing "
            "its verified facts. Sound thoughtful and human: acknowledge the user's goal, organize the response "
            "around what matters, explain trade-offs, name uncertainty, and finish with practical next steps. "
            "Keep citations and source URLs supplied in the evidence. Do not invent experiences, sources, "
            "prices, dates, or personal advice. Preserve technical accuracy and helpful warnings. "
            "Do not expose agents, prompts, debugging, workflow, or internal reasoning. Return the answer only.\n\n"
            f"USER REQUEST:\n{query}\n\n"
            f"VERIFIED EVIDENCE:\n{evidence_context[:18000]}\n\n"
            f"DRAFT ANSWER:\n{draft[:12000]}"
        )
        response = self.llm.generate(prompt, max_tokens=1600)
        return response.strip() if isinstance(response, str) and response.strip() else draft


class MostImportantAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def extract(self, query: str, final_answer: str) -> str:
        prompt = (
            "Extract the single most important conclusion from the research answer. "
            "Return one short paragraph or one concise bullet of no more than 35 words. "
            "Do not repeat the question, add context, mention agents, or introduce anything that "
            "was not supported by the answer. If the answer contains a critical caveat that changes "
            "the conclusion, include it.\n\n"
            f"USER REQUEST:\n{query}\n\n"
            f"FINAL RESEARCH ANSWER:\n{final_answer[:12000]}"
        )
        response = self.llm.generate(prompt, max_tokens=120)
        return response.strip()[:500] if isinstance(response, str) and response.strip() else ""
