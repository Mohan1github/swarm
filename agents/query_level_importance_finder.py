from __future__ import annotations

import json


class QueryClassificationError(ValueError):
    pass


class QueryLevelImportanceFinder:
    def __init__(self, llm):
        self.llm = llm

    def classify(self, query: str, attachment_context: str = "", has_images: bool = False) -> dict:
        prompt = (
            "You are the query level and importance finder. Decide whether this request needs the staged "
            "reasoning workflow or can be answered directly by the LLM. Choose `direct` only for simple, "
            "low-importance requests such as greetings, definitions, or straightforward everyday questions. "
            "Choose `reasoning` for research, technical work, analysis, comparisons, multi-step tasks, "
            "decisions with meaningful consequences, or requests needing careful verification. "
            "Decide whether external public-web evidence is necessary; set `requires_web_search` true "
            "for current or changing facts, current news, fact-checking, explicit source/citation "
            "requests, or longer fact-intensive research where reliable public evidence is needed. "
            "Sensitive public-interest or high-stakes medical, legal, financial, or civic questions "
            "need current authoritative public evidence when the answer depends on changing facts. "
            "Do not search for ordinary definitions, timeless/basic facts, code help, private or "
            "personally identifying information, credentials, or content in attachments. Private "
            "account-specific questions and personal details must never be sent to web search. "
            "Web search is permitted "
            "only when `route` is `reasoning`. Set `search_topic` to `news` for news, `finance` for "
            "current financial/market information, otherwise `general`. Return JSON only with `route` (`direct` or `reasoning`), "
            "`importance` (integer 1-5), `reason` (short string), `requires_web_search` (boolean), and "
            "`search_topic` (`general`, `news`, or `finance`).\n\n"
            f"USER REQUEST:\n{query}\n\n"
            f"ATTACHED FILE CONTEXT:\n{attachment_context or '(none)'}\n"
            f"IMAGE ATTACHMENTS: {'yes' if has_images else 'no'}"
        )
        for attempt in range(2):
            output = self.llm.generate(prompt, max_tokens=160)
            try:
                start = output.find("{")
                end = output.rfind("}")
                if start < 0 or end < start:
                    raise ValueError("Query classifier response does not contain a JSON object")

                result = json.loads(output[start : end + 1])
                route = result["route"]
                importance = result["importance"]
                reason = result["reason"]
                if route not in {"direct", "reasoning"}:
                    raise ValueError("Query classifier route is invalid")
                if isinstance(importance, bool) or not isinstance(importance, int) or not 1 <= importance <= 5:
                    raise ValueError("Query classifier importance must be an integer from 1 to 5")
                if not isinstance(reason, str):
                    raise ValueError("Query classifier reason must be a string")
                if importance >= 4:
                    route = "reasoning"
                requires_web_search = result.get("requires_web_search", False)
                search_topic = result.get("search_topic", "general")
                if not isinstance(requires_web_search, bool):
                    raise ValueError("The web-search decision must be a boolean")
                if search_topic not in {"general", "news", "finance"}:
                    raise ValueError("The web-search topic is invalid")
                if route == "direct":
                    requires_web_search = False
                if not requires_web_search:
                    search_topic = "general"
                return {
                    "decision": route,
                    "importance": importance,
                    "reason": reason,
                    "requires_web_search": requires_web_search,
                    "search_topic": search_topic,
                }
            except (ValueError, KeyError, TypeError) as error:
                if attempt:
                    raise QueryClassificationError(
                        "The query classifier returned an invalid response after one retry."
                    ) from error
                prompt += (
                    "\nYour previous response did not match the required schema. "
                    "Return exactly one valid JSON object with route, integer importance from 1 to 5, "
                    "string reason, boolean requires_web_search, and search_topic of general, news, or finance. "
                    "Do not include markdown or commentary."
                )

        raise QueryClassificationError("The query classifier could not classify the request.")