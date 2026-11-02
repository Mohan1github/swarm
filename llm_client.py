from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from logging import getLogger
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


LOGGER = getLogger(__name__)
EMPTY_RESPONSE_MAX_ATTEMPTS = 3


class EmptyLLMResponseError(RuntimeError):
    def __init__(self, attempts: int, finish_reason: str | None = None) -> None:
        self.attempts = attempts
        self.finish_reason = finish_reason
        detail = f"The LLM provider returned no text after {attempts} attempts."
        if finish_reason:
            detail += f" Last finish reason: {finish_reason}."
        super().__init__(detail)


class LLMAdapter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        groq_configured = bool(
            os.getenv("GROQ_API_KEY") or os.getenv("GROQ_BASE_URL") or os.getenv("GROQ_MODEL")
        )
        openai_configured = bool(os.getenv("OPENAI_API_KEY")) and not groq_configured
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("GROQ_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or (
                "https://api.openai.com/v1"
                if openai_configured
                else "https://api.groq.com/openai/v1"
            )
        )
        self.model = (
            model
            or os.getenv("GROQ_MODEL")
            or os.getenv("OPENAI_MODEL")
            or ("gpt-4o-mini" if openai_configured else "openai/gpt-oss-120b")
        )
        self.vision_model = os.getenv("GROQ_VISION_MODEL") or self.model
        self.client = None

        if self.api_key and OpenAI is not None:
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 800,
        image_data_urls: Optional[list[str]] = None,
    ) -> str:
        return self._generate_completion(prompt, max_tokens, image_data_urls)

    def generate_json(self, prompt: str, max_tokens: int = 512) -> str:
        return self._generate_completion(
            prompt,
            max_tokens,
            response_format={"type": "json_object"},
            temperature=0,
        )

    def _generate_completion(
        self,
        prompt: str,
        max_tokens: int,
        image_data_urls: Optional[list[str]] = None,
        response_format: dict | None = None,
        temperature: float = 0.3,
    ) -> str:
        if not self.enabled or self.client is None:
            raise RuntimeError("No LLM backend is configured. Set GROQ_API_KEY or provide a compatible client.")

        has_images = bool(image_data_urls)
        content = prompt
        if has_images:
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": image_data_url}}
                for image_data_url in image_data_urls or []
            )

        request = {
            "model": self.vision_model if has_images else self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format
        last_finish_reason = None
        for attempt in range(1, EMPTY_RESPONSE_MAX_ATTEMPTS + 1):
            response = self.client.chat.completions.create(**request)
            choices = getattr(response, "choices", None) or []
            if choices:
                choice = choices[0]
                last_finish_reason = getattr(choice, "finish_reason", None)
                message = getattr(choice, "message", None)
                output = self._message_text(message)
                if output:
                    return output
                refusal = getattr(message, "refusal", None)
                if isinstance(refusal, str) and refusal.strip():
                    return refusal.strip()

            LOGGER.warning(
                "LLM provider returned an empty completion (model=%s, attempt=%s/%s, finish_reason=%s)",
                request["model"],
                attempt,
                EMPTY_RESPONSE_MAX_ATTEMPTS,
                last_finish_reason or "unknown",
            )
            if attempt < EMPTY_RESPONSE_MAX_ATTEMPTS:
                time.sleep(0.25 * attempt)
                if last_finish_reason == "length":
                    request["max_tokens"] = min(max_tokens * 2, 4096)

        raise EmptyLLMResponseError(EMPTY_RESPONSE_MAX_ATTEMPTS, last_finish_reason)

    @staticmethod
    def _message_text(message: object) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ]
            return "".join(text_parts).strip()
        return ""

    def stream(
        self,
        prompt: str,
        max_tokens: int = 800,
        image_data_urls: Optional[list[str]] = None,
    ) -> Iterator[str]:
        if not self.enabled or self.client is None:
            raise RuntimeError("No LLM backend is configured. Set GROQ_API_KEY or provide a compatible client.")

        has_images = bool(image_data_urls)
        content = prompt
        if has_images:
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": image_data_url}}
                for image_data_url in image_data_urls or []
            )

        request = {
            "model": self.vision_model if has_images else self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": True,
        }
        for attempt in range(1, EMPTY_RESPONSE_MAX_ATTEMPTS + 1):
            response = self.client.chat.completions.create(**request)
            has_text = False
            for chunk in response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                text = getattr(choices[0].delta, "content", None)
                if isinstance(text, str) and text:
                    has_text = True
                    yield text
            if has_text:
                return

            LOGGER.warning(
                "LLM provider returned an empty stream (model=%s, attempt=%s/%s)",
                request["model"],
                attempt,
                EMPTY_RESPONSE_MAX_ATTEMPTS,
            )
            if attempt < EMPTY_RESPONSE_MAX_ATTEMPTS:
                time.sleep(0.25 * attempt)

        raise EmptyLLMResponseError(EMPTY_RESPONSE_MAX_ATTEMPTS)


def build_llm_adapter() -> LLMAdapter:
    return LLMAdapter()


def describe_provider_error(error: Exception) -> str:
    if isinstance(error, EmptyLLMResponseError):
        return str(error)

    details = [type(error).__name__]
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        details.append(f"HTTP {status_code}")

    error_code = getattr(error, "code", None)
    if not error_code:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            provider_error = body.get("error")
            if isinstance(provider_error, dict):
                error_code = provider_error.get("code") or provider_error.get("type")
    if isinstance(error_code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", error_code):
        details.append(f"code {error_code}")

    return f"LLM provider request failed ({', '.join(details)})."
