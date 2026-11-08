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
    from openai import APIError, OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
    APIError = None
LOGGER = getLogger(__name__)
API_ERRORS = (APIError,) if APIError is not None else ()
EMPTY_RESPONSE_MAX_ATTEMPTS = 3
DEFAULT_GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
DEFAULT_GROQ_FALLBACK_MODEL = "qwen/qwen3.8-27b"
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
        self.vision_model = os.getenv("GROQ_VISION_MODEL") or (
            DEFAULT_GROQ_VISION_MODEL
            if "api.groq.com" in self.base_url
            else self.model
        )
        self.last_used_model = self.model
        self.client = None

        if self.api_key and OpenAI is not None:
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _model_candidates(self, has_images: bool) -> list[str]:
        primary_model = self.vision_model if has_images else self.model
        configured = os.getenv(
            "GROQ_VISION_FALLBACK_MODELS" if has_images else "GROQ_FALLBACK_MODELS",
            "",
        )
        fallback_models = [model.strip() for model in configured.split(",") if model.strip()]
        if not has_images and "api.groq.com" in self.base_url and not fallback_models:
            fallback_models = [DEFAULT_GROQ_FALLBACK_MODEL]
        models = [primary_model]
        models.extend(model for model in fallback_models if model and model not in models)
        return models

    def _uses_groq_gpt_oss(self, model: str) -> bool:
        return "api.groq.com" in self.base_url and model.startswith("openai/gpt-oss-")

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

        models = self._model_candidates(has_images)
        request = {
            "model": models[0],
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
        }
        if response_format is not None:
            request["response_format"] = response_format
        last_finish_reason = None
        last_provider_error = None
        for model in models:
            request["model"] = model
            uses_groq_gpt_oss = self._uses_groq_gpt_oss(model)
            token_limit_key = "max_completion_tokens" if uses_groq_gpt_oss else "max_tokens"
            request.pop("max_tokens" if uses_groq_gpt_oss else "max_completion_tokens", None)
            current_max_tokens = max_tokens
            for attempt in range(1, EMPTY_RESPONSE_MAX_ATTEMPTS + 1):
                request[token_limit_key] = current_max_tokens
                if uses_groq_gpt_oss and last_finish_reason == "length" and attempt > 1:
                    request["reasoning_effort"] = "low"
                else:
                    request.pop("reasoning_effort", None)
                try:
                    response = self.client.chat.completions.create(**request)
                except API_ERRORS as error:
                    LOGGER.warning(
                        "LLM request failed; trying the next configured model "
                        "(model=%s, error=%s)",
                        model,
                        type(error).__name__,
                    )
                    last_provider_error = error
                    break
                choices = getattr(response, "choices", None) or []
                if choices:
                    choice = choices[0]
                    last_finish_reason = getattr(choice, "finish_reason", None)
                    message = getattr(choice, "message", None)
                    output = self._message_text(message)
                    if output:
                        self.last_used_model = model
                        return output
                    refusal = getattr(message, "refusal", None)
                    if isinstance(refusal, str) and refusal.strip():
                        return refusal.strip()

                LOGGER.warning(
                    "LLM provider returned an empty completion "
                    "(model=%s, attempt=%s/%s, finish_reason=%s)",
                    model,
                    attempt,
                    EMPTY_RESPONSE_MAX_ATTEMPTS,
                    last_finish_reason or "unknown",
                )
                if attempt < EMPTY_RESPONSE_MAX_ATTEMPTS:
                    time.sleep(0.25 * attempt)
                    if last_finish_reason == "length":
                        if uses_groq_gpt_oss:
                            current_max_tokens = min(max(current_max_tokens * 2, 2048), 8192)
                        else:
                            current_max_tokens = min(current_max_tokens * 2, 4096)

        if last_provider_error is not None:
            raise last_provider_error
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

    def stream( self,prompt: str,max_tokens: int = 800,image_data_urls: Optional[list[str]] = None,) -> Iterator[str]:
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

        models = self._model_candidates(has_images)
        request = {
            "model": models[0],
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": True,
        }
        last_provider_error = None
        for model in models:
            request["model"] = model
            for attempt in range(1, EMPTY_RESPONSE_MAX_ATTEMPTS + 1):
                has_text = False
                try:
                    response = self.client.chat.completions.create(**request)
                    for chunk in response:
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        text = getattr(choices[0].delta, "content", None)
                        if isinstance(text, str) and text:
                            has_text = True
                            self.last_used_model = model
                            yield text
                except API_ERRORS as error:
                    if has_text:
                        raise
                    LOGGER.warning(
                        "LLM stream failed; trying the next configured model "
                        "(model=%s, error=%s)",
                        model,
                        type(error).__name__,
                    )
                    last_provider_error = error
                    break
                if has_text:
                    return

                LOGGER.warning(
                    "LLM provider returned an empty stream (model=%s, attempt=%s/%s)",
                    model,
                    attempt,
                    EMPTY_RESPONSE_MAX_ATTEMPTS,
                )
                if attempt < EMPTY_RESPONSE_MAX_ATTEMPTS:
                    time.sleep(0.25 * attempt)

        if last_provider_error is not None:
            raise last_provider_error
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
