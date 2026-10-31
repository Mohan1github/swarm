from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


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

        response = self.client.chat.completions.create(
            model=self.vision_model if has_images else self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

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

        response = self.client.chat.completions.create(
            model=self.vision_model if has_images else self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if isinstance(text, str) and text:
                yield text


def build_llm_adapter() -> LLMAdapter:
    return LLMAdapter()


def describe_provider_error(error: Exception) -> str:
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
