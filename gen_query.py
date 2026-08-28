"""Query-generation contracts and the first baseline implementation.

To add a query rewriting approach, implement ``QueryGenerator.generate`` and
pass the instance to ``Evaluator``.  The evaluation and retrieval code does
not need to change.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from prompt import BASELINE_PROMPT


class QueryGenerationError(RuntimeError):
    """Raised when a model response cannot produce a non-empty query."""


class ChatCompletionsClient(Protocol):
    """The small subset of an OpenAI-compatible client used by the baseline."""

    @property
    def chat(self) -> Any:
        """Expose the OpenAI chat namespace."""


class QueryGenerator(ABC):
    """Stable extension point for all query rewriting strategies."""

    @abstractmethod
    def generate(self, dialogue: str) -> str:
        """Return one retrieval query for a complete dialogue."""


@dataclass(frozen=True)
class LLMConfig:
    """Connection parameters for an OpenAI-compatible chat-completions API."""

    base_url: str
    model_name: str
    api_key: str

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("base_url", self.base_url),
                ("model_name", self.model_name),
                ("api_key", self.api_key),
            )
            if not value or not value.strip()
        ]
        if missing:
            raise ValueError(f"missing LLM configuration: {', '.join(missing)}")


def build_openai_client(config: LLMConfig) -> ChatCompletionsClient:
    """Create the optional SDK client only when a real model call is needed."""

    config.validate()
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment-specific path
        raise RuntimeError(
            "The 'openai' package is required for model calls. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def extract_query(model_content: str) -> str:
    """Extract ``query`` from the JSON response required by ``BASELINE_PROMPT``.

    Some model providers include a Markdown fence while others return the JSON
    object directly, so both forms are accepted.  We intentionally reject a
    malformed answer instead of silently sending arbitrary text to retrieval.
    """

    if not isinstance(model_content, str) or not model_content.strip():
        raise QueryGenerationError("model returned empty content")

    candidates = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        model_content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates.append(model_content.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("query"), str):
            query = parsed["query"].strip()
            if query:
                return query

    raise QueryGenerationError("model response must contain a non-empty JSON 'query'")


class BaselineQueryGenerator(QueryGenerator):
    """One-shot baseline defined by ``prompt.BASELINE_PROMPT``."""

    def __init__(self, client: ChatCompletionsClient, model_name: str) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")
        self._client = client
        self._model_name = model_name

    @classmethod
    def from_config(cls, config: LLMConfig) -> "BaselineQueryGenerator":
        return cls(build_openai_client(config), config.model_name)

    def generate(self, dialogue: str) -> str:
        if not dialogue or not dialogue.strip():
            raise ValueError("dialogue must not be empty")

        # The prompt includes a literal JSON example, whose braces must not be
        # interpreted as ``str.format`` fields.
        prompt = BASELINE_PROMPT.replace("{dialogue}", dialogue)
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise QueryGenerationError("model response has no first message content") from exc
        return extract_query(content)
