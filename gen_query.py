"""Query-generation contracts and prompt-based query rewriting methods.

To add a query rewriting approach, implement ``QueryGenerator.generate`` and
pass the instance to ``Evaluator``.  The evaluation and retrieval code does
not need to change.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from prompt import BASELINE_PROMPT, METHOD_V1_PROMPT, MULTI_QUERY_PROMPT


PROMPT_TEMPLATES = {
    "baseline": BASELINE_PROMPT,
    "method_v1": METHOD_V1_PROMPT,
    "multi_query": MULTI_QUERY_PROMPT,
}
CUSTOM_METHOD = "custom"
SUPPORTED_QUERY_METHODS = tuple(PROMPT_TEMPLATES) + (CUSTOM_METHOD,)
MAX_MULTI_QUERIES = 3
DIALOGUE_PLACEHOLDER = "{dialogue}"


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


class MultiQueryGenerator(QueryGenerator):
    """Extension point for strategies that can propose several queries."""

    @abstractmethod
    def generate_queries(self, dialogue: str) -> list[str]:
        """Return one or more retrieval queries for a complete dialogue."""

    def generate(self, dialogue: str) -> str:
        return self.generate_queries(dialogue)[0]


@dataclass(frozen=True)
class LLMConfig:
    """Connection parameters for an OpenAI-compatible chat-completions API."""

    base_url: str
    model_name: str
    api_key: str
    temperature: float = 0.0

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


def normalize_query_method(method: str) -> str:
    """Return the canonical name for a configured prompt method.

    ``METHOD_V1`` and ``METHOD_V1_PROMPT`` are accepted as convenient aliases
    for the constant in ``prompt.py``; artifacts always use ``method_v1``.
    """

    if not isinstance(method, str) or not method.strip():
        raise ValueError("query method must be a non-empty string")
    normalized = method.strip().lower().replace("-", "_")
    if normalized.endswith("_prompt"):
        normalized = normalized[: -len("_prompt")]
    if normalized not in SUPPORTED_QUERY_METHODS:
        supported = ", ".join(SUPPORTED_QUERY_METHODS)
        raise ValueError(f"unsupported query method '{method}'; supported methods: {supported}")
    return normalized


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


def extract_queries(model_content: str, *, max_queries: int = MAX_MULTI_QUERIES) -> list[str]:
    """Extract one or more queries from the JSON response required by ``MULTI_QUERY_PROMPT``.

    ``query`` may be a JSON array (the intended shape) or a single string, so
    both forms are accepted. Duplicate and empty entries are dropped while
    preserving order, and the result is capped at ``max_queries``.
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
        if not isinstance(parsed, dict):
            continue
        raw_query = parsed.get("query")
        if isinstance(raw_query, str):
            raw_values: list[Any] = [raw_query]
        elif isinstance(raw_query, list):
            raw_values = raw_query
        else:
            continue

        queries: list[str] = []
        for value in raw_values:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped and stripped not in queries:
                queries.append(stripped)
        if queries:
            return queries[:max_queries]

    raise QueryGenerationError("model response must contain a non-empty JSON 'query'")


def _call_model(
    client: ChatCompletionsClient,
    model_name: str,
    prompt: str,
    temperature: float,
) -> str:
    """Call the chat-completions endpoint with thinking disabled and return its text."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise QueryGenerationError("model response has no first message content") from exc


def load_prompt_template(path: str) -> str:
    """Read a custom prompt template from disk and validate its placeholder.

    The file must contain the literal ``{dialogue}`` placeholder, the same
    marker used by the built-in prompt constants, so the dialogue can be
    substituted in without silently appending it in an unexpected place.
    """

    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"prompt file does not exist: {source}") from exc
    except OSError as exc:
        raise ValueError(f"could not read prompt file: {source}: {exc}") from exc

    if not content.strip():
        raise ValueError(f"prompt file is empty: {source}")
    if DIALOGUE_PLACEHOLDER not in content:
        raise ValueError(
            f"prompt file must contain a '{DIALOGUE_PLACEHOLDER}' placeholder: {source}"
        )
    return content


class _PromptGeneratorBase:
    """Shared validation and prompt binding for prompt-based generators."""

    def __init__(
        self,
        client: ChatCompletionsClient,
        model_name: str,
        method: str,
        temperature: float = 0.0,
        prompt_template: str | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")
        self._client = client
        self._model_name = model_name
        self._method = normalize_query_method(method)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self._temperature = float(temperature)
        if self._method == CUSTOM_METHOD:
            if not prompt_template or not prompt_template.strip():
                raise ValueError("prompt_template is required for the 'custom' method")
            self._prompt_template = prompt_template
        else:
            self._prompt_template = PROMPT_TEMPLATES[self._method]

    @property
    def method(self) -> str:
        """Canonical method name used to create this generator."""

        return self._method

    def _render_prompt(self, dialogue: str) -> str:
        if not dialogue or not dialogue.strip():
            raise ValueError("dialogue must not be empty")
        # The prompt includes a literal JSON example, whose braces must not be
        # interpreted as ``str.format`` fields.
        return self._prompt_template.replace("{dialogue}", dialogue)


class PromptQueryGenerator(_PromptGeneratorBase, QueryGenerator):
    """One-shot generator backed by one named prompt template."""

    def generate(self, dialogue: str) -> str:
        prompt = self._render_prompt(dialogue)
        content = _call_model(self._client, self._model_name, prompt, self._temperature)
        return extract_query(content)


class PromptMultiQueryGenerator(_PromptGeneratorBase, MultiQueryGenerator):
    """Generator that proposes several queries, backed by ``MULTI_QUERY_PROMPT``."""

    def generate_queries(self, dialogue: str) -> list[str]:
        prompt = self._render_prompt(dialogue)
        content = _call_model(self._client, self._model_name, prompt, self._temperature)
        return extract_queries(content)


class BaselineQueryGenerator(PromptQueryGenerator):
    """One-shot baseline defined by ``prompt.BASELINE_PROMPT``."""

    def __init__(self, client: ChatCompletionsClient, model_name: str) -> None:
        super().__init__(client, model_name, "baseline")

    @classmethod
    def from_config(cls, config: LLMConfig) -> "BaselineQueryGenerator":
        return cls(build_openai_client(config), config.model_name)


class MethodV1QueryGenerator(PromptQueryGenerator):
    """Prompt-improved query generator defined by ``prompt.METHOD_V1_PROMPT``."""

    def __init__(self, client: ChatCompletionsClient, model_name: str) -> None:
        super().__init__(client, model_name, "method_v1")

    @classmethod
    def from_config(cls, config: LLMConfig) -> "MethodV1QueryGenerator":
        return cls(build_openai_client(config), config.model_name)


def create_query_generator(
    config: LLMConfig,
    method: str,
    *,
    prompt_file: str | None = None,
) -> QueryGenerator:
    """Create a supported prompt method from LLM configuration.

    ``prompt_file`` is required when ``method`` normalizes to ``custom`` and
    ignored otherwise. Future prompt-only methods need only be added to
    ``PROMPT_TEMPLATES``; the two-stage evaluation code remains unchanged.
    """

    normalized = normalize_query_method(method)
    client = build_openai_client(config)
    if normalized == CUSTOM_METHOD:
        if not prompt_file or not prompt_file.strip():
            raise ValueError("prompt_file is required for the 'custom' method")
        prompt_template = load_prompt_template(prompt_file)
        return PromptQueryGenerator(
            client, config.model_name, normalized, prompt_template=prompt_template
        )
    if normalized == "multi_query":
        return PromptMultiQueryGenerator(client, config.model_name, normalized)
    return PromptQueryGenerator(client, config.model_name, normalized)
