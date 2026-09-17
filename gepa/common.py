"""Shared configuration, data loading, and artifact helpers for GEPA runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LMSettings:
    base_url: str
    model_name: str
    api_key: str
    temperature: float = 0.0


@dataclass(frozen=True)
class RetrievalSettings:
    url: str
    timeout: float
    top_k: int


@dataclass(frozen=True)
class RunSettings:
    task_lm: LMSettings
    reflection_lm: LMSettings
    retrieval: RetrievalSettings
    input_path: Path
    train_ratio: float
    split_seed: int
    metric_cutoff: int
    max_metric_calls: int
    num_threads: int
    output_dir: Path


@dataclass(frozen=True)
class DialogueExample:
    sample_index: int
    call_sno: str | None
    dialogue: str
    expected_case_id: str


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration field '{name}' must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration field '{name}' must be a non-empty string")
    return value.strip()


def _lm(payload: Mapping[str, Any], name: str) -> LMSettings:
    section = _mapping(payload, name)
    env_name = _string(section.get("api_key_env", "OPENAI_API_KEY"), f"{name}.api_key_env")
    api_key = section.get("api_key") or os.getenv(env_name)
    temperature = section.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
        raise ValueError(f"configuration field '{name}.temperature' must be between 0 and 2")
    return LMSettings(
        base_url=_string(section.get("base_url"), f"{name}.base_url"),
        model_name=_string(section.get("model_name"), f"{name}.model_name"),
        api_key=_string(api_key, f"{name}.api_key or environment variable {env_name}"),
        temperature=float(temperature),
    )


def load_settings(path: str | Path) -> RunSettings:
    config_path = Path(path).resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file does not exist: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be a JSON object")

    retrieval = _mapping(payload, "retrieval")
    dataset = _mapping(payload, "dataset")
    optimization = _mapping(payload, "optimization")
    root = config_path.parent
    input_path = Path(_string(dataset.get("input_path"), "dataset.input_path"))
    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()
    output_dir = Path(_string(payload.get("output_dir", "runs"), "output_dir"))
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    def positive_int(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"configuration field '{name}' must be a positive integer")
        return value

    train_ratio = dataset.get("train_ratio", 0.8)
    if not isinstance(train_ratio, (int, float)) or isinstance(train_ratio, bool) or not 0 < train_ratio < 1:
        raise ValueError("configuration field 'dataset.train_ratio' must be between 0 and 1")
    return RunSettings(
        task_lm=_lm(payload, "task_lm"),
        reflection_lm=_lm(payload, "reflection_lm"),
        retrieval=RetrievalSettings(
            url=_string(retrieval.get("url"), "retrieval.url"),
            timeout=float(retrieval.get("timeout", 30)),
            top_k=positive_int(retrieval.get("top_k", 10), "retrieval.top_k"),
        ),
        input_path=input_path,
        train_ratio=float(train_ratio),
        split_seed=int(dataset.get("split_seed", 42)),
        metric_cutoff=positive_int(optimization.get("metric_cutoff", 5), "optimization.metric_cutoff"),
        max_metric_calls=positive_int(optimization.get("max_metric_calls", 100), "optimization.max_metric_calls"),
        num_threads=positive_int(optimization.get("num_threads", 1), "optimization.num_threads"),
        output_dir=output_dir,
    )


def load_examples(path: str | Path) -> list[DialogueExample]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"dialogue input does not exist: {path}") from exc
    if not isinstance(payload, list):
        raise ValueError("dialogue input must be a JSON array")
    examples: list[DialogueExample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            continue
        dialogue, case_id = item.get("chat_content"), item.get("caseID")
        if not isinstance(dialogue, str) or not dialogue.strip() or case_id is None or not str(case_id).strip():
            continue
        call_sno = item.get("call_sno")
        examples.append(DialogueExample(index, str(call_sno).strip() if call_sno is not None else None, dialogue.strip(), str(case_id).strip()))
    if len(examples) < 2:
        raise ValueError("at least two valid dialogue samples with chat_content and caseID are required")
    return examples


def split_examples(examples: Sequence[DialogueExample], train_ratio: float, seed: int) -> tuple[list[DialogueExample], list[DialogueExample]]:
    import random

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train_size = min(max(1, round(len(shuffled) * train_ratio)), len(shuffled) - 1)
    return shuffled[:train_size], shuffled[train_size:]


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(target)
