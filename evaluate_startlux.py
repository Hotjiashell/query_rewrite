"""Multi-query filtering with StartLux-Decision-4B; shared evaluation pipeline.

By default this calls the local StartLux HTTP server at
http://127.0.0.1:8090/v1/systemone. Use --startlux-endpoint to change it.
The local-checkpoint path remains available with --startlux-model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_laya
import requests


DEFAULT_MODEL = "StartLux-Decision-4B"


class StartLuxBatchAdapter:
    """Translate StartLux answer lists into the shared probability-filter contract."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_batch(self, states, questions, *, batch_size, **kwargs):
        results = []
        for start in range(0, len(states), batch_size):
            chunk = states[start:start + batch_size]
            answers = self.model.decide_batch([(state, questions) for state in chunk])
            if len(answers) != len(chunk):
                raise ValueError("StartLux returned an unexpected number of results")
            results.extend({"answers": answer, "usage": {}} for answer in answers)
        return results


class StartLuxHTTPAdapter:
    """Call the upstream single-request endpoint for each case state."""

    def __init__(self, endpoint: str, *, timeout: float = 120.0) -> None:
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint:
            raise ValueError("StartLux endpoint must not be empty")
        self.endpoint = endpoint if endpoint.endswith("/v1/systemone") else endpoint + "/v1/systemone"
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("StartLux timeout must be greater than zero")
        self.timeout = float(timeout)
        self.session = requests.Session()

    def predict_batch(self, states, questions, *, batch_size, **kwargs):
        results = []
        for start in range(0, len(states), batch_size):
            for state in states[start:start + batch_size]:
                try:
                    response = self.session.post(
                        self.endpoint, json={"state": state, "questions": questions},
                        headers={"Content-Type": "application/json"}, timeout=self.timeout,
                    )
                    response.raise_for_status()
                    payload = response.json()
                except requests.RequestException as exc:
                    raise RuntimeError(f"StartLux request failed at {self.endpoint}: {exc}") from exc
                except ValueError as exc:
                    raise RuntimeError("StartLux response was not valid JSON") from exc
                if not isinstance(payload, Mapping) or not isinstance(payload.get("answers"), Mapping):
                    detail = json.dumps(payload, ensure_ascii=False)[:500]
                    raise RuntimeError(f"StartLux response has no answers: {detail}")
                results.append({"answers": payload["answers"], "usage": payload.get("usage", {})})
        return results


def create_filter(args: argparse.Namespace, settings: Mapping[str, Any]):
    def setting(name, default):
        value = getattr(args, name, None)
        return value if value is not None else settings.get(name, default)

    model_path = args.startlux_model or settings.get("model", DEFAULT_MODEL)
    endpoint = args.startlux_endpoint or settings.get("endpoint", "http://127.0.0.1:8090/v1/systemone")
    timeout = args.startlux_timeout if args.startlux_timeout is not None else settings.get("timeout", 120.0)
    threshold = setting("threshold", 0.5)
    batch_size = setting("batch_size", 16)
    max_len = setting("max_len", 4096)
    max_batch_tokens = setting("max_batch_tokens", 65536)
    device = setting("device", None)
    # This model has a total prompt limit, with no separate head budget.
    relevance_filter = evaluate_laya.LayaRelevanceFilter(
        None, threshold=threshold, batch_size=batch_size, max_len=max_len,
        head_max_len=1,
    )
    if isinstance(max_batch_tokens, bool) or not isinstance(max_batch_tokens, int) or max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be a positive integer")
    if args.startlux_model is None:
        relevance_filter.agent = StartLuxHTTPAdapter(endpoint, timeout=timeout)
        return relevance_filter, {
            "backend": "http", "endpoint": relevance_filter.agent.endpoint,
            "threshold": threshold, "batch_size": batch_size,
            "max_len": max_len, "max_batch_tokens": max_batch_tokens,
            "timeout": float(timeout), "images": False,
        }
    if not isinstance(model_path, str) or not Path(model_path).is_dir():
        raise ValueError(
            f"StartLux requires a local checkpoint directory: {model_path}. Download with: "
            "hf download startlux-models/StartLux-Decision-4B --local-dir StartLux-Decision-4B"
        )
    try:
        from startlux_decision import StartLuxDecision
    except ImportError as exc:
        raise RuntimeError(
            "startlux_decision is not importable. Install the upstream model folder's "
            "requirements.txt and add its parent package directory to PYTHONPATH; "
            "see README.md for StartLux setup."
        ) from exc
    model = StartLuxDecision(
        model_path, device=device, max_length=max_len,
        max_batch_tokens=max_batch_tokens, images=False,
    )
    relevance_filter.agent = StartLuxBatchAdapter(model)
    return relevance_filter, {
        "backend": "local", "model": model_path, "threshold": threshold, "batch_size": batch_size,
        "max_len": max_len, "max_batch_tokens": max_batch_tokens,
        "device": device, "images": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return evaluate_laya.parse_args(argv, backend="startlux")


def main(argv: Sequence[str] | None = None) -> int:
    return evaluate_laya.main(args=parse_args(argv), filter_factory=create_filter)


if __name__ == "__main__":
    raise SystemExit(main())
