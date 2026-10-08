"""Multi-query filtering with StartLux-Decision-4B; shared evaluation pipeline.

Download the checkpoint to ./StartLux-Decision-4B first. Install the upstream
startlux_decision package from the model folder or a source checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_laya


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


def create_filter(args: argparse.Namespace, settings: Mapping[str, Any]):
    def setting(name, default):
        value = getattr(args, name, None)
        return value if value is not None else settings.get(name, default)

    model_path = args.startlux_model or settings.get("model", DEFAULT_MODEL)
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
        "model": model_path, "threshold": threshold, "batch_size": batch_size,
        "max_len": max_len, "max_batch_tokens": max_batch_tokens,
        "device": device, "images": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return evaluate_laya.parse_args(argv, backend="startlux")


def main(argv: Sequence[str] | None = None) -> int:
    return evaluate_laya.main(args=parse_args(argv), filter_factory=create_filter)


if __name__ == "__main__":
    raise SystemExit(main())
