"""The one-predictor DSPy program whose instruction GEPA evolves."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt import BASELINE_PROMPT  # noqa: E402


def make_program():
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("DSPy is not installed. Run: pip install -r gepa/requirements.txt") from exc

    class RewriteQuery(dspy.Signature):
        """Generate one concise retrieval query from a customer-service dialogue.

        Preserve product, system, error, action, and business-context terms that
        distinguish the target support case. Return only the query, with no
        analysis, Markdown, or JSON wrapper.
        """

        dialogue: str = dspy.InputField(desc="Complete customer-service dialogue")
        query: str = dspy.OutputField(desc="One retrieval query")

    program = dspy.Predict(RewriteQuery)
    # Use the business instruction from the existing baseline as GEPA's seed.
    # DSPy owns output serialization, so the old JSON/fence instructions must
    # not be included in the candidate that GEPA evolves.
    seed = BASELINE_PROMPT.split("将你的结果用", maxsplit=1)[0].replace("{dialogue}", "").strip()
    program.signature = program.signature.with_instructions(seed)
    return program


def make_lm(settings):
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DSPy is not installed. Run: pip install -r gepa/requirements.txt") from exc
    return dspy.LM(
        f"openai/{settings.model_name}",
        api_base=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
