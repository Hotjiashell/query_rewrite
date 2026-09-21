"""Render and parse the standalone prompt format used by the DSPy evaluator."""

from __future__ import annotations

import re


PROMPT_TEMPLATE = """{instruction}

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## dialogue ## ]]
{dialogue}

[[ ## query ## ]]
One retrieval query as a string.

[[ ## completed ## ]]

Respond with the corresponding output fields, starting with the field `[[ ## query ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
"""

QUERY_PATTERN = re.compile(r"\[\[\s*##\s*query\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)", re.IGNORECASE | re.DOTALL)


def render_prompt(instruction: str, dialogue: str) -> str:
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")
    return PROMPT_TEMPLATE.format(instruction=instruction, dialogue=dialogue)


def parse_query(completion: str) -> str:
    match = QUERY_PATTERN.search(completion)
    if not match:
        raise ValueError("model response does not contain the DSPy query field")
    query = match.group(1).strip()
    if not query:
        raise ValueError("DSPy query field is empty")
    return query
