"""Export a GEPA/DSPy instruction as the custom prompt format used by gen_query.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROMPT_TEMPLATE = """{instruction}

Return the result as exactly one JSON object in a JSON code block:
```json
{{"query":"the retrieval query"}}
```

Dialogue:
{{dialogue}}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a DSPy GEPA instruction for the existing custom query generator.")
    parser.add_argument("--instruction", required=True, help="Path to optimized_instruction.txt")
    parser.add_argument("--output", required=True, help="Destination custom-prompt text file")
    args = parser.parse_args(argv)
    instruction_path, output_path = Path(args.instruction), Path(args.output)
    try:
        instruction = instruction_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"instruction file does not exist: {instruction_path}") from exc
    if not instruction:
        raise ValueError("instruction file is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(PROMPT_TEMPLATE.format(instruction=instruction), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
