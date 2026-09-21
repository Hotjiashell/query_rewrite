"""Export an optimized instruction with the standalone DSPy field protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dspy_prompt import PROMPT_TEMPLATE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a GEPA instruction with DSPy field markers.")
    parser.add_argument("--instruction", required=True, help="Path to optimized_instruction.txt")
    parser.add_argument("--output", required=True, help="Destination prompt file")
    args = parser.parse_args(argv)
    instruction_path, output_path = Path(args.instruction), Path(args.output)
    try:
        instruction = instruction_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"instruction file does not exist: {instruction_path}") from exc
    if not instruction:
        raise ValueError("instruction file is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(PROMPT_TEMPLATE.format(instruction=instruction, dialogue="{dialogue}"), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
