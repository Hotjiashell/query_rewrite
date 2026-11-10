"""Synonym experiments layered on the existing evaluation CLI."""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path
import re
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate as base

HERE = Path(__file__).resolve().parent


def load_synonyms(path):
    pairs = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = re.split(r"[,，\t]", line, maxsplit=1)
        if len(fields) != 2 or not all(field.strip() for field in fields):
            raise ValueError(f"{path}:{number}: expected 原词, 同义词")
        source, target = (field.strip() for field in fields)
        # Keep the first mapping in file order for duplicate source terms.
        pairs.setdefault(source, target)
    return pairs


def replace_synonyms(query, pairs):
    if not pairs:
        return query
    pattern = re.compile("|".join(re.escape(word) for word in sorted(pairs, key=len, reverse=True)))
    return pattern.sub(lambda match: pairs[match.group()], query)


def render_suffix(dialogue, pairs, template):
    matched = [f"- {source}——{target}" for source, target in pairs.items() if source in dialogue]
    return template.replace("{syonyms_list}", "\n".join(matched)).replace("{}", "\n".join(matched)) if matched else ""


def run_with_retries(argv, interval=None):
    args = base.parse_args(argv)
    if interval is not None and (not math.isfinite(interval) or interval <= 0):
        raise ValueError("--internal-time must be a finite number greater than zero")
    if interval is not None and args.stage not in ("retrieve", "all"):
        raise ValueError("--internal-time requires retrieve or all")
    result = base.main(argv)
    if result or interval is None:
        return result
    retry_args = argparse.Namespace(**vars(args))
    retry_args.stage = "retrieve"
    if args.stage == "all":
        generation_args = argparse.Namespace(**vars(args))
        generation_args.input = args.query_input if args.query_input is not None else args.input
        generation_args.output = args.query_output
        generation_args.concurrency = args.query_concurrency if args.query_concurrency is not None else args.concurrency
        generation = base.resolve_query_generation_config(generation_args)
        retry_args.input = args.retrieval_input
        retry_args.output = args.retrieval_output if args.retrieval_output is not None else args.output
        retry_args.concurrency = args.retrieval_concurrency if args.retrieval_concurrency is not None else args.concurrency
        config = base.resolve_retrieval_config(retry_args)
        retry_args.input = generation.output_path
    config = base.resolve_retrieval_config(retry_args)
    retry_args.retry_errors = config.output_path
    while True:
        artifact = base._load_json(config.output_path, "retrieval result")
        pending = sum(record.get("retrieval_status") in ("failed", "partial_success")
                      or bool(record.get("retrieval_error") and record.get("retrieval_status") != "skipped")
                      for record in artifact["records"])
        if not pending:
            print("No retrieval errors remain.", flush=True)
            return 0
        print(f"{pending} samples have retrieval errors; retrying in {interval:g} seconds.", flush=True)
        time.sleep(interval)
        base._run_retrieve(retry_args)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--synonym-mode", choices=("replace", "prompt"), required=True)
    parser.add_argument("--synonyms", default=str(HERE / "syonyms.txt"))
    parser.add_argument("--synonym-prompt", default=str(HERE / "prompt_template.txt"))
    parser.add_argument("--internal-time", type=float, help="Seconds to wait between retrieval-error retries")
    options, remaining = parser.parse_known_args(argv)
    original_factory = base.create_query_generator
    original_loader = base.load_generated_query_records
    try:
        pairs = load_synonyms(options.synonyms)
        if options.synonym_mode == "replace":
            def load_records(path):
                return [replace(record,
                    query=replace_synonyms(record.query, pairs) if record.query is not None else None,
                    queries=[replace_synonyms(query, pairs) for query in record.queries] if record.queries else None,
                ) for record in original_loader(path)]
            base.load_generated_query_records = load_records
        else:
            template = Path(options.synonym_prompt).read_text(encoding="utf-8")
            if "{syonyms_list}" not in template and "{}" not in template:
                raise ValueError("synonym prompt template must contain {syonyms_list} or {}")
            def factory(*args, **kwargs):
                generator = original_factory(*args, **kwargs)
                original_render = generator._render_prompt
                def render(self, dialogue):
                    prompt = original_render(dialogue)
                    suffix = render_suffix(dialogue, pairs, template)
                    return prompt + "\n\n" + suffix if suffix else prompt
                generator._render_prompt = MethodType(render, generator)
                return generator
            base.create_query_generator = factory
        return run_with_retries(remaining, options.internal_time)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Synonym evaluation failed: {exc}", file=sys.stderr)
        return 2
    finally:
        base.create_query_generator = original_factory
        base.load_generated_query_records = original_loader


if __name__ == "__main__":
    raise SystemExit(main())
