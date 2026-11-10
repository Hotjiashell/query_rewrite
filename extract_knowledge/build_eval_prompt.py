"""Build final_prompt.txt from a run's knowledge snapshot and top-k new entries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile


def build_prompt(run_dir, template_file, k):
    if type(k) is not int or k < 0:
        raise ValueError("k 必须为非负整数")
    directory = Path(run_dir)
    template = Path(template_file).read_text(encoding="utf-8")
    if "{knowledge}" not in template:
        raise ValueError("模板中缺少 {knowledge} 占位符")
    snapshot = json.loads((directory / "existing_knowledge.json").read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("content"), str):
        raise ValueError("existing_knowledge.json 必须包含字符串 content")
    entries = json.loads((directory / "knowledge.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("knowledge.json 必须为数组")
    items = []
    for line in snapshot["content"].splitlines():
        value = re.sub(r'^\s*(?:\d+[.、．)]\s*|[-*]\s+)', '', line).strip()
        if value:
            items.append(value)
    for index, entry in enumerate(entries[:k], 1):
        if not isinstance(entry, dict) or not isinstance(entry.get("knowledge"), str) or not entry["knowledge"].strip():
            raise ValueError(f"knowledge.json 第 {index} 条缺少非空 knowledge 字符串")
        # Keep each entry on one line without losing its text.
        items.append(" ".join(entry["knowledge"].split()))
    knowledge = "\n".join(f"{index}. {value}" for index, value in enumerate(items, 1))
    rendered = template.replace("{knowledge}", knowledge)
    output = directory / "final_prompt.txt"
    # Validate and render completely before replacing an existing prompt.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                     prefix=".final_prompt-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
    temporary.replace(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="完整的知识抽取输出目录")
    parser.add_argument("template_file", help="含 {knowledge} 的提示词模板")
    parser.add_argument("k", type=int, help="追加的新知识数量；0 表示仅使用已有知识")
    args = parser.parse_args(argv)
    try:
        output = build_prompt(args.run_dir, args.template_file, args.k)
    except (OSError, ValueError) as exc:
        print(f"生成提示词失败：{exc}", file=sys.stderr)
        return 2
    print(f"提示词已生成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
