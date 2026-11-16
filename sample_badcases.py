"""Randomly sample a badcase report while preserving annotation-compatible records."""
from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from label_badcase.server import atomic_write


def sample_badcases(payload, count, seed=42):
    if not isinstance(payload, dict) or not isinstance(payload.get('records'), list):
        raise ValueError('输入必须是包含 records 数组的 badcase JSON 文件')
    records = payload['records']
    if type(count) is not int or not 1 <= count <= len(records):
        raise ValueError(f'采样数量必须在 1 到 {len(records)} 之间')
    seen = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'第 {position + 1} 条记录不是对象')
        key = str(record.get('sample_index') if record.get('sample_index') is not None else position)
        if key in seen:
            raise ValueError(f'重复 sample_index：{key}，无法适配标注器')
        seen.add(key)
    positions = sorted(random.Random(seed).sample(range(len(records)), count))
    result = copy.deepcopy(payload)
    sampled = []
    for position in positions:
        record = copy.deepcopy(records[position])
        # The annotator otherwise uses the position in the *sampled* file as key.
        if record.get('sample_index') is None:
            record['sample_index'] = position
        sampled.append(record)
    result['records'] = sampled
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    result['sampling'] = {'seed': seed, 'source_record_count': len(records), 'sample_count': count}
    summary = result.setdefault('summary', {})
    summary['badcase_count'] = count
    summary['unresolved_gt_case_count'] = sum(not r.get('gt_case_found') for r in sampled)
    groups = {}
    for r in sampled:
        cid = r.get('expected_case_id')
        cid = str(cid) if cid is not None else None
        group = groups.setdefault(cid, {'gt_case_id': cid, 'gt_case_title': r.get('gt_case_title'), 'frequency': 0, 'conversations': []})
        group['frequency'] += 1
        group['conversations'].append({k: r.get(k) for k in ('sample_index', 'call_sno', 'chat_content')})
    result['gt_case_frequency'] = sorted(groups.values(), key=lambda g: -g['frequency'])
    return result


def main():
    parser = argparse.ArgumentParser(description='随机、不重复采样 badcase，输出可直接用于标注器')
    parser.add_argument('input', help='collect_badcases.py 输出文件')
    parser.add_argument('-n', '--count', type=int, required=True, help='采样数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认42')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        source, output = Path(args.input).resolve(), Path(args.output).resolve()
        if source == output:
            raise ValueError('输出不能覆盖输入文件')
        payload = json.loads(source.read_text(encoding='utf-8'))
        result = sample_badcases(payload, args.count, args.seed)
        result['sampling']['source_path'] = str(source)
        atomic_write(output, result)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'采样失败：{exc}\n')
    print(f'已采样 {args.count}/{len(payload["records"])} 条，seed={args.seed}，输出：{output}')


if __name__ == '__main__':
    main()
