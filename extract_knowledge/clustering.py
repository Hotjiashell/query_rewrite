"""String candidate grouping and lossless cluster merging (no model I/O)."""
import re
import unicodedata


def compact(word):
    return re.sub(r"[\s\-‐‑‒–—―_]+", "", unicodedata.normalize('NFKC', word).casefold())


def similarity(left, right, edit_threshold=0.8, dice_threshold=0.8, short_threshold=0.9):
    a, b = compact(left), compact(right)
    if not a or not b:
        return {'edit': 0.0, 'dice': 0.0, 'candidate': False}
    if a == b:
        return {'edit': 1.0, 'dice': 1.0, 'candidate': True}
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    edit = 1 - previous[-1] / max(len(a), len(b))
    aa = {a[i:i+2] for i in range(len(a)-1)}
    bb = {b[i:i+2] for i in range(len(b)-1)}
    dice = 2 * len(aa & bb) / (len(aa) + len(bb)) if aa and bb else 0.0
    # Short names need stricter recall to avoid sending arbitrary two-letter pairs.
    if min(len(a), len(b)) <= 3:
        edit_threshold = max(edit_threshold, short_threshold)
        dice_threshold = max(dice_threshold, short_threshold)
    return {'edit': edit, 'dice': dice, 'candidate': edit >= edit_threshold or dice >= dice_threshold}


def candidates(clusters, edit_threshold=0.8, dice_threshold=0.8, short_threshold=0.9):
    parent = list(range(len(clusters)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    for i, left in enumerate(clusters):
        for j in range(i+1, len(clusters)):
            right = clusters[j]
            # Cheap length bound before edit distance / bigram calculation.
            a, b = compact(left['keyword']), compact(right['keyword'])
            if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < 0.5:
                continue
            scores = similarity(a, b, edit_threshold, dice_threshold, short_threshold)
            if scores['candidate']:
                edges.append({'left': left['keyword'], 'right': right['keyword'], **scores})
                parent[find(j)] = find(i)
    groups = {}
    for i, cluster in enumerate(clusters):
        groups.setdefault(find(i), []).append(cluster['keyword'])
    return [words for words in groups.values() if len(words) > 1], edges


def validate_partition(result, words):
    groups = result.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('groups 必须为非空分组数组')
    flat = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError('每个分组必须是对象')
        members = group.get('keywords')
        if not isinstance(members, list) or not members or any(not isinstance(w, str) for w in members):
            raise ValueError('每组 keywords 必须为非空字符串数组')
        if group.get('keyword') not in members or not isinstance(group.get('reason'), str) or not group['reason'].strip():
            raise ValueError('每组 keyword 须来自组内，reason 不能为空')
        flat.extend(members)
    if len(flat) != len(set(flat)) or set(flat) != set(words):
        raise ValueError('分组必须不重不漏地覆盖输入关键词，不允许新增词')
    return result


def fuse(clusters, tasks, responses, digest):
    lookup = {c.get('concept', c['keyword']): c for c in clusters}
    partitions = []
    consumed = set()
    for task, response in zip(tasks, responses):
        if response['status'] == 'success':
            groups = validate_partition(response['result'], task['keywords'])['groups']
            partitions.extend(groups)
            consumed.update(task['keywords'])
    partitions.extend({'keyword': c.get('concept', c['keyword']), 'keywords': [c.get('concept', c['keyword'])], 'reason': '保留原簇'}
                      for c in clusters if c.get('concept', c['keyword']) not in consumed)
    result = []
    for group in partitions:
        records, evidence = {}, []
        for word in group['keywords']:
            cluster = lookup[word]
            records.update({r['id']: r for r in cluster['records']})
            evidence.extend(cluster['keyword_evidence'])
        representative = lookup[group['keyword']]
        result.append({'keyword': representative['keyword'], 'desc': representative.get('desc', ''),
                       'concepts': group['keywords'], 'concept': group['keyword'],
                       'aliases': list(dict.fromkeys(lookup[word]['keyword'] for word in group['keywords'])),
                       'id': digest(sorted(group['keywords']))[:16], 'records': list(records.values()),
                       'count': len(records), 'gt_case_count': len({r['expected_case_id'] for r in records.values()}),
                       'keyword_evidence': evidence, 'merge_reason': group['reason']})
    return sorted(result, key=lambda c: (-c['count'], c['keyword'].casefold()))
