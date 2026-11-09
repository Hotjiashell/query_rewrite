"""OpenAI-compatible batched embedding preclustering with resumable calls."""
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


def vectors_valid(vectors, count):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError('embedding 返回数量不匹配')
    dimension = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            raise ValueError('embedding 向量必须为非空有限数值数组')
        if dimension is not None and len(vector) != dimension:
            raise ValueError('embedding 向量维度不一致')
        dimension = len(vector)
        if not math.isfinite(math.hypot(*vector)) or math.hypot(*vector) == 0:
            raise ValueError('embedding 向量不能为零')
    return vectors


def semantic_groups(clusters, vectors, threshold):
    vectors_valid(vectors, len(clusters))
    normalized = [[x / math.hypot(*v) for x in v] for v in vectors]
    parent = list(range(len(clusters)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    for i, left in enumerate(normalized):
        for j in range(i+1, len(normalized)):
            score = max(-1.0, min(1.0, sum(a*b for a,b in zip(left, normalized[j]))))
            if score >= threshold:
                edges.append({'left': clusters[i].get('concept', clusters[i]['keyword']), 'right': clusters[j].get('concept', clusters[j]['keyword']), 'cosine': score})
                parent[find(j)] = find(i)
    groups = {}
    for i, cluster in enumerate(clusters):
        groups.setdefault(find(i), []).append(cluster.get('concept', cluster['keyword']))
    return [v for v in groups.values() if len(v)>1], edges


def precluster(clusters, args, save, read_json, digest):
    config = args.embedding
    public = {k:v for k,v in config.items() if k != 'api_key'}
    if len(clusters) < 2:
        return [], [], 0
    from openai import OpenAI
    client = OpenAI(base_url=config['base_url'], api_key=config['api_key'], timeout=config['timeout'], max_retries=0)
    def batch(words):
        key = digest([words, public])
        path = args.output_dir / 'calls' / 'embedding' / (key+'.json')
        if args.resume and path.exists():
            try:
                cached = read_json(path)
                if cached['status'] == 'success':
                    return vectors_valid(cached['vectors'],len(words)), None
            except (ValueError, OSError, KeyError, TypeError):
                pass
        trace = {'input': words, 'configuration': public, 'status':'failed', 'attempts':[]}
        for attempt in range(config['retries']+1):
            try:
                response = client.embeddings.create(model=config['model_name'],input=words,encoding_format='float')
                indexed = sorted(response.data,key=lambda item:item.index)
                if [item.index for item in indexed] != list(range(len(words))):
                    raise ValueError('embedding index 不完整或重复')
                vectors = vectors_valid([item.embedding for item in indexed],len(words))
                trace.update(status='success',vectors=vectors)
                trace['attempts'].append({'status':'success'})
                save(path,trace)
                return vectors,None
            except Exception as exc:
                trace['attempts'].append({'error_type':type(exc).__name__})
                save(path,trace)
                if attempt < config['retries']:
                    time.sleep(min(20,2**attempt)+random.random())
        return None,trace
    words = [c.get('concept', c['keyword']) for c in clusters]
    batches = [words[i:i+config['batch_size']] for i in range(0,len(words),config['batch_size'])]
    results, failures = {}, []
    with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
        futures = {pool.submit(batch,w):i for i,w in enumerate(batches)}
        for done,future in enumerate(as_completed(futures),1):
            i = futures[future]
            values,error = future.result()
            if error:
                failures.append(error)
            else:
                results.update(zip(batches[i],values))
            print(f'\r[embedding] {done}/{len(batches)} failed={len(failures)}',end='\n' if done==len(batches) else '',file=sys.stderr,flush=True)
    save(args.output_dir/'04_embeddings.json',{'configuration':public,'vectors':results,'failed_batches':failures})
    usable = [c for c in clusters if c.get('concept', c['keyword']) in results]
    try:
        groups,edges = semantic_groups(usable,[results[c.get('concept', c['keyword'])] for c in usable],args.precluster_similarity_threshold)
    except ValueError as exc:
        save(args.output_dir/'embedding_error.json',{'error':str(exc)})
        return [],[],len(failures)+1
    return groups,edges,len(failures)
