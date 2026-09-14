"""Self-evolving retrieval prompt optimizer."""
from __future__ import annotations
import inspect
import json, os, re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Any
import requests
from search import DEFAULT_RETRIEVAL_URL, test_retrieval

from .prompt import INITIAL_PROMPT, ANALYZER_PROMPT, OPTIMIZER_PROMPT

DEFAULT_ANALYZER = ANALYZER_PROMPT
DEFAULT_OPTIMIZER = OPTIMIZER_PROMPT


def _format_prompt(template: str, **values: str) -> str:
    """Substitute known placeholders without interpreting JSON braces in prompts."""
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace('{' + name + '}', value)
    return rendered


def _retrieval_items(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    results = payload.get('retrieval_result', {}) if isinstance(payload, dict) else {}
    if not isinstance(results, dict):
        return []

    def rank(entry: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        match = re.search(r'(\d+)$', entry[0])
        return (int(match.group(1)) if match else 10**9, entry[0])

    return [(key, value) for key, value in sorted(results.items(), key=rank) if isinstance(value, dict)]


def _print_progress(stage: str, completed: int, total: int, *, iteration: int) -> None:
    percentage = 100.0 if total == 0 else completed / total * 100
    print(
        f"\r[PromptEov][第 {iteration} 轮][{stage}] "
        f"{completed}/{total} ({percentage:5.1f}%)",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if completed >= total:
        print(file=sys.stderr)

def _llm(prompt, *, model, base_url, api_key, temperature=0.0, timeout=60):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError('temperature must be between 0 and 2')
    r=requests.post(base_url.rstrip('/')+'/chat/completions', headers={'Authorization':f'Bearer {api_key}'}, json={'model':model,'messages':[{'role':'user','content':prompt}], 'temperature':temperature}, timeout=timeout)
    r.raise_for_status(); return r.json()['choices'][0]['message']['content']

class PromptEov:
    def __init__(self, initial_prompt:str = INITIAL_PROMPT, dataset:list[dict] = None, *, analyzer_prompt=DEFAULT_ANALYZER, optimizer_prompt=DEFAULT_OPTIMIZER, generate_query:Callable[[str,str],str]|None=None, retrieve:Callable[[str],dict]=test_retrieval, llm:Callable[[str],str]|None=None, temperature:float=0.0, concurrency:int=1, analysis_concurrency:int=4):
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be between 0 and 2')
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError('concurrency must be at least 1')
        if isinstance(analysis_concurrency, bool) or not isinstance(analysis_concurrency, int) or analysis_concurrency < 1:
            raise ValueError('analysis_concurrency must be at least 1')
        self.initial_prompt,self.dataset=initial_prompt,dataset or []; self.analyzer_prompt,self.optimizer_prompt=analyzer_prompt,optimizer_prompt; self.generate_query=generate_query; self.retrieve=retrieve; self.llm=llm; self.temperature=temperature; self.concurrency=concurrency; self.analysis_concurrency=analysis_concurrency
    def _call_llm(self, prompt):
        """Call injected LLMs with temperature when their signature supports it."""
        if not self.llm: raise ValueError('provide generate_query or llm')
        try:
            parameters = inspect.signature(self.llm).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_temperature = 'temperature' in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_temperature:
            return self.llm(prompt, temperature=float(self.temperature))
        return self.llm(prompt)
    def _gen(self, prompt, dialogue):
        if self.generate_query: return self.generate_query(prompt, dialogue)
        out=self._call_llm(_format_prompt(prompt, dialogue=dialogue)); m=re.search(r'"query"\s*:\s*"(.*?)"',out,re.S); return m.group(1) if m else out.strip()
    def _analyze_badcase(self, badcase: dict[str, Any]) -> str:
        prompt = _format_prompt(
            self.analyzer_prompt,
            bad_cases=json.dumps(badcase, ensure_ascii=False, indent=2),
        )
        try:
            return self._call_llm(prompt).strip()
        except Exception as exc:
            return f"分析失败: {type(exc).__name__}: {exc}"

    def _analyze_badcases(self, badcases: list[dict[str, Any]], *, iteration: int, progress: bool) -> str:
        if not self.llm or not badcases:
            return ''
        if len(badcases) == 1:
            analyses = [self._analyze_badcase(badcases[0])]
            if progress:
                _print_progress('分析 badcase', 1, 1, iteration=iteration)
        else:
            with ThreadPoolExecutor(max_workers=min(self.analysis_concurrency, len(badcases))) as executor:
                futures = [executor.submit(self._analyze_badcase, badcase) for badcase in badcases]
                analyses = [''] * len(futures)
                completed = 0
                future_indexes = {future: index for index, future in enumerate(futures)}
                for future in as_completed(futures):
                    analyses[future_indexes[future]] = future.result()
                    completed += 1
                    if progress:
                        _print_progress('分析 badcase', completed, len(futures), iteration=iteration)
        return '\n\n'.join(
            f"### Badcase {index}\n{analysis}"
            for index, analysis in enumerate(analyses, start=1)
        )
    def _process_row(self, row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        try:
            q=self._gen(self._current_prompt,row['chat_content'])
            result=self.retrieve(q)
            ranked=_retrieval_items(result)
            top10=ranked[:10]
            ids=[value.get('case_id') for _, value in top10 if value.get('case_id')]
            titles=[value.get('case_title') for _, value in top10 if value.get('case_title')]
            hit=row.get('caseID') in ids
        except Exception as e: q=''; ids=[]; titles=[]; hit=False; result={'error':str(e)}
        return ({**row,'query':q,'extract_query':q,'top10_case_ids':ids,'top_10titles':titles,'gt_caseId':row.get('caseID'),'gt_caseId_title':row.get('case_title', row.get('caseID')),'hit':hit,'retrieval':result}, hit)
    def run(self, iterations=3, output_dir='promptEov/runs', *, progress=True):
        Path(output_dir).mkdir(parents=True,exist_ok=True); prompt=self.initial_prompt; history=[]
        for i in range(1,iterations+1):
            snap=[]; bad=[]
            self._current_prompt = prompt
            if self.concurrency == 1:
                processed = []
                for sample_index, row in enumerate(self.dataset, start=1):
                    processed.append(self._process_row(row))
                    if progress:
                        _print_progress('生成 query + 检索', sample_index, len(self.dataset), iteration=i)
            else:
                with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                    futures = [executor.submit(self._process_row, row) for row in self.dataset]
                    processed = [None] * len(futures)
                    future_indexes = {future: index for index, future in enumerate(futures)}
                    completed = 0
                    for future in as_completed(futures):
                        processed[future_indexes[future]] = future.result()
                        completed += 1
                        if progress:
                            _print_progress('生成 query + 检索', completed, len(futures), iteration=i)
            for item, hit in processed:
                snap.append(item)
                if not hit:
                    bad.append(item)
            analysis=self._analyze_badcases(bad, iteration=i, progress=progress)
            if progress and self.llm:
                print(f"[PromptEov][第 {i} 轮] 正在优化 prompt...", file=sys.stderr, flush=True)
            new_prompt=self._call_llm(_format_prompt(self.optimizer_prompt, initial_prompt=prompt, analysis=analysis)) if self.llm else prompt
            if progress:
                print(f"[PromptEov][第 {i} 轮] 完成，badcase={len(bad)}", file=sys.stderr, flush=True)
            record={'iteration':i,'prompt':prompt,'snapshot':snap,'bad_cases':bad,'analysis':analysis,'new_prompt':new_prompt}; json.dump(record,open(Path(output_dir)/f'iteration_{i}.json','w'),ensure_ascii=False,indent=2); history.append(record); prompt=new_prompt
        return history

def run_evolution(
    dataset_path,
    initial_prompt,
    *,
    iterations=3,
    output_dir='promptEov/runs',
    progress=True,
    **kwargs,
):
    """Load a dataset, configure the evolver, and run the requested iterations."""
    with open(dataset_path, encoding='utf-8') as dataset_file:
        data = json.load(dataset_file)
    return PromptEov(initial_prompt, data, **kwargs).run(
        iterations=iterations,
        output_dir=output_dir,
        progress=progress,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run promptEov prompt evolution.')
    parser.add_argument('--config', default='config.json', help='JSON config file (default: config.json)')
    parser.add_argument('--dataset', help='JSON dataset containing chat_content and caseID')
    parser.add_argument('--initial-prompt-file', help='File containing the initial query-generation prompt')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output-dir', default='promptEov/runs')
    parser.add_argument('--model', help='Override llm.model_name')
    parser.add_argument('--base-url', help='Override llm.base_url')
    parser.add_argument('--api-key', help='API key; defaults to the configured environment variable')
    parser.add_argument('--api-key-env', help='Override llm.api_key_env')
    parser.add_argument('--temperature', type=float, help='Override llm.temperature (0 to 2)')
    parser.add_argument('--retrieval-url', help='Override retrieval.url')
    parser.add_argument('--retrieval-timeout', type=float, help='Override retrieval.timeout')
    parser.add_argument('--concurrency', type=int, help='Concurrent query-generation and retrieval workers')
    parser.add_argument('--analysis-concurrency', type=int, default=4)
    return parser


def cli(argv=None) -> int:
    args = _build_cli_parser().parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('config file root must be an object')
    llm_config = config.get('llm', {})
    retrieval_config = config.get('retrieval', {})
    query_config = config.get('query_generation', {})
    evolution_config = config.get('prompt_evolution', {})
    if not isinstance(evolution_config, dict):
        raise ValueError('config section prompt_evolution must be an object')
    if not all(isinstance(section, dict) for section in (llm_config, retrieval_config, query_config)):
        raise ValueError('config sections llm, retrieval, and query_generation must be objects')
    iterations = args.iterations if args.iterations != 3 else evolution_config.get('iterations', 3)
    output_dir = args.output_dir if args.output_dir != 'promptEov/runs' else evolution_config.get('output_dir', 'promptEov/runs')
    analysis_concurrency = args.analysis_concurrency if args.analysis_concurrency != 4 else evolution_config.get('analysis_concurrency', 4)
    concurrency = args.concurrency if args.concurrency is not None else query_config.get('concurrency', 1)
    if iterations < 1:
        raise ValueError('iterations must be at least 1')
    dataset = args.dataset or query_config.get('input_path')
    model = args.model or llm_config.get('model_name')
    base_url = args.base_url or llm_config.get('base_url')
    api_key_env = args.api_key_env or llm_config.get('api_key_env') or 'OPENAI_API_KEY'
    api_key = args.api_key or llm_config.get('api_key') or os.getenv(api_key_env)
    temperature = args.temperature if args.temperature is not None else llm_config.get('temperature', 0.0)
    retrieval_url = args.retrieval_url or retrieval_config.get('url') or DEFAULT_RETRIEVAL_URL
    retrieval_timeout = args.retrieval_timeout if args.retrieval_timeout is not None else retrieval_config.get('timeout', 30.0)
    if not dataset:
        raise ValueError('missing dataset; pass --dataset or set query_generation.input_path')
    if not model or not base_url:
        raise ValueError('missing LLM config; set llm.model_name and llm.base_url or pass CLI overrides')
    if not api_key:
        raise ValueError(f'missing API key; pass --api-key or set {api_key_env}')
    initial_prompt = INITIAL_PROMPT
    if args.initial_prompt_file:
        initial_prompt = Path(args.initial_prompt_file).read_text(encoding='utf-8').strip()
        if not initial_prompt:
            raise ValueError('initial prompt file must not be empty')

    def llm(prompt, *, temperature=args.temperature):
        return _llm(
            prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )

    history = run_evolution(
        dataset,
        initial_prompt,
        llm=llm,
        retrieve=lambda query: test_retrieval(
            query, url=retrieval_url, timeout=retrieval_timeout
        ),
        iterations=iterations,
        output_dir=output_dir,
        temperature=temperature,
        concurrency=concurrency,
        analysis_concurrency=analysis_concurrency,
    )
    print(f'完成 {len(history)} 轮迭代，结果已写入: {output_dir}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(cli())
    except (OSError, ValueError, requests.RequestException) as exc:
        raise SystemExit(f'错误: {exc}')
