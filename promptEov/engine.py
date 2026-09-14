"""Self-evolving retrieval prompt optimizer."""
from __future__ import annotations
import inspect
import json, os, re
import argparse
from concurrent.futures import ThreadPoolExecutor
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

def _llm(prompt, *, model, base_url, api_key, temperature=0.0, timeout=60):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError('temperature must be between 0 and 2')
    r=requests.post(base_url.rstrip('/')+'/chat/completions', headers={'Authorization':f'Bearer {api_key}'}, json={'model':model,'messages':[{'role':'user','content':prompt}], 'temperature':temperature}, timeout=timeout)
    r.raise_for_status(); return r.json()['choices'][0]['message']['content']

class PromptEov:
    def __init__(self, initial_prompt:str = INITIAL_PROMPT, dataset:list[dict] = None, *, analyzer_prompt=DEFAULT_ANALYZER, optimizer_prompt=DEFAULT_OPTIMIZER, generate_query:Callable[[str,str],str]|None=None, retrieve:Callable[[str],dict]=test_retrieval, llm:Callable[[str],str]|None=None, temperature:float=0.0, analysis_concurrency:int=4):
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be between 0 and 2')
        if isinstance(analysis_concurrency, bool) or not isinstance(analysis_concurrency, int) or analysis_concurrency < 1:
            raise ValueError('analysis_concurrency must be at least 1')
        self.initial_prompt,self.dataset=initial_prompt,dataset or []; self.analyzer_prompt,self.optimizer_prompt=analyzer_prompt,optimizer_prompt; self.generate_query=generate_query; self.retrieve=retrieve; self.llm=llm; self.temperature=temperature; self.analysis_concurrency=analysis_concurrency
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

    def _analyze_badcases(self, badcases: list[dict[str, Any]]) -> str:
        if not self.llm or not badcases:
            return ''
        if len(badcases) == 1:
            analyses = [self._analyze_badcase(badcases[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.analysis_concurrency, len(badcases))) as executor:
                analyses = list(executor.map(self._analyze_badcase, badcases))
        return '\n\n'.join(
            f"### Badcase {index}\n{analysis}"
            for index, analysis in enumerate(analyses, start=1)
        )
    def run(self, iterations=3, output_dir='promptEov/runs'):
        Path(output_dir).mkdir(parents=True,exist_ok=True); prompt=self.initial_prompt; history=[]
        for i in range(1,iterations+1):
            snap=[]; bad=[]
            for row in self.dataset:
                try:
                    q=self._gen(prompt,row['chat_content'])
                    result=self.retrieve(q)
                    ranked=_retrieval_items(result)
                    top10=ranked[:10]
                    ids=[value.get('case_id') for _, value in top10 if value.get('case_id')]
                    titles=[value.get('case_title') for _, value in top10 if value.get('case_title')]
                    hit=row.get('caseID') in ids
                except Exception as e: q=''; ids=[]; titles=[]; hit=False; result={'error':str(e)}
                item={**row,'query':q,'extract_query':q,'top10_case_ids':ids,'top_10titles':titles,'gt_caseId':row.get('caseID'),'gt_caseId_title':row.get('case_title', row.get('caseID')),'hit':hit,'retrieval':result}; snap.append(item)
                if not hit: bad.append(item)
            analysis=self._analyze_badcases(bad)
            new_prompt=self._call_llm(_format_prompt(self.optimizer_prompt, initial_prompt=prompt, analysis=analysis)) if self.llm else prompt
            record={'iteration':i,'prompt':prompt,'snapshot':snap,'bad_cases':bad,'analysis':analysis,'new_prompt':new_prompt}; json.dump(record,open(Path(output_dir)/f'iteration_{i}.json','w'),ensure_ascii=False,indent=2); history.append(record); prompt=new_prompt
        return history

def run_evolution(dataset_path, initial_prompt, **kwargs):
    data=json.load(open(dataset_path,encoding='utf-8')); return PromptEov(initial_prompt,data,**kwargs).run()


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run promptEov prompt evolution.')
    parser.add_argument('--dataset', required=True, help='JSON dataset containing chat_content and caseID')
    parser.add_argument('--initial-prompt-file', help='File containing the initial query-generation prompt')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output-dir', default='promptEov/runs')
    parser.add_argument('--model', required=True, help='LLM model name')
    parser.add_argument('--base-url', required=True, help='OpenAI-compatible API base URL')
    parser.add_argument('--api-key', help='API key; defaults to the configured environment variable')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--temperature', type=float, default=0.0, help='LLM temperature from 0 to 2')
    parser.add_argument('--retrieval-url', default=DEFAULT_RETRIEVAL_URL)
    parser.add_argument('--retrieval-timeout', type=float, default=30.0)
    parser.add_argument('--analysis-concurrency', type=int, default=4)
    return parser


def cli(argv=None) -> int:
    args = _build_cli_parser().parse_args(argv)
    if args.iterations < 1:
        raise ValueError('iterations must be at least 1')
    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f'missing API key; pass --api-key or set {args.api_key_env}')
    initial_prompt = INITIAL_PROMPT
    if args.initial_prompt_file:
        initial_prompt = Path(args.initial_prompt_file).read_text(encoding='utf-8').strip()
        if not initial_prompt:
            raise ValueError('initial prompt file must not be empty')

    def llm(prompt, *, temperature=args.temperature):
        return _llm(
            prompt,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=temperature,
        )

    history = run_evolution(
        args.dataset,
        initial_prompt,
        llm=llm,
        retrieve=lambda query: test_retrieval(
            query, url=args.retrieval_url, timeout=args.retrieval_timeout
        ),
        iterations=args.iterations,
        output_dir=args.output_dir,
        temperature=args.temperature,
        analysis_concurrency=args.analysis_concurrency,
    )
    print(f'完成 {len(history)} 轮迭代，结果已写入: {args.output_dir}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(cli())
    except (OSError, ValueError, requests.RequestException) as exc:
        raise SystemExit(f'错误: {exc}')
