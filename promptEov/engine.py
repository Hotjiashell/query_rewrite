"""Self-evolving retrieval prompt optimizer."""
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Callable, Any
import requests
from search import test_retrieval

from .prompt import INITIAL_PROMPT, ANALYZER_PROMPT, OPTIMIZER_PROMPT

DEFAULT_ANALYZER = ANALYZER_PROMPT
DEFAULT_OPTIMIZER = OPTIMIZER_PROMPT

def _llm(prompt, *, model, base_url, api_key, timeout=60):
    r=requests.post(base_url.rstrip('/')+'/chat/completions', headers={'Authorization':f'Bearer {api_key}'}, json={'model':model,'messages':[{'role':'user','content':prompt}], 'temperature':0.2}, timeout=timeout)
    r.raise_for_status(); return r.json()['choices'][0]['message']['content']

class PromptEov:
    def __init__(self, initial_prompt:str = INITIAL_PROMPT, dataset:list[dict] = None, *, analyzer_prompt=DEFAULT_ANALYZER, optimizer_prompt=DEFAULT_OPTIMIZER, generate_query:Callable[[str,str],str]|None=None, retrieve:Callable[[str],dict]=test_retrieval, llm:Callable[[str],str]|None=None):
        self.initial_prompt,self.dataset=initial_prompt,dataset or []; self.analyzer_prompt,self.optimizer_prompt=analyzer_prompt,optimizer_prompt; self.generate_query=generate_query; self.retrieve=retrieve; self.llm=llm
    def _gen(self, prompt, dialogue):
        if self.generate_query: return self.generate_query(prompt, dialogue)
        if not self.llm: raise ValueError('provide generate_query or llm')
        out=self.llm(prompt.format(dialogue=dialogue)); m=re.search(r'"query"\s*:\s*"(.*?)"',out,re.S); return m.group(1) if m else out.strip()
    def run(self, iterations=3, output_dir='promptEov/runs'):
        Path(output_dir).mkdir(parents=True,exist_ok=True); prompt=self.initial_prompt; history=[]
        for i in range(1,iterations+1):
            snap=[]; bad=[]
            for row in self.dataset:
                try: q=self._gen(prompt,row['chat_content']); result=self.retrieve(q); tops=result.get('retrieval_result',{}); ids=[v.get('case_id') for _,v in sorted(tops.items())[:10] if isinstance(v,dict)]; hit=row.get('caseID') in ids
                except Exception as e: q=''; ids=[]; hit=False; result={'error':str(e)}
                item={**row,'query':q,'top10_case_ids':ids,'hit':hit,'retrieval':result}; snap.append(item)
                if not hit: bad.append(item)
            analysis=self.llm(self.analyzer_prompt.format(bad_cases=json.dumps(bad,ensure_ascii=False,indent=2))) if self.llm else ''
            new_prompt=self.llm(self.optimizer_prompt.format(initial_prompt=self.initial_prompt,analysis=analysis)) if self.llm else prompt
            record={'iteration':i,'prompt':prompt,'snapshot':snap,'bad_cases':bad,'analysis':analysis,'new_prompt':new_prompt}; json.dump(record,open(Path(output_dir)/f'iteration_{i}.json','w'),ensure_ascii=False,indent=2); history.append(record); prompt=new_prompt
        return history

def run_evolution(dataset_path, initial_prompt, **kwargs):
    data=json.load(open(dataset_path,encoding='utf-8')); return PromptEov(initial_prompt,data,**kwargs).run()
