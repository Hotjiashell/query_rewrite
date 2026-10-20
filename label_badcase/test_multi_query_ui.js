/* Frontend regression without browser dependencies: async record ownership and safe rendering. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'',innerHTML:'',disabled:false,handlers:{},
    addEventListener(name, fn) {(this.handlers[name]??=[]).push(fn);}});
  return elements.get(id);
}
let respond, request;
const context = vm.createContext({
  document:{getElementById:element,addEventListener(){}},
  window:{addEventListener(){}},localStorage:{getItem(){return null;}},setInterval(){},AbortSignal,
  fetch:async (url, options)=>{request={url,body:JSON.parse(options.body)};return new Promise(resolve=>{respond=resolve;});},
});
const source = fs.readFileSync(__dirname+'/static/app.js','utf8');
vm.runInContext(source.slice(0,source.lastIndexOf('(async()=>')),context);
const run = code => vm.runInContext(code,context);
(async()=>{
  run(`data={multi_query_prompt:'default {dialogue}',records:[]};currentKey='12';`);
  assert.match(run('multiPanel()'), /多 query 提示词测试/);
  element('multi-query-prompt').value='custom {dialogue}';
  element('multi-query-fusion').value='round_robin';
  element('multi-query-top-k').value='10';
  const pending=run('tryMultiQuery()');
  assert.equal(request.url,'/api/multi-query');
  assert.deepEqual(request.body,{key:'12',prompt:'custom {dialogue}',fusion_method:'round_robin',top_k:10});
  assert.equal(element('multi-query-submit').disabled,true);
  run(`currentKey='27';`);
  assert.equal(run('multiTrialFor().prompt'),'custom {dialogue}');
  const blankPage=run('multiPanel()');
  element('multi-query-results').innerHTML='different record';
  const recalled={query:'<script>query</script>',gt_available:true,gt_case_id:'GT',gt_hit:true,matched_rank:1,
    cases:[{rank:1,case_id:'GT',title:'<img onerror=bad>',text:'body',score:.8,is_gt:true,sources:[{query_index:1,rank:1}]}]};
  respond({ok:true,json:async()=>({record_key:'12',prompt:'custom {dialogue}',queries:[recalled.query,'q2'],
    fusion_method:'round_robin',top_k:10,status:'partial_success',
    query_results:[{...recalled,status:'success'},{query:'q2',status:'failed',error:'timeout',cases:[]}],fused:recalled})});
  await pending;
  assert.equal(element('multi-query-results').innerHTML,'different record');
  assert.equal(run('multiTrialFor().result'),null);
  run(`currentKey='12';renderMultiTrial();`);
  const rendered=element('multi-query-results').innerHTML;
  assert.match(rendered,/Query 1/);assert.match(rendered,/Query 2/);
  assert.match(rendered,/最终融合案例列表/);assert.match(rendered,/Query 1 排名 #1/);
  assert.match(rendered,/timeout/);assert.match(rendered,/部分 query 检索失败/);
  assert.match(rendered,/&lt;img onerror=bad&gt;/);assert.doesNotMatch(rendered,/<script>/);
  assert.equal(element('multi-query-submit').disabled,false);
  element('multi-query-prompt').value='no placeholder';
  await run('tryMultiQuery()');
  assert.match(element('multi-query-results').innerHTML,/必须包含/);
  assert.match(blankPage,/custom \{dialogue\}/);
  console.log('Multi-query frontend checks passed: request payload, loading, navigation, record ownership, prompt reuse, all result sections, source ranks, escaping and validation.');
})().catch(error=>{console.error(error);process.exitCode=1;});
