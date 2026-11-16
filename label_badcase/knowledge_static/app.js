const $=id=>document.getElementById(id);let data, index=0, dirty=false, settingsDirty=false;const trials={};let recordBase=0,settingsBase=0,annotationConflict=false,settingsConflict=false,busy=false;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const v=await r.json();if(!r.ok)throw Error(v.error||'请求失败');return v}
function message(s){$('status').textContent=s}function current(){return data.records[index]}
function cards(cs){const hit=cs.find(c=>c.is_gt);const banner=hit?`<div class="gt-banner"><strong>✓ 命中 GT 正确案例</strong><span>排名 #${esc(hit.rank)} · ${esc(hit.case_id)}</span></div>`:'';return banner+(cs.map(c=>`<details class="${c.is_gt?'gt-hit':''}"><summary>${c.is_gt?'★ GT ':''}${esc(c.rank||'')} ${esc(c.title)} (${esc(c.case_id)}) · ${esc(c.score)}</summary><pre>${esc(c.text)}</pre></details>`).join('')||'<p>没有返回案例</p>')}
function perQueryDetails(r){
  const traces=r.per_query_traces||[],queries=r.queries?.length?r.queries:[r.query];
  if(!traces.length)return '<p>该 badcase 未保存逐 Query 检索结果，请从包含 per_query_traces 的评测结果重新导出。</p>';
  const aligned=traces.length===queries.length;
  return (aligned?'':'<p class="gt-miss">逐 Query 结果数量与 query 数量不一致，可能存在检索失败。以下仅按保存顺序展示，不推断对应的 query。</p>')+traces.map((trace,i)=>`<details><summary>${aligned?'Query':'检索结果'} ${i+1} · ${trace.length} 条案例</summary>${aligned?`<pre>${esc(queries[i]||'未提供 query')}</pre>`:''}${cards(trace.map(c=>({...c,is_gt:c.case_id===r.gt.case_id,text:c.text||'案例正文缺失'})))}</details>`).join('');
}
function matches(r){const a=data.state.annotations[r.key],f=$('filter').value,s=$('search').value.toLowerCase();return(f==='all'||f==='pending'&&!a||f==='yes'&&a?.can_solve===true||f==='no'&&a?.can_solve===false)&&JSON.stringify(r).toLowerCase().includes(s)}
function fields(){$('yesFields').hidden=$('decision').value!=='yes';$('noFields').hidden=$('decision').value!=='no'}
function render(){const r=current();$('stats').textContent=`共 ${data.summary.total} 条 · 已标注 ${data.summary.completed} · 能 ${data.summary.yes} · 不能 ${data.summary.no}`;if(!r){message('没有记录');return}const a=data.state.annotations[r.key]||{};const filtered=data.records.filter(matches),page=filtered.findIndex(item=>item.key===r.key)+1;$('position').textContent=`第 ${page||'—'}/${filtered.length} 页（原始第 ${index+1} 条）`;$('pageNumber').max=filtered.length;$('pageNumber').value=page||'';$('recordTitle').textContent=`记录 ${r.key}`;$('dialogue').textContent=r.chat_content;$('original').textContent=JSON.stringify(r.queries.length?r.queries:r.query,null,2);$('gt').innerHTML=cards([r.gt]);$('retrieved').innerHTML=cards(r.retrieved);$('perQuery').innerHTML=perQueryDetails(r);$('decision').value=a.can_solve===true?'yes':a.can_solve===false?'no':'';$('entry').value=a.knowledge_entry||'';const config=data.state.record_settings?.[r.key]||data.initial_settings;settingsBase=data.state.record_settings?.[r.key]?.revision||0;$('prompt').value=config.prompt;$('knowledge').value=config.knowledge;settingsDirty=false;$('errorType').value=a.error_type||'';$('reason').value=a.reason||'';$('query').value=r.query||r.queries[0]||'';$('results').innerHTML=trials[r.key]||'';$('types').innerHTML=[...new Set(Object.values(data.state.annotations).map(a=>a.error_type).filter(Boolean))].map(t=>`<option value="${esc(t)}">`).join('');dirty=false;recordBase=a.revision||0;fields()}
async function save(){if(!dirty)return;if(annotationConflict)throw Error('请先处理标注冲突');const editVersion=editCounter;if(!$('decision').value)throw Error('请先选择能否通过知识解决');const r=current(),yes=$('decision').value==='yes';const v=await api('/api/save',{action:'annotation',key:r.key,record_revision:recordBase,annotation:{owner:$('owner').value.trim(),prompt:$('prompt').value,knowledge:$('knowledge').value,can_solve:yes,knowledge_entry:yes?$('entry').value:'',error_type:yes?'':$('errorType').value,reason:yes?'':$('reason').value}});Object.assign(data,v);recordBase=v.state.annotations[r.key].revision;dirty=editCounter!==editVersion;message(dirty?'已保存，另有新的修改待保存':'标注已保存到磁盘')}
async function settings(){if(settingsConflict)throw Error('请先处理本条提示词和知识冲突');const version=settingsCounter;const v=await api('/api/save',{action:'settings',key:current().key,settings_revision:settingsBase,settings:{prompt:$('prompt').value,knowledge:$('knowledge').value}});Object.assign(data,v);settingsBase=v.state.record_settings[current().key].revision;settingsDirty=settingsCounter!==version;message('本条提示词与知识已保存')}
async function move(dir,pending=false){await save();if(settingsDirty)await settings();if(dirty)throw Error('还有新的修改，请保存后再翻页');await sync();if(annotationConflict)throw Error('请先处理冲突');for(let n=1;n<data.records.length;n++){const j=(index+dir*n+data.records.length)%data.records.length;if(matches(data.records[j])&&(!pending||!data.state.annotations[data.records[j].key])){index=j;render();return}}message(pending?'标注已保存；当前筛选下没有其他未标注记录':'当前筛选下没有其他符合条件的记录')}
function run(fn){return async()=>{if(busy)return;busy=true;try{await fn()}catch(e){message(e.message)}finally{busy=false}}}
for(const id of ['decision','entry','errorType','reason'])$(id).oninput=()=>{dirty=true;editCounter++;fields()};for(const id of ['prompt','knowledge'])$(id).oninput=()=>{settingsDirty=true;settingsCounter++};
$('save').onclick=run(async()=>{dirty=true;await save();if(!dirty)render()});$('saveNext').onclick=run(async()=>{dirty=true;await save();await move(1,true)});$('prev').onclick=run(()=>move(-1));$('next').onclick=run(()=>move(1));for(const id of ['filter','search'])$(id).onchange=run(async()=>{await save();if(settingsDirty)await settings();if(dirty)throw Error('还有新的修改，请保存后再筛选');const j=data.records.findIndex(matches);if(j>=0){index=j;render()}else message('没有符合筛选条件的记录')});$('saveSettings').onclick=run(settings);$('useEntry').onclick=()=>{$('knowledge').value+='\n'+$('entry').value;settingsDirty=true;settingsCounter++};
function retrievalCards(result){return (result.gt_available&&!result.cases.some(c=>c.is_gt)?'<div class="gt-miss">本次结果未命中 GT 正确案例</div>':'')+cards(result.cases)}
async function trial(multi){const r=current(),key=r.key,button=$(multi?'generate':'retrieve');button.disabled=true;message('正在请求…');try{const v=await api(multi?'/api/multi-query':'/api/retrieve',multi?{key,prompt:$('prompt').value,knowledge:$('knowledge').value,fusion_method:'round_robin',top_k:10}:{key,query:$('query').value});const html=multi?(v.query_results.map(q=>`<h3>${esc(q.query)}</h3><p>${esc(q.error||'')}</p>${retrievalCards(q)}`).join('')+'<h3>融合结果</h3>'+retrievalCards(v.fused)):retrievalCards(v);trials[key]=html;if(current().key===key)$('results').innerHTML=html;message('请求完成')}finally{button.disabled=false}}
$('generate').onclick=run(()=>trial(true));$('retrieve').onclick=run(()=>trial(false));window.onbeforeunload=e=>{if(dirty||settingsDirty){e.preventDefault();e.returnValue=''}};
run(async()=>{data=await api('/api/data');data.initial_prompt=data.state.settings?.prompt||'';data.initial_knowledge=data.state.settings?.knowledge||'';render()})();

async function jump(){const page=Number($('pageNumber').value);await save();if(settingsDirty)await settings();if(dirty)throw Error('还有新的修改，请保存后再跳转');const filtered=data.records.filter(matches);if(!Number.isInteger(page)||page<1||page>filtered.length)throw Error(`请输入 1–${filtered.length} 的页码`);index=data.records.findIndex(r=>r.key===filtered[page-1].key);render();message('已跳转')}
$('jump').onclick=run(jump);$('pageNumber').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();run(jump)()}};

let editCounter=0,settingsCounter=0;
$('owner').value=localStorage.getItem('knowledge-annotator-name')||'';
$('owner').oninput=()=>{localStorage.setItem('knowledge-annotator-name',$('owner').value);dirty=true;editCounter++};
function showConflicts(){
 $('conflicts').hidden=!(annotationConflict||settingsConflict);
 for(const id of ['acceptRemote','keepLocal'])$(id).hidden=!annotationConflict;
 for(const id of ['acceptSettings','keepSettings'])$(id).hidden=!settingsConflict;
 $('remoteChanges').textContent=JSON.stringify({最新标注:annotationConflict?data.state.annotations[current().key]:undefined,最新提示词与知识:settingsConflict?data.state.record_settings?.[current().key]:undefined},null,2);
}
async function sync(){
 if(!data)return;
 try{
 const v=await api('/api/sync');
 const changed=(v.state.annotations[current()?.key]?.revision||0)!==recordBase;
 const settingsChanged=(v.state.record_settings?.[current()?.key]?.revision||0)!==settingsBase;
 Object.assign(data,v);
 if(changed&&dirty)annotationConflict=true;
 if(settingsChanged&&settingsDirty)settingsConflict=true;
 if(!dirty&&!settingsDirty&&(changed||settingsChanged))render();
 else $('stats').textContent=`共 ${data.summary.total} 条 · 已标注 ${data.summary.completed} · 能 ${data.summary.yes} · 不能 ${data.summary.no}`;
 if(!settingsDirty&&settingsChanged){const config=data.state.record_settings?.[current().key]||data.initial_settings;$('prompt').value=config.prompt;$('knowledge').value=config.knowledge;settingsBase=config.revision||0;}
 showConflicts();$('syncStatus').textContent='已同步 · 每 3 秒自动更新';
 }catch(e){$('syncStatus').textContent='同步中断 · 将自动重试';throw e}
}
$('syncNow').onclick=run(sync);
$('acceptRemote').onclick=run(async()=>{await sync();annotationConflict=false;render();showConflicts();message('已采用最新标注')});
$('keepLocal').onclick=run(async()=>{await sync();recordBase=data.state.annotations[current().key]?.revision||0;annotationConflict=false;dirty=true;await save();showConflicts()});
$('acceptSettings').onclick=run(async()=>{await sync();settingsConflict=false;settingsDirty=false;const config=data.state.record_settings?.[current().key]||data.initial_settings;settingsBase=config.revision||0;$('prompt').value=config.prompt;$('knowledge').value=config.knowledge;showConflicts()});
$('keepSettings').onclick=run(async()=>{await sync();settingsBase=data.state.record_settings?.[current().key]?.revision||0;settingsConflict=false;await settings();showConflicts()});
setInterval(()=>{if(!busy)run(sync)()},3000);
