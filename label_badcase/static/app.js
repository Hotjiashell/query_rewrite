/* All writes are serialized; navigation waits for the current draft to reach disk. */
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, state, stats, currentKey, draft, dirty = false, timer, queue = Promise.resolve(), fatalConflict = false;
let filter = 'all', search = '';
let defaultOwner = '';
try {defaultOwner=localStorage.getItem('badcase-default-owner')||'';} catch {}
let baseRevision = 0, remotePending = false, syncing = false;
const revisionOf = key => state.annotations[key]?.revision || 0;
const backupKey = key => `badcase-draft:${state.dataset_fingerprint}:${key}`;
function backup() {
  try { localStorage.setItem(backupKey(currentKey), JSON.stringify({draft, baseRevision})); }
  catch { notice('浏览器草稿备份失败，请保持页面打开并检查保存状态。'); }
}
function clearBackup(key) { try {localStorage.removeItem(backupKey(key));} catch {} }
function showConflict() {
  fatalConflict=true;remotePending=true;$('conflict').hidden=false;
  $('remote-annotation').textContent=JSON.stringify(state.annotations[currentKey]||blank(),null,2);
  saving('当前记录有冲突 · 草稿已保留',true);pageInfo();
}
function updateTypes() {
  const select=$('error-type');
  if(select){const names=[...state.error_types];if(draft.error_type&&!names.includes(draft.error_type))names.push(draft.error_type);select.innerHTML='<option value="">请选择错误类型</option>'+names.map(t=>`<option value="${esc(t)}" ${draft.error_type===t?'selected':''}>${esc(t)}${state.error_types.includes(t)?'':'（已删除，请重选）'}</option>`).join('');}
  if($('types-dialog').open)renderTypes();
  updateOwners();
}
function applyRemote(result) {
  if(result.state.revision < state.revision)return;
  const before=revisionOf(currentKey), typesBefore=state.types_revision, ownersBefore=JSON.stringify(state.owners);
  state=result.state;stats=result.summary;renderStats();
  if(typesBefore!==state.types_revision)updateTypes();
  else if(ownersBefore!==JSON.stringify(state.owners))updateOwners();
  const hist=document.querySelector('.histogram');if(hist)hist.outerHTML=histogram();
  if(revisionOf(currentKey)!==before || remotePending){
    if(dirty||fatalConflict){showConflict();}
    else if($('workspace').contains(document.activeElement)){
      remotePending=true;notice('当前记录已有更新。离开输入框后自动同步；继续编辑时将保留你的草稿并提示冲突。');
    }else{remotePending=false;baseRevision=revisionOf(currentKey);draft=structuredClone(state.annotations[currentKey]||blank());renderWorkspace();notice('已同步其他人的最新标注。');}
  }
  pageInfo();
}
async function sync() {
  if(syncing||!state)return;
  syncing=true;
  const job=queue.catch(()=>{}).then(async()=>{
    const response=await fetch('/api/sync',{signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw Error('同步失败');applyRemote(await response.json());
  });queue=job;
  try{await job;$('sync-state').textContent='自动同步 · 每 3 秒';}
  catch{$('sync-state').textContent='同步中断 · 正在自动重连';}
  finally{syncing=false;}
}
const hasYes = a => Object.values(a?.reviews || {}).some(r => r.verdict === 'yes');
const hasUncertain = a => Object.values(a?.reviews || {}).some(r => r.verdict === 'uncertain');
const record = () => data.records.find(r => r.key === currentKey);
const blank = () => ({reviews:{}, error_type:'', error_detail:'', owner:'', completed:false});
function ownerOptions(selected='') {
  const names=[...(state.owners||[])];if(selected&&!names.includes(selected))names.push(selected);
  return '<option value="">请选择姓名</option>'+names.map(name=>`<option value="${esc(name)}" ${name===selected?'selected':''}>${esc(name)}</option>`).join('');
}
function selectedOwner(owner='') {return owner||defaultOwner;}
function applyDefaultOwner() {
  if(!defaultOwner||!record())return;
  if(!draft.owner)draft.owner=defaultOwner;
  for(const c of record().retrieved){
    if(c.is_gt || (c.model_same!==true&&!draft.reviews[c.case_id]))continue;
    draft.reviews[c.case_id]??={verdict:'',reason:'',owner:''};
    if(!draft.reviews[c.case_id].owner)draft.reviews[c.case_id].owner=defaultOwner;
  }
}
function renderDefaultOwner() {
  $('default-owner').innerHTML='<option value="">不设置默认负责人</option>'+[...new Set([...(state.owners||[]),...(defaultOwner?[defaultOwner]:[])])].map(name=>`<option value="${esc(name)}" ${name===defaultOwner?'selected':''}>${esc(name)}</option>`).join('');
  $('default-owner-status').textContent=defaultOwner?`我的默认负责人：${defaultOwner}`:'我的默认负责人：未设置';
}
function renderOwners() {
  renderDefaultOwner();
  $('owners-list').innerHTML=(state.owners||[]).map(name=>`<div class="type-row">${esc(name)}</div>`).join('')||'<p class="muted">尚无负责人，请先添加姓名。</p>';
}
function updateOwners() {
  document.querySelectorAll('select[data-review-field="owner"],select#error-owner').forEach(select=>{const value=select.value;const html=ownerOptions(value);if(select.innerHTML!==html)select.innerHTML=html;select.value=value;});
  if($('owners-dialog').open)renderOwners();
}
function notice(message='') { $('notice').textContent=message; $('notice').hidden=!message; }
function saving(message, error=false) { $('save-state').textContent=message; $('save-state').classList.toggle('error',error); $('retry').hidden=!error; }
function visible() {
  return data.records.filter(r => {
    const a=state.annotations[r.key];
    const match=filter==='all' || filter==='pending'&&!a?.completed || filter==='complete'&&a?.completed || filter==='alternative'&&hasYes(a) || filter==='genuine'&&a?.completed&&!hasYes(a)&&!hasUncertain(a) || filter==='uncertain'&&hasUncertain(a)&&!hasYes(a) || filter==='model'&&r.retrieved.some(c=>c.model_same===true&&!c.is_gt) || filter==='model_none'&&r.model_status==='success'&&!r.retrieved.some(c=>c.model_same===true&&!c.is_gt);
    return match && (!search || [r.key,r.call_sno,r.chat_content,r.query,...r.queries,r.gt.title,...r.retrieved.map(c=>c.title)].join(' ').toLowerCase().includes(search));
  });
}
function renderStats() {
  const percent=stats.total ? Math.round(stats.completed/stats.total*100) : 0;
  $('stats').innerHTML=[
    ['全部 badcase',stats.total,'本次待复核数据'],
    ['已完成标注',`${stats.completed} / ${stats.total}`,`${percent}% · ${stats.drafts} 条草稿`],
    ['人工确认有其他正确案例',stats.alternative,'包含已确认“是”的草稿'],
    ['真实 badcase',stats.genuine,'仅统计完成且没有其他正确案例'],
    ['模型认为有其他正确案例',stats.model_positive,'模型建议，等待人工核验']
  ].map(([title,value,sub],i)=>`<div class="stat ${i===2?'highlight':''}"><span>${title}</span><strong>${value}</strong><small>${sub}</small>${i===1?`<div class="progress"><i style="width:${percent}%"></i></div>`:''}</div>`).join('');
}
function histogram() {
  const max=Math.max(1,...Object.values(stats.error_counts));
  return `<div class="histogram"><h3>错误类型分布</h3><div class="model-note">仅统计已完成的真实 badcase</div>${Object.entries(stats.error_counts).map(([name,count])=>`<div class="hist-row"><span>${esc(name)}</span><div class="hist-bar"><i style="width:${count/max*100}%"></i></div><b>${count}</b></div>`).join('')}</div>`;
}
function pageInfo() {
  const list=visible(), index=list.findIndex(r=>r.key===currentKey);
  const originalIndex=data.records.findIndex(r=>r.key===currentKey);
  $('page-info').textContent=index<0?`当前记录不在筛选结果中 · ${list.length} 条`:`${index+1} / ${list.length} 条`;
  $('jump').max=list.length; $('jump').value=index>=0?index+1:'';
  $('prev').disabled=index>=0?index<=0:!list.some(r=>data.records.indexOf(r)<originalIndex);
  $('next').disabled=index>=0?index>=list.length-1:!list.some(r=>data.records.indexOf(r)>originalIndex);
  $('complete').disabled=!record() || fatalConflict;
  $('record-state').innerHTML=record()?`<b>样本 ${esc(currentKey)}</b> · ${draft.completed?'<span class="tag green">已完成</span>':'<span class="tag gold">待完成</span>'} <span class="muted">修改后自动保存</span>`:'';
  $('complete').textContent=draft?.completed?'重新确认完成':'完成标注';
}
function reviewCard(c) {
  const r=draft.reviews[c.case_id]||{};
  return `<article class="candidate ${c.model_same===true?'positive':''}" data-cid="${esc(c.case_id)}"><div class="card-heading"><h3>#${c.rank} ${esc(c.title)}</h3><span class="tag ${c.model_same===true?'green':'gold'}">${c.model_same===true?'模型认为正确':'人工补充核验'}</span></div><span class="case-id">${esc(c.case_id)}</span><div class="content">${c.text?esc(c.text):'<span class="tag red">案例正文缺失，请补齐案例库再核验</span>'}</div>${c.model_reason?`<div class="model-reason"><b>模型理由</b><br>${esc(c.model_reason)}</div>`:''}<div class="verdict"><button data-verdict="yes" class="${r.verdict==='yes'?'selected':''}" aria-pressed="${r.verdict==='yes'}">是，能解决</button><button data-verdict="no" class="${r.verdict==='no'?'selected':''}" aria-pressed="${r.verdict==='no'}">否，不能解决</button><button data-verdict="uncertain" class="${r.verdict==='uncertain'?'selected':''}" aria-pressed="${r.verdict==='uncertain'}">不确定</button></div><label class="form-field"><span>判断理由（不确定可留空）${c.model_same===true&&c.model_reason?.trim()?'（选“是”也可留空，接受模型理由；选“否”必填）':'（选“是”或“否”必填）'}</span><textarea data-review-field="reason" placeholder="${c.model_same===true&&c.model_reason?.trim()?'选是可留空接受模型理由，也可填写自己的理由':'请说明判断理由'}">${esc(r.reason||'')}</textarea></label><label class="form-field"><span>核验负责人 *</span><select data-review-field="owner">${ownerOptions(selectedOwner(r.owner))}</select></label></article>`;
}
function errorPanel() {
  if(hasYes(draft)) return `<div class="card"><h2>已有其他正确案例</h2><p>已人工确认至少一个召回案例能够解决用户问题。完成其余待核验案例后，可点击“完成标注”。</p><p class="muted">如改为全部否决，将显示真实 badcase 分类表单。</p></div>`;
  if(hasUncertain(draft)) return '<div class="card"><h2>待进一步确认</h2><p>目前有案例标为“不确定”，且没有确认正确的案例。可以完成本轮标注，无需填写错误分类；该记录不会计入真实 badcase 或错误类型统计。</p></div>';
  return `<div class="card"><h2>真实 badcase 分类</h2><p class="muted">当所有待核验案例都不能解决问题，或没有待核验案例时，请填写以下信息。尚未完成核验的记录不会计入真实 badcase 数量。</p><label class="form-field"><span>错误类型 *</span><select id="error-type"><option value="">请选择错误类型</option>${state.error_types.map(t=>`<option value="${esc(t)}" ${draft.error_type===t?'selected':''}>${esc(t)}</option>`).join('')}</select></label><label class="form-field"><span>详细说明 *</span><textarea id="error-detail" placeholder="说明漏召回的原因、query的问题、正确处理方向等">${esc(draft.error_detail)}</textarea></label><label class="form-field"><span>分类负责人 *</span><select id="error-owner">${ownerOptions(selectedOwner(draft.owner))}</select></label></div>`;
}
function renderWorkspace() {
  const r=record();
  if(!r) { $('workspace').innerHTML='<div class="empty"><h2>没有匹配的 badcase</h2><p>调整筛选条件或检查输入数据。</p></div>';pageInfo();return; }
  const selected=r.retrieved.filter(c=>!c.is_gt&&(c.model_same===true||draft.reviews[c.case_id]));
  $('workspace').innerHTML=`<div class="column"><section class="card"><div class="card-heading"><h2>用户与客服对话</h2><span class="tag">样本 ${esc(r.key)}</span></div><div class="model-note">会话编号 ${esc(r.call_sno||'未提供')}</div><div class="content scroll-content">${esc(typeof r.chat_content==='string'?r.chat_content:JSON.stringify(r.chat_content,null,2))||'未提供对话'}</div></section><section class="card"><h2>检索 Query</h2><div class="label">QUERY</div><div class="query">${esc(r.query)||'未提供'}</div><div class="label">QUERIES · ${r.queries.length} 条</div>${r.queries.length?r.queries.map((q,i)=>`<div class="query">${i+1}. ${esc(q)}</div>`).join(''):'<p class="muted">未提供多路 query</p>'}</section><section class="card"><div class="card-heading"><h2>GT 案例</h2><span class="tag green">标注正确案例</span></div><h3>${esc(r.gt.title)}</h3><span class="case-id">${esc(r.gt.case_id)}</span><div class="content scroll-content">${esc(r.gt.text)||'<span class="tag red">GT正文缺失</span>'}</div></section><section class="card"><h2>召回 Top-10</h2><p class="model-note">可查看正文，并补充核验模型漏选的非GT案例。</p><table class="retrieval"><tbody>${r.retrieved.map(c=>`<tr><td>${c.rank}</td><td><details><summary>${esc(c.title)}</summary><span class="case-id">${esc(c.case_id)}</span><div class="content">${esc(c.text)||'案例正文缺失'}</div>${c.model_reason?`<div class="model-note">模型理由：${esc(c.model_reason)}</div>`:''}</details>${c.is_gt?'<span class="tag">GT本身，不计入其他正确案例</span>':c.model_same===true?'<span class="tag green">模型认为正确</span>':c.model_same===false?'<span class="tag">模型否决</span>':'<span class="tag gold">未判断 / 未选中</span>'}</td><td>${c.is_gt?'':`<button data-add="${esc(c.case_id)}" ${selected.includes(c)?'disabled':''}>${selected.includes(c)?'已列入':'补充核验'}</button>`}</td></tr>`).join('')}</tbody></table>${histogram()}</section></div><div class="column"><section class="card"><div class="card-heading"><h2>人工核验</h2><span class="tag">${selected.length} 个待核验案例</span></div><p class="muted">核验召回案例能否直接或间接解决对话中的用户问题。每个案例分别填写判断和负责人。对模型推荐案例选“是”时可留空理由，表示接受模型理由；选“否”时需填写理由，“不确定”无需理由。</p>${r.model_status!=='success'?`<div class="model-reason"><span class="tag gold">模型分析${r.model_status==='missing'?'缺失':'失败'}</span><p>${esc(r.model_error||'请查看召回列表并手动补充需要核验的案例。')}</p></div>`:''}${r.stage1_reason?`<details><summary>查看模型初筛依据</summary><p class="content model-note">${esc(r.stage1_reason)}</p></details>`:''}${selected.length?selected.map(reviewCard).join(''):'<p class="muted">模型没有建议核验的正确案例。请复核左侧召回列表，必要时补充核验；确认没有其他正确案例后填写错误分类。</p>'}</section><div id="classification">${errorPanel()}</div></div>`;
  pageInfo();
}
function openRecord(key) {
  currentKey=key;baseRevision=revisionOf(key);draft=structuredClone(state.annotations[key]||blank());dirty=false;fatalConflict=false;remotePending=false;$('conflict').hidden=true;notice();
  try{const saved=JSON.parse(localStorage.getItem(backupKey(key))||'null');if(saved){draft=saved.draft;baseRevision=saved.baseRevision;dirty=true;notice('已恢复本浏览器尚未保存的草稿。');}}catch{}
  renderWorkspace();if(dirty){if(baseRevision!==revisionOf(key))showConflict();else{saving('已恢复草稿 · 待保存');timer=setTimeout(()=>flush().catch(()=>{}),450);}}
}
function changed() {applyDefaultOwner();draft.completed=false;dirty=true;backup();if(remotePending||baseRevision!==revisionOf(currentKey))showConflict();else saving('待保存…');clearTimeout(timer);if(!fatalConflict)timer=setTimeout(()=>flush().catch(()=>{}),450);pageInfo();}
async function request(body) {
  const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),15000);
  try {
    const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,types_revision:state.types_revision}),signal:controller.signal});
    const result=await response.json();
    if(!response.ok) { if(response.status===409&&body.action==='annotation'){fatalConflict=true;$('conflict').hidden=false;}throw Error(result.error||'保存失败'); }
    if(body.action==='annotation'){state=result.state;stats=result.summary;renderStats();updateTypes();const hist=document.querySelector('.histogram');if(hist)hist.outerHTML=histogram();}
    else applyRemote(result);
    return result;
  } finally {clearTimeout(timeout);}
}
function flush() {
  clearTimeout(timer);
  const job=queue.catch(()=>{}).then(async()=>{
    if(!dirty)return;
    if(fatalConflict)throw Error('这条badcase存在冲突，请查看最新标注并选择处理方式；你的草稿仍保留。');
    const key=currentKey, snapshot=structuredClone(draft);
    saving('正在写入文件…');
    await request({action:'annotation',key,annotation:snapshot,record_revision:baseRevision});
    baseRevision=revisionOf(key);
    if(key===currentKey&&JSON.stringify(snapshot)===JSON.stringify(draft)) {draft=structuredClone(state.annotations[key]);dirty=false;clearBackup(key);}else backup();
    saving(dirty?'待保存…':`已保存 · ${new Date().toLocaleTimeString()}`);pageInfo();
  });
  queue=job;
  return job.catch(error=>{saving('保存失败 · 草稿仍在页面中',true);notice(error.message);throw error;});
}
async function navigate(delta) {
  try {await flush();const list=visible();const i=list.findIndex(r=>r.key===currentKey), originalIndex=data.records.findIndex(r=>r.key===currentKey);const next=i>=0?list[i+delta]:(delta>0?list.find(r=>data.records.indexOf(r)>originalIndex):list.findLast(r=>data.records.indexOf(r)<originalIndex));if(next){openRecord(next.key);window.scrollTo(0,0);}}catch{}
}
async function nextPending() {
  const button=$('next-pending');button.disabled=true;
  try {
    await flush();
    await sync();
    // Changes made while syncing must also reach disk before switching records.
    await flush();
    const originalIndex=data.records.findIndex(r=>r.key===currentKey);
    const candidates=visible().filter(r=>r.key!==currentKey&&!state.annotations[r.key]?.completed);
    const next=candidates.find(r=>data.records.indexOf(r)>originalIndex)||candidates[0];
    if(!next){notice('当前筛选和搜索范围内没有其他未完成的 badcase。');return;}
    const wrapped=data.records.indexOf(next)<=originalIndex;
    openRecord(next.key);window.scrollTo(0,0);
    if(wrapped)notice('已到末尾，从当前筛选范围的开头继续查找未完成记录。');
  }catch{}finally{button.disabled=false;}
}
async function changeView() {
  try {await flush();filter=$('filter').value;search=$('search').value.trim().toLowerCase();const list=visible();openRecord(list.some(r=>r.key===currentKey)?currentKey:list[0]?.key);}catch{}
}
$('workspace').addEventListener('input',event=>{
  const el=event.target, card=el.closest('[data-cid]');
  if(card&&el.dataset.reviewField){const cid=card.dataset.cid;draft.reviews[cid]??={verdict:'',reason:'',owner:''};draft.reviews[cid][el.dataset.reviewField]=el.value;changed();}
  else if(['error-type','error-detail','error-owner'].includes(el.id)){draft[{'error-type':'error_type','error-detail':'error_detail','error-owner':'owner'}[el.id]]=el.value;changed();}
});
$('workspace').addEventListener('click',event=>{
  const el=event.target.closest('button');if(!el)return;
  if(el.dataset.verdict){const card=el.closest('[data-cid]'),cid=card.dataset.cid;draft.reviews[cid]??={verdict:'',reason:'',owner:''};draft.reviews[cid].verdict=el.dataset.verdict;changed();card.querySelectorAll('[data-verdict]').forEach(b=>{b.classList.toggle('selected',b===el);b.setAttribute('aria-pressed',b===el?'true':'false');});$('classification').innerHTML=errorPanel();}
  if(el.dataset.add){draft.reviews[el.dataset.add]={verdict:'',reason:'',owner:''};changed();renderWorkspace();}
});
$('prev').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);
$('next-pending').onclick=nextPending;
$('jump-go').onclick=async()=>{try{await flush();const list=visible(),n=Number($('jump').value);if(Number.isInteger(n)&&n>=1&&n<=list.length){openRecord(list[n-1].key);window.scrollTo(0,0);}else notice('请输入筛选范围内的有效页码');}catch{}};
$('filter').onchange=changeView;$('search').onchange=changeView;
$('complete').onclick=async()=>{
  const r=record();if(!r)return;
  applyDefaultOwner();
  const selected=r.retrieved.filter(c=>!c.is_gt&&(c.model_same===true||draft.reviews[c.case_id]));
  if(selected.some(c=>{const x=draft.reviews[c.case_id];const acceptsModel=x?.verdict==='yes'&&c.model_same===true&&c.model_reason?.trim();return !x?.verdict||(x.verdict!=='uncertain'&&!x.reason?.trim()&&!acceptsModel)||!x.owner?.trim();})){notice('请为每个待核验案例选择判断和负责人；不确定可留空理由，选是可接受模型理由，选否必须填写理由');return;}
  if(!hasYes(draft)&&!hasUncertain(draft)&&(!draft.error_type||!draft.error_detail.trim()||!draft.owner.trim())){notice('请填写真实 badcase 的错误类型、详细说明和负责人');return;}
  draft.completed=true;dirty=true;backup();try{await flush();notice();renderWorkspace();}catch{}
};
$('retry').onclick=()=>flush().catch(()=>{});
$('export').onclick=async()=>{try{await flush();const response=await fetch('/api/export');if(!response.ok)throw Error('导出失败');const blob=new Blob([JSON.stringify(await response.json(),null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='badcase_annotations.json';a.click();URL.revokeObjectURL(url);}catch(e){notice(e.message);}};
function renderTypes() {$('types-list').innerHTML=state.error_types.map(t=>`<div class="type-row"><span>${esc(t)}</span>${t==='其他'?'<span class="muted">保留类型</span>':`<button data-delete-type="${esc(t)}">删除</button>`}</div>`).join('');}
$('types-open').onclick=async()=>{try{await flush();renderTypes();$('types-dialog').showModal();}catch{}};
$('types-close').onclick=()=>$('types-dialog').close();
$('owners-open').onclick=()=>{renderOwners();$('owners-message').textContent='';$('owners-dialog').showModal();};
$('owners-close').onclick=()=>$('owners-dialog').close();
$('default-owner').onchange=()=>{
  const value=$('default-owner').value;
  try{localStorage.setItem('badcase-default-owner',value);defaultOwner=value;if(dirty){applyDefaultOwner();backup();}renderDefaultOwner();renderWorkspace();$('owners-message').textContent=value?`已设置默认负责人：${value}（仅当前浏览器）`:'已取消默认负责人';}
  catch{$('owners-message').textContent='浏览器未允许保存设置，请检查本地存储权限。';renderDefaultOwner();}
};
$('owner-form').onsubmit=async event=>{
  event.preventDefault();const name=$('owner-name').value.trim();
  const button=$('owner-form').querySelector('button');button.disabled=true;
  const job=queue.catch(()=>{}).then(()=>request({action:'add_owner',name}));queue=job;
  try{await job;updateOwners();$('owner-name').value='';$('owners-message').textContent=`已保存负责人：${name}`;}
  catch(error){$('owners-message').textContent=error.message;}
  finally{button.disabled=false;}
};
async function typeMutation(action,name) {
  try{await flush();const job=queue.catch(()=>{}).then(()=>request({action,name}));queue=job;await job;renderTypes();saving('错误类型已保存');}
  catch(e){notice(e.message);alert(e.message);}
}
$('type-form').onsubmit=async event=>{event.preventDefault();const name=$('type-name').value.trim();await typeMutation('add_type',name);if(state.error_types.includes(name))$('type-name').value='';};
$('types-list').onclick=event=>{const name=event.target.dataset.deleteType;if(name&&confirm(`删除“${name}”？已有该类型标注将转为“其他”，并保留原类型说明。`))typeMutation('delete_type',name);};
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||$('types-dialog').open||$('owners-dialog').open)return;if(event.key==='ArrowLeft')navigate(-1);if(event.key==='ArrowRight')navigate(1);});
$('use-remote').onclick=async()=>{if(!confirm('采用最新标注将放弃本条尚未保存的草稿。是否继续？'))return;await sync();clearBackup(currentKey);openRecord(currentKey);saving('已采用最新标注');};
$('use-draft').onclick=async()=>{
  await sync();if(!confirm('确认以页面中的草稿替换刚刚展示的服务端最新标注？'))return;
  baseRevision=revisionOf(currentKey);fatalConflict=false;remotePending=false;$('conflict').hidden=true;dirty=true;backup();try{await flush();notice();}catch{}
};
$('download-draft').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify({record_key:currentKey,baseRevision,draft},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`badcase-${currentKey}-draft.json`;a.click();URL.revokeObjectURL(url);};
document.addEventListener('focusout',()=>{setTimeout(()=>{if(remotePending&&!dirty&&!fatalConflict&&!$('workspace').contains(document.activeElement)){remotePending=false;baseRevision=revisionOf(currentKey);draft=structuredClone(state.annotations[currentKey]||blank());renderWorkspace();notice('已同步当前记录的最新标注。');}},0);});
setInterval(sync,3000);
(async()=>{try{const response=await fetch('/api/data');if(!response.ok)throw Error('数据加载失败');data=await response.json();state=data.state;stats=data.summary;renderStats();openRecord(data.records[0]?.key);saving('已连接 · 自动保存');$('save-state').title=`输出文件：${data.output}`;renderDefaultOwner();}catch(e){notice(e.message);saving('加载失败',true);$('workspace').innerHTML='<div class="empty">请检查本地服务及输入文件后刷新页面。</div>';}})();
