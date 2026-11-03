// Offline interaction smoke test with a minimal DOM. No browser dependency.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.style={};this.textContent='';}
 append(...nodes){this.children.push(...nodes)} replaceChildren(...nodes){this.children=nodes}
 setAttribute(k,v){this[k]=v} focus(){} select(){}
 querySelectorAll(tag){return this.children.flatMap(n=>[...(n.tag===tag?[n]:[]),...n.querySelectorAll(tag)])}
}
const nodes={},messages=[],requests=[];
const fixture={run_dir:'/demo/run',candidates:[{keyword:'会议',knowledge:'模型原文',reason:'依据',cluster_count:2,records:[{gt_case_title:'预订会议',chat_content:'用户：如何预订',queries:['预订会议']}],example_ids:[]},{keyword:'VPN',knowledge:'远程办公',records:[]}],state:{fingerprint:'f',revision:0,index:0,confirmed:[{id:'old',text:'已有知识',source:'existing'}],drafts:{},statuses:{}}};
const context=vm.createContext({document:{getElementById:id=>nodes[id]??=(new Element()),createElement:tag=>new Element(tag)},window:{addEventListener(){},scrollTo(){}},setTimeout:()=>1,clearTimeout(){},crypto:{randomUUID:()=>`new-${Math.random()}`},console,fetch:async(url,options)=>{if(url==='/api/data')return {ok:true,json:async()=>structuredClone(fixture)};const body=JSON.parse(options.body);requests.push({url,body});return {ok:true,json:async()=>({revision:body.revision+1,paths:['/demo/run/final_knowledge.txt']})}}});
const source=fs.readFileSync(__dirname+'/index.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
vm.runInContext(source,context);
const drain=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{await drain();assert.equal(nodes.draft.value,'模型原文');
 assert.equal(nodes['confirmed-list'].querySelectorAll('textarea')[0].value,'已有知识');
 nodes.draft.value='人工修正';nodes.draft.oninput();nodes.add.onclick();await drain();
 assert.equal(nodes['confirmed-list'].querySelectorAll('textarea').length,2);
 nodes.add.onclick();await drain();assert.equal(nodes['confirmed-list'].querySelectorAll('textarea').length,2);
 nodes.next.onclick();assert.equal(nodes.draft.value,'远程办公');nodes.skip.onclick();nodes.prev.onclick();assert.equal(nodes.draft.value,'人工修正');
 const old=nodes['confirmed-list'].querySelectorAll('textarea')[0];old.value='已编辑的初始知识';old.oninput();
 nodes.export.onclick();await drain();await drain();
 const saved=requests.findLast(r=>r.url==='/api/export');assert(saved);assert.equal(saved.body.confirmed[0].text,'已编辑的初始知识');assert.equal(saved.body.confirmed[1].text,'人工修正');assert.equal(saved.body.statuses['1'],'skipped');
 assert(nodes.sources.children.length>0);console.log('UI logic smoke test passed: edit, add, dedup, navigate, skip, export');
})().catch(e=>{console.error(e);process.exitCode=1});
