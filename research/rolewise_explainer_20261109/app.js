(function(){
const root=document.getElementById('relay-lab'),$=id=>document.getElementById(id),E=makeRelayEngine();
let data=null,hop=0,phase='final',step=0,position=20,query=0,block='total',playing=false,timer=null,cache={},worker=null,job=0;
const names=['Planner','Critic','Refiner','Judger'],roleNames=['self / sink','Planner','Critic','Refiner'],maxEnds=[170,324,510];
const fmt=x=>x===0?'0':x.toExponential(2),mb=x=>(x/1e6).toFixed(2),rgba=(c,a)=>{const h=c.replace('#','');return `rgba(${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)},${a})`;};
function colors(){const c=getComputedStyle(document.documentElement);return Object.fromEntries(['surface','ink','muted','soft','line','g','r','p','c','f','latent','sink'].map(k=>[k,c.getPropertyValue('--'+k).trim()]));}
function setup(canvas,height){if(height)canvas.style.height=height+'px';const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return {c,w,h};}
function mix(a,b,t){const expand=c=>{const h=c.replace('#','');return h.length===3?h.split('').map(x=>x+x).join(''):h;},x=expand(a),y=expand(b),v=[0,2,4].map(i=>Math.round(parseInt(x.slice(i,i+2),16)*(1-t)+parseInt(y.slice(i,i+2),16)*t));return `rgb(${v.join(',')})`;}
function heat(v,max,C){return mix(C.surface,v<0?C.p:C.f,Math.min(1,Math.abs(v)/(max||1))*.9);}
function current(s){const key=s.mode+hop+phase+step;if(cache[s.mode]?.key===key)return cache[s.mode];const keep=E.stateMask(s,phase,step),metrics=phase==='final'?s.finalMetrics:E.metrics(s,keep),value={key,keep,metrics};cache[s.mode]=value;return value;}
function vector(s,array,q,b){const out=new Float64Array(s.d);for(let d=0;d<s.d;d++){if(b==='total'){for(let r=0;r<4;r++)out[d]+=array[(r*s.q+q)*s.d+d];}else out[d]=array[(Number(b)*s.q+q)*s.d+d];}return out;}
function updateStory(){
 $('hop-label').textContent=`HANDOFF 0${hop+1} / ${names[hop].toUpperCase()} → ${names[hop+1].toUpperCase()}`;
 const title={full:'接收者先观察整段历史。',bank:'先限定候选池，再决定删谁。',delete:'预算不变，逐行看看两种取舍。',final:`交付${110+hop*10}行原K/V，进入${names[hop+1]}。`};$('stage-title').textContent=title[phase];
 const text={full:`${names[hop+1]}的prompt只在probe中临时加入；本次携带${10*(hop+1)}个已有latent和4个sink。`,bank:'最多192行prompt进入候选池，最新发送方末8行强制入池。latent、sink和接收者self都固定保留。',delete:'G依据注意力排名；R每删一行都重新计算四块读出误差，所有块共享新的归一化分母Z。',final:hop===0?'第一次选择从同一个Planner缓存和同一个probe起点分叉。Critic随后分别沿G、R自己的历史生成。':hop===1?'前一跳删掉的prompt不再出现。新Critic的prompt和旧Planner幸存行一起竞争96行预算。':'第三跳合并剩余三种来源竞争96行prompt。Judger接收96 prompt + 30 latent + 4 sink。'};
 $('story').textContent=text[phase];
 document.querySelectorAll('[data-hop]').forEach(b=>{const active=Number(b.dataset.hop)===hop;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
 document.querySelectorAll('[data-phase]').forEach(b=>{const active=b.dataset.phase===phase;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
 const n=data.paths.R[hop].trace.length;$('delete-step').max=n;$('delete-step').value=phase==='full'||phase==='bank'?0:phase==='final'?n:step;
 $('step-count').textContent=`${$('delete-step').value} / ${n}`;$('step-caption').textContent=phase==='final'?'96行prompt，已交付':phase==='delete'?`候选池内第${step}次删除`:'候选池以外行先移除';
 $('query-label').textContent=`GQA头${Math.floor(query/16)+1} · 倒数第${16-query%16}个query`;
 $('position').max=maxEnds[hop]-1;
}
function drawMap(method,s,keep){
 const canvas=$('map-'+method),C=colors(),cols=Math.max(20,Math.floor(canvas.clientWidth/12)),rows=Math.ceil(s.end/cols),{c,w,h}=setup(canvas,rows*13+22),cw=w/cols,ch=12;
 const lookup=new Map(s.rows.slice(0,s.relayN).map((r,i)=>[r.id,{r,i}]));canvas._map={cols,cw,ch,lookup,s,keep};
 for(let id=0;id<s.end;id++){
  const x=id%cols*cw,y=Math.floor(id/cols)*13,item=lookup.get(id);
  if(!item){c.fillStyle=rgba(C.muted,.07);c.fillRect(x,y,cw-2,ch-1);continue;}
  const {r,i}=item,base=r.kind==='latent'?C.latent:r.kind==='sink'?C.sink:[C.sink,C.p,C.c,C.f][r.role];
  c.fillStyle=keep[i]?base:rgba(base,.13);c.fillRect(x,y,cw-2,ch-1);
  if(!keep[i]){c.strokeStyle=rgba(base,.55);c.lineWidth=.7;c.beginPath();c.moveTo(x+2,y+ch-3);c.lineTo(x+cw-4,y+2);c.stroke();}
  if(s.bankInfo.mandatory[i]){c.fillStyle=C.surface;c.fillRect(x+1,y+1,3,3);}
  if(id===position){c.strokeStyle=C.ink;c.lineWidth=1.5;c.strokeRect(x-1,y-1,cw,ch+1);}
 }
 c.fillStyle=C.muted;c.font='11px system-ui';c.fillText('原位置 0',0,h-3);c.textAlign='right';c.fillText(String(s.end-1),w,h-3);c.textAlign='left';
}
function drawReadout(method,s,metrics){
 const C=colors(),{c,w}=setup($('readout-'+method)),before=vector(s,s.target,query,block),after=vector(s,metrics.out,query,block),err=Float64Array.from(after,(v,i)=>v-before[i]);
 const max=Math.max(...before.map(Math.abs),...after.map(Math.abs),1e-10),emax=Math.max(...err.map(Math.abs),1e-10),left=76,cw=(w-left)/s.d,values=[before,after,err],labels=['原读出','删后读出','Δ 独立色标'];
 for(let r=0;r<3;r++){const y=r*28;c.fillStyle=C.muted;c.font='11px system-ui';c.fillText(labels[r],0,y+13);for(let d=0;d<s.d;d++){c.fillStyle=heat(values[r][d],r===2?emax:max,C);c.fillRect(left+d*cw,y,cw+.2,18);}}
 c.fillStyle=C.muted;c.font='11px ui-monospace';for(const d of [0,31,63,95,127]){c.textAlign=d===127?'right':'left';c.fillText('d'+(d+1),left+d*cw,105);}c.textAlign='left';
 $('readout-'+method)._data={values,left,cw};
}
function lane(method){
 const s=data.paths[method][hop],{keep,metrics}=current(s),kept=s.rows.slice(0,s.relayN).filter((_,i)=>keep[i]),count=kept.length,bytes=count*147456,latent=kept.filter(r=>r.kind==='latent').length,prompts=kept.filter(r=>r.kind==='prompt').length;
 $('stats-'+method).innerHTML=`<div class="stat"><small>交付历史KV</small><strong>${count}<span> 行</span></strong></div><div class="stat"><small>其中prompt / latent</small><strong>${prompts}<span> / ${latent}</span></strong></div><div class="stat"><small>BF16全模型换算</small><strong>${mb(bytes)}<span> MB</span></strong></div>`;
 $('map-info-'+method).textContent=`${s.relayN}行可用历史 · self ${s.n-s.relayN}行仅probe`;
 drawMap(method,s,keep);
 $('roles-'+method).innerHTML=[1,2,3].map((r,j)=>{const color=[colors().p,colors().c,colors().f][j],n=kept.filter(x=>x.kind==='prompt'&&x.role===r).length;return `<span><i style="background:${color}"></i>${roleNames[r]} ${n}</span>`;}).join('');
 const b=s.bankInfo,deleted=phase==='delete'&&step>0?s.trace[Math.min(step,s.trace.length)-1]:null;
 $('bank-'+method).textContent=`prompt ${b.candidates.length} → 候选池 ${b.bankSize} → 最终 ${b.budget}。`+(deleted?` 本步剔除原位置 ${deleted.id}。`:'')+(method==='G'?' G直接选Top96；剔除顺序按排名展示。':' R逐次选“删后分块误差最小”的行。');
 drawReadout(method,s,metrics);$('errors-'+method).innerHTML=`<span>四块平均 MSE <b>${fmt(metrics.blockMSE)}</b></span><span>整体读出 MSE <b>${fmt(metrics.totalMSE)}</b></span>`;
 return {s,keep,metrics};
}
function blockBars(g,r){
 const max=Math.max(...g.metrics.roleMSE,...r.metrics.roleMSE,1e-15);$('block-bars').innerHTML='<div class="block-legend"><span><i style="background:var(--g)"></i>G 当前局部probe</span><span><i style="background:var(--r)"></i>R 当前局部probe</span></div>'+roleNames.map((n,i)=>`<div class="block-row"><span>${n}</span><div class="bar-stack"><div class="bar-track"><div class="bar gbar" style="width:${100*g.metrics.roleMSE[i]/max}%"></div></div><div class="bar-track"><div class="bar rbar" style="width:${100*r.metrics.roleMSE[i]/max}%"></div></div></div><div class="bar-value">G ${fmt(g.metrics.roleMSE[i])}<br>R ${fmt(r.metrics.roleMSE[i])}</div></div>`).join('');
}
function inspect(method,s,keep){
 const canvas=$('kv-'+method),C=colors(),row=s.rows.slice(0,s.relayN).findIndex(r=>r.id===position),{c,w,h}=setup(canvas),gap=3,cw=(w-gap*15)/16,ch=(h-18-gap*7)/8;
 let vector=null;
 if(row>=0){const r=s.rows[row];vector=r[$('kv-kind').value];$('inspect-label-'+method).textContent=`${method==='G'?'GGG':'RRR'} · 原位置${position} · ${roleNames[r.role]} / ${r.kind} · ${keep[row]?'保留，原值直接gather':'本次删除'}`;}
 else $('inspect-label-'+method).textContent=`${method==='G'?'GGG':'RRR'} · 原位置${position} · 前跳已删除，本跳不再候选`;
 const max=vector?Math.max(...vector.map(Math.abs),1e-10):1;
 for(let d=0;d<128;d++){const x=d%16*(cw+gap),y=Math.floor(d/16)*(ch+gap);c.fillStyle=vector?heat(vector[d],max,C):C.soft;c.fillRect(x,y,cw,ch);}
 c.fillStyle=C.muted;c.font='11px system-ui';c.fillText(vector?`128维 · min ${Math.min(...vector).toFixed(3)} / max ${Math.max(...vector).toFixed(3)}`:'该行不在当前缓存',0,h-2);canvas._data={vector,cw,ch,gap};
}
function journey(){
 $('journey').innerHTML=[0,1,2].map(h=>{const n=110+h*10,original=maxEnds[h];return `<tr><td>${names[h]} → ${names[h+1]}</td><td>${original}行</td><td>96，共同竞争</td><td>${10*(h+1)} latent + 4 sink</td><td>${n}行</td><td>${mb(n*147456)} MB</td><td>${(100*n/original).toFixed(1)}%</td></tr>`;}).join('');
}
function render(){if(!data)return;updateStory();const g=lane('G'),r=lane('R');blockBars(g,r);inspect('G',g.s,g.keep);inspect('R',r.s,r.keep);window.__relayView={hop,phase,step,query,block,position,rowsG:g.s.rows.slice(0,g.s.relayN).filter((_,i)=>g.keep[i]).length,rowsR:r.s.rows.slice(0,r.s.relayN).filter((_,i)=>r.keep[i]).length};}
function stop(){playing=false;clearInterval(timer);$('play').textContent='▶ 顺序演示';}
function calculate(){
 stop();const cfg={seed:Number($('seed').value)||42,layer:Number($('layer').value),head:Number($('head').value)},id=++job;$('status').textContent='计算64 × 128读出与逐行剔除…';$('rebuild').disabled=true;
 const done=result=>{if(id!==job)return;data=result;window.__relayData=data;cache={};step=data.paths.R[hop].trace.length;phase='final';$('rebuild').disabled=false;$('status').textContent=`已计算三次交接 · 第${cfg.layer}层 / KV头${cfg.head} · seed ${cfg.seed}`;journey();render();};
 const fallback=()=>setTimeout(()=>{try{done(E.run(cfg));}catch(e){$('status').textContent='计算错误：'+e.message;$('rebuild').disabled=false;}},30);
 try{if(worker)worker.terminate();const source=`const E=(${makeRelayEngine.toString()})();onmessage=e=>{try{postMessage({result:E.run(e.data)})}catch(err){postMessage({error:err.message})}}`;
  const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));worker=new Worker(url);URL.revokeObjectURL(url);worker.onmessage=e=>{if(e.data.error){$('status').textContent=e.data.error;$('rebuild').disabled=false;}else done(e.data.result);};worker.onerror=()=>{worker.terminate();fallback();};worker.postMessage(cfg);
 }catch{fallback();}
}
for(let i=1;i<=36;i++)$('layer').add(new Option(`第${i} / 36层`,i));for(let i=1;i<=8;i++)$('head').add(new Option(`第${i} / 8头`,i));
document.querySelectorAll('[data-hop]').forEach(b=>b.onclick=()=>{stop();hop=Number(b.dataset.hop);phase='final';step=data?data.paths.R[hop].trace.length:0;position=Math.min(position,maxEnds[hop]-1);$('position').value=position;render();});
document.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>{stop();phase=b.dataset.phase;step=phase==='delete'?0:data?.paths.R[hop].trace.length||0;render();});
$('delete-step').oninput=()=>{stop();phase='delete';step=Number($('delete-step').value);render();};$('query').oninput=()=>{query=Number($('query').value);render();};$('block').onchange=()=>{block=$('block').value;render();};$('position').oninput=()=>{position=Math.max(0,Math.min(maxEnds[hop]-1,Number($('position').value)||0));render();};$('kv-kind').onchange=render;$('rebuild').onclick=calculate;
$('play').onclick=()=>{if(playing){stop();return;}if(!data)return;playing=true;hop=0;phase='full';step=0;$('play').textContent='Ⅱ 暂停';render();let tick=0;timer=setInterval(()=>{if(phase==='full')phase='bank';else if(phase==='bank'){phase='delete';step=0;}else if(phase==='delete'){step=Math.min(data.paths.R[hop].trace.length,step+8);if(step===data.paths.R[hop].trace.length)phase='final';}else if(hop<2){hop++;phase='full';step=0;}else{stop();}tick++;render();},600);};
function tooltip(text,e){const t=$('tooltip');t.textContent=text;t.hidden=false;t.style.left=Math.max(8,Math.min(window.innerWidth-290,e.clientX+12))+'px';t.style.top=Math.max(8,Math.min(window.innerHeight-80,e.clientY+12))+'px';}
for(const method of ['G','R']){
 const canvas=$('map-'+method);function hit(e){const a=canvas._map;if(!a)return null;const r=canvas.getBoundingClientRect(),col=Math.floor((e.clientX-r.left)/a.cw),row=Math.floor((e.clientY-r.top)/13),id=row*a.cols+col;return id<a.s.end?id:null;}
 canvas.onclick=e=>{const id=hit(e);if(id===null)return;position=id;$('position').value=id;render();};
 canvas.onmousemove=e=>{const id=hit(e),a=canvas._map;if(id===null||!a){$('tooltip').hidden=true;return;}const x=a.lookup.get(id);tooltip(x?`位置${id} · ${roleNames[x.r.role]} ${x.r.kind} · ${a.keep[x.i]?'保留':'本次删除'}`:`位置${id} · 前跳已删除`,e);};canvas.onmouseleave=()=>$('tooltip').hidden=true;
 $('kv-'+method).onmousemove=e=>{const canvas=$('kv-'+method),v=canvas._data;if(!v?.vector)return;const r=canvas.getBoundingClientRect(),col=Math.floor((e.clientX-r.left)/(v.cw+v.gap)),row=Math.floor((e.clientY-r.top)/(v.ch+v.gap)),d=row*16+col;if(d>=0&&d<128)tooltip(`d${d+1} = ${v.vector[d].toFixed(6)}`,e);};$('kv-'+method).onmouseleave=()=>$('tooltip').hidden=true;
 $('readout-'+method).onmousemove=e=>{const c=$('readout-'+method),v=c._data;if(!v)return;const r=c.getBoundingClientRect(),d=Math.floor((e.clientX-r.left-v.left)/v.cw),row=Math.floor((e.clientY-r.top)/28);if(d>=0&&d<128&&row>=0&&row<3)tooltip(`${['原读出','删后读出','差值'][row]} · d${d+1} = ${v.values[row][d].toFixed(7)}`,e);};$('readout-'+method).onmouseleave=()=>$('tooltip').hidden=true;
}
$('layer').onchange=calculate;$('head').onchange=calculate;$('seed').onchange=calculate;
window.addEventListener('resize',()=>requestAnimationFrame(render));window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',render);calculate();
})();
