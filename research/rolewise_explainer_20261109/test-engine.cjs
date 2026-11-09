const assert=require('node:assert/strict'),E=require('./engine.js');
let tests=0;
function check(v,msg){assert.ok(v,msg);tests++;}
for(const seed of [42,7,123]){
 const run=E.run({seed,layer:1,head:1});
 for(const m of ['G','R'])for(let h=0;h<3;h++){
  const s=run.paths[m][h],kept=s.rows.slice(0,s.relayN).filter((_,i)=>s.keep[i]);
  check(kept.length===110+10*h,'110/120/130');check(kept.filter(x=>x.kind==='latent').length===10*(h+1),'all latent');check(kept.filter(x=>x.kind==='sink').length===4,'all sink');
  check(kept.filter(x=>x.kind==='prompt').length===96,'global prompt budget');check(s.rows.every((r,i)=>!s.bankInfo.mandatory[i]||s.keep[i]),'mandatory recent8');
  check(kept.every(x=>x.kind!=='self'),'self not relayed');check(s.finalMetrics.z.every(z=>z>0&&z<=1+1e-12),'common normalizer');
  for(let i=0;i<s.relayN;i++)if(s.keep[i])check(s.rows[i]===kept.find(x=>x.id===s.rows[i].id),'original row reference');
  if(h>0){const previous=run.paths[m][h-1],ids=new Set(previous.rows.slice(0,previous.relayN).filter((_,i)=>previous.keep[i]).map(x=>x.id));check(s.rows.slice(0,s.relayN).filter(r=>r.id<previous.end).every(r=>ids.has(r.id)),'no resurrection');}
 }
 check(run.paths.G[0].A.every((v,i)=>v===run.paths.R[0].A[i]),'shared first probe');
 check(run.paths.G[0].target.every((v,i)=>v===run.paths.R[0].target[i]),'shared first target');
 const before=run.paths.G[0].keep;const changed=structuredClone(run.paths.G[0]);for(let i=0;i<changed.values.length;i++)changed.values[i]=Math.cos(i)*100;E.select(changed,'G');check(before.every((v,i)=>v===changed.keep[i]),'G ranking independent of V');
}
// Independent exhaustive recomputation of each trial deletion, including self/sink.
const random=E.rng(99),s={q:5,d:12,n:14,rows:Array.from({length:14},(_,i)=>({role:i===0||i>=11?0:1+i%3})),A:new Float64Array(5*14),values:Float64Array.from({length:14*12},()=>random()*2-1)};
for(let q=0;q<s.q;q++){let z=0;for(let i=0;i<s.n;i++){s.A[q*s.n+i]=.1+random();z+=s.A[q*s.n+i];}for(let i=0;i<s.n;i++)s.A[q*s.n+i]/=z;}
const protectedMask=Uint8Array.from({length:14},(_,i)=>Number(i===0||i>=11)),mandatory=Uint8Array.from({length:14},(_,i)=>Number(i===9||i===10)),bank=Array.from({length:10},(_,i)=>i+1);
s.target=E.components(s,new Uint8Array(14).fill(1)).out;
const actual=E.chooseR(s,{bank,protectedMask,mandatory,budget:6}),keep=new Uint8Array(14).fill(1);
function brute(mask){let error=0;for(let q=0;q<s.q;q++){let denom=0;for(let i=0;i<s.n;i++)if(mask[i])denom+=s.A[q*s.n+i];for(let b=0;b<4;b++)for(let d=0;d<s.d;d++){let v=0;for(let i=0;i<s.n;i++)if(mask[i]&&s.rows[i].role===b)v+=s.A[q*s.n+i]*s.values[i*s.d+d]/denom;error+=(v-s.target[(b*s.q+q)*s.d+d])**2;}}return error/(s.q*4*s.d);}
for(const step of actual.trace){let best=-1,loss=Infinity;for(const i of bank)if(keep[i]&&!mandatory[i]){const trial=keep.slice();trial[i]=0;const l=brute(trial);if(l<loss){best=i;loss=l;}}check(best===step.index,'exact greedy deletion matches independent brute force');check(Math.abs(loss-step.loss)<1e-12,'exact loss including common Z');keep[best]=0;}
check(keep.every((v,i)=>v===actual.keep[i]),'final R mask');
console.log(JSON.stringify({passed:true,checks:tests,seeds:[42,7,123],dimensions:'64 queries × 128 dimensions',greedy_bruteforce_fixture:'5 queries × 12 dimensions / 4 deletion steps',model_calls:0}));
