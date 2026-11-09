/* Synthetic 128-D role forwards + frozen G/R selection mechanics. No model/API. */
(function(root){
function makeRelayEngine(){
  const D=128,Q=64,PROMPTS=[160,144,176],SELF=[144,176,152];
  function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
  const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;};
  function summary(rows){const v=new Float64Array(D);if(!rows.length)return v;for(const r of rows)for(let d=0;d<D;d++)v[d]+=r.v[d]/rows.length;return v;}
  function promptRows(role,n,start,history,seed,self=false){
    const random=rng(seed+role*100003),ctx=summary(history),out=[];
    for(let j=0;j<n;j++){
      const k=new Float64Array(D),v=new Float64Array(D),rare=j%19===7;
      for(let d=0;d<D;d++){
        const inRole=d>=16+(role-1)*32&&d<16+role*32;
        k[d]=(random()-.5)*1.4+.10*ctx[d]+(inRole?.45:0);
        v[d]=(random()-.5)*.48+.22*ctx[d]+(d<16?.22*Math.sin(j*.19+d*.2):0)+(inRole?(rare?2.8:.42)*Math.sin(d*.23+j*.09):0);
      }
      k[0]=(rare?.12:1.4+.65*Math.sin(j*.7));k[1]=role*.65;
      out.push({id:start+j,role:self||(role===1&&j<4)?0:role,source:role,kind:!self&&role===1&&j<4?'sink':self?'self':'prompt',k,v});
    }
    return out;
  }
  function oneRead(rows,q){const scores=rows.map(r=>dot(q,r.k)/Math.sqrt(D)),max=Math.max(...scores),w=scores.map(x=>Math.exp(x-max)),z=w.reduce((a,b)=>a+b,0),v=new Float64Array(D);for(let i=0;i<rows.length;i++)for(let d=0;d<D;d++)v[d]+=w[i]/z*rows[i].v[d];return v;}
  function roleForward(history,role,start,seed){
    const fresh=promptRows(role,PROMPTS[role-1],start,history,seed),rows=history.concat(fresh),random=rng(seed+role*70001);
    let state=summary(rows);
    for(let t=0;t<10;t++){
      const q=Float64Array.from(state,(x,d)=>x+(d<16?.6:0)+Math.sin(d*.13+t*.17)*.2),read=oneRead(rows,q),k=new Float64Array(D),v=new Float64Array(D);
      for(let d=0;d<D;d++){v[d]=.62*read[d]+.25*state[d]+(random()-.5)*.12;k[d]=.7*v[d]+Math.sin(d*.17+t)*.14;}
      rows.push({id:start+fresh.length+t,role,source:role,kind:'latent',k,v});state=v;
    }
    return rows;
  }
  function probe(history,hop,seed){
    const role=hop+2,end=[170,324,510][hop],self=promptRows(role,SELF[hop],end,history,seed,true),rows=history.concat(self),N=rows.length;
    const A=new Float64Array(Q*N),random=rng(seed+role*32003),ctx=summary(history);
    for(let q=0;q<Q;q++){
      const query=new Float64Array(D),group=Math.floor(q/16),position=self.length-16+q%16;
      for(let d=0;d<D;d++)query[d]=(random()-.5)*1.4+.15*ctx[d]+(d>=16+(hop%3)*32&&d<48+(hop%3)*32?.5:0);
      query[0]=4+group*.2;query[1]=.8;let max=-Infinity;
      for(let i=0;i<N;i++){const x=i>=history.length&&i-history.length>position?-Infinity:dot(query,rows[i].k)/Math.sqrt(D);A[q*N+i]=x;if(x>max)max=x;}
      let z=0;for(let i=0;i<N;i++){A[q*N+i]=Math.exp(A[q*N+i]-max);z+=A[q*N+i];}for(let i=0;i<N;i++)A[q*N+i]/=z;
    }
    const values=new Float64Array(N*D);rows.forEach((r,i)=>values.set(r.v,i*D));
    return {d:D,q:Q,n:N,rows,A,values,relayN:history.length,hop,end};
  }
  function components(s,keep){
    const out=new Float64Array(4*s.q*s.d),z=new Float64Array(s.q);
    for(let q=0;q<s.q;q++)for(let i=0;i<s.n;i++)if(keep[i])z[q]+=s.A[q*s.n+i];
    for(let i=0;i<s.n;i++)if(keep[i]){const r=s.rows[i].role;for(let q=0;q<s.q;q++){const w=s.A[q*s.n+i]/z[q],base=(r*s.q+q)*s.d;for(let d=0;d<s.d;d++)out[base+d]+=w*s.values[i*s.d+d];}}
    return {out,z};
  }
  function metrics(s,keep){
    const after=components(s,keep),perRole=[0,0,0,0];let total=0;
    for(let q=0;q<s.q;q++)for(let d=0;d<s.d;d++){let e=0;for(let b=0;b<4;b++){const ix=(b*s.q+q)*s.d+d,delta=after.out[ix]-s.target[ix];perRole[b]+=delta*delta/(s.q*s.d);e+=delta;}total+=e*e/(s.q*s.d);}
    return {out:after.out,z:after.z,roleMSE:perRole,blockMSE:perRole.reduce((a,b)=>a+b,0)/4,totalMSE:total};
  }
  function makeBank(s,budget=96){
    const mass=new Float64Array(s.n),protectedMask=new Uint8Array(s.n),mandatory=new Uint8Array(s.n);let candidates=[];
    const lastStart=[0,170,324][s.hop],lastEnd=[160,314,500][s.hop];
    for(let i=0;i<s.n;i++){
      for(let q=0;q<s.q;q++)mass[i]+=s.A[q*s.n+i];
      const r=s.rows[i];if(r.kind==='prompt'){candidates.push(i);if(r.id>=lastEnd-8&&r.id<lastEnd&&r.id>=lastStart)mandatory[i]=1;}else protectedMask[i]=1;
    }
    budget=Math.min(budget,candidates.length);
    const required=candidates.filter(i=>mandatory[i]),rank=candidates.filter(i=>!mandatory[i]).sort((a,b)=>mass[b]-mass[a]||a-b),bankSize=Math.min(2*budget,candidates.length);
    const bank=required.concat(rank.slice(0,bankSize-required.length)).sort((a,b)=>a-b);
    return {mass,protectedMask,mandatory,candidates,rank,required,bank,budget,bankSize};
  }
  function chooseR(s,bankInfo){
    const {bank,protectedMask,mandatory,budget}=bankInfo,C=bank.length,d=s.d,qc=s.q,N=s.n,keep=Uint8Array.from(protectedMask);for(const i of bank)keep[i]=1;
    const initial=components(s,keep),num=new Float64Array(initial.out.length),z=initial.z.slice();
    for(let b=0;b<4;b++)for(let q=0;q<qc;q++)for(let d0=0;d0<d;d0++){const t=(b*qc+q)*d+d0;num[t]=initial.out[t]*z[q];}
    const gram=new Float64Array(C*C),nv=new Float64Array(C*qc),tv=new Float64Array(C*qc),base2=new Float64Array(qc),nt=new Float64Array(qc),target2=new Float64Array(qc);
    for(let c=0;c<C;c++)for(let j=0;j<=c;j++){let x=0;for(let k=0;k<d;k++)x+=s.values[bank[c]*d+k]*s.values[bank[j]*d+k];gram[c*C+j]=gram[j*C+c]=x;}
    for(let c=0;c<C;c++){const b=s.rows[bank[c]].role;for(let q=0;q<qc;q++){let n=0,t=0;for(let k=0;k<d;k++){const v=s.values[bank[c]*d+k],i=(b*qc+q)*d+k;n+=num[i]*v;t+=s.target[i]*v;}nv[c*qc+q]=n;tv[c*qc+q]=t;}}
    for(let q=0;q<qc;q++)for(let b=0;b<4;b++)for(let k=0;k<d;k++){const i=(b*qc+q)*d+k;base2[q]+=num[i]*num[i];nt[q]+=num[i]*s.target[i];target2[q]+=s.target[i]*s.target[i];}
    const active=new Uint8Array(C).fill(1),trace=[],candidateLosses=[];
    for(let step=0;step<C-budget;step++){
      let best=-1,bestLoss=Infinity;
      const losses=[];
      for(let c=0;c<C;c++){
        if(!active[c]||mandatory[bank[c]])continue;
        let loss=0;for(let q=0;q<qc;q++){const w=s.A[q*N+bank[c]],zp=z[q]-w;if(zp<=0){loss=Infinity;break;}loss+=(base2[q]-2*w*nv[c*qc+q]+w*w*gram[c*C+c])/(zp*zp)-2*(nt[q]-w*tv[c*qc+q])/zp+target2[q];}
        loss/=qc*4*d;if(step===0)losses.push({index:bank[c],loss});
        if(loss<bestLoss){bestLoss=loss;best=c;}
      }
      if(best<0)throw Error('没有可删除候选');
      const index=bank[best],role=s.rows[index].role;
      for(let q=0;q<qc;q++){
        const w=s.A[q*N+index];base2[q]+=-2*w*nv[best*qc+q]+w*w*gram[best*C+best];nt[q]-=w*tv[best*qc+q];z[q]-=w;
        for(let c=0;c<C;c++)if(active[c]&&s.rows[bank[c]].role===role)nv[c*qc+q]-=w*gram[c*C+best];
      }
      active[best]=0;keep[index]=0;trace.push({index,id:s.rows[index].id,loss:Math.max(0,bestLoss)});if(step===0)candidateLosses.push(...losses);
    }
    return {keep,trace,candidateLosses};
  }
  function select(s,mode){
    s.target=components(s,new Uint8Array(s.n).fill(1)).out;s.bankInfo=makeBank(s);
    const b=s.bankInfo;
    if(mode==='R'){const r=chooseR(s,b);s.trace=r.trace;s.firstLosses=r.candidateLosses;s.keep=r.keep;}
    else{
      const keep=Uint8Array.from(b.protectedMask);for(const i of b.required.concat(b.rank.slice(0,b.budget-b.required.length)))keep[i]=1;
      s.keep=keep;s.trace=b.bank.filter(i=>!keep[i]).sort((a,c)=>b.mass[a]-b.mass[c]||c-a).map(index=>({index,id:s.rows[index].id,loss:null}));
    }
    s.mode=mode;s.finalMetrics=metrics(s,s.keep);return s;
  }
  function stateMask(s,phase,step){
    if(phase==='full')return new Uint8Array(s.n).fill(1);
    const mask=Uint8Array.from(s.bankInfo.protectedMask);for(const i of s.bankInfo.bank)mask[i]=1;
    if(phase==='final')return s.keep.slice();
    if(phase==='delete')for(let j=0;j<Math.min(step,s.trace.length);j++)mask[s.trace[j].index]=0;
    return mask;
  }
  function run({seed=42,layer=1,head=1}={}){
    const salt=(Number(seed)+layer*101+head*1009)>>>0,planner=roleForward([],1,0,salt),result={seed:Number(seed),layer,head,d:D,q:Q,paths:{G:[],R:[]}};
    for(const mode of ['G','R']){
      let history=planner;
      for(let hop=0;hop<3;hop++){
        if(hop>0)history=roleForward(history,hop+1,[0,170,324][hop],salt);
        const s=select(probe(history,hop,salt),mode);result.paths[mode].push(s);
        history=s.rows.slice(0,s.relayN).filter((_,i)=>s.keep[i]);
        if(history.length!==110+hop*10)throw Error('交接长度不一致');
        if(history.filter(r=>r.kind==='latent').length!==10*(hop+1))throw Error('latent保护失效');
        if(history.filter(r=>r.kind==='sink').length!==4)throw Error('sink保护失效');
      }
    }
    return result;
  }
  return {run,components,metrics,makeBank,chooseR,stateMask,select,rng};
}
if(typeof module!=='undefined'&&module.exports)module.exports=makeRelayEngine();
else root.makeRelayEngine=makeRelayEngine;
})(globalThis);
