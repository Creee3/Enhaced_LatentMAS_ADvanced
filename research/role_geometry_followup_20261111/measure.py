"""Forward-free numeric measurements, with explicit finite-sample bounds."""
import numpy as np
def sample_indices(n,cap):
 assert n>=2 and cap>=2
 return np.linspace(0,n-1,min(n,cap),dtype=np.int64)
def describe(x):
 """x [...,N,128]; output spectrum metrics and at most four feature directions."""
 x=np.asarray(x,dtype=np.float64);assert x.shape[-2]>=2 and x.shape[-1]==128 and np.isfinite(x).all()
 c=x-x.mean(-2,keepdims=True);_,s,vh=np.linalg.svd(c,full_matrices=False);e=s*s;total=e.sum(-1,keepdims=True)
 p=e/np.maximum(total,1e-300);entropy=np.exp(-(p*np.log(np.maximum(p,1e-300))).sum(-1));pr=total[...,0]**2/np.maximum((e*e).sum(-1),1e-300)
 rank=(s>np.maximum(x.shape[-2],128)*np.finfo(float).eps*s[...,:1]).sum(-1)
 metrics=np.stack([entropy,pr,rank],-1);metrics[total[...,0]<=1e-300]=0
 return metrics,vh[...,:4,:],np.minimum(rank,4)
def overlap(a,ra,b,rb):
 q=np.minimum(ra,rb);z=a@np.swapaxes(b,-1,-2);n=z.shape[-1]
 mask=(np.arange(n)<q[...,None]);energy=((z*z)*mask[...,None,:]*mask[...,:,None]).sum((-2,-1));return np.where(q>0,energy/np.maximum(q,1),np.nan)
def compare(samples,descriptions):
 """Six branches ordered planner0,planner1,critic0,critic1,refiner0,refiner1."""
 assert len(samples)==len(descriptions)==6
 out=[]
 for i in range(6):
  for j in range(i+1,6):
   _,a,ra=descriptions[i];_,b,rb=descriptions[j];ov=overlap(a,ra,b,rb)
   den=np.maximum(np.linalg.norm(samples[i],axis=(-2,-1)),1e-30)
   rel=np.linalg.norm(samples[i]-samples[j],axis=(-2,-1))/den
   out.append(dict(i=i,j=j,same_role=i//2==j//2,subspace_distance=1-ov,relative_l2=rel))
 return out
def role_effect(pairs):
 same=np.stack([p['subspace_distance'] for p in pairs if p['same_role']]);different=np.stack([p['subspace_distance'] for p in pairs if not p['same_role']])
 return dict(within_role=same.mean(0),between_role=different.mean(0),between_minus_within=different.mean(0)-same.mean(0))
def windows(prompt_len,cap=32):
 n=min(cap,prompt_len//3);assert n>=2
 return [np.arange(0,n),np.arange((prompt_len-n)//2,(prompt_len-n)//2+n),np.arange(prompt_len-n,prompt_len)]
