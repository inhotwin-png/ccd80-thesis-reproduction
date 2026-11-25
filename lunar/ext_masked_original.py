"""No chemistry labels accepted by this module. CPU deterministic masked loss."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import random,copy,hashlib
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class Model(nn.Module):
 def __init__(self,out=1):
  super().__init__();self.encoder=nn.Linear(8,8);self.head=nn.Linear(8,out)
 def forward(self,x):return self.head(torch.relu(self.encoder(x)))
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def statehash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()
def initial(seed,out):
 seed_all(seed);m=Model(out)
 torch.manual_seed(seed+(1000 if out==1 else 2000));m.head.reset_parameters()
 return m
def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001,betas=(.9,.999),eps=1e-8,foreach=False)
def group_indices(groups):return [np.flatnonzero(groups==g) for g in sorted(set(groups))]
def draw_records(indices,rng):return np.array([rng.choice(ind) for ind in indices])
def pretrain(x,groups,seed,shuffled=False):
 assert x.ndim==2 and x.shape[1]==8 and np.isfinite(x).all()
 m=initial(seed,8);opt=optimizer(m)
 rng=np.random.default_rng(seed+10000);maskrng=np.random.default_rng(seed+20000);permrng=np.random.default_rng(seed+30000)
 inds=group_indices(groups);n=len(inds)
 assert n>1
 trace=[];counts=np.zeros(n,dtype=int);samplehash=hashlib.sha256();maskhash=hashlib.sha256();permhash=hashlib.sha256()
 # Fixed source reconstruction probe, not a validation set, no model selection.
 reps=np.stack([x[ind].mean(0) for ind in inds]).astype(np.float32)
 def probe():
  with torch.no_grad():
   losses=[]
   for k in range(8):
    z=torch.from_numpy(reps.copy());mask=torch.zeros_like(z,dtype=torch.bool);mask[:,k]=True;mask[:,(k+1)%8]=True;z[mask]=0
    losses.append(float(((m(z)-torch.from_numpy(reps))[mask]**2).mean()))
   return float(np.mean(losses))
 startprobe=probe()
 for step in range(1,201):
  chosen=rng.integers(n,size=64);counts+=np.bincount(chosen,minlength=n)
  if shuffled:
   # All-parent representatives ensure each band marginal is preserved exactly
   # before the common equal-parent minibatch is drawn.
   rowids=draw_records(inds,rng);base=x[rowids];batchall=np.empty_like(base)
   for band in range(8):
    perm=permrng.permutation(n)
    while (perm==np.arange(n)).any():perm=permrng.permutation(n)
    assert sorted(perm.tolist())==list(range(n))
    batchall[:,band]=base[perm,band];permhash.update(perm.tobytes())
    assert np.array_equal(np.sort(batchall[:,band]),np.sort(base[:,band]))
   batch=batchall[chosen];samplehash.update(rowids.tobytes())
  else:
   rowids=np.array([rng.choice(inds[k]) for k in chosen]);batch=x[rowids];samplehash.update(rowids.tobytes())
  mask=np.argsort(maskrng.random((64,8)),axis=1)[:,:2]
  maskhash.update(mask.tobytes());samplehash.update(chosen.tobytes())
  truth=torch.tensor(batch,dtype=torch.float32);z=truth.clone();rr=torch.arange(64)[:,None];mm=torch.tensor(mask);z[rr,mm]=0
  opt.zero_grad();err=((m(z)[rr,mm]-truth[rr,mm])**2).mean()
  assert torch.isfinite(err)
  err.backward();opt.step();trace.append(dict(stage='ssl',step=step,loss=float(err.detach())))
 return copy.deepcopy(m.encoder.state_dict()),trace,dict(source_groups=n,group_draw_counts=counts.tolist(),group_order=sorted(set(groups)),sample_order_sha256=samplehash.hexdigest(),mask_sha256=maskhash.hexdigest(),permutation_sha256=permhash.hexdigest() if shuffled else None,probe_start=startprobe,probe_final=probe(),encoder_sha256=statehash(m.encoder.state_dict()),ssl_state=copy.deepcopy(m.state_dict()))
