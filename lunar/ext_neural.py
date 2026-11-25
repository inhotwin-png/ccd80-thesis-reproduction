"""Fixed NumPy neural learners. CPU Adam/AdamW; no early stopping or selection.

Architecture/loss ports of C52/C65 and C60/C66/C68. NumPy RNG differs from
historical PyTorch RNG: these are new fixed candidates, not saved-weight replay.
"""
import numpy as np
from scipy.special import softmax,expit

class Network:
 def __init__(self,kind,params):self.kind=kind;self.params=params
 def forward(self,x,w,training=False,rng=None):
  z=x;cache=[]
  if 'conv' in w:
   ch=x.shape[1]//8;v=x.reshape(len(x),ch,8).transpose(0,2,1)
   windows=np.lib.stride_tricks.sliding_window_view(np.pad(v,((0,0),(1,1),(0,0))),3,axis=1).reshape(len(x),8,-1)
   a=windows@w['conv']+w['cb'];z=np.maximum(a,0).reshape(len(x),-1);conv=(windows,a)
  else:conv=None
  for j in range(self.depth):
   inp=z;a=z@w[f'w{j}']+w[f'b{j}'];mask=None
   if j<self.depth-1:
    z=np.maximum(a,0)
    if training and self.kind=='StrongMLP':mask=(rng.random(z.shape)>=.2)/.8;z=z*mask
   else:z=a
   cache.append((inp,a,mask))
  if 'skip' in w:z=z+x@w['skip']
  return z,(cache,conv)
 def backward(self,x,w,cache,grad):
  layers,conv=cache;g={}
  if 'skip' in w:g['skip']=x.T@grad
  for j in reversed(range(self.depth)):
   inp,a,mask=layers[j]
   if j<self.depth-1:
    if mask is not None:grad=grad*mask
    grad=grad*(a>0)
   g[f'w{j}']=inp.T@grad;g[f'b{j}']=grad.sum(0);grad=grad@w[f'w{j}'].T
  if conv is not None:
   windows,a=conv;grad=grad.reshape(a.shape)*(a>0);g['conv']=np.einsum('npk,npw->kw',windows,grad);g['cb']=grad.sum((0,1))
  return g
 def fit(self,x,y):
  y=np.asarray(y,float)
  if y.ndim==1:y=y[:,None]
  finite=np.isfinite(y);self.ym=np.nanmean(y,0);self.ys=np.maximum(np.nanstd(y,0),1e-8)
  target=np.where(finite,(y-self.ym)/self.ys,0);mask=finite/np.maximum(finite.sum(0),1)/y.shape[1]
  arch=self.params.get('architecture',[16,8]);arch=arch[:-1] if self.kind=='MultiMLP' and arch[-1]==6 else arch
  if self.kind=='BoundedMLP':arch=[32,16]
  no=7 if self.kind=='BoundedMLP' else y.shape[1];self.depth=len(arch)+1;self.weights=[];self.losses=[]
  for seed in self.params.get('seeds',[42,43,44,45,46]):
   rng=np.random.default_rng(seed);w={};n=x.shape[1]
   if self.kind=='SpectralCNN':
    if n not in (8,16):raise ValueError('CNN_REQUIRES_ORDERED_EIGHT_BAND_CHANNELS')
    ch=n//8;n=32;w['conv']=rng.uniform(-1/np.sqrt(ch*3),1/np.sqrt(ch*3),(ch*3,4));w['cb']=np.zeros(4);arch=[8];self.depth=2
   for j,m in enumerate(arch+[no]):
    lim=1/np.sqrt(n);w[f'w{j}']=rng.uniform(-lim,lim,(n,m));w[f'b{j}']=rng.uniform(-lim,lim,m);n=m
   if self.kind=='ResMLP':w['skip']=np.zeros((x.shape[1],no))
   mom={k:np.zeros_like(v) for k,v in w.items()};vel={k:v.copy() for k,v in mom.items()}
   lr=self.params.get('lr',.003);decay=self.params.get('weight_decay',.1 if self.kind=='StrongMLP' else .01)
   for step in range(1,self.params.get('epochs',300)+1):
    pred,cache=self.forward(x,w,True,rng)
    if self.kind=='BoundedMLP':
     probs=softmax(pred,axis=1);chem=100*probs[:,:6];delta=(chem-np.where(finite,y,0))/self.ys;gchem=2*mask*delta/self.ys
     gp=np.c_[100*gchem,np.zeros(len(x))];grad=probs*(gp-(gp*probs).sum(1,keepdims=True));loss=np.sum(mask*delta**2)
    elif self.kind=='SoftplusMLP':
     raw=pred*self.ys+self.ym;chem=np.logaddexp(0,raw);delta=(chem-np.where(finite,y,0))/self.ys;grad=2*mask*delta*expit(raw);loss=np.sum(mask*delta**2)
    else:delta=pred-target;grad=2*mask*delta;loss=np.sum(mask*delta**2)
    grads=self.backward(x,w,cache,grad)
    for k in w:
     g=grads[k]
     if self.kind in ('FirstMLP','SpectralCNN'):g=g+2*self.params.get('alpha',.01)*w[k]/len(x)
     else:w[k]*=1-lr*decay
     mom[k]=.9*mom[k]+.1*g;vel[k]=.999*vel[k]+.001*g*g
     w[k]-=lr*(mom[k]/(1-.9**step))/(np.sqrt(vel[k]/(1-.999**step))+1e-8)
   if not np.isfinite(loss):raise ValueError('NONFINITE_NEURAL_LOSS')
   self.weights.append(w);self.losses.append(float(loss))
  return self
 def predict(self,x):
  pp=[]
  for w in self.weights:
   p=self.forward(x,w)[0]
   if self.kind=='BoundedMLP':p=100*softmax(p,axis=1)[:,:6]
   else:
    p=p*self.ys+self.ym
    if self.kind=='SoftplusMLP':p=np.logaddexp(0,p)
   pp.append(p)
  return np.mean(pp,axis=0)
