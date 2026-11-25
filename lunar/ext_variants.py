"""Fixed target, ensemble, local gate and training-only masked reconstruction variants."""
import numpy as np

class Variant:
 def __init__(self,family,params):self.family=family;self.params=params
 def fit(self,x,y,sites):
  from .ext_estimators import Regressor
  a=self.family;self.audit={'variant':a,'selection':'none'}
  if a=='MaskedSSL':
   import torch
   from .ext_masked_original import pretrain,initial,optimizer
   if x.shape[1]!=8:raise ValueError('SSL_REQUIRES_8_BANDS')
   self.xm=x.mean(0);self.xs=np.maximum(x.std(0),1e-8);z=((x-self.xm)/self.xs).astype('float32');self.ym=y.mean();self.ys=max(y.std(),1e-8);self.states=[]
   xx=torch.tensor(z);yy=torch.tensor((y-self.ym)/self.ys,dtype=torch.float32);self.audit['seeds']=[]
   for seed in self.params['seeds']:
    enc,_,info=pretrain(z,np.arange(len(z)).astype(str),seed);m=initial(seed,1);m.encoder.load_state_dict(enc);opt=optimizer(m)
    for _ in range(self.params['epochs']):
     opt.zero_grad();loss=((m(xx)[:,0]-yy)**2).mean();loss.backward();opt.step()
    self.states.append({k:v.detach().numpy().copy() for k,v in m.state_dict().items()});self.audit['seeds'].append(dict(seed=seed,loss=float(loss.detach()),training_rows=len(z),ssl_updates=200))
   return self
  if a=='Log1pRidge':
   if np.any(y<0):raise ValueError('LOG1P_REQUIRES_NONNEGATIVE_CHEMISTRY')
   self.models=[Regressor('Ridge',self.params).fit(x,np.log1p(y),sites)];return self
  configs=[('Ridge',{'alpha':10.})]
  if a=='FixedMeanEnsemble':configs += [('SVR',{'C':1.,'epsilon':.1,'gamma':'1/p'}),('GPR',{'length_scale':'sqrt(p)','noise':.5})]
  else:configs += [('LocalKernel',{'gamma_factor':1.})]
  self.models=[Regressor(f,p).fit(x,y,sites) for f,p in configs];self.low=y.min();self.high=y.max();self.audit['members']=configs;return self
 def predict(self,x):
  if self.family=='MaskedSSL':
   z=(x-self.xm)/self.xs
   return np.mean([(np.maximum(z@w['encoder.weight'].T+w['encoder.bias'],0)@w['head.weight'].T+w['head.bias'])[:,0] for w in self.states],axis=0)*self.ys+self.ym
  p=np.array([m.predict(x) for m in self.models])
  if self.family=='Log1pRidge':return np.expm1(p[0])
  if self.family=='FixedMeanEnsemble':return p.mean(0)
  gate=self.params['gate'];weight=.5 if gate=='half' else p[0]<0 if gate=='negative' else (p[0]<self.low)|(p[0]>self.high)
  return (1-weight)*p[0]+weight*p[1]
