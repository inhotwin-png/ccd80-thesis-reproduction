"""C69 half local RBF / half measured-context kernel, fixed KRR settings."""
import numpy as np
from scipy.spatial.distance import cdist,pdist
from .ext_spatial import Spatial
from .ext_inputs import need
class ContextFeatures:
 def __init__(self,name,width=1.):self.name=name;self.width=width
 def fit(self,b):
  if self.name=='B0':return self
  self.spatial=Spatial('KME5').fit(b);self.spatial.bandwidth*=self.width
  self.norm=None;z=self.context(b)
  if self.name.startswith('B2') or self.name=='B3':
   self.norm=np.array([max(float(np.mean(np.sum(v*v,1))),1e-12) for v in z])
  return self
 def context(self,b):
  supports=['ring0','ring1','ring2'] if self.name=='B3' else ['full'+self.name.split('_')[1]];s=self.spatial;out=[]
  for key in supports:
   z=(need(b,'bag_'+key)-s.mu)/s.sd;w=need(b,'weight_'+key)
   if self.name.startswith('B1'):v=np.einsum('nk,nkd->nd',w,z)/s.bandwidth
   else:v=np.einsum('nk,nkf->nf',w,np.sqrt(2/256)*np.cos(z@s.omega/s.bandwidth+s.phase))
   out.append(v)
  return out
 def transform(self,b,base):
  local=np.c_[b['raw'],base]
  if self.name=='B0':return local
  z=self.context(b)
  if self.norm is not None:z=[v/np.sqrt(n*len(z)) for v,n in zip(z,self.norm)]
  return np.c_[local,np.concatenate(z,axis=1)]

class ContextKRR:
 def __init__(self,params):self.params=params
 def fit(self,x,y):
  local=x[:,:14];self.mu=local.mean(0);self.sd=local.std(0);self.sd[self.sd<1e-12]=1
  self.local=(local-self.mu)/self.sd;d=pdist(self.local);d=d[d>1e-12];self.bw=np.median(d) if len(d) else 1.;self.context=x[:,14:]
  k=self.kernel(x);self.kmean=k.mean(0);self.kgrand=k.mean();kc=k-self.kmean[None,:]-self.kmean[:,None]+self.kgrand;self.ym=y.mean()
  self.coef=np.linalg.solve(kc+self.params.get('alpha',1.)*np.eye(len(y)),y-self.ym);return self
 def kernel(self,x):
  z=(x[:,:14]-self.mu)/self.sd;kl=np.exp(-cdist(z,self.local,'sqeuclidean')/(2*self.bw*self.bw));kind=self.params['kind']
  if kind=='B0':return kl
  if kind.startswith('B1'):kc=np.exp(-cdist(x[:,14:],self.context,'sqeuclidean')/2)
  else:kc=x[:,14:]@self.context.T
  return .5*kl+.5*kc
 def predict(self,x):
  k=self.kernel(x);return self.ym+(k-self.kmean[None,:]-k.mean(1,keepdims=True)+self.kgrand)@self.coef
