"""Training-only PCA and measured-pixel kernel mean representations from C69/C70."""
import numpy as np
from scipy.spatial.distance import pdist
from .ext_inputs import need
NAMES=('RingsNear','RingsNearMiddle','RingsAll','SLIC','KME025','KME1','KME5','KMERings','Mean025','Mean1','Mean5')

class Spatial:
 def __init__(self,name):self.name=name
 def descriptors(self,b):
  if self.name=='SLIC':return [np.c_[need(b,'slic_mean')-b['raw'],np.sqrt(np.maximum(need(b,'slic_var'),0))]]
  return [np.c_[need(b,'rings_mean')[:,j]-b['raw'],np.sqrt(np.maximum(need(b,'rings_var')[:,j],0))] for j in range({'RingsNear':1,'RingsNearMiddle':2,'RingsAll':3}[self.name])]
 def fit(self,b):
  self.pcas=[]
  if self.name.startswith('Rings') or self.name=='SLIC':
   for x in self.descriptors(b):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1;z=(x-mu)/sd;_,_,vt=np.linalg.svd(z,full_matrices=False);v=vt[:2].T
    for j in range(v.shape[1]):
     if v[np.argmax(abs(v[:,j])),j]<0:v[:,j]*=-1
    self.pcas.append((mu,sd,v))
  else:
   x=need(b,'bag_full5').reshape(-1,8);w=need(b,'weight_full5').ravel()/len(b['raw']);self.mu=np.average(x,axis=0,weights=w);self.sd=np.sqrt(np.average((x-self.mu)**2,axis=0,weights=w));self.sd[self.sd<1e-12]=1
   z=(x-self.mu)/self.sd;positive=w>0;z=z[positive];w=w[positive];n=min(1024,len(z));ix=np.searchsorted(np.cumsum(w)/w.sum(),(np.arange(n)+.5)/n);d=pdist(z[ix]);d=d[d>1e-12];self.bandwidth=float(np.median(d)) if len(d) else 1
   rng=np.random.default_rng(6901);self.omega=rng.normal(size=(8,256));self.phase=rng.uniform(0,2*np.pi,256)
  return self
 def transform(self,b):
  if self.pcas:
   z=[((x-m)/s)@v for x,(m,s,v) in zip(self.descriptors(b),self.pcas)]
   if self.name=='SLIC':return np.c_[b['raw'],z[0],need(b,'slic_fraction'),need(b,'slic_coverage'),need(b,'slic_contrast')]
   return np.c_[b['raw'],np.concatenate([np.c_[q,need(b,'rings_coverage')[:,j]] for j,q in enumerate(z)],axis=1)]
  supports=['ring0','ring1','ring2'] if self.name=='KMERings' else ['full'+self.name.removeprefix('KME').removeprefix('Mean')];vectors=[]
  for support in supports:
   x=(need(b,'bag_'+support)-self.mu)/self.sd;w=need(b,'weight_'+support)
   if self.name.startswith('Mean'):v=np.einsum('nk,nkd->nd',w,x)
   else:v=np.einsum('nk,nkf->nf',w,np.sqrt(2/256)*np.cos(x@self.omega/self.bandwidth+self.phase))
   vectors.append(v)
  return np.c_[b['raw'],np.concatenate(vectors,axis=1)]
