"""Memoize identical training-only prefixes; frozen numerical fit body preserved."""
import hashlib
from collections import OrderedDict
import numpy as np
from .common import canonical
from .ext_pipeline import Pipeline as OriginalPipeline
from .ext_estimators import Regressor,JOINT
from .ext_inputs import Input,subset

class Cache(OrderedDict):
 def __init__(self,limit=256):super().__init__();self.limit=limit;self.hits=0;self.misses=0

def training_key(recipe,b,y,sites,support):
 h=hashlib.sha256(canonical(recipe).encode())
 for name,value in [('labels',y),('sites',sites),('support',support)]+sorted(b.items()):
  h.update(name.encode())
  if value is None:h.update(b'NONE');continue
  a=np.asarray(value);h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
  h.update(canonical(a.tolist()).encode() if a.dtype.hasobject else np.ascontiguousarray(a).tobytes())
 return h.hexdigest()

def fit_cached(recipe,b,y,sites,support,cache):
 if cache is None:cache=Cache()
 key=training_key(recipe,b,y,sites,support)
 if key in cache:
  cache.hits+=1;cache.move_to_end(key);return cache[key]
 cache.misses+=1
 model=Pipeline(recipe).fit(b,y,sites,support,cache)
 cache[key]=model
 while len(cache)>cache.limit:cache.popitem(last=False)
 return model

class Pipeline(OriginalPipeline):
 def fit(self,batch,y,sites,label_support=None,cache=None):
  y=np.asarray(y,float);sites=np.asarray(sites);self.warnings=[];self.audit=[];self.previous=None
  cfg=self.recipe['stages'][-1];self.cfg=cfg;self.feature=None
  if len(self.recipe['stages'])>1:
   prefix=dict(self.recipe,stages=self.recipe['stages'][:-1]);base=np.empty_like(y);chem={}
   for site in sorted(set(sites)):
    tr=np.flatnonzero(sites!=site);te=np.flatnonzero(sites==site)
    if len(set(sites[tr]))<2:raise ValueError('INSUFFICIENT_CROSSFIT_TRAINING_SITES')
    teacher=fit_cached(prefix,subset(batch,tr),y[tr],sites[tr],None if label_support is None else label_support[tr],cache);base[te]=teacher.predict(subset(batch,te),raw=True)
    if cfg['input'].startswith(('C_','D_')):chem[site]=teacher.broad(subset(batch,te),cfg['input'].split('_')[1])
    self.audit.append(dict(role='crossfit_teacher',held_site=str(site),training_sites=sorted(set(sites[tr])),training_rows=len(tr)))
   self.previous=fit_cached(prefix,batch,y,sites,label_support,cache)
   z=self.design(batch,base,fit=True)
   if chem:
    means=np.empty_like(y);sds=np.empty_like(y)
    for site,(m,s) in chem.items():means[sites==site]=m;sds[sites==site]=s
    z=self.append_chem(z,batch,base,means,sds)
  else:base=np.zeros_like(y);self.feature=Input(cfg['input']).fit(batch);z=self.feature.transform(batch)
  target=y-base if cfg['mode']=='residual' else y
  self.models=[];a=cfg['family'];p=cfg['params']
  if a in JOINT:
   ix=np.flatnonzero(np.isfinite(target).all(1) if a.startswith('Multi') and a!='MultiMLP' else np.isfinite(target).any(1))
   if len(ix)<3:raise ValueError('INSUFFICIENT_JOINT_TRAINING_ROWS')
   self.models=[Regressor(a,p).fit(z[ix],target[ix],sites[ix])];self.joint=True
  else:
   self.joint=False
   for oi in range(6):
    ix=np.flatnonzero(np.isfinite(target[:,oi]))
    if len(ix)<3:raise ValueError('INSUFFICIENT_OXIDE_TRAINING_ROWS')
    self.models.append(Regressor(a,p).fit(z[ix],target[ix,oi],sites[ix],None if label_support is None else label_support[ix,oi]))
  for model in self.models:self.warnings.extend(model.warnings);self.audit.append(model.audit)
  self.training_sites=sorted(set(sites));return self
