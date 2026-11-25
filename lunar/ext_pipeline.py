"""Six-oxide fixed pipelines; crossfitting is feature construction, never selection."""
import numpy as np
from .ext_estimators import Regressor,JOINT
from .ext_inputs import Input,subset,need,from_raw

class Pipeline:
 def __init__(self,recipe):self.recipe=recipe
 def fit(self,batch,y,sites,label_support=None):
  y=np.asarray(y,float);sites=np.asarray(sites);self.warnings=[];self.audit=[];self.previous=None
  cfg=self.recipe['stages'][-1];self.cfg=cfg;self.feature=None
  if len(self.recipe['stages'])>1:
   prefix=dict(self.recipe,stages=self.recipe['stages'][:-1]);base=np.empty_like(y);chem={}
   for site in sorted(set(sites)):
    tr=np.flatnonzero(sites!=site);te=np.flatnonzero(sites==site)
    if len(set(sites[tr]))<2:raise ValueError('INSUFFICIENT_CROSSFIT_TRAINING_SITES')
    teacher=Pipeline(prefix).fit(subset(batch,tr),y[tr],sites[tr],None if label_support is None else label_support[tr]);base[te]=teacher.predict(subset(batch,te),raw=True)
    if cfg['input'].startswith(('C_','D_')):chem[site]=teacher.broad(subset(batch,te),cfg['input'].split('_')[1])
    self.audit.append(dict(role='crossfit_teacher',held_site=str(site),training_sites=sorted(set(sites[tr])),training_rows=len(tr)))
   self.previous=Pipeline(prefix).fit(batch,y,sites,label_support)
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
 def design(self,batch,base,fit=False):
  name=self.cfg['input'];x=batch['raw']
  if name.startswith('Kernel_'):
   from .ext_context_kernel import ContextFeatures
   if fit:self.context_feature=ContextFeatures(name[7:],self.cfg['params'].get('width',1.)).fit(batch)
   return self.context_feature.transform(batch,base)
  if name=='A2' or name.startswith(('B_','C_','D_')):
   z=np.c_[x,base]
   if name.startswith(('B_','D_')):
    broad=need(batch,'broad_'+name.split('_')[1]);z=np.c_[z,broad[:,:16],broad[:,:8]-x,broad[:,16]]
   return z
  if fit:self.feature=Input(name).fit(batch)
  return np.c_[self.feature.transform(batch),base]
 def append_chem(self,z,batch,base,mu,sd):
  h=self.cfg['input'].split('_')[1];return np.c_[z,mu,sd,mu-base,need(batch,'broad_'+h)[:,16]]
 def broad(self,batch,h):
  pixels=need(batch,'pixels_'+h);weights=need(batch,'weights_'+h);n,k,_=pixels.shape
  if weights.shape!=(n,k) or np.any(weights<0) or not np.allclose(weights.sum(1),1):raise ValueError('INVALID_PIXEL_AREA_WEIGHTS')
  pp=self.predict(from_raw(pixels.reshape(-1,8)),raw=True).reshape(n,k,6);mu=np.einsum('nk,nko->no',weights,pp);sd=np.sqrt(np.einsum('nk,nko->no',weights,(pp-mu[:,None])**2));return mu,sd
 def predict(self,batch,raw=False):
  if not isinstance(batch,dict):batch=from_raw(batch)
  if self.previous is None:base=np.zeros((len(batch['raw']),6));z=self.feature.transform(batch)
  else:
   base=self.previous.predict(batch,raw=True);z=self.design(batch,base)
   if self.cfg['input'].startswith(('C_','D_')):z=self.append_chem(z,batch,base,*self.previous.broad(batch,self.cfg['input'].split('_')[1]))
  pred=self.models[0].predict(z) if self.joint else np.column_stack([m.predict(z) for m in self.models])
  if self.cfg['mode']=='residual':pred=base+pred
  if not np.isfinite(pred).all():raise ValueError('NONFINITE_EXTENDED_PREDICTION')
  return pred if raw else np.maximum(pred,0)

class OxideView:
 def __init__(self,pipeline,index):self.pipeline=pipeline;self.index=index
 def predict(self,batch):return self.pipeline.predict(batch)[:,self.index]
