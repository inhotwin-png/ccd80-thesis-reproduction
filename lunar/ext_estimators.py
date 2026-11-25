"""Fixed historical regression families, including genuinely distinct neural/rank losses."""
import inspect,warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet,BayesianRidge,HuberRegressor,GammaRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from .ext_neural import Network
from .ext_rank import HistoricalRanker,RANK_FAMILIES

JOINT={'MultiMLP','BoundedMLP','MultiRandomForest','MultiExtraTrees'}
NEURAL={'FirstMLP','SpectralCNN','SecondMLP','ResMLP','StrongMLP','MultiMLP','BoundedMLP','SoftplusMLP'}
EARLY={'EarlyMLP','EarlyResidualMLP','EarlyRegularizedMLP','TinyCNN','YangCNN'}

class WeightedPLS:
 def __init__(self,n_components=2):self.n_components=n_components
 def fit(self,x,y,sample_weight):
  w=sample_weight/sample_weight.sum();self.xm=np.average(x,axis=0,weights=w);self.ym=np.average(y,weights=w);xx=x-self.xm;yy=y-self.ym;ww=[];pp=[];qq=[]
  for _ in range(min(self.n_components,x.shape[1],len(x)-1)):
   v=xx.T@(w*yy);n=np.linalg.norm(v)
   if n<1e-12:break
   v/=n;t=xx@v;den=np.dot(w,t*t)
   if den<1e-12:break
   p=xx.T@(w*t)/den;q=np.dot(w*t,yy)/den;xx=xx-np.outer(t,p);yy=yy-t*q;ww.append(v);pp.append(p);qq.append(q)
  self.coef=np.zeros(x.shape[1]) if not ww else np.array(ww).T@np.linalg.pinv(np.array(pp)@np.array(ww).T)@np.array(qq);return self
 def predict(self,x):return (x-self.xm)@self.coef+self.ym

class Regressor:
 def __init__(self,family,params):self.family=family;self.params=params
 def fit(self,x,y,sites=None,label_support=None):
  a=self.family;p=dict(self.params);self.audit={};self.warnings=[];x=np.asarray(x,float);y=np.asarray(y,float)
  if a in ('Log1pRidge','FixedMeanEnsemble','RidgeLocalBlend','MaskedSSL'):
   from .ext_variants import Variant
   self.model=Variant(a,p).fit(x,y,sites);self.scaler=None;self.ym=0.;self.ys=1.;self.audit=self.model.audit;return self
  if a=='ContextKRR':
   from .ext_context_kernel import ContextKRR
   self.model=ContextKRR(p).fit(x,y);self.scaler=None;self.ym=0.;self.ys=1.;return self
  self.scaler=StandardScaler().fit(x);z=self.scaler.transform(x);self.ym=np.nanmean(y,axis=0);self.ys=np.maximum(np.nanstd(y,axis=0),1e-8)
  target=(y-self.ym)/self.ys
  if a.startswith('JointRank'):
   from .ext_joint_rank import JointRank
   self.model=JointRank(a,p).fit(x,y,sites,label_support);self.scaler=None;self.ym=0.;self.ys=1.;self.audit=self.model.audit;return self
  if a in EARLY:
   from .ext_torch import TorchRegressor
   self.model=TorchRegressor(a,p).fit(z,y);self.ym=0.;self.ys=1.;self.audit={'losses':self.model.losses,'seed_aggregation':'mean predictions','backend':'PyTorch CPU fixed budget'};return self
  if a in RANK_FAMILIES:
   self.model=HistoricalRanker(a,p).fit(x,y,sites,groups=np.arange(len(y)));self.audit=self.model.audit;self.scaler=None;self.ym=0.;self.ys=1.;return self
  if a in NEURAL:
   self.model=Network(a,p).fit(z,y);self.ym=0.;self.ys=1.;self.audit={'losses':self.model.losses,'seed_aggregation':'mean predictions','backend':'NumPy CPU Adam/AdamW port'};return self
  if 'gamma_factor' in p:p['gamma']=p.pop('gamma_factor')/z.shape[1]
  if p.get('gamma')=='1/p':p['gamma']=1/z.shape[1]
  p.pop('random_state',None)
  if a=='Ridge':self.model=Ridge(**p);target=y;self.ym=0.;self.ys=1.
  elif a=='Lasso':self.model=Lasso(**p)
  elif a=='ElasticNet':self.model=ElasticNet(**p)
  elif a=='BayesianRidge':self.model=BayesianRidge(**p)
  elif a=='Huber':self.model=HuberRegressor(**p)
  elif a=='Gamma':
   if np.any(y<=0):raise ValueError('GAMMA_REQUIRES_POSITIVE_LABELS')
   self.model=GammaRegressor(**p);target=y;self.ym=0.;self.ys=1.
  elif a=='PLS':self.model=WeightedPLS(**p)
  elif a=='SVR':self.model=SVR(**p)
  elif a=='KRR':self.model=KernelRidge(kernel='rbf',**p)
  elif a=='GPR':
   length=p.pop('length_scale',3);length=np.sqrt(z.shape[1]) if length=='sqrt(p)' else length;noise=p.pop('noise',.5);p.pop('optimizer',None)
   self.model=GaussianProcessRegressor(kernel=RBF(length,length_scale_bounds='fixed')+WhiteKernel(noise,noise_level_bounds='fixed'),alpha=1e-8,optimizer=None,normalize_y=False)
  elif a in ('RandomForest','MultiRandomForest'):self.model=RandomForestRegressor(**p,random_state=42,n_jobs=1)
  elif a in ('ExtraTrees','MultiExtraTrees'):self.model=ExtraTreesRegressor(**p,random_state=42,n_jobs=1)
  elif a=='GradientBoosting':self.model=GradientBoostingRegressor(**p,random_state=42)
  elif a=='HistGradientBoosting':self.model=HistGradientBoostingRegressor(**p,random_state=42,early_stopping=False)
  elif a=='XGBoost':
   from xgboost import XGBRegressor
   self.model=XGBRegressor(**p,random_state=42,n_jobs=1,verbosity=0)
  elif a=='LightGBM':
   from lightgbm import LGBMRegressor
   self.model=LGBMRegressor(**p,random_state=42,n_jobs=1,verbosity=-1)
  elif a=='CatBoost':
   from catboost import CatBoostRegressor
   self.model=CatBoostRegressor(**p,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)
  elif a in ('KNN','LocalKernel'):
   self.train=z;self.target=target;self.model=None;self.audit['training_rows']=len(y)
   distances=cdist(z,z);valid=distances[distances>1e-10];self.bandwidth=max(np.median(valid) if len(valid) else 1.,1e-10);return self
  elif a in ('TrainingMean','TrainingMedian'):
   self.constant=np.mean(y,0) if a=='TrainingMean' else np.median(y,0);self.model=None;return self
  elif a=='IRLSHuber':
   self.model=Ridge(alpha=p.get('alpha',1.));w=np.ones(len(y));scale=max(np.median(np.abs(y-np.median(y)))*1.4826,1e-3)
   for _ in range(p.get('max_iter',30)):
    self.model.fit(z,target,sample_weight=w);err=(target-self.model.predict(z))*self.ys;w=np.minimum(1,p.get('huber_factor',1.345)*scale/np.maximum(abs(err),1e-12))
   return self
  else:raise ValueError('UNKNOWN_HISTORICAL_FAMILY:'+a)
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');kw={'sample_weight':np.ones(len(y))} if 'sample_weight' in inspect.signature(self.model.fit).parameters else {};self.model.fit(z,target,**kw)
  self.warnings=[str(v.message) for v in ws];return self
 def predict(self,x):
  a=self.family
  if self.scaler is None:return self.model.predict(x)
  z=self.scaler.transform(x)
  if a in ('TrainingMean','TrainingMedian'):return np.broadcast_to(self.constant,(len(x),)+np.shape(self.constant)).copy()
  if a in ('KNN','LocalKernel'):
   dd=cdist(z,self.train)
   if a=='KNN':
    k=min(self.params.get('k',3),len(self.train));lim=np.partition(dd,k-1,axis=1)[:,k-1];w=(dd<=lim[:,None]+1e-12)/np.maximum(dd,1e-12)
   else:
    if 'c' in self.params:lw=-.5*(dd/(self.params['c']*self.bandwidth))**2
    else:lw=-self.params.get('gamma_factor',1.)*dd**2/z.shape[1]
    w=np.exp(lw-lw.max(1,keepdims=True))
   p=w@self.target/w.sum(1) if self.target.ndim==1 else (w@self.target)/w.sum(1,keepdims=True)
  else:p=np.asarray(self.model.predict(z))
  if p.ndim==2 and p.shape[1]==1:p=p[:,0]
  return p*self.ys+self.ym
