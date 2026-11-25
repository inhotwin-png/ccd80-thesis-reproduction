"""Historical learners, explicit unit calibration and training-only pair construction.

Original mathematical kernels are in historical_pair.py and historical_rank.py.
This adapter is a new AutoLab integration, NOT an exact replay of full old pipelines.
"""
import numpy as np, warnings, importlib.util
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.base import BaseEstimator,RegressorMixin
from . import ext_historical_pair as hp
from . import ext_historical_rank as hr

REG_PARAMS={
 'XGBoost':dict(n_estimators=40,max_depth=2,learning_rate=.05,reg_lambda=10,reg_alpha=1,min_child_weight=2),
 'LightGBM':dict(n_estimators=40,max_depth=2,num_leaves=3,min_child_samples=3,learning_rate=.05,reg_lambda=10,verbosity=-1),
 'CatBoost':dict(iterations=40,depth=2,learning_rate=.05,l2_leaf_reg=10,loss_function='RMSE'),
 'GradientBoosting':dict(n_estimators=40,max_depth=2,min_samples_leaf=2,learning_rate=.05),
 'HistGradientBoosting':dict(max_iter=40,max_leaf_nodes=3,min_samples_leaf=3,l2_regularization=10,learning_rate=.05,early_stopping=False),
 'BayesianRidge':{},'ElasticNet':dict(alpha=.05,l1_ratio=.5,max_iter=5000),
 'Lasso':dict(alpha=.05,max_iter=5000),'Huber':dict(epsilon=1.35,alpha=.1,max_iter=500),
 'KNN':dict(k=3),'LocalKernel':dict(gamma_factor=1.),'Gamma':dict(alpha=1.,max_iter=500)}
RANK_FAMILIES=['RankNet','LinearRankNet','RankMLP','RankElasticNet','LinearRankSVM','RBFRankSVM','RBFPairLogistic','PairGBR','LightGBMRanker']
NEW_FAMILIES=list(REG_PARAMS)+RANK_FAMILIES
DEPS={'XGBoost':'xgboost','LightGBM':'lightgbm','LightGBMRanker':'lightgbm','CatBoost':'catboost'}

def eligible_config(c,stage):
 a=c['algorithm'];p=c.get('params',{})
 if a in DEPS and importlib.util.find_spec(DEPS[a]) is None:return False,'DEPENDENCY_MISSING:'+DEPS[a]
 if a=='Gamma' and stage>0 and c.get('mode')!='direct':return False,'GAMMA_REQUIRES_POSITIVE_DIRECT_TARGET_NOT_RESIDUAL'
 if a in RANK_FAMILIES:
  if stage>0 and c.get('mode')!='direct':return False,'RANK_HEAD_REQUIRES_DIRECT_ABUNDANCE_NOT_RESIDUAL_RANK'
  policy=p.get('pair_policy','B');filt=p.get('pair_filter','ALL')
  if policy not in ['B','C','D1','D2']:return False,'PAIR_POLICY_UNSUPPORTED'
  if filt not in ['ALL','GAP']:return False,'EVID_REQUIRES_UNAVAILABLE_MEASUREMENT_UNCERTAINTIES'
  if a in ['RankNet','LightGBMRanker'] and (policy!='B' or filt!='ALL'):return False,'ORIGINAL_LEARNER_SUPPORTS_SITE_MEAN_ALL_ONLY'
  if a=='LightGBMRanker' and p.get('objective','lambdarank') not in ['lambdarank','rank_xendcg']:return False,'NOT_A_RANKING_OBJECTIVE'
 return True,None

def data_eligible(c,y,sites):
 if c['algorithm']=='Gamma' and (np.asarray(y)<=0).any():return False,'GAMMA_NONPOSITIVE_TRAINING_LABEL_NO_SHIFT'
 if c['algorithm'] in RANK_FAMILIES:
  try:pair_design(np.asarray(y),np.asarray(sites),c.get('params',{}))
  except ValueError as e:return False,str(e)
 return True,None

def pair_design(y,sites,p):
 policy=p.get('pair_policy','B');n=len(y);rows=[];keys=[]
 if policy=='B':
  units=sorted(set(sites));members=[np.flatnonzero(sites==s) for s in units]
  for a in range(len(units)):
   for b in range(a+1,len(units)):
    d=np.zeros(n);d[members[a]]=1/len(members[a]);d[members[b]]=-1/len(members[b]);rows.append(d);keys.append((units[a],units[b]))
 else:
  for a in range(n):
   for b in range(a+1,n):
    if policy in ['D1','D2'] and sites[a]==sites[b]:continue
    d=np.zeros(n);d[a]=1;d[b]=-1;rows.append(d);keys.append(tuple(sorted((str(sites[a]),str(sites[b])))))
 if not rows:raise ValueError('NO_TRAINING_RANK_PAIRS')
 D=np.array(rows);gap=D@y;threshold=p.get('gap_min',.1) if p.get('pair_filter','ALL')=='GAP' else 1e-12
 mask=np.abs(gap)>threshold;D=D[mask]*np.sign(gap[mask])[:,None];keys=[k for k,m in zip(keys,mask) if m]
 if not len(D):raise ValueError('NO_NON_TIED_ELIGIBLE_TRAINING_PAIRS')
 w=np.ones(len(D))
 if policy=='D2':w=np.array([1/keys.count(k) for k in keys])
 w/=w.sum()
 return D,w,keys

class HistoricalRegressor(BaseEstimator,RegressorMixin):
 def __init__(self,algorithm,params=None):self.algorithm=algorithm;self.params=params
 def fit(self,x,y,sites=None,groups=None):
  x=np.asarray(x);y=np.asarray(y);p=dict(REG_PARAMS[self.algorithm]);p.update(self.params or {});a=self.algorithm
  groups=np.asarray(groups if groups is not None else sites if sites is not None else np.arange(len(y)))
  prior=np.array([1/np.sum(groups==g) for g in groups]);prior*=len(prior)/prior.sum();self.prior=prior
  self.sc=StandardScaler().fit(x,sample_weight=prior);z=self.sc.transform(x)
  self.ym=float(np.average(y,weights=prior));self.ys=max(float(np.sqrt(np.average((y-self.ym)**2,weights=prior))),1e-8)
  yy=(y-self.ym)/self.ys
  from sklearn.linear_model import BayesianRidge,ElasticNet,Lasso,HuberRegressor,GammaRegressor
  from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
  if a=='XGBoost':
   from xgboost import XGBRegressor
   self.est=XGBRegressor(**p,random_state=42,n_jobs=1,verbosity=0)
  elif a=='LightGBM':
   from lightgbm import LGBMRegressor
   self.est=LGBMRegressor(**p,random_state=42,n_jobs=1)
  elif a=='CatBoost':
   from catboost import CatBoostRegressor
   self.est=CatBoostRegressor(**p,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)
  elif a=='GradientBoosting':self.est=GradientBoostingRegressor(**p,random_state=42)
  elif a=='HistGradientBoosting':self.est=HistGradientBoostingRegressor(**p,random_state=42)
  elif a=='BayesianRidge':self.est=BayesianRidge(**p)
  elif a=='ElasticNet':self.est=ElasticNet(**p)
  elif a=='Lasso':self.est=Lasso(**p)
  elif a=='Huber':self.est=HuberRegressor(**p)
  elif a=='Gamma':
   if np.any(y<=0):raise ValueError('GAMMA_NONPOSITIVE_TRAINING_LABEL_NO_SHIFT')
   self.est=GammaRegressor(**p);yy=y;self.ym=0.;self.ys=1.
  elif a in ['KNN','LocalKernel']:self.trainz=z;self.trainy=yy;self.p=p;self.est=None
  else:raise ValueError(a)
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always')
   if self.est is not None:self.est.fit(z,yy,sample_weight=prior)
  self.audit=dict(objective='gamma_deviance_log_link' if a=='Gamma' else 'abundance_regression',chemical_units=True,pair_loss_used=False,weight_policy='equal_related_group_training_only',warnings=[str(w.message) for w in ws],n_iter=str(getattr(self.est,'n_iter_',None)),implementation='C66_ESTIMATOR_AND_LOCAL_KERNEL_PORT_WITH_GROUP_WEIGHT_ADAPTER',params=p)
  return self
 def predict(self,x):
  z=self.sc.transform(x)
  if self.est is not None:yp=np.asarray(self.est.predict(z)).ravel()
  else:
   dd=cdist(z,self.trainz,'sqeuclidean')
   if self.algorithm=='KNN':
    k=min(int(self.p['k']),len(self.trainy));lim=np.partition(dd,k-1,axis=1)[:,k-1];w=self.prior[None,:]/np.maximum(np.sqrt(dd),1e-10)*(dd<=lim[:,None]+1e-12)
   else:
    lw=-self.p['gamma_factor']*dd/z.shape[1]+np.log(self.prior)[None,:];w=np.exp(lw-lw.max(1,keepdims=True))
   w/=w.sum(1,keepdims=True);yp=w@self.trainy
  return yp*self.ys+self.ym

class HistoricalRanker(BaseEstimator,RegressorMixin):
 def __init__(self,algorithm,params=None):self.algorithm=algorithm;self.params=params
 def fit(self,x,y,sites=None,groups=None):
  x=np.asarray(x,float);y=np.asarray(y,float);sites=np.asarray(sites);p=dict(self.params or {});a=self.algorithm
  policy=p.get('pair_policy','B')
  # Default B is one real mean per training site, not 28 independent sites.
  self.training_sites=sorted(set(sites));self.original_training_rows=len(y)
  if policy=='B':x=np.array([x[sites==s].mean(0) for s in self.training_sites]);y=np.array([y[sites==s].mean() for s in self.training_sites]);sites=np.array(self.training_sites)
  D,w,keys=pair_design(y,sites,p);self.base=float(y.mean());b=np.full(len(y),self.base)
  self.joint=a=='RankNet' and not p.get('rank_only',False)
  if a in ['RankNet','LightGBMRanker']:
   if a=='RankNet':cfg=dict(family='ranknet',width=4,lambda_rank=1.,rank_only=False,weight_decay=.01,correction_penalty=.1,maxiter=80,abundance_loss='smooth_L1',smooth_l1_epsilon=.1)
   else:cfg=dict(family='lightgbm',objective='lambdarank',learning_rate=.05,calibration_rho=.1,max_depth=2,num_leaves=3,n_estimators=30,reg_lambda=10.,min_child_samples=1)
   cfg.update({k:v for k,v in p.items() if k not in ['pair_policy','pair_filter','gap_min']})
   res=hr.fit_predict(x,y,b,x,b,cfg,seeds=p.get('seeds',[42]));self.state=res['state'];self.kind='recent';logs=res['logs']
   raw=res['native_score'];self.native_chemical=self.joint
   if a=='LightGBMRanker':self.cal=self.state['calibration']
  else:
   fam='RankNet' if a=='LinearRankNet' else a
   delta,self.state=hp.fit_predict(x,x,b,y,(D,D@b,w),fam,p.get('strength',1.));self.kind='legacy';logs=self.state['logs'];raw=delta+self.base;self.native_chemical=False
  if a!='LightGBMRanker' and not self.joint:
   _,_,self.cal,_,_=hr._positive_affine_calibration(raw,y,raw,p.get('calibration_rho',.1))
  objective=('smooth_L1_plus_pair_logistic' if p.get('lambda_rank',1.)>0 else 'smooth_L1_only') if self.joint else p.get('objective','lambdarank') if a=='LightGBMRanker' else 'pair_squared_hinge' if 'SVM' in a else 'pair_logistic'
  self.audit=dict(algorithm=a,pair_loss_used=p.get('lambda_rank',1.)>0,objective=objective,pair_policy=policy,pair_filter=p.get('pair_filter','ALL'),pair_count=len(D),pair_site_keys=[list(k) for k in keys],pair_weight_sum=float(w.sum()),training_sites=self.training_sites,training_rows=len(y),original_training_rows=self.original_training_rows,output_units='wt_percent',native_score_units='wt_percent' if self.native_chemical else 'ranking_score_not_abundance',calibration='NONE_JOINT_ABUNDANCE_LOSS' if self.joint else 'TRAINING_IN_SAMPLE_POSITIVE_AFFINE_NO_TEST_LABELS',logs=logs,implementation='ORIGINAL_LEARNER_KERNEL_NEW_AUTOLAB_TRAINING_CONTEXT',params=p)
  return self
 def predict_score(self,x):
  if self.kind=='recent':return hr.predict_from_state(self.state,x,np.full(len(x),self.base))['native_score']
  s=self.state;z=(np.asarray(x)-s['mu'])/s['sd'];a=self.algorithm
  if a=='RankMLP':raw=np.mean([hp.mlp_pack(z,v)[0] for v in s['weights']],axis=0)
  elif a=='PairGBR':raw=sum(t.predict(z) for t in s['trees'])*s['rate']
  elif a in ['RBFRankSVM','RBFPairLogistic']:raw=np.exp(-cdist(z,s['train_z'],'sqeuclidean')/(2*z.shape[1]))@s['T']@s['beta']
  else:raw=z@s['beta']
  return self.base+raw*s['scale']
 def predict(self,x):
  score=self.predict_score(x)
  if self.joint:return score
  c=self.cal;return c['intercept']+c['slope']*(score-c['score_mean'])/c['score_sd']

def fit_new(x,y,xt,cfg,sites,groups):
 a=cfg['algorithm'];ok,why=data_eligible(cfg,y,sites)
 if not ok:raise ValueError(why)
 est=(HistoricalRanker if a in RANK_FAMILIES else HistoricalRegressor)(a,cfg.get('params',{})).fit(x,y,sites,groups)
 pred=est.predict(xt);state=dict(scaler=FunctionTransformer(),estimator=est,audit=est.audit)
 if a in RANK_FAMILIES:state['native_scores']=est.predict_score(xt)
 return pred,state,[str(v) for v in est.audit.get('warnings',[])]+[str(v) for v in est.audit.get('logs',[]) if not v.get('success',True)]

def configs():
 out=[]
 for a,p in REG_PARAMS.items():
  for f in ['Raw8','RatioMSC16','SSA8']:out.append(dict(stages=[dict(algorithm=a,feature=f,params=p,seed=42)],source_id='C66_PARAMETER_PORT' if a!='Gamma' else 'GAMMA_REIMPLEMENTATION_POSITIVE_LABEL_ONLY'))
 for a in RANK_FAMILIES:
  for f in ['Raw8','RatioMSC16']:
   p=dict(pair_policy='B',pair_filter='ALL',strength=1.) if a not in ['RankNet','LightGBMRanker'] else dict(pair_policy='B',pair_filter='ALL')
   out.append(dict(stages=[dict(algorithm=a,feature=f,params=p,seed=42)],source_id='HISTORICAL_PAIR_LEARNER_PORT'))
 # Explicitly distinct pair populations and losses, not aliases.
 for a in RANK_FAMILIES:
  if a in ['RankNet','LightGBMRanker']:continue
  for policy in ['C','D1','D2']:out.append(dict(stages=[dict(algorithm=a,feature='Raw8',params=dict(pair_policy=policy,pair_filter='ALL',strength=1.),seed=42)],source_id='SAMPLE_PAIR_POLICY_EXTENSION'))
  out.append(dict(stages=[dict(algorithm=a,feature='Raw8',params=dict(pair_policy='D2',pair_filter='GAP',gap_min=.1,strength=10.),seed=42)],source_id='EXPLICIT_GAP_NOT_UNCERTAINTY_EVID'))
 for lam in [0.,.1,3.]:out.append(dict(stages=[dict(algorithm='RankNet',feature='Raw8',params=dict(lambda_rank=lam,pair_policy='B'),seed=42)],source_id='HISTORICAL_JOINT_LOSS_VARIANT'))
 out.append(dict(stages=[dict(algorithm='RankNet',feature='Raw8',params=dict(rank_only=True,pair_policy='B'),seed=42)],source_id='PURE_RANK_REQUIRES_CALIBRATION'))
 out.append(dict(stages=[dict(algorithm='LightGBMRanker',feature='Raw8',params=dict(objective='rank_xendcg',pair_policy='B'),seed=42)],source_id='HISTORICAL_LGBM_OBJECTIVE'))
 return out
