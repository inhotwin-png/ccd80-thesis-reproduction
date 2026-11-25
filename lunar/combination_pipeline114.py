"""New mixed-origin candidates; original v1/v104/v105 fitting remains frozen."""
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from .models import estimator,Features
from .ext_estimators import Regressor,JOINT
from .ext_inputs import Input as HistoricalInput,subset
from .ext_pipeline import Pipeline as HistoricalPipeline
from .performance_pipeline import Cache,training_key

class Input(HistoricalInput):
    def fit(self,batch):
        if self.name=='M3':self.basic=Features('M3').fit(batch['raw']);return self
        return super().fit(batch)
    def transform(self,batch):
        return self.basic.transform(batch['raw']) if self.name=='M3' else super().transform(batch)

class BasicRegressor:
    def __init__(self,family,params):self.family=family;self.params=params
    def fit(self,x,y,sites=None,label_support=None):
        self.scaler=StandardScaler().fit(x);self.model=estimator(self.family,self.params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');self.model.fit(self.scaler.transform(x),y)
        self.warnings=[str(w.message) for w in caught];self.audit=dict(backend='basic',target_transform='identity',training_rows=len(y));return self
    def predict(self,x):return np.asarray(self.model.predict(self.scaler.transform(x))).reshape(-1)

def fit_cached(recipe,b,y,sites,support,cache=None):
    if cache is None:cache=Cache()
    key=training_key(recipe,b,y,sites,support)
    if key in cache:cache.hits+=1;cache.move_to_end(key);return cache[key]
    cache.misses+=1;model=Pipeline(recipe).fit(b,y,sites,support,cache);cache[key]=model
    while len(cache)>cache.limit:cache.popitem(last=False)
    return model

class Pipeline(HistoricalPipeline):
    def design(self,batch,base,fit=False):
        if self.cfg['input']=='M3':
            if fit:self.feature=Input('M3').fit(batch)
            return np.c_[self.feature.transform(batch),base]
        return super().design(batch,base,fit)
    def fit(self,batch,y,sites,label_support=None,cache=None):
        y=np.asarray(y,float);sites=np.asarray(sites);self.warnings=[];self.audit=[];self.previous=None
        cfg=self.recipe['stages'][-1];self.cfg=cfg;self.feature=None
        if len(self.recipe['stages'])>1:
            prefix=dict(self.recipe,stages=self.recipe['stages'][:-1]);base=np.empty_like(y);chem={}
            for site in sorted(set(sites)):
                tr=np.flatnonzero(sites!=site);te=np.flatnonzero(sites==site)
                if len(set(sites[tr]))<2:raise ValueError('INSUFFICIENT_CROSSFIT_TRAINING_SITES')
                teacher=fit_cached(prefix,subset(batch,tr),y[tr],sites[tr],None if label_support is None else label_support[tr],cache)
                base[te]=teacher.predict(subset(batch,te),raw=True)
                if cfg['input'].startswith(('C_','D_')):chem[site]=teacher.broad(subset(batch,te),cfg['input'].split('_')[1])
                self.audit.append(dict(role='crossfit_teacher',held_site=str(site),training_sites=sorted(set(sites[tr])),training_rows=len(tr)))
            self.previous=fit_cached(prefix,batch,y,sites,label_support,cache);z=self.design(batch,base,fit=True)
            if chem:
                means=np.empty_like(y);sds=np.empty_like(y)
                for site,(m,s) in chem.items():means[sites==site]=m;sds[sites==site]=s
                z=self.append_chem(z,batch,base,means,sds)
        else:base=np.zeros_like(y);self.feature=Input(cfg['input']).fit(batch);z=self.feature.transform(batch)
        target=y-base if cfg['mode']=='residual' else y
        a=cfg['family'];p=cfg['params'];basic=cfg['backend']=='basic';factory=BasicRegressor if basic else Regressor
        self.models=[];self.joint=not basic and a in JOINT
        if self.joint:
            ix=np.flatnonzero(np.isfinite(target).all(1) if a.startswith('Multi') and a!='MultiMLP' else np.isfinite(target).any(1))
            if len(ix)<3:raise ValueError('INSUFFICIENT_JOINT_TRAINING_ROWS')
            self.models=[factory(a,p).fit(z[ix],target[ix],sites[ix])]
        else:
            for oi in range(6):
                ix=np.flatnonzero(np.isfinite(target[:,oi]))
                if len(ix)<3:raise ValueError('INSUFFICIENT_OXIDE_TRAINING_ROWS')
                self.models.append(factory(a,p).fit(z[ix],target[ix,oi],sites[ix],None if label_support is None else label_support[ix,oi]))
        for model in self.models:self.warnings.extend(model.warnings);self.audit.append(model.audit)
        self.training_sites=sorted(set(sites));return self
