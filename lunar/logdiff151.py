"""Versioned LogDiff extension. Released v104/v105/v114 numerical code is unchanged."""
import numpy as np
from .models import validate_mi
from .ext_inputs import need,subset
from .ext_estimators import Regressor,JOINT
from .ext_pipeline import Pipeline as HistoricalPipeline
from .combination_pipeline114 import Input as ExistingInput,BasicRegressor
from .performance_pipeline import Cache,training_key

SCHEMA='unified_fixed_six_v151'
NAMES=('LogDiff8','F0|LogDiff8|M2')

class Input(ExistingInput):
    def fit(self,batch):
        if self.name in NAMES:
            validate_mi(batch['raw']);return self
        return super().fit(batch)
    def transform(self,batch):
        if self.name not in NAMES:return super().transform(batch)
        x=validate_mi(batch['raw']);z=np.log(x);features=np.c_[z[:,0],np.diff(z,axis=1)]
        return np.c_[features,need(batch,'M2')] if self.name==NAMES[1] else features

def fit_cached(recipe,b,y,sites,support,cache=None):
    if cache is None:cache=Cache()
    key=training_key(recipe,b,y,sites,support)
    if key in cache:cache.hits+=1;cache.move_to_end(key);return cache[key]
    cache.misses+=1;model=Pipeline(recipe).fit(b,y,sites,support,cache);cache[key]=model
    while len(cache)>cache.limit:cache.popitem(last=False)
    return model

class Pipeline(HistoricalPipeline):
    def design(self,batch,base,fit=False):
        if self.cfg['input'] in (*NAMES,'M3'):
            if fit:self.feature=Input(self.cfg['input']).fit(batch)
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
        family=cfg['family'];params=cfg['params'];basic=cfg['backend']=='basic';factory=BasicRegressor if basic else Regressor
        self.models=[];self.joint=not basic and family in JOINT
        if self.joint:
            ix=np.flatnonzero(np.isfinite(target).all(1) if family.startswith('Multi') and family!='MultiMLP' else np.isfinite(target).any(1))
            if len(ix)<3:raise ValueError('INSUFFICIENT_JOINT_TRAINING_ROWS')
            self.models=[factory(family,params).fit(z[ix],target[ix],sites[ix])]
        else:
            for oi in range(6):
                ix=np.flatnonzero(np.isfinite(target[:,oi]))
                if len(ix)<3:raise ValueError('INSUFFICIENT_OXIDE_TRAINING_ROWS')
                self.models.append(factory(family,params).fit(z[ix],target[ix,oi],sites[ix],None if label_support is None else label_support[ix,oi]))
        for model in self.models:self.warnings.extend(model.warnings);self.audit.append(model.audit)
        self.training_sites=sorted(set(sites));return self

def validated_recipe():
    return dict(stages=[dict(family='BayesianRidge',params={},input=NAMES[1],mode='direct',seed=42,backend='historical')],postprocess='max(0,prediction)',semantics=SCHEMA,weight_policy='equal_62m_area')
