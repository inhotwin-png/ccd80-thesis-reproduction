"""Fixed estimators. No model/parameter selection and no validation-target access."""
from __future__ import annotations
import itertools, warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from .common import digest

FEATURES=('Raw8','Ratio8','MSC8','RatioMSC16','SSA8','Log8','Sqrt8','M3')
FAMILIES=('Ridge','BayesianRidge','ElasticNet','Huber','RandomForest','ExtraTrees','GradientBoosting','HistGradientBoosting','SVR','PLS','KNN','MLP','CNN','RankNet','LightGBMRanker')
DEFAULTS={
 'Ridge':{'alpha':10.0}, 'BayesianRidge':{}, 'ElasticNet':{'alpha':0.05,'l1_ratio':0.5,'max_iter':5000},
 'Huber':{'alpha':1.,'max_iter':500}, 'RandomForest':{'n_estimators':80,'min_samples_leaf':2,'max_depth':5},
 'ExtraTrees':{'n_estimators':80,'min_samples_leaf':2,'max_depth':5},
 'GradientBoosting':{'n_estimators':60,'learning_rate':0.05,'max_depth':2},
 'HistGradientBoosting':{'max_iter':80,'max_leaf_nodes':4,'min_samples_leaf':3,'l2_regularization':10.,'early_stopping':False},
 'SVR':{'C':10.,'epsilon':0.1,'kernel':'rbf'},'PLS':{'n_components':2},'KNN':{'n_neighbors':3,'weights':'distance'},
 'MLP':{'hidden_layer_sizes':[16,8],'alpha':0.1,'max_iter':300,'solver':'lbfgs'},
 'CNN':{'width':4,'max_iter':150,'penalty':0.05},'RankNet':{'width':8,'max_iter':150,'rank_weight':1.,'penalty':0.1},
 'LightGBMRanker':{'n_estimators':40,'num_leaves':4,'min_child_samples':2,'learning_rate':0.05,'reg_lambda':10.}}

def validate_mi(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=8 or not len(x) or not np.isfinite(x).all() or np.any(x<=0) or np.any(x>1):
        raise ValueError('MI input must contain eight finite reflectances in (0,1], in MV1..MN4 order')
    return x

class Features:
    def __init__(self,name):
        if name not in FEATURES: raise ValueError('UNKNOWN_INPUT: '+name)
        self.name=name
    def fit(self,x): self.reference=np.mean(x,axis=0); return self
    def transform(self,x):
        x=np.asarray(x,float); name=self.name
        ratio=x/x[:,[1]]; ratio[:,1]=x[:,1]
        if name=='Raw8': return x.copy()
        if name=='Ratio8': return ratio
        if name in ('MSC8','RatioMSC16'):
            ref=self.reference; rc=ref-ref.mean(); den=rc@rc
            slopes=(x-x.mean(1)[:,None])@rc/max(den,1e-15)
            if np.any(np.abs(slopes)<1e-8):raise ValueError('MSC_SLOPE_NEAR_ZERO')
            msc=(x-(x.mean(1)-slopes*ref.mean())[:,None])/slopes[:,None]
            return msc if name=='MSC8' else np.c_[ratio,msc]
        if name=='SSA8':
            # Exact source C48 restricted Hapke grid, inc=30 deg, emergence=0, K=1.
            w=np.linspace(0,1-1e-12,100001); mu0=np.cos(np.deg2rad(30.)); g=np.sqrt(1-w)
            rr=w/4*mu0/(mu0+1)*((1+2*mu0)/(1+2*mu0*g))*(3/(1+2*g))
            if np.any(x>=rr[-1]):raise ValueError('OUTSIDE_RESTRICTED_HAPKE_DOMAIN')
            return np.interp(x,rr,w)
        if name=='Log8': return np.log(x)
        if name=='Sqrt8': return np.sqrt(x)
        if name=='M3': return np.c_[x,(x[:,1]-x[:,0])/x[:,1],1-x[:,3]/((x[:,2]+x[:,4])/2),x[:,7]/x[:,1]]

class Neural:
    """Small deterministic tanh RankNet or convolutional regression with analytic gradients."""
    def __init__(self,kind,**params): self.kind=kind; self.params=params
    def setup(self,x):
        self.dim=x.shape[1]; self.width=int(self.params.get('width',8)); self.rng=np.random.default_rng(42)
        if self.kind=='CNN':
            self.positions=self.dim-2; self.qdim=self.positions*self.width
            self.sizes=(3*self.width,self.width,self.qdim,1)
        else: self.sizes=(self.dim*self.width,self.width,self.width,1)
    def unpack(self,w):
        a,b,c,_=self.sizes; k=3 if self.kind=='CNN' else self.dim
        return w[:a].reshape(k,self.width),w[a:a+b],w[a+b:a+b+c],w[-1]
    def forward(self,x,w):
        w1,b,w2,b2=self.unpack(w)
        if self.kind=='CNN':
            windows=np.lib.stride_tricks.sliding_window_view(x,3,axis=1)
            h=np.tanh(windows@w1+b); flat=h.reshape(len(x),-1)
            return flat@w2+b2,(windows,h,flat)
        h=np.tanh(x@w1+b); return h@w2+b2,(x,h,h)
    def fit(self,x,y):
        self.setup(x); self.ym=float(np.mean(y)); self.ys=max(float(np.std(y)),.1); target=(y-self.ym)/self.ys
        ii,jj=np.triu_indices(len(y),1); valid=np.abs(y[ii]-y[jj])>1e-10; ii,jj=ii[valid],jj[valid]; sign=np.sign(y[ii]-y[jj])
        lam=float(self.params.get('rank_weight',1)) if self.kind=='RankNet' else 0.
        penalty=float(self.params.get('penalty',.1))
        def fun(w):
            p,(xx,h,flat)=self.forward(x,w); loss=np.mean((p-target)**2); dp=2*(p-target)/len(y)
            if lam and len(ii):
                delta=sign*(p[ii]-p[jj]); loss+=lam*np.mean(np.logaddexp(0,-delta))
                grad=-lam*sign*expit(-delta)/len(ii); np.add.at(dp,ii,grad); np.add.at(dp,jj,-grad)
            w1,b,w2,b2=self.unpack(w)
            gw2=flat.T@dp
            if self.kind=='CNN':
                dh=dp[:,None,None]*w2.reshape(self.positions,self.width)[None,:,:]*(1-h*h)
                gw1=np.einsum('npk,npw->kw',xx,dh); gb=dh.sum((0,1))
            else:
                dh=dp[:,None]*w2[None,:]*(1-h*h); gw1=x.T@dh; gb=dh.sum(0)
            grad=np.r_[gw1.ravel(),gb,gw2,dp.sum()]
            return loss+penalty*np.mean(w*w),grad+2*penalty*w/len(w)
        result=minimize(fun,self.rng.normal(0,.1,sum(self.sizes)),jac=True,method='L-BFGS-B',options={'maxiter':int(self.params.get('max_iter',150)),'ftol':1e-10})
        if not np.isfinite(result.fun): raise ValueError('NONFINITE_OPTIMIZER')
        self.weights=result.x; self.optimizer=dict(success=bool(result.success),iterations=int(result.nit),message=str(result.message)); return self
    def predict(self,x): return self.forward(x,self.weights)[0]*self.ys+self.ym

class LGBRank:
    def __init__(self,params): self.params=params
    def fit(self,x,y):
        from lightgbm import LGBMRanker
        relevance=np.searchsorted(np.unique(y),y)
        self.estimator=LGBMRanker(**self.params,objective='lambdarank',label_gain=list(range(len(np.unique(y)))),random_state=42,n_jobs=1,verbosity=-1)
        self.estimator.fit(x,relevance,group=[len(y)])
        scores=self.estimator.predict(x); centered=scores-scores.mean()
        self.slope=max(float(centered@(y-y.mean())/(centered@centered+1.)),0.)
        self.offset=float(y.mean()-self.slope*scores.mean()); return self
    def predict(self,x): return self.offset+self.slope*self.estimator.predict(x)

def estimator(family,params):
    params=dict(params)
    if family=='Ridge':return Ridge(**params)
    if family=='BayesianRidge':return BayesianRidge(**params)
    if family=='ElasticNet':return ElasticNet(**params,random_state=42)
    if family=='Huber':return HuberRegressor(**params)
    if family=='RandomForest':return RandomForestRegressor(**params,random_state=42,n_jobs=1)
    if family=='ExtraTrees':return ExtraTreesRegressor(**params,random_state=42,n_jobs=1)
    if family=='GradientBoosting':return GradientBoostingRegressor(**params,random_state=42)
    if family=='HistGradientBoosting':return HistGradientBoostingRegressor(**params,random_state=42)
    if family=='SVR':return SVR(**params)
    if family=='PLS':return PLSRegression(**params)
    if family=='KNN':return KNeighborsRegressor(**params)
    if family=='MLP':return MLPRegressor(**params,random_state=42)
    if family in ('RankNet','CNN'):return Neural(family,**params)
    if family=='LightGBMRanker':return LGBRank(params)
    raise ValueError('UNKNOWN_MODEL: '+family)

def validate_template(t):
    if set(t)-{'name','family','params','description'}: raise ValueError('UNKNOWN_MODEL_FIELD')
    if not t.get('name') or t['family'] not in FAMILIES: raise ValueError('INVALID_MODEL')
    estimator(t['family'],t['params'])
    # Probe actual fit before adding the recipe to the queue.
    x=np.arange(80,dtype=float).reshape(10,8)/100+0.01; y=np.linspace(1,9,10)
    est=estimator(t['family'],t['params']).fit(x,y)
    if not np.isfinite(est.predict(x)).all(): raise ValueError('MODEL_PROBE_FAILED')
    return t

def recipes(templates,inputs,max_stages=3):
    if not 1<=max_stages<=3 or not inputs or any(f not in FEATURES for f in inputs): raise ValueError('INVALID_CANDIDATE_SPACE')
    stages=[dict(family=t['family'],params=t['params'],input=f,seed=42,mode='residual') for t,f in itertools.product(templates,inputs)]
    for depth in range(1,max_stages+1):
        for combo in itertools.product(stages,repeat=depth):
            conf=dict(stages=list(combo),postprocess='max(0,prediction)',semantics='sequential_residual_v1')
            yield digest(conf),conf

class Pipeline:
    def __init__(self,recipe): self.recipe=recipe
    def fit(self,x,y):
        x=validate_mi(x); y=np.asarray(y,float)
        if not np.isfinite(y).all() or len(y)!=len(x) or len(y)<3: raise ValueError('INVALID_TRAINING_TARGETS')
        self.layers=[]; base=np.zeros(len(x)); self.warnings=[]
        for cfg in self.recipe['stages']:
            feat=Features(cfg['input']).fit(x); z=feat.transform(x)
            if self.layers: z=np.c_[z,base]
            scaler=StandardScaler().fit(z); z=scaler.transform(z)
            model=estimator(cfg['family'],cfg['params'])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); model.fit(z,y-base)
            self.warnings.extend(str(w.message) for w in caught)
            self.layers.append((feat,scaler,model)); base+=np.asarray(model.predict(z)).ravel()
        return self
    def predict(self,x,raw=False):
        x=validate_mi(x); base=np.zeros(len(x))
        for i,(feat,scaler,model) in enumerate(self.layers):
            z=feat.transform(x)
            if i: z=np.c_[z,base]
            base+=np.asarray(model.predict(scaler.transform(z))).ravel()
        if not np.isfinite(base).all(): raise ValueError('NONFINITE_PREDICTION')
        return base if raw else np.maximum(0,base)
