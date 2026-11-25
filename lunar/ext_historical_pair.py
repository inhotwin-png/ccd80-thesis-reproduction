"""Seven genuine pair-loss corrections and nine residual regressors.
All chemical units are recovered after training-only scaling; baseline is fixed.
"""
import numpy as np,time,warnings
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
FAMILIES=['RankNet','RankElasticNet','LinearRankSVM','RBFRankSVM','RBFPairLogistic','RankMLP','PairGBR','Ridge','ElasticNet','SVR','KRR','GPR','RF','ExtraTrees','GBR','MLP']
STRENGTHS=[1.,10.];SEEDS=list(range(42,47))
def mlp_pack(z,w):
 d=z.shape[1];a=w[:d*4].reshape(d,4);b=w[d*4:d*4+4];v=w[d*4+4:d*4+8];c=w[-1];h=np.tanh(z@a+b);return h@v+c,h,a,v
def mlp_fit(z,zt,fun,lam):
 pp=[];states=[];logs=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed);w=rng.normal(0,.1,z.shape[1]*4+9)
  def obj(w):
   p,h,a,v=mlp_pack(z,w);loss,dp=fun(p);dh=(dp[:,None]*v)*(1-h*h)
   grad=np.r_[(z.T@dh).ravel(),dh.sum(0),h.T@dp,dp.sum()]+lam*w
   return loss+.5*lam*w@w,grad
  fit=minimize(obj,w,jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-10,'gtol':1e-6,'maxls':30})
  pp.append(mlp_pack(zt,fit.x)[0]);states.append(fit.x);logs.append(dict(seed=seed,success=bool(fit.success),message=str(fit.message),niter=int(fit.nit),loss=float(fit.fun)))
 return np.array(pp).T,states,logs
def fit_predict(x,xt,b,y,pairs,family,strength):
 start=time.perf_counter();mu=x.mean(0);sd=x.std(0);sd[sd<1e-10]=1;z=(x-mu)/sd;zt=(xt-mu)/sd;n,d=z.shape
 scale=max(float(np.std(y)),.1);state={'family':family,'strength':strength,'mu':mu,'sd':sd,'scale':scale};logs=[];seedpred=None
 if family in FAMILIES[:7]:
  D,offset,weight=pairs;offset=offset/scale;lam=.1*strength;mag=.1*strength
  def lossfun(p):
   margin=offset+D@p
   if 'SVM' in family:
    v=np.maximum(0,1-margin);loss=weight@(v*v);der=-2*weight*v
   else:loss=weight@np.logaddexp(0,-margin);der=-weight*expit(-margin)
   return loss+.5*mag*np.mean(p*p),D.T@der+mag*p/n
  if family=='RankMLP':
   seedpred,weights,logs=mlp_fit(z,zt,lossfun,lam);p=seedpred.mean(1);state.update(weights=weights)
  elif family=='PairGBR':
   ptrain=np.zeros(n);p=np.zeros(len(zt));trees=[]
   for j in range(24):
    _,grad=lossfun(ptrain);tree=DecisionTreeRegressor(max_depth=2,min_samples_leaf=2,random_state=42+j);tree.fit(z,-grad*n);ptrain+=.05*tree.predict(z);p+=.05*tree.predict(zt);trees.append(tree)
   state.update(trees=trees,rate=.05);logs=[dict(success=True,message='24 functional pair-logistic gradient steps',loss=float(lossfun(ptrain)[0]))]
  else:
   if family in ['RBFRankSVM','RBFPairLogistic']:
    K=np.exp(-cdist(z,z,'sqeuclidean')/(2*d));ev,U=np.linalg.eigh(K);keep=ev>1e-10;T=U[:,keep]/np.sqrt(ev[keep]);design=K@T;test=np.exp(-cdist(zt,z,'sqeuclidean')/(2*d))@T;state.update(train_z=z,T=T)
   else:design=z;test=zt
   def obj(beta):
    loss,dp=lossfun(design@beta);return loss+.5*lam*(beta@beta),design.T@dp+lam*beta
   if family=='RankElasticNet':
    k=design.shape[1];l1=.05*strength
    def fun(uv):
     l,gr=obj(uv[:k]-uv[k:]);return l+l1*uv.sum(),np.r_[gr+l1,-gr+l1]
    fit=minimize(fun,np.zeros(2*k),jac=True,bounds=[(0,None)]*(2*k),method='L-BFGS-B',options={'maxiter':300,'ftol':1e-11});beta=fit.x[:k]-fit.x[k:]
   else:
    fit=minimize(obj,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-11});beta=fit.x
   p=test@beta;state['beta']=beta;logs=[dict(success=bool(fit.success),message=str(fit.message),niter=int(fit.nit),loss=float(fit.fun))]
  delta=p*scale
  if seedpred is not None:seedpred*=scale
  state.update(objective='pair_squared_hinge' if 'SVM' in family else 'pair_logistic',pair_count=len(weight),pair_weight_sum=float(weight.sum()))
 else:
  residual=y-b;ym=residual.mean();ys=max(float(residual.std()),.1);target=(residual-ym)/ys;state.update(ym=ym,ys=ys,objective='residual_regression')
  if family=='MLP':
   def fun(p):return .5*np.mean((p-target)**2),(p-target)/n
   seedpred,weights,logs=mlp_fit(z,zt,fun,.1*strength);seedpred=ym+ys*seedpred;delta=seedpred.mean(1);state['weights']=weights
  else:
   if family=='Ridge':est=Ridge(alpha=10*strength)
   elif family=='ElasticNet':est=ElasticNet(alpha=.1*strength,l1_ratio=.5,max_iter=3000,tol=1e-8)
   elif family=='SVR':est=SVR(C=1/strength,epsilon=.1,gamma=1/d)
   elif family=='KRR':est=KernelRidge(alpha=strength,kernel='rbf',gamma=1/d)
   elif family=='GPR':est=GaussianProcessRegressor(kernel=ConstantKernel(1,constant_value_bounds='fixed')*RBF(np.sqrt(d),length_scale_bounds='fixed'),alpha=strength,optimizer=None,normalize_y=False)
   elif family=='RF':est=RandomForestRegressor(n_estimators=24,max_depth=2,min_samples_leaf=2 if strength==1 else 3,random_state=42,n_jobs=1)
   elif family=='ExtraTrees':est=ExtraTreesRegressor(n_estimators=24,max_depth=2,min_samples_leaf=2 if strength==1 else 3,random_state=42,n_jobs=1)
   elif family=='GBR':est=GradientBoostingRegressor(n_estimators=24,max_depth=2,min_samples_leaf=2,learning_rate=.05/np.sqrt(strength),random_state=42)
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always');est.fit(z,target);delta=ym+ys*est.predict(zt)
   state['estimator']=est;logs=[dict(success=not bool(ww),message='; '.join(str(w.message) for w in ww) or 'fit_completed')]
 if seedpred is None:seedpred=delta[:,None]
 assert np.all(np.isfinite(delta))
 state.update(logs=logs,seconds=time.perf_counter()-start,seed_predictions=seedpred)
 return delta,state
