"""Chemical regression plus pair-logistic loss; fixed C67/DirectRank controls."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit
from .ext_rank import pair_design

class JointRank:
 def __init__(self,family,params):self.family=family;self.params=params
 def fit(self,x,y,sites,label_support=None):
  self.mu=x.mean(0);self.sd=x.std(0);self.sd[self.sd<1e-12]=1;z=(x-self.mu)/self.sd
  self.ym=y.mean();self.ys=max(y.std(),1e-8);yy=(y-self.ym)/self.ys;n,p=z.shape;cf=self.params
  from .ext_evidence import evidence_pairs
  D,pw,keys=evidence_pairs(y,sites,cf,label_support) if cf.get('pair_filter') in ('EVID','GAP_EVID') else pair_design(y,sites,cf);lam=cf['lambda_rank'];factor=cf['factor'];self.train=z
  if self.family=='JointRankMLP':return self.fit_mlp(z,yy,D,pw)
  if self.family=='JointRankGP':
   K=np.exp(-cdist(z,z,'sqeuclidean')/(2*p));ev,U=np.linalg.eigh(K);keep=ev>np.finfo(float).eps*n*max(1.,ev.max());self.basis=U[:,keep]/np.sqrt(ev[keep]);design=K@self.basis;penalty=(.5*factor+1e-8)/n
  else:design=z;penalty=cf.get('alpha',10.)*factor/n if self.family=='JointRankRidge' else .05*factor*.5
  L=D@design;l1=.05*factor*.5 if self.family=='JointRankElasticNet' else 0.
  def obj(beta):
   err=design@beta-yy;margin=L@beta
   loss=.5*np.mean(err**2)+.5*penalty*(beta@beta)+lam*(pw@np.logaddexp(0,-margin))
   grad=design.T@err/n+penalty*beta-lam*L.T@(pw*expit(-margin));return loss,grad
  k=design.shape[1];b=np.linalg.solve(design.T@design/n+penalty*np.eye(k),design.T@yy/n)
  if l1:
   def fun(v):
    loss,g=obj(v[:k]-v[k:]);return loss+l1*v.sum(),np.r_[g+l1,-g+l1]
   result=minimize(fun,np.r_[np.maximum(b,0),np.maximum(-b,0)],jac=True,bounds=[(0,None)]*(2*k),method='L-BFGS-B',options={'maxiter':4000,'ftol':1e-13,'gtol':1e-8,'maxls':50});self.beta=result.x[:k]-result.x[k:]
  else:
   result=minimize(obj,b,jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-13,'gtol':1e-8,'maxls':50});self.beta=result.x
  self.audit=dict(success=bool(result.success),message=str(result.message),loss=float(result.fun),pair_count=len(pw),objective='chemical_squared_error_plus_pair_logistic',pair_policy=cf['pair_policy'],units='wt_percent');return self
 def fit_mlp(self,z,y,D,pw):
  import torch
  from torch import nn
  torch.set_num_threads(1);self.states=[];self.losses=[]
  x=torch.tensor(z,dtype=torch.float32);target=torch.tensor(y,dtype=torch.float32);dd=torch.tensor(D,dtype=torch.float32);ww=torch.tensor(pw,dtype=torch.float32)
  for seed in range(42,47):
   torch.manual_seed(seed);net=nn.Sequential(nn.Linear(z.shape[1],16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1));opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01*self.params['factor'])
   for _ in range(300):
    opt.zero_grad();p=net(x)[:,0];loss=((p-target)**2).mean()+self.params['lambda_rank']*(ww*nn.functional.softplus(-(dd@p))).sum();loss.backward();opt.step()
   self.states.append({k:v.detach().numpy().copy() for k,v in net.state_dict().items()});self.losses.append(float(loss.detach()))
  self.audit=dict(losses=self.losses,pair_count=len(pw),objective='chemical_squared_error_plus_pair_logistic',units='wt_percent');return self
 def predict(self,x):
  z=(x-self.mu)/self.sd
  if self.family=='JointRankGP':p=np.exp(-cdist(z,self.train,'sqeuclidean')/(2*z.shape[1]))@self.basis@self.beta
  elif self.family=='JointRankMLP':
   pp=[]
   for w in self.states:
    h=np.maximum(z@w['0.weight'].T+w['0.bias'],0);h=np.maximum(h@w['2.weight'].T+w['2.bias'],0);pp.append((h@w['4.weight'].T+w['4.bias'])[:,0])
   p=np.mean(pp,axis=0)
  else:p=z@self.beta
  return self.ym+self.ys*p
