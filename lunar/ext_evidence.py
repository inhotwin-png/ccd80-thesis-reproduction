"""Report-dispersion pair filters, never measurement uncertainty or prediction inputs."""
import numpy as np
from .common import ASSETS

def load_support(dataset):
 with np.load(ASSETS/'expansion104/label_support.npz') as p:
  if list(p['ids'])!=dataset.ids:raise ValueError('LABEL_SUPPORT_ALIGNMENT')
  return p['halfwidth'].copy()

def evidence_pairs(y,sites,params,halfwidth):
 y=np.asarray(y);sites=np.asarray(sites);n=len(y);policy=params['pair_policy'];filt=params['pair_filter']
 if halfwidth is None:raise ValueError('REPORT_DISPERSION_REQUIRED_FOR_EVID')
 h=np.asarray(halfwidth,float)
 if h.shape!=y.shape or not np.isfinite(h).all() or np.any(h<0):raise ValueError('INVALID_REPORT_DISPERSION')
 units=[np.flatnonzero(sites==s) for s in sorted(set(sites))] if policy=='B' else [np.array([i]) for i in range(n)]
 means=np.array([y[ix].mean() for ix in units]);hw=np.array([h[ix].mean() for ix in units]);iqr=max(np.subtract(*np.percentile(means,[75,25])),1e-8)
 rows=[];weights=[];keys=[]
 for a in range(len(units)):
  for b in range(a+1,len(units)):
   ia,ib=units[a],units[b]
   if policy in ('D1','D2') and sites[ia[0]]==sites[ib[0]]:continue
   gap=means[a]-means[b];bound=hw[a]+hw[b]
   if filt=='GAP_EVID':bound=max(.25*iqr,bound)
   if abs(gap)<=max(bound,1e-12):continue
   row=np.zeros(n);row[ia]=np.sign(gap)/len(ia);row[ib]=-np.sign(gap)/len(ib);rows.append(row)
   keys.append(tuple(sorted((str(sites[ia[0]]),str(sites[ib[0]])))))
   weights.append(min(abs(gap)/iqr,3)/(1+(hw[a]+hw[b])/iqr) if filt=='GAP_EVID' else 1.)
 w=np.asarray(weights)
 if policy=='D2':w=np.array([v/keys.count(k) for v,k in zip(w,keys)])
 if len(w):w/=w.sum()
 return np.asarray(rows).reshape(-1,n),w,keys
