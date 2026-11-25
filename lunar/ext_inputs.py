"""Observed inputs only; no chemical labels or saved teacher predictions in packs."""
import numpy as np
from .common import ASSETS
from .models import Features,validate_mi

SPECTRAL=('Raw8','Ratio8','MSC8','RatioMSC16','SSA8','Log8','Sqrt8','RatioSSA16','SSARatioMSC16')
LOCAL=('MI_M3','SNV8','RawSNV16','Sat7','V8theta')
OBSERVED=('F0R_62','F4_62','F0R_250','F1_250','F2_250','F3_250','F4_250','F5_250','M1','M2','M4','CLASS3','CLASS6','B1','B5','S1','S5')
CLASS_SUBSETS={'CLASS_X1':[0],'CLASS_X2':[1],'CLASS_X3':[2],'CLASS_X4':[3],'CLASS_X5':[4],'CLASS_X6':[5],'CLASS_X7':[0,1],'CLASS_X10':[0,1,2,3],'CLASS_X11':[0,1,2,4],'CLASS_X12':[0,1,2,5],'CLASS_M5FC':[0,1,2,3,4],'CLASS_M5FO':[0,1,2,3,5],'CLASS_M5CO':[0,1,2,4,5]}
CLASS_INPUTS=tuple(CLASS_SUBSETS)+('CLASS_ONLY2','CLASS_ONLY3','CLASS_ONLY6','MI_CLASS2','WAC3','MI_WAC3','MI_MATCHED_WAC3','CLASS_CELL296_3','CLASS_CELL296_6','CLASS_CELL96_3','CLASS_CELL96_4','CLASS_SITE96_3','CLASS_SITE96_4','MI_M3_CLASS3','MI_M3_CLASS6','COORD2','MI_COORD2','GEO3')
COMPOSITE=tuple(f'{f}|{s}|{m}' for f in ('F0','F0R_62','F4_62','F0R_250','F1_250','F2_250','F3_250','F4_250','F5_250') for s in ('Raw8','RatioMSC16') for m in ('M0','M1','M2','M3','M4') if not (s=='Raw8' and m=='M0'))
def m3(x):return np.c_[(x[:,1]-x[:,0])/.335/x[:,1],1-x[:,3]/(x[:,1]+(x[:,6]-x[:,1])*(950-749)/(1248-749)),x[:,7]/x[:,1]]
def load_observations(dataset):
 with np.load(ASSETS/'expansion104/observations.npz') as p:
  if list(p['ids'])!=dataset.ids:raise ValueError('OBSERVATION_AREA_ALIGNMENT')
  return dict(raw=dataset.x,**{k:p[k].copy() for k in p.files if k!='ids'})
def subset(batch,ix):return {k:np.asarray(v)[ix] for k,v in batch.items()}
def from_raw(x):return {'raw':validate_mi(x)}
def need(batch,name):
 if name not in batch:raise ValueError('ADDITIONAL_OBSERVATION_REQUIRED:'+name)
 x=np.asarray(batch[name],float)
 if not np.isfinite(x).all():raise ValueError('INVALID_OBSERVATION:'+name)
 return x

class Input:
 def __init__(self,name):self.name=name
 def fit(self,batch):
  from .ext_spatial import Spatial,NAMES
  self.spatial=Spatial(self.name).fit(batch) if self.name in NAMES else None
  if '|' in self.name:
   f,s,m=self.name.split('|');v=batch['raw'] if f=='F0' else need(batch,f);self.primary=9 if f.startswith(('F4','F5')) else 8;self.composite_ref=v[:,:self.primary].mean(0)
  x=validate_mi(batch['raw']);self.raw_ref=x.mean(0);self.ssa_ref=Features('SSA8').fit(x).transform(x).mean(0) if self.name=='SSARatioMSC16' else None
  return self
 def transform(self,batch):
  x=validate_mi(batch['raw']);n=self.name
  if self.spatial:return self.spatial.transform(batch)
  if '|' in n:
   f,s,m=n.split('|');v=x if f=='F0' else need(batch,f);a=v[:,:self.primary];tail=v[:,self.primary:]
   if s=='RatioMSC16':
    ratio=a/a[:,[1]];ratio[:,1]=a[:,1];ref=self.composite_ref;rc=ref-ref.mean();slopes=(a-a.mean(1,keepdims=True))@rc/max(rc@rc,1e-15)
    if np.any(abs(slopes)<1e-8):raise ValueError('MSC_SLOPE_NEAR_ZERO')
    msc=(a-(a.mean(1)-slopes*ref.mean())[:,None])/slopes[:,None];a=np.c_[ratio,msc]
   z=np.c_[a,tail]
   if m!='M0':z=np.c_[z,m3(x) if m=='M3' else need(batch,m)]
   return z
  if n in SPECTRAL:
   f=Features('Raw8' if n in ('RatioSSA16','SSARatioMSC16') else n);f.reference=self.raw_ref
   if n=='RatioSSA16':return np.c_[Features('Ratio8').fit(x).transform(x),Features('SSA8').fit(x).transform(x)]
   if n=='SSARatioMSC16':
    s=Features('SSA8').fit(x).transform(x);f=Features('RatioMSC16');f.reference=self.ssa_ref;return f.transform(s)
   return f.transform(x)
  if n=='MI_M3':return np.c_[x,m3(x)]
  if n in ('SNV8','RawSNV16'):
   sd=x.std(1,keepdims=True)
   if np.any(sd<1e-12):raise ValueError('SNV_CONSTANT_SPECTRUM')
   snv=(x-x.mean(1,keepdims=True))/sd;return snv if n=='SNV8' else np.c_[x,snv]
  if n=='Sat7':return x[:,1:]
  if n=='V8theta':
   # Lucey-style indices from measured bands; fixed, not fitted chemistry.
   with np.errstate(divide='ignore'):theta=-np.arctan((x[:,3]/x[:,1]-1.25)/(x[:,1]-.037))
   return np.c_[x,theta,np.arctan((x[:,0]/x[:,1]-.208)/(x[:,1]+.108))]
  if n.startswith('CLASS_ONLY'):return need(batch,'CLASS6')[:,:int(n[-1])]
  if n=='MI_CLASS2':return np.c_[x,need(batch,'CLASS6')[:,:2]]
  if n in ('WAC3','MI_WAC3','MI_MATCHED_WAC3'):
   w=need(batch,'WAC2')
   if np.any(w<=0):raise ValueError('WAC_REQUIRES_POSITIVE_REFLECTANCE')
   z=np.c_[w,w[:,0]/w[:,1]]
   return z if n=='WAC3' else np.c_[need(batch,'MI_WAC_support') if n=='MI_MATCHED_WAC3' else x,z]
  if n in ('M1','M2','M4'):return np.c_[x,need(batch,n)]
  if n in ('CLASS3','CLASS6'):return np.c_[x,need(batch,n)]
  if n in CLASS_SUBSETS:return np.c_[x,need(batch,'CLASS6')[:,CLASS_SUBSETS[n]]]
  if n.startswith(('CLASS_CELL','CLASS_SITE')):
   size=int(n.rsplit('_',1)[1]);source='CLASS_296s_cell' if '296' in n else 'CLASS_96s_site' if 'SITE' in n else 'CLASS_96s_cell';return np.c_[x,need(batch,source)[:,:size]]
  if n.startswith('MI_M3_CLASS'):return np.c_[x,m3(x),need(batch,'CLASS6')[:,:int(n[-1])]]
  if n in ('COORD2','MI_COORD2','GEO3'):
   coordinates=need(batch,'coordinates')
   if n=='COORD2':return coordinates
   if n=='MI_COORD2':return np.c_[x,coordinates]
   lat,lon=np.deg2rad(coordinates).T;return np.c_[x,np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]
  if n in ('B1','B5'):return need(batch,'broad_'+n[-1])[:,:8]
  if n in ('S1','S5'):return np.c_[x,need(batch,'broad_'+n[-1])[:,:8]]
  if n.startswith('F'):return need(batch,n)
  raise ValueError('UNKNOWN_EXTENDED_INPUT:'+n)

def input_requirements(name):
 from .ext_spatial import NAMES
 if name.startswith('Kernel_'):return 'C69 local fourteen features + measured joint-eight pixel kernel; training-only scaler, bandwidth, centering and KRR'
 if name in NAMES:return 'Measured spatial MI pixels: C69 rings / SLIC / joint-8 distribution; KME pixel scaler and PCA fitted on training areas only'
 if '|' in name:
  f,s,m=name.split('|');return (input_requirements(f) if f!='F0' else '62m MI')+'; '+s+('; '+input_requirements(m) if m not in ('M0','M3') else '; local auxiliary '+m)
 if name in SPECTRAL+LOCAL:return '62m MI 8 bands'
 if name in ('F0R_62','F4_62'):return '62m re-extracted MI; actual MN1 required' if name=='F4_62' else '62m re-extracted MI'
 if name.startswith('F'):return '250m full-side MI observations'+(' including actual MN1' if name in ('F4_250','F5_250') else '')
 if name in ('M1','M2'):return '500m full-side MI distribution'
 if name in ('M4','B1','S1'):return '2km full-side MI observations (M4 also 500m)'
 if name in ('B5','S5'):return '10km full-side MI observations'
 if 'WAC' in name:return 'Measured WAC321/415 cell reflectance, about 474m at equator; MI_MATCHED also needs MI averaged over that exact cell'
 if 'CLASS' in name:return 'Measured CLASS ratios; not available from MI alone'
 if name in ('COORD2','MI_COORD2','GEO3'):return 'Measured latitude/longitude; historical geography control, not spectral chemistry evidence'
 if name=='A2':return '62m MI + six training-only crossfit teacher predictions'
 if name.startswith(('B_','C_','D_')):return {'025':'500m','1':'2km','5':'10km'}[name.split('_')[1]]+' full-side MI observations; teachers retrained within each outer fold'
 raise ValueError(name)
