import itertools
import numpy as np
from .common import ASSETS, OXIDES, read, sha

class Dataset:
    def __init__(self,path=None):
        self.path=path or ASSETS/'dataset.json'; self.hash=sha(self.path); self.document=read(self.path)
        self.areas=self.document['areas']; self.ids=[a['id'] for a in self.areas]
        if len(set(self.ids))!=len(self.ids): raise ValueError('DUPLICATE_AREA')
        self.x=np.asarray([a['mi8'] for a in self.areas],float)
        self.y=np.array([[a['oxides'][o] if a['oxides'][o] is not None else np.nan for o in OXIDES] for a in self.areas])
        self.site=np.array([a['site'] for a in self.areas]); self.sites=sorted(set(self.site))
        self.overlaps={frozenset((o['a'],o['b'])) for o in self.document['overlaps']}
    def folds(self):
        return [tuple(c) for n in (1,2) for c in itertools.combinations(self.sites,n)]
    def split(self,oxide,excluded):
        y=self.y[:,OXIDES.index(oxide)]; valid=np.isfinite(y)
        return np.flatnonzero(valid&~np.isin(self.site,excluded)), np.flatnonzero(np.isin(self.site,excluded))
    def metrics(self,oxide,predictions):
        y=self.y[:,OXIDES.index(oxide)]; errors=[]; site_errors=[]; pairs=[]
        for site in self.sites:
            p=np.asarray(predictions[(site,)]); ix=np.flatnonzero((self.site==site)&np.isfinite(y))
            if not np.isfinite(p[ix]).all(): raise ValueError('MISSING_LOLO_PREDICTIONS')
            ee=np.abs(y[ix]-p[ix]); errors.extend(ee.tolist())
            if len(ee): site_errors.append(float(ee.mean()))
        ties=0
        for i,j in itertools.combinations(np.flatnonzero(np.isfinite(y)),2):
            key=tuple(sorted({self.site[i],self.site[j]})); p=np.asarray(predictions[key]); dy=y[i]-y[j]; dp=p[i]-p[j]
            if not np.isfinite(dp): raise ValueError('MISSING_PAIR_PREDICTIONS')
            if abs(dy)<=1e-10: ties+=1; continue
            pairs.append(dict(a=self.ids[i],b=self.ids[j],excluded=list(key),actual_delta=float(dy),predicted_delta=float(dp),correct=bool(dy*dp>0 and abs(dp)>1e-10),predicted_tie=bool(abs(dp)<=1e-10),overlapping=frozenset((self.ids[i],self.ids[j])) in self.overlaps))
        correct=sum(p['correct'] for p in pairs)
        return dict(mae_area=float(np.mean(errors)),mae_site_balanced=float(np.mean(site_errors)),max_error=float(max(errors)),valid_areas=len(errors),rank_correct=correct,rank_total=len(pairs),rank_accuracy=correct/len(pairs) if pairs else None,true_ties_excluded=ties,predicted_ties=sum(p['predicted_tie'] for p in pairs),overlapping_pairs=sum(p['overlapping'] for p in pairs),pairs=pairs,protocol='area62_fixed_v1',selection_scope='observed best on repeatedly reused eight sites; no independent new-site confirmation')
