"""Regional coarse proposals from saved predictions; bounded working memory."""
import heapq,math,time
import numpy as np
from . import adaptive_grid131 as g
from .common import OXIDES,read
from .adaptive_select131 import candidate,separated
from .region_score172 import POLICY,statistics,score,anchored_thresholds
from .map_store127 import MapPaused

def check(stop):
    if stop.is_set():raise MapPaused()

def gather(raster,centre,radius,stride=8):
    parts=[]
    for key in g.circle_buckets(centre,radius):
        a,b,c,d=g.box_for(key);rr=np.arange((a+7)//8,b//8+(b%8>0));cc=np.arange((c+7)//8,d//8+(d%8>0))
        r,c=np.meshgrid(rr,cc,indexing='ij');rc=np.c_[r.ravel()*8,c.ravel()*8]
        take=g.distance(rc,centre)<=radius
        if take.any():parts.append((rc[take],raster[r.ravel()[take],c.ravel()[take]]))
    if not parts:return np.empty((0,2),np.int32),np.empty((0,6),np.float32)
    return np.concatenate([x[0] for x in parts]),np.concatenate([x[1] for x in parts])

def select_coarse(job,p,stop,notify):
    from .regions172 import ready_record,source_digest
    ready=job.ws.path(job.base+'/coarse_ready172.json')
    if ready.exists():
        try:
            if read(ready)==ready_record(job,p):return read(job.ws.path(job.base+'/coarse_candidates.json'))
        except (OSError,KeyError):pass
    policy=job.policy;ws=job.ws;shape=((g.HEIGHT+7)//8,(g.WIDTH+7)//8,6)
    path=ws.path(job.base+'/coarse172.part');checkpoint=ws.path(job.base+'/coarse_cache172.json')
    rows=job.chunks(phase='coarse');progress=read(checkpoint) if checkpoint.exists() else {}
    from .map_recovery174 import summary
    revision=summary(ws.root,job.original.base).get('revision',0)
    if progress.get('recovery_revision174',0)!=revision:progress={}
    expected=int(np.prod(shape))*4
    if not path.exists() or path.stat().st_size!=expected or progress.get('model_set')!=job.manifest()['selection']['model_set']:
        ws.reserve(expected+1000000);raster=np.memmap(path,mode='w+',dtype=np.float32,shape=shape);raster[:]=np.nan;raster.flush();progress={}
    else:raster=np.memmap(path,mode='r+',dtype=np.float32,shape=shape)
    start=int(progress.get('loaded',0));hist=np.asarray(progress.get('hist',np.zeros((6,10001))),float)
    try:
        from .region_parallel176 import coarse_chunks
        for i,(row,(rc,v,q)) in enumerate(coarse_chunks(job,rows[start:],stop),start):
            check(stop);good=(q==0)&np.isfinite(v)&(v>=0)&(v<=100)
            raster[rc[:,0]//8,rc[:,1]//8]=np.where(good,v,np.nan)
            weights=g.weight(rc)
            for j in range(6):
                ix=np.rint(v[good[:,j],j]*100).astype(int);hist[j]+=np.bincount(ix,weights=weights[good[:,j]],minlength=10001)
            if (i+1)%512==0 or i+1==len(rows):
                raster.flush();ws.put_json(job.base+'/coarse_cache172.json',dict(loaded=i+1,model_set=job.manifest()['selection']['model_set'],hist=hist.tolist(),recovery_revision174=revision),kind='scratch')
                notify(dict(phase='select_coarse',operation='regional_cache',current=i+1,total=len(rows)))
        refs={}
        for j,o in enumerate(OXIDES):
            cum=np.cumsum(hist[j]);cdf=cum/max(cum[-1],1e-12);value=float(np.searchsorted(cdf,policy['high_quantile']))/100 if cum[-1] else None
            refs[o]=dict(value=value,edges=(np.arange(10001)/100).tolist(),cdf=cdf.tolist(),quantile=policy['high_quantile'],resolution_wt_percent=.01,scope='fixed area-weighted saved coarse reference')
        refs=anchored_thresholds(refs,job.manifest()['selection'],policy)
        ws.put_json(job.base+'/thresholds.json',refs)
        heaps=[[] for _ in range(6)];regional={};cap=max(64,p['initial_candidates']*3)
        # Exhaustive fixed-physical-width patches propose locations. Exact 2km
        # circles below decide ranking. No highest-pixel preselection is used.
        for a in range(0,shape[0],8):
            check(stop);b=min(a+8,shape[0]);centre_r=min(g.HEIGHT-1,(a+b-1)*4)
            lat=float(g.coordinates(np.array([centre_r,0]))[1]);width=max(8,int(round(8/max(.08,math.cos(math.radians(lat))))))
            cols=((shape[1]+width-1)//width)*width
            if not cols:continue
            slab=np.array(raster[a:b]);slab=np.concatenate([slab,slab[:,:cols-shape[1]]],axis=1) if cols>shape[1] else slab
            block=slab.reshape(b-a,-1,width,6).transpose(1,0,2,3).reshape(-1,(b-a)*width,6)
            valid=np.isfinite(block);counts=valid.sum(1);sorted_=np.sort(block,axis=1);idx=np.arange(block.shape[1])[None,:,None]
            lo=counts[:,None,:]*.05;hi=counts[:,None,:]*.95
            weights=np.maximum(0,np.minimum(idx+1,hi)-np.maximum(idx,lo))
            weighted=np.where(np.isfinite(sorted_),sorted_,0)*weights
            means=weighted.sum(1)/np.maximum(weights.sum(1),1);means[counts<block.shape[1]*policy['preliminary_valid']]=-np.inf
            centres=np.c_[np.full(len(means),centre_r),((np.arange(len(means))*width+width//2)*8)%g.WIDTH]
            zones=g.region(centres)
            for j in range(6):
                ids=np.flatnonzero(np.isfinite(means[:,j]));top=ids[np.argsort(means[ids,j])[-min(cap,len(ids)):]]
                for k in top:
                    item=(float(means[k,j]),int(centres[k,0]),int(centres[k,1]));heap=heaps[j]
                    if len(heap)<cap:heapq.heappush(heap,item)
                    elif item>heap[0]:heapq.heapreplace(heap,item)
                for z in np.unique(zones):
                    ids=np.flatnonzero((zones==z)&np.isfinite(means[:,j]));ids=ids[np.argsort(means[ids,j])[-32:]];heap=regional.setdefault((j,int(z)),[])
                    for k in ids:
                        item=(float(means[k,j]),int(centres[k,0]),int(centres[k,1]))
                        if len(heap)<32:heapq.heappush(heap,item)
                        elif item>heap[0]:heapq.heapreplace(heap,item)
            if a%128==0:notify(dict(phase='select_coarse',operation='regional_proposals',current=a,total=shape[0]))
        candidates=[];assessments=[]
        for j,o in enumerate(OXIDES):
            pool=set(heaps[j]);regional_ids=set()
            for z in range(200):
                for item in regional.get((j,z),[]):pool.add(item);regional_ids.add((item[1],item[2]))
            evaluated=[]
            for i,(_,r,c) in enumerate(sorted(pool,reverse=True)):
                check(stop);rc,v=gather(raster,[r,c],policy['radius_m']);q=np.where(np.isfinite(v[:,j]),0,1)
                s=statistics(rc,v[:,j],q,[r,c],policy['radius_m'],8,refs[o]['value'],policy);rank=score(s,refs[o],policy=policy)
                item=candidate(o,[r,c],s['peak'] or 0,'regional_mean172');item.update(assessment=s,regional172=rank,expected_peak=s['peak'],expected_area_km2=s['area_km2'],geographic_proposal=(r,c) in regional_ids)
                evaluated.append(item)
                if i%128==0:notify(dict(phase='select_coarse',operation='regional_scores',oxide=o,current=i,total=len(pool)))
            eligible=sorted((x for x in evaluated if x['regional172']['admissible']),key=lambda x:(-x['regional172']['score'],x['id']))
            chosen=[];global_cap=max(0,p['initial_candidates']-p['regional_candidates'])
            for c in eligible:
                if len(chosen)>=global_cap:break
                if separated(c['rc'],chosen,p['separation_m']):chosen.append(c)
            by_zone={}
            for c in eligible:by_zone.setdefault(int(g.region(np.array([c['rc']]))[0]),[]).append(c)
            while len(chosen)<p['initial_candidates']:
                added=False
                for z,items in sorted(by_zone.items()):
                    while items:
                        c=items.pop(0)
                        if separated(c['rc'],chosen,p['separation_m']):chosen.append(c);added=True;break
                    if len(chosen)>=p['initial_candidates']:break
                if not added:break
            assessments.extend(evaluated);candidates.extend(chosen)
        ws.put_json(job.base+'/coarse_assessments172.json',assessments);ws.put_json(job.base+'/coarse_candidates.json',candidates)
        ws.put_json(job.base+'/coarse_ready172.json',ready_record(job,p))
        return candidates
    finally:
        raster.flush();raster._mmap.close()
        # Completed cache can always be rebuilt from immutable map chunks.
        if ws.path(job.base+'/coarse_candidates.json').exists():
            path.unlink(missing_ok=True);checkpoint.unlink(missing_ok=True);ws.usage(True)
