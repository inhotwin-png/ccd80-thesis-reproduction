"""Versioned regional selection, isolated from immutable map predictions."""
import csv,io,json,sqlite3
from pathlib import Path
from collections import OrderedDict
import numpy as np
from .common import OXIDES,read,sha,now,digest,atomic_bytes
from . import adaptive_grid131 as g
from .adaptive_select131 import candidate,separated
from .region_score172 import POLICY,statistics,score
from .region_sources172 import Auditor
from .map_store127 import MapPaused

def enabled(job):return job.ws.path(job.base+'/selection172/policy.json').is_file()

def source_digest(job):
    from .map_recovery174 import overlays
    corrected=overlays(job.ws.root,job.original.base)
    with sqlite3.connect(job.ws.path(job.original.base+'/journal.sqlite').as_uri()+'?mode=ro',uri=True,timeout=30) as db:
        rows=db.execute("SELECT COALESCE(b.path,a.path,c.path),COALESCE(b.sha,a.sha,c.sha),c.bucket,c.phase,c.n FROM chunks c LEFT JOIN repairs136 a ON a.original=c.path LEFT JOIN repairs142 b ON b.original=c.path WHERE c.phase='coarse' ORDER BY c.path").fetchall()
    return digest([rows,sorted(corrected.items())]) if corrected else digest(rows)

def ready_record(job,p):
    return dict(schema=172,manifest_sha256=sha(job.ws.path(job.original.base+'/map.json')),source_digest=source_digest(job),policy_digest=digest({k:job.policy[k] for k in POLICY}),parameters_digest=digest(p),files={name:sha(job.ws.path(job.base+'/'+name)) for name in ('thresholds.json','coarse_candidates.json')})

def enable(ws,jid):
    base='jobs/'+jid;manifest=ws.path(base+'/map.json');state=read(ws.path(base+'/state.json'))
    if state.get('stage') not in ('coarse','select_coarse'):raise ValueError('REGIONAL_SELECTION_REQUIRES_NEW_JOB_OR_PRESELECTION_CHECKPOINT')
    name=base+'/selection172/policy.json'
    if not ws.path(name).exists():ws.put_json(name,dict(POLICY,created=now(),original_manifest_sha256=sha(manifest),preservation='Existing map chunks, frozen models and study results unchanged'))

class SelectionJob:
    def __init__(self,job):
        self.original=job;self.ws=job.ws;self.id=job.id;self.base=job.base+'/selection172';self.policy=read(self.ws.path(self.base+'/policy.json'));self.cache=OrderedDict()
        if self.policy['original_manifest_sha256']!=sha(self.ws.path(job.base+'/map.json')):raise ValueError('REGIONAL_FROZEN_MANIFEST_CHANGED')
    def manifest(self):return self.original.manifest()
    def chunks(self,**kw):return self.original.chunks(**kw)
    def read_chunk(self,row):return self.original.read_chunk(row)
    def points(self,key):
        rc,v,q,_=self.arrays(key);return rc,v,q
    def arrays(self,key):
        from .map_recovery174 import revision
        current=revision(self.ws.root,self.original.base)
        if current!=getattr(self,'recovery_revision174',0):self.cache.clear();self.recovery_revision174=current
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        root=self.ws.root;dbpath=self.ws.path(self.original.base+'/journal.sqlite')
        with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True,timeout=30) as db:
            rows=db.execute('SELECT c.path,COALESCE(b.path,a.path,c.path),COALESCE(b.sha,a.sha,c.sha) FROM chunks c LEFT JOIN repairs136 a ON a.original=c.path LEFT JOIN repairs142 b ON b.original=c.path WHERE c.bucket=? ORDER BY c.rowid',(key,)).fetchall()
        from .map_recovery174 import overlays
        corrected=overlays(root,self.original.base)
        parts=[]
        for original,name,h in rows:
            name,h=corrected.get(original,(name,h))
            path=root/name
            if sha(path)!=h:raise ValueError('REGIONAL_CHUNK_HASH')
            with np.load(path,allow_pickle=False) as z:parts.append((z['rc'],z['prediction'],z['quality'],z['mi_version'] if 'mi_version' in z.files else np.full(z['quality'].shape,3,np.uint8)))
        if parts:
            rc,v,q,ver=(np.concatenate([x[k] for x in parts]) for k in range(4));ids=rc[:,0].astype(np.int64)*g.WIDTH+rc[:,1];order=np.argsort(ids)
            if len(np.unique(ids))!=len(ids):raise ValueError('REGIONAL_DUPLICATE_COORDINATES')
            result=(rc[order],v[order],q[order],ver[order])
        else:result=(np.empty((0,2),np.int32),np.empty((0,6),np.float32),np.empty((0,6),np.uint8),np.empty((0,6),np.uint8))
        self.cache[key]=result
        while len(self.cache)>4:self.cache.popitem(last=False)
        return result

def gather(job,c,radius,stride,stop):
    parts=[]
    for key in g.circle_buckets(c['rc'],radius):
        if stop.is_set():raise MapPaused()
        rc,v,q,ver=job.arrays(key);take=(rc[:,0]%stride==0)&(rc[:,1]%stride==0)&(g.distance(rc,c['rc'])<=radius)
        if take.any():parts.append((rc[take],v[take],q[take],ver[take]))
    if not parts:return np.empty((0,2),np.int32),np.empty((0,6)),np.empty((0,6)),np.empty((0,6),np.uint8)
    return tuple(np.concatenate([x[k] for x in parts]) for k in range(4))

def assess(job,c,stride,refs,stop,auditor=None,final=False,radius=None,phase='select_review'):
    radius=radius or job.policy['radius_m'];rc,v,q,ver=gather(job,c,radius,stride,stop);j=OXIDES.index(c['oxide']);source=None
    if auditor is not None:
        good=np.flatnonzero((q[:,j]==0)&np.isfinite(v[:,j])&(v[:,j]>=0)&(v[:,j]<=100));codes=np.full(len(rc),3,np.uint8)
        chosen=good if final else good[np.unique(np.linspace(0,max(0,len(good)-1),min(32,len(good))).astype(int))]
        inspected=auditor.inspect(rc[chosen],ver[chosen,j],job.manifest()['selection']['models'][c['oxide']]['inputs'],c['oxide'],phase)
        codes[chosen]=inspected['codes'];source=dict(inspected,codes=codes)
        # A suspicious pilot triggers the full-footprint check before a decision;
        # never extrapolate a few source failures into a claimed regional rate.
        if not final and np.isin(inspected['codes'],[1,2]).any():
            inspected=auditor.inspect(rc[good],ver[good,j],job.manifest()['selection']['models'][c['oxide']]['inputs'],c['oxide'],phase);codes[good]=inspected['codes'];source=dict(inspected,codes=codes)
    s=statistics(rc,v[:,j],q[:,j],c['rc'],radius,stride,refs[c['oxide']]['value'],job.policy,source)
    rank=score(s,refs[c['oxide']],final,job.policy)
    if source:s['source_examples']=source.get('examples',[])
    result = dict(c,assessment=s,regional172=rank,expected_peak=c.get('expected_peak',s['peak']),expected_area_km2=s['area_km2'])
    if phase == 'report' and source is not None and job.manifest()['selection'].get('ranking', {}).get('models'):
        from .region_rank181 import targets, options
        result['rank_cells181'] = options(targets(c['rc'], radius), rc, v[:, j], q[:, j], ver[:, j], source['codes'])
    return result

def pick(items,p,cap):
    selected=[]
    for o in OXIDES:
        chosen=[]
        for c in sorted((x for x in items if x['oxide']==o and x['regional172']['admissible']),key=lambda x:(-x['regional172']['score'],x['id'])):
            if separated(c['rc'],chosen,p['separation_m']):chosen.append(c)
            if len(chosen)>=cap:break
        selected.extend(chosen)
    return selected

def promote(job,parents,p,phase,stride,cap,stop,notify,auditor_factory=None):
    from .region_parallel176 import evaluate
    refs=read(job.ws.path(job.base+'/thresholds.json'));path=job.ws.path(job.base+'/'+phase+'_progress172.json')
    signature=digest([[(c['id'],c['rc']) for c in parents],job.policy,stride])
    previous=read(path) if path.exists() else {};scored=previous.get('items',[]) if previous.get('signature')==signature else []
    by_id={c['id']:c for c in scored}
    # Old files remain intact. Do not import their scores after an input repair.
    from .pipeline176 import legacy_recovery_compatible
    if path.exists() and not legacy_recovery_compatible(job.ws.root,job.original.base,path):by_id={}
    legacy=[(i,by_id[c['id']]) for i,c in enumerate(parents) if c['id'] in by_id]
    scored=evaluate(job,parents,phase,stop,notify,lambda frozen,a,c:assess(frozen,c,stride,refs,stop,a,phase=='select_review',phase=phase),auditor_factory or Auditor,signature=[stride,refs],legacy=legacy)
    selected=pick(scored,p,cap)
    job.ws.put_json(job.base+'/'+phase+'_assessments.json',scored);job.ws.put_json(job.base+'/'+phase+'_candidates.json',selected)
    return selected

def audit_review(job,p,stop,notify):
    # Every independent dense patch gets a chance to propose a centre, even
    # when coarse acquisition failed there. This is a proposal quota; the
    # subsequent scout stage compares all candidates on the same 2km core.
    initial=read(job.ws.path(job.base+'/coarse_candidates.json'));refs=read(job.ws.path(job.base+'/thresholds.json'));plan=read(job.ws.path(job.original.base+'/audit_plan.json'));found=[]
    from .region_parallel176 import evaluate
    proposals=[]
    for i,(a,b,c,d) in enumerate(plan):
        centre=[(a+b)//2,(c+d)//2];lat=float(g.coordinates(np.asarray(centre))[1])
        radius=min(job.policy['radius_m'],.49*62*min(b-a,(d-c)*np.cos(np.radians(lat))))
        for o in OXIDES:
            if radius<62:continue
            item=candidate(o,centre,0.,'audit_regional172');item['audit_proposal172']=dict(radius_m=radius,needs_common_2km_refinement=True);proposals.append(item)
    measured=evaluate(job,proposals,'audit_review',stop,notify,lambda frozen,a,c:assess(frozen,c,1,refs,stop,radius=c['audit_proposal172']['radius_m']),auditor_factory=None,signature=refs)
    found=[item for item in measured if item['regional172']['admissible'] and item['assessment']['trimmed_mean']>=(refs[item['oxide']]['value'] or 0)]
    discoveries=pick(found,p,p['review_candidates']);selected=[]
    for o in OXIDES:
        extra=[c for c in discoveries if c['oxide']==o and separated(c['rc'],[x for x in initial if x['oxide']==o],p['separation_m'])]
        kept=sorted([c for c in initial if c['oxide']==o],key=lambda c:(-c['regional172']['score'],c['id']))[:max(0,p['initial_candidates']-len(extra))]
        selected.extend(kept+extra)
    selected_ids={x['id'] for x in selected}
    result=dict(discoveries=found,deferred_prior_candidates=[x for x in initial if x['id'] not in selected_ids],patches_considered=len(plan),scope='All planned dense patches considered before choosing bounded audit proposal slots; all undergo common 2km scout evaluation; not global absence certification')
    job.ws.put_json(job.base+'/audit_review.json',result);job.ws.put_json(job.base+'/scout_candidates.json',selected);return selected

def expand(job,parents,radius,stop,notify,phase=None):
    refs=read(job.ws.path(job.base+'/thresholds.json'));items=[];all_=[]
    # Outer expansion measurements remain distinct from the fixed 2km ranking.
    from .region_parallel176 import evaluate
    phase=phase or ('expand4' if radius<=2000 else 'expand8')
    all_=evaluate(job,parents,phase,stop,notify,lambda frozen,a,c:assess(frozen,c,1,refs,stop,a,True,radius,phase),Auditor,signature=[radius,refs])
    for c,item in zip(parents,all_):
        s=item['assessment']
        if s['boundary']=='open_high' and s['source_checked_fraction']>=job.policy['source_checked_min'] and s['source_suspect_fraction']<=job.policy['suspect_max']:items.append(c)
    job.ws.put_json(job.base+f'/boundary_{int(radius)}.json',all_);return items

def report(runner,job):
    items=read(job.ws.path(job.base+'/select_review_candidates.json'));refs=read(job.ws.path(job.base+'/thresholds.json'));final=[]
    ranked=bool(job.manifest()['selection'].get('ranking',{}).get('models'))
    from .map_recovery174 import revision
    from .region_rank181 import METHOD, measure, standings
    key=digest([METHOD,job.policy,refs,items,job.manifest()['selection'],revision(job.ws.root,job.original.base)])
    rank_checkpoint=job.base+'/rank_report181.json'
    rank_path=job.ws.path(rank_checkpoint)
    saved=read(rank_path) if ranked and rank_path.is_file() else {}
    ranks={cid: record for cid,record in saved.get('ranks',{}).items()
           if record.get('method')==METHOD and not record.get('errors')} if saved.get('signature')==key else {}
    runner.report_counter177=dict(candidates=len(items),total=len(items)*(2 if ranked else 1)+1,ranked=len(ranks))
    from .region_parallel176 import evaluate
    final=evaluate(job,items,'report',runner.stop,runner.emit,lambda frozen,a,c:assess(frozen,c,1,refs,runner.stop,a,True,phase='report'),Auditor,signature=[refs,METHOD])
    for item in final:item['actual']=item.pop('assessment')
    if ranked:
        from .rank_candidates144 import Scorer
        try:ranker=Scorer(job,runner.stop,runner.emit,'report')
        except MapPaused:raise
        except Exception as e:
            ranker=None
            for c in final:
                groups=c.pop('rank_cells181', [])
                option_key=digest([c['regional172']['status'],groups])
                saved_rank=ranks.get(c['id'],{})
                c['rank_supplement172']=saved_rank if saved_rank.get('options_sha256')==option_key else dict(method=METHOD,points=[],median=None,errors=[dict(error=type(e).__name__,message=str(e))])
            runner.emit(dict(phase='report',current=len(final),total=len(final),operation='rank_supplement'))
        try:
            for c in final if ranker is not None else []:
                groups=c.pop('rank_cells181')
                option_key=digest([c['regional172']['status'],groups])
                saved_rank=ranks.get(c['id'],{})
                if saved_rank.get('options_sha256')==option_key and not saved_rank.get('errors'):
                    c['rank_supplement172']=ranks[c['id']]
                    continue
                if c['regional172']['status']!='eligible':
                    groups=[]
                c['rank_supplement172']=measure(groups, lambda rc, version: ranker.reference_value(c['oxide'],rc,version))
                c['rank_supplement172']['options_sha256']=option_key
                ranks[c['id']]=c['rank_supplement172']
                job.ws.put_json(rank_checkpoint,dict(signature=key,ranks=ranks))
                runner.emit(dict(phase='report',current=len(ranks),total=len(final),operation='rank_supplement',progress_scope='stage'))
        finally:
            if ranker is not None:ranker.close()
    standings(final)
    final.sort(key=lambda c:(OXIDES.index(c['oxide']),-c['regional172']['score'],c['id']))
    result=dict(schema=172,job=job.id,created=now(),scope=job.policy['scope'],policy=job.policy,review=final,thresholds=refs,points=runner.job.count(),status='local_complete',rank_reference_method=METHOD,audit=read(job.ws.path(job.base+'/audit_review.json')),selection_note='Fixed 2km regional mean, connected high area, valid footprint and spatial retention; source-checked. RANK never overrides this order.')
    job.ws.put_json(job.base+'/report.json',result)
    stream=io.StringIO(newline='');w=csv.writer(stream);w.writerow(['oxide','longitude','latitude','mean_wt_percent','trimmed_mean_wt_percent','median','p25','connected_high_km2','valid_fraction','source_checked_fraction','source_suspect_fraction','stability','priority_score','status','candidate_id','rank_reference','rank_assessed_quadrants'])
    for c in final:
        reference=c.get('rank_supplement172',{})
        position=f"{reference['position']}/{reference['pool']}" if 'position' in reference else ''
        s=c['actual'];r=c['regional172'];w.writerow([c['oxide'],c['lon'],c['lat'],s['mean'],s['trimmed_mean'],s['median'],s['p25'],s['area_km2'],s['valid_fraction'],s['source_checked_fraction'],s['source_suspect_fraction'],s['stability'],r['score'],r['status'],c['id'],position,len(reference.get('points',[]))])
    name=job.base+'/candidate_summary.csv';atomic_bytes(job.ws.path(name),stream.getvalue().encode('utf-8-sig'));job.ws.record(name)
    runner.emit(dict(phase='report',operation='report_saved'))
