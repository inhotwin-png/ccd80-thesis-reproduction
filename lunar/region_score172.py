"""Area-based candidate statistics. This module never modifies map predictions."""
import math
import numpy as np
from scipy.ndimage import label
from .adaptive_grid131 import coordinates,distance,weight,WIDTH
from .map_inputs127 import RADIUS

POLICY=dict(schema=172,version='regional_mean_area_quality_v1',radius_m=2000.,trim=.05,
    preliminary_valid=.80,final_valid=.90,source_checked_min=.95,suspect_max=.02,
    high_quantile=.95,quantile_resolution=.01,weights=dict(concentration=.40,area=.30,valid=.15,stability=.15),
    observed_high_cap_quantile=.75,
    source_ratio_min=2/3,source_ratio_max=1.5,source_signal_dn=500,
    concentration='area-weighted 5%-per-tail trimmed mean; all valid pixels, not high pixels only',
    connectivity='4-neighbour; missing cells never bridged',
    scope='Regional prediction priority, not a calibrated accuracy probability or confirmed resource',
    rank='supplementary regional observations only; never reranks regional priority',
    thresholds='Fixed recorded reference within oxide; initial heuristic, not established optimum')

def anchored_thresholds(refs,selection,policy=None):
    """Limit a contaminated prediction tail's influence on the high-area
    reference. This observed-label anchor never clips map predictions."""
    from .data import Dataset
    from .common import OXIDES
    p=policy or POLICY;ds=Dataset()
    if ds.hash!=selection.get('dataset_sha256'):
        for ref in refs.values():ref['observed_anchor']='unavailable_dataset_mismatch'
        return refs
    for j,o in enumerate(OXIDES):
        values=ds.y[:,j];values=values[np.isfinite(values)];ref=refs[o]
        if not len(values):continue
        cap=float(np.quantile(values,p['observed_high_cap_quantile']));original=ref['value'];ref['unanchored_global_threshold']=original
        ref['value']=cap if original is None else min(cap,original)
        ref['observed_anchor']=dict(dataset_sha256=ds.hash,quantile=p['observed_high_cap_quantile'],value=cap,scope='fixed observed training-label reference; not proof of global abundance distribution')
    return refs

def quantile(values,weights,prob):
    x=np.asarray(values,float);w=np.asarray(weights,float)
    take=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not take.any():return None
    x=x[take];w=w[take];order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    i=np.searchsorted(np.cumsum(w),float(prob)*w.sum(),side='left')
    return float(x[min(len(x)-1,int(i))])

def trimmed_mean(values,weights,trim=.05):
    x=np.asarray(values,float);w=np.asarray(weights,float)
    keep=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not keep.any():return None
    order=np.argsort(x[keep],kind='stable');x=x[keep][order];w=w[keep][order]
    cumulative=np.cumsum(w);total=cumulative[-1]
    part=np.maximum(0,np.minimum(cumulative,(1-trim)*total)-np.maximum(cumulative-w,trim*total))
    return float(part@x/part.sum())

def component(rc,high,areas,stride,centre):
    if not np.any(high):return 0.,np.zeros(len(rc),bool)
    yy=(rc[:,0]-int(rc[:,0].min()))//stride
    dx=(rc[:,1].astype(np.int64)-int(centre[1])+WIDTH//2)%WIDTH-WIDTH//2
    xx=(dx-dx.min())//stride
    shape=(int(yy.max())+1,int(xx.max())+1)
    if np.prod(shape)>12_000_000:raise ValueError('REGIONAL_GRID_TOO_LARGE')
    grid=np.zeros(shape,bool);grid[yy,xx]=high
    labels,n=label(grid);ids=labels[yy,xx]
    sums=np.bincount(ids[high],weights=areas[high],minlength=n+1);sums[0]=0
    k=int(np.argmax(sums));return float(sums[k]),high&(ids==k)

def statistics(rc,values,quality,centre,radius,stride,threshold,policy=None,source=None):
    """Fixed-footprint denominator includes missing and uncomputed area.

    source.codes: 0 checked usable, 1 confirmed unusable, 2 source anomaly,
    3 unverified/read error, 4 warning-only usable. No chemistry labels enter.
    """
    p=policy or POLICY;rc=np.asarray(rc,np.int32);v=np.asarray(values,float);q=np.asarray(quality)
    area=weight(rc)*(.062*stride)**2
    nominal=2*math.pi*(RADIUS/1000)**2*(1-math.cos(radius/RADIUS))
    original=(q==0)&np.isfinite(v)&(v>=0)&(v<=100)
    codes=np.full(len(rc),3,np.uint8) if source is None else np.asarray(source['codes'],np.uint8)
    if codes.shape!=v.shape:raise ValueError('REGIONAL_SOURCE_SHAPE')
    # Statistical warnings never change eligibility. Confirmed/suspect input
    # exclusions affect this derived candidate analysis, not the stored map.
    usable=original&~np.isin(codes,[1,2])
    observed=float(area.sum());valid_area=float(area[usable].sum())
    fraction=min(1.,valid_area/max(nominal,1e-12))
    checked_area=float(area[original&np.isin(codes,[0,1,2,4])].sum())
    checked=checked_area/max(float(area[original].sum()),1e-12)
    bad=float(area[original&(codes==1)].sum())/max(nominal,1e-12)
    suspect=float(area[original&(codes==2)].sum())/max(nominal,1e-12)
    result=dict(schema172=172,radius_m=radius,stride=stride,nominal_area_km2=nominal,high_threshold=threshold,
        computed_fraction=min(1.,observed/max(nominal,1e-12)),valid_fraction=fraction,
        source_checked_fraction=min(1.,checked),source_invalid_fraction=bad,source_suspect_fraction=suspect,
        source_scope=(source or {}).get('scope','not inspected'),source_reasons=(source or {}).get('reasons',{}),
        sample_count=len(rc),usable_count=int(usable.sum()),mean=None,trimmed_mean=None,median=None,p25=None,
        original_mean=float(np.average(v[original],weights=area[original])) if original.any() else None,
        peak=None,peak_rc=None,area_km2=0.,stability=0.,boundary='unknown',high_count=0,
        assessment_scope='sampled area estimate' if stride>1 else 'computed 62m prediction cells')
    if not usable.any():return result
    x=v[usable];a=area[usable];idx=np.flatnonzero(usable)
    peak=idx[np.lexsort((rc[idx,1],rc[idx,0],-v[idx]))[0]]
    mu=trimmed_mean(x,a,p['trim']);high=usable&(v>=threshold) if threshold is not None else np.zeros(len(v),bool)
    connected,part=component(rc,high,area,stride,centre)
    lon,lat=coordinates(rc);cx,cy=coordinates(np.asarray(centre));dx=(lon-cx+180)%360-180
    blocks=(lat>=cy).astype(int)*2+(dx>=0).astype(int);retained=[]
    for block in range(4):
        keep=usable&(blocks!=block)
        if not keep.any():retained.append(0.);continue
        m=trimmed_mean(v[keep],area[keep],p['trim'])
        ratio=min(1.,max(0.,m/max(mu,1e-12))) if mu and mu>0 else 0.
        if connected>0:
            a2,_=component(rc,high&(blocks!=block),area,stride,centre)
            expected=connected-float(area[part&(blocks==block)].sum())
            ratio=min(ratio,min(1.,a2/max(expected,1e-12)) if expected>0 else 0.)
        retained.append(ratio)
    ring=distance(rc,centre)>=max(0,radius-496.)
    ring_nominal=math.pi*(radius**2-max(0,radius-496.)**2)/1e6
    ring_good=min(1.,float(area[ring&usable].sum())/max(ring_nominal,1e-12))
    ring_high_area=float(area[ring&part].sum())
    # At least 0.05 km² or 1% of ring, with adequate coverage; never one spike.
    boundary='open_high' if ring_good>=p['preliminary_valid'] and ring_high_area>=max(.05,.01*ring_nominal) else 'closed_at_threshold' if ring_good>=.95 and not np.any(high&ring) else 'unknown'
    result.update(mean=float(np.average(x,weights=a)),trimmed_mean=mu,median=quantile(x,a,.5),p25=quantile(x,a,.25),
        peak=float(v[peak]),peak_rc=rc[peak].tolist(),area_rc=rc[np.flatnonzero(part)[np.argmax(v[part])]].tolist() if part.any() else rc[peak].tolist(),
        area_km2=connected,stability=float(min(retained)),block_retention=retained,boundary=boundary,
        ring_valid_fraction=ring_good,ring_high_area_km2=ring_high_area,high_count=int(high.sum()))
    return result

def classify(s,final=False,policy=None):
    p=policy or POLICY
    if s.get('trimmed_mean') is None:return 'unavailable',['NO_USABLE_PREDICTION']
    if s['valid_fraction']<(p['final_valid'] if final else p['preliminary_valid']):return 'insufficient',['INSUFFICIENT_VALID_AREA']
    if s.get('source_suspect_fraction',0)>p['suspect_max']:return 'review',['SOURCE_ANOMALY']
    if final and s.get('source_checked_fraction',0)<p['source_checked_min']:return 'unverified',['SOURCE_CHECK_INCOMPLETE']
    if s.get('area_km2',0)<=0:return 'below_threshold',['NO_CONNECTED_HIGH_AREA']
    return ('eligible' if final else 'provisional'),[]

def score(s,reference,final=False,policy=None):
    p=policy or POLICY;status,reasons=classify(s,final,p)
    edges=np.asarray(reference['edges'],float);cdf=np.asarray(reference['cdf'],float)
    c=0. if s.get('trimmed_mean') is None else float(np.interp(s['trimmed_mean'],edges,cdf,left=0,right=1))
    a=float(np.clip(np.log1p(s.get('area_km2',0))/np.log1p(s['nominal_area_km2']),0,1))
    terms=dict(concentration=c,area=a,valid=s['valid_fraction'],stability=s['stability'])
    return dict(status=status,reasons=reasons,score=100*sum(p['weights'][k]*v for k,v in terms.items()),terms=terms,
                admissible=status in ('eligible','provisional'),score_semantics='heuristic regional priority, not probability')
