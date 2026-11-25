"""Nested 62 m target centres, geodesic regions, and recorded search settings."""
import math, json
import numpy as np
from .common import digest, OXIDES
from .map_jobs127 import WIDTH, HEIGHT, STEP_DEG, bounds
from .map_inputs127 import RADIUS

SIDE=1024
DEFAULTS=dict(coarse_stride=8,scout_stride=2,initial_candidates=2000,regional_candidates=400,
    regional_count=200,detail_candidates=200,review_candidates=20,quantile=.99,
    separation_m=1000.,scout_radius_m=4000.,detail_radius_m=2000.,expansion_radii_m=[4000.,8000.],
    ring_width_m=248.,audit_count=1000,audit_side=32,audit_seed=4966201,
    scout_budget=64_000_000,detail_budget=16_000_000)

def parameters(values=None):
    p=json.loads(json.dumps(DEFAULTS));p.update(values or {})
    if set(p)!=set(DEFAULTS):raise ValueError('ADAPTIVE_SETTINGS_KEYS')
    for k in ('initial_candidates','detail_candidates','review_candidates','scout_budget','detail_budget'):
        if type(p[k]) is not int or p[k]<1:raise ValueError('ADAPTIVE_SETTINGS_'+k)
    if not 0<=p['regional_candidates']<=p['initial_candidates'] or p['regional_count']!=200:raise ValueError('ADAPTIVE_REGIONS')
    if not 0<=p['audit_count']<=10000 or p['audit_side']<1:raise ValueError('ADAPTIVE_AUDIT')
    if p['coarse_stride']!=8 or p['scout_stride']!=2:raise ValueError('ADAPTIVE_NESTED_GRID')
    if not 0<p['quantile']<1:raise ValueError('ADAPTIVE_QUANTILE')
    radii=[p['detail_radius_m'],*p['expansion_radii_m']]
    if len(radii)!=3 or radii!=sorted(set(radii)) or radii[0]<=0 or radii[-1]>100000:raise ValueError('ADAPTIVE_RADII')
    if not 0<p['scout_radius_m']<=100000 or not 0<p['separation_m']<=100000:raise ValueError('ADAPTIVE_RADIUS')
    if not 0<p['ring_width_m']<p['detail_radius_m']:raise ValueError('ADAPTIVE_RING')
    return p

def coordinates(rc):
    rc=np.asarray(rc);return -180+(rc[...,1]+.5)*STEP_DEG,90-(rc[...,0]+.5)*STEP_DEG

def distance(rc,centre):
    lon,lat=coordinates(rc);x,y=coordinates(np.asarray(centre));a=np.radians(lat);b=np.radians(y)
    h=np.sin((a-b)/2)**2+np.cos(a)*np.cos(b)*np.sin(np.radians(lon-x)/2)**2
    return 2*RADIUS*np.arcsin(np.sqrt(np.clip(h,0,1)))

def weight(rc):return np.cos(np.radians(coordinates(rc)[1]))
def region(rc):
    lon,lat=coordinates(rc)
    return np.minimum(9,((np.sin(np.radians(lat))+1)*5).astype(int))*20+np.minimum(19,((lon+180)/18).astype(int))
def bucket(rc):return np.asarray(rc)//SIDE
def key(br,bc):return f'r{int(br):04d}_c{int(bc):04d}'
def box_for(key_):
    r,c=(int(x[1:])*SIDE for x in key_.split('_'))
    return r,min(HEIGHT,r+SIDE),c,min(WIDTH,c+SIDE)
def in_extent(rc,extent):
    a,b,c,d=bounds(extent);return (rc[:,0]>=a)&(rc[:,0]<b)&(rc[:,1]>=c)&(rc[:,1]<d)

def coarse_tasks(extent):
    a,b,c,d=bounds(extent)
    rows=sorted(range(a//SIDE,(b-1)//SIDE+1),key=lambda r:abs((r+.5)*SIDE-HEIGHT/2))
    return [dict(key=key(r,col),stride=8) for r in rows for col in range(c//SIDE,(d-1)//SIDE+1)]

def circle_buckets(centre,radius):
    lon,lat=coordinates(np.asarray(centre));delta=radius/RADIUS*180/math.pi
    a=max(0,int(math.floor((90-lat-delta)/STEP_DEG))-1);b=min(HEIGHT,int(math.ceil((90-lat+delta)/STEP_DEG))+1)
    if abs(lat)+delta>=90:intervals=[(0,WIDTH)]
    else:
        dx=math.degrees(math.asin(min(1.,math.sin(math.radians(delta))/max(1e-12,math.cos(math.radians(lat))))))
        lo=int(math.floor((lon-dx+180)/STEP_DEG))-1;hi=int(math.ceil((lon+dx+180)/STEP_DEG))+1
        intervals=[]
        for shift in (-WIDTH,0,WIDTH):
            l=max(0,lo+shift);h=min(WIDTH,hi+shift)
            if l<h:intervals.append((l,h))
    return sorted({key(r,c) for l,h in intervals for r in range(a//SIDE,(b-1)//SIDE+1) for c in range(l//SIDE,(h-1)//SIDE+1)})

def region_tasks(candidates,radius,stride):
    out={}
    for candidate in candidates:
        centre=candidate['rc']
        for k in circle_buckets(centre,radius):
            task=out.setdefault(k,dict(key=k,stride=stride,circles=[]))
            if list(centre) not in task['circles']:task['circles'].append(list(centre))
    return [dict(out[k],radius_m=radius) for k in sorted(out)]

def audit_plan(extent,p):
    a,b,c,d=bounds(extent);rng=np.random.default_rng(p['audit_seed']);items={};side=p['audit_side']
    w,e,s,n=extent;lo,hi=np.sin(np.radians([s,n]));attempt=0
    while len(items)<p['audit_count'] and attempt<max(100,p['audit_count']*20):
        cell=attempt%200;attempt+=1;iy,ix=divmod(cell,20)
        x=w+(e-w)*(ix+rng.random())/20;y=math.degrees(math.asin(lo+(hi-lo)*(iy+rng.random())/10))
        r=max(a,min(max(a,b-side),int((90-y)/STEP_DEG)-side//2));col=max(c,min(max(c,d-side),int((x+180)/STEP_DEG)-side//2))
        rect=[r,min(b,r+side),col,min(d,col+side)];items[tuple(rect)]=rect
        if (b-a)*(d-c)<=side*side:break
    return list(items.values())

def audit_tasks(rects):
    out={}
    for a,b,c,d in rects:
        for r in range(a//SIDE,(b-1)//SIDE+1):
            for col in range(c//SIDE,(d-1)//SIDE+1):out.setdefault(key(r,col),[]).append([a,b,c,d])
    return [dict(key=k,stride=1,rectangles=v) for k,v in sorted(out.items())]

def task_points(task,extent):
    a,b,c,d=box_for(task['key']);stride=task['stride']
    ea,eb,ec,ed=bounds(extent);a=max(a,ea);b=min(b,eb);c=max(c,ec);d=min(d,ed)
    if 'rectangles' in task:
        rects=task['rectangles'];a=max(a,min(x[0] for x in rects));b=min(b,max(x[1] for x in rects));c=max(c,min(x[2] for x in rects));d=min(d,max(x[3] for x in rects))
    if 'circles' in task:
        radius=task['radius_m'];dr=int(math.ceil(radius/62))+3;centres=task['circles'];a=max(a,min(x[0] for x in centres)-dr);b=min(b,max(x[0] for x in centres)+dr+1);intervals=[]
        for centre in centres:
            lat=float(coordinates(np.asarray(centre))[1]);dc=int(math.ceil(radius/(62*max(.00001,math.cos(math.radians(min(89.9999,abs(lat)+radius/RADIUS*180/math.pi)))))))+3
            for shift in (-WIDTH,0,WIDTH):
                lo=max(c,centre[1]+shift-dc);hi=min(d,centre[1]+shift+dc+1)
                if lo<hi:intervals.append((lo,hi))
        if intervals:c=min(x[0] for x in intervals);d=max(x[1] for x in intervals)
        else:return np.empty((0,2),np.int32)
    if a>=b or c>=d:return np.empty((0,2),np.int32)
    rr,cc=np.meshgrid(np.arange(math.ceil(a/stride)*stride,b,stride,dtype=np.int32),np.arange(math.ceil(c/stride)*stride,d,stride,dtype=np.int32),indexing='ij')
    rc=np.c_[rr.ravel(),cc.ravel()];mask=in_extent(rc,extent)
    if 'circles' in task:
        selected=np.zeros(len(rc),bool)
        for centre in task['circles']:selected|=distance(rc,centre)<=task['radius_m']
        mask&=selected
    if 'rectangles' in task:
        selected=np.zeros(len(rc),bool)
        for a,b,c,d in task['rectangles']:selected|=(rc[:,0]>=a)&(rc[:,0]<b)&(rc[:,1]>=c)&(rc[:,1]<d)
        mask&=selected
    return rc[mask]

def candidate(oxide,rc,value,origin):
    rc=[int(x) for x in rc];lon,lat=coordinates(np.asarray(rc))
    return dict(id=digest([oxide,rc])[:20],oxide=oxide,rc=rc,lon=float(lon),lat=float(lat),predicted_peak=float(value),origin=origin)

def separated(rc,selected,metres):
    return not selected or bool(np.all(distance(np.array([x['rc'] for x in selected]),rc)>=metres))
