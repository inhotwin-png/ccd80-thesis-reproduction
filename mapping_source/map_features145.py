"""Measured F / C69 spatial inputs. Original 127 inputs retain their arithmetic.

F: archived extract.py (2026-10-25), common nine-band DN>500, unweighted
center inclusion and numpy linear quantiles. C69: native intersection areas,
native ID ordering, 256 area quantiles, mask-aware SLIC. No chemistry labels.
"""
import math
import numpy as np
from .map_inputs127 import STEP,KM,IND,area
from .map_source136 import mi_stem
from .ext_spatial import NAMES

F_NAMES=('F0R_62','F4_62','F0R_250','F1_250','F2_250','F3_250','F4_250','F5_250')

def fields(name):
    if name.startswith('archive:'):
        _,tag,m=name.split(':');return ['archive_pixels_'+tag,'archive_weights_'+tag,'archive_raw_'+tag]+(['archive_'+m+'_'+tag] if m in ('M1','M2','M4') else [])
    if '|' in name:
        f,_,m=name.split('|');return ['raw']+([f] if f!='F0' else [])+([m] if m in ('M1','M2','M4') else [])
    if name in F_NAMES:return ['raw',name]
    if name in ('COORD2','MI_COORD2','GEO3'):return ['raw','coordinates']
    if name.startswith('CLASS_SITE96'):return ['raw','CLASS_96s_site']
    if name in ('B1','B5','S1','S5'):return ['raw','broad_'+name[-1]]
    if name in ('WAC3','MI_WAC3','MI_MATCHED_WAC3'):return ['raw','WAC2']+(['MI_WAC_support'] if name=='MI_MATCHED_WAC3' else [])
    if name=='SLIC':return ['raw','slic_mean','slic_var','slic_fraction','slic_coverage','slic_contrast']
    if name.startswith('Rings'):return ['raw','rings_mean','rings_var','rings_coverage']
    if name in NAMES or name.startswith('Kernel_'):
        k=name.removeprefix('Kernel_')
        if k=='B0':return ['raw']
        tags=['ring0','ring1','ring2'] if k in ('B3','KMERings') else ['full'+(k.split('_')[1] if k.startswith(('B1_','B2_')) else k.removeprefix('KME').removeprefix('Mean'))]
        return ['raw']+[prefix+tag for tag in tags for prefix in ('bag_','weight_')]
    return None

def halfwidth(names):
    from .alme_features152 import expand_names
    names=expand_names(names)
    h=.031
    for n in names:
        f=n.split('|')[0]
        if f in F_NAMES:h=max(h,.125 if f.endswith('250') else .031)
        if n in ('B1','S1'):h=max(h,1.)
        if n in ('B5','S5') or n in NAMES or n.startswith('Kernel_'):h=max(h,5.)
        if 'WAC' in n:h=max(h,.5)
        if n.startswith('archive:'):h=max(h,6.)
    return h

def f_observations(provider,lon,lat,width,requested=None):
    lon=lon%360;dy=(width/2000)/KM;dx=dy/max(1e-6,math.cos(math.radians(lat)))
    w,e,s,n=lon-dx,lon+dx,lat-dy,lat+dy;values=[];inside=[];selected=0
    mean_only=requested is not None and set(requested)<={'F4_'+str(width),'F0R_'+str(width)}
    for nt in range(math.floor(s)+1,math.floor(n)+2):
        for wt in range(math.floor(w),math.floor(e)+1):
            rr=np.arange(max(0,math.ceil((nt-n)/STEP-1e-9)),min(2047,math.floor((nt-s)/STEP+1e-9))+1)
            cc=np.arange(max(0,math.ceil((w-wt)/STEP-1e-9)),min(2047,math.floor((e-wt)/STEP+1e-9))+1)
            if not len(rr) or not len(cc):continue
            rr,cc=np.meshgrid(rr,cc,indexing='ij');rr=rr.ravel();cc=cc.ravel();selected+=len(rr);im=provider.tile(nt,wt)
            if im is None:continue
            dn=np.column_stack([im[k,rr,cc] for k in range(9)]);good=(dn>500).all(1);good8=(dn[:,IND]>500).all(1)
            if not np.array_equal(good,good8):return {} # Historical paired mask contract.
            values.append(dn[good].astype(float)*2e-5)
            if not mean_only:inside.append(((np.abs((wt+cc*STEP-lon)*KM*1000*math.cos(math.radians(lat)))<=31)&(np.abs((nt-rr*STEP-lat)*KM*1000)<=31))[good])
    if not values or not selected:return {}
    x=np.concatenate(values)
    if len(x)/selected<.8 or not len(x):return {}
    mean=x.mean(0)
    out={'F0R_'+str(width):mean[IND],'F4_'+str(width):mean}
    # The mean-only F0R/F4 heads do not consume quantiles or inner/outer means.
    # Keep the original reductions intact whenever those outputs are requested.
    if mean_only:return out
    inner=np.concatenate(inside)
    iqr=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
    if width==250:
        out.update(F1_250=np.median(x,axis=0)[IND],F2_250=np.r_[mean[IND],iqr[IND]],F5_250=np.r_[mean,iqr])
        if inner.any() and (~inner).any():
            a=x[inner][:,IND].mean(0);b=x[~inner][:,IND].mean(0);out['F3_250']=np.r_[a,b,a-b]
    return out

def region(provider,lon,lat,h=None,bounds=None,threshold=0):
    lon=(lon+180)%360-180
    if bounds is None:
        dy=h/1737.4*180/np.pi;dx=dy/max(1e-6,np.cos(np.deg2rad(lat)));w,e,s,n=lon-dx,lon+dx,lat-dy,lat+dy
    else:w,e,s,n=bounds
    if max(abs(s),abs(n))>85 or e-w>60:return None
    parts=[]
    for nt in range(math.floor(s-STEP/2)+1,math.ceil(n-STEP/2)+1):
        for wt in range(math.floor(w+STEP/2),math.ceil(e+STEP/2)):
            tw,tn=wt-STEP/2,nt+STEP/2
            rr=np.arange(max(0,math.floor((tn-n)/STEP)),min(2048,math.ceil((tn-s)/STEP)))
            cc=np.arange(max(0,math.floor((w-tw)/STEP)),min(2048,math.ceil((e-tw)/STEP)))
            if not len(rr) or not len(cc):continue
            im=provider.tile(nt,wt)
            if im is None:continue
            x=np.column_stack([im[k,rr[:,None],cc[None,:]].ravel() for k in IND]).astype(float)*2e-5
            ww=np.maximum(w,tw+cc*STEP);ee=np.minimum(e,tw+(cc+1)*STEP);nn=np.minimum(n,tn-rr*STEP);ss=np.maximum(s,tn-(rr+1)*STEP)
            a=area(ww[None,:],ee[None,:],ss[:,None],nn[:,None]).ravel();good=(x>threshold*2e-5).all(1)&(a>1e-15)
            tile_id=provider.order.get(mi_stem(nt,wt),len(provider.order)+(nt+90)*360+wt%360)
            ids=(tile_id*2048**2+rr[:,None]*2048+cc[None,:]).ravel()
            xc,yc=np.meshgrid(wt+cc*STEP,nt-rr*STEP)
            parts.append((ids,x,a,good,xc.ravel(),yc.ravel()))
    if not parts:return None
    ids,x,a,good,xc,yc=(np.concatenate([p[j] for p in parts]) for j in range(6));order=np.argsort(ids,kind='stable')
    return dict(ids=ids[order],x=x[order],area=a[order],valid=good[order],lon=xc[order],lat=yc[order],nominal=float(area(w,e,s,n)),bounds=(w,e,s,n))

def bag(r,weights):
    from .rank_inputs144 import quantile_bag
    return quantile_bag(r,np.where(r['valid'],weights,0))

def moments(r,weights):
    good=r['valid']&(weights>1e-15);x=r['x'][good];a=weights[good]
    if not len(x):return None
    mu=np.average(x,weights=a,axis=0);return mu,np.average((x-mu)**2,weights=a,axis=0)

def slic_features(r,lon,lat):
    from scipy.ndimage import binary_dilation
    from skimage.segmentation import slic
    w,e,s,n=r['bounds'];c0=int(np.floor(w/STEP-.5))+1;c1=int(np.ceil(e/STEP+.5))-1;r0=int(np.floor(-n/STEP-.5))+1;r1=int(np.ceil(-s/STEP+.5))-1
    shape=(r1-r0+1,c1-c0+1);iy=np.rint(-r['lat']/STEP).astype(int)-r0;ix=np.rint(r['lon']/STEP).astype(int)-c0
    mask=np.zeros(shape,bool);x=np.zeros((*shape,8));a=np.zeros(shape);mask[iy,ix]=r['valid'];x[iy,ix]=r['x'];a[iy,ix]=r['area']
    labels=slic(x,n_segments=32,compactness=.1,max_num_iter=10,sigma=0,spacing=(1.,np.cos(np.deg2rad(lat))),convert2lab=False,enforce_connectivity=True,min_size_factor=.5,max_size_factor=3,slic_zero=False,start_label=1,mask=mask,channel_axis=-1)
    cy=int(np.floor(-lat/STEP+.5))-r0;cx=int(np.floor(lon/STEP+.5))-c0
    if not(0<=cy<shape[0] and 0<=cx<shape[1] and mask[cy,cx]) or labels[cy,cx]<=0:return {}
    chosen=(labels==labels[cy,cx])&mask;edge=binary_dilation(chosen,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))&mask&~chosen;adjacent=np.isin(labels,np.unique(labels[edge]))&mask
    mu=np.average(x[chosen],weights=a[chosen],axis=0);var=np.average((x[chosen]-mu)**2,weights=a[chosen],axis=0);other=np.average(x[adjacent],weights=a[adjacent],axis=0) if adjacent.any() else mu
    return dict(slic_mean=mu,slic_var=var,slic_fraction=np.array([a[chosen].sum()/r['nominal']]),slic_coverage=np.array([r['area'][r['valid']].sum()/r['nominal']]),slic_contrast=np.array([np.linalg.norm(mu-other)]))

def spatial(provider,lon,lat,names):
    out={};previous=None;means=[];variances=[];coverages=[]
    for j,(h,tag) in enumerate(((.25,'025'),(1.,'1'),(5.,'5'))):
        r=region(provider,lon,lat,h)
        if r is None or r['area'][r['valid']].sum()<=0:return {}
        pair=bag(r,r['area'])
        if pair is None:return {}
        out['bag_full'+tag],out['weight_full'+tag]=pair;a=r['area'].copy();nominal=r['nominal']
        if previous is not None:
            pos=np.searchsorted(r['ids'],previous['ids'])
            if np.any(pos>=len(r['ids'])) or not np.array_equal(r['ids'][pos],previous['ids']):return {}
            a[pos]-=previous['area'];nominal-=previous['nominal']
            if a.min()<-1e-11:raise ValueError('SPATIAL_RING_AREA_PARTITION')
            a[np.abs(a)<1e-15]=0;a=np.maximum(a,0)
        cov=a[r['valid']].sum()/nominal
        if cov<=0:return {}
        pair=bag(r,a)
        if pair is None:return {}
        out['bag_ring'+str(j)],out['weight_ring'+str(j)]=pair;mu,var=moments(r,a);means.append(mu);variances.append(var);coverages.append(cov);previous=r
    out.update(rings_mean=np.array(means),rings_var=np.array(variances),rings_coverage=np.array(coverages))
    if 'SLIC' in names:out.update(slic_features(previous,(lon+180)%360-180,lat))
    return out

def augment(provider,point,lon,lat,names):
    if point is None:return None
    primary={n.split('|')[0] for n in names};out=point
    if primary&set(F_NAMES):
        for width in (62,250):
            wanted={n for n in primary if n in F_NAMES and n.endswith(str(width))}
            if wanted:out.update(f_observations(provider,lon,lat,width,requested=wanted))
    if names&{'COORD2','MI_COORD2','GEO3'}:out['coordinates']=np.array([lat,(lon+180)%360-180])
    if any(n.startswith('CLASS_SITE96') for n in names):out['CLASS_96s_site']=out['CLASS_96s_cell']
    for tag in ('1','5'):
        if names&{'B'+tag,'S'+tag}:
            q=provider.context(lon,lat,float(tag))
            if q is not None:out['broad_'+tag]=q[0]
    if any(n in NAMES or n.startswith('Kernel_') for n in names):out.update(spatial(provider,lon,lat,names))
    if any('WAC' in n for n in names):
        from .map_wac145 import observations
        out.update(observations(provider,lon,lat,names))
    for token in names:
        if token.startswith('archive:'):
            _,tag,m=token.split(':');out.update(archived(provider,lon,lat,tag,m,out['raw']))
    return out

def archived(provider,lon,lat,tag,m,local):
    """Original C64 stratified sample, C65 quadrature, C72 native-node extras."""
    from .map_inputs129 import summary
    from .map_inputs127 import spectral
    h={'025':.25,'1':1.,'5':5.}[tag];r=region(provider,lon,lat,h)
    if r is None or not r['valid'].any():return {}
    good=r['valid'];x=r['x'];a=r['area'];w,e,s,n=r['bounds']
    xc=(np.maximum(w,r['lon']-STEP/2)+np.minimum(e,r['lon']+STEP/2))/2
    yc=(np.maximum(s,r['lat']-STEP/2)+np.minimum(n,r['lat']+STEP/2))/2
    block=np.minimum(7,((xc-w)/(e-w)*8).astype(int))+8*np.minimum(7,((yc-s)/(n-s)*8).astype(int));rng=np.random.default_rng(6401);selected=[];weight=[]
    for b in np.unique(block[good]):
        ix=np.flatnonzero(good&(block==b));count=min(128,len(ix));selected.extend(rng.choice(ix,count,replace=False));weight.extend([a[ix].sum()/count]*count)
    selected=np.array(selected,int);weight=np.array(weight)
    if len(weight)>256:
        ix=np.searchsorted(np.cumsum(weight)/weight.sum(),(np.arange(256)+.5)/256);ix,counts=np.unique(ix,return_counts=True);selected=selected[ix];weight=counts/256
    else:weight=weight/weight.sum()
    mu,var=moments(r,a);count=len(selected);out={'archive_pixels_'+tag:np.pad(x[selected],((0,256-count),(0,0)),constant_values=.1),'archive_weights_'+tag:np.pad(weight,(0,256-count)), 'archive_raw_'+tag:np.r_[mu,np.sqrt(var),mu-local,a[good].sum()/r['nominal']]}
    if m in ('M1','M2','M4'):
        window=native_window(region(provider,lon,lat,h+1.))
        if window is None:return {}
        extras=[]
        for index in selected:
            st=window(r['lon'][index],r['lat'][index],.25);v=x[index]
            if st is None:return {}
            if m=='M1':extra=np.array([st['sd'][1],st['sd'][8],st['sd'][9]])
            elif m=='M2':extra=np.array([st['q'][1,2]-st['mean'][1],st['q'][1,3]-st['q'][1,1],st['q'][9,4]-st['q'][9,0]])
            else:
                large=window(r['lon'][index],r['lat'][index],1.)
                if large is None:return {}
                sp=spectral(v[None])[0]
                extra=np.array([v[1]-large['mean'][1],sp[0]-large['mean'][8],sp[1]-large['mean'][9],st['sd'][1]/st['mean'][1]-large['sd'][1]/large['mean'][1]])
            extras.append(extra)
        out['archive_'+m+'_'+tag]=np.pad(extras,((0,256-count),(0,0)))
    return out


def native_window(pool):
    """C72 native-node window, including its cKDTree quantile tie ordering.

    Training pools were the measured union of site ROIs. At a new coordinate
    the pool covers the requested ROI plus its 1 km contextual margin. The
    same operator uses only actual pixels; archived edge coverage can differ.
    """
    from scipy.spatial import cKDTree
    from .map_inputs129 import summary
    if pool is None or not pool['valid'].any():return None
    good=pool['valid'];lon=pool['lon'][good];lat=pool['lat'][good];x=pool['x'][good]
    km=1737.4*np.pi/180;cc=np.cos(np.deg2rad(float(np.mean(lat))))
    tree=cKDTree(np.c_[lon*km*cc,lat*km])
    def window(lo,la,h):
        dl=h/(km*np.cos(np.deg2rad(la)));da=h/km
        ids=np.array(tree.query_ball_point([lo*km*cc,la*km],max(dl*km*cc,da*km)+STEP*km,p=np.inf),int)
        a=area(np.maximum(lon[ids]-STEP/2,lo-dl),np.minimum(lon[ids]+STEP/2,lo+dl),np.maximum(lat[ids]-STEP/2,la-da),np.minimum(lat[ids]+STEP/2,la+da));ok=a>0
        return summary(x[ids[ok]],a[ok]) if ok.any() else None
    return window
