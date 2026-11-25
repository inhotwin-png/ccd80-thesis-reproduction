"""Measured MI MAP pixels and CLASS maps; bounded, georeferenced inference inputs.

Raw8 keeps the legacy arithmetic 62 m pixel-centre window. Context integration
uses clipped spherical pixel areas and the archived C64/C65 quadrature rule.
"""
from __future__ import annotations
import gzip, json, math, os, re, struct
from pathlib import Path
from collections import OrderedDict
import numpy as np
import requests
from .common import ASSETS, read, sha
from .map_store127 import MapWait, MapPaused

RADIUS=1737400.; KM=RADIUS*np.pi/180/1000; STEP=1/2048
IND=[0,1,2,3,4,6,7,8]; IMG_BYTES=92_274_688
RATIOS=['Mg_Al','Al_Si','Mg_Si','FeL_Si','CaKa_Si','O_Si']
BASE='https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-mi-5-map-v3.0'

def area(w,e,s,n):
    return (RADIUS/1000)**2*np.deg2rad(np.maximum(0,e-w))*np.maximum(0,np.sin(np.deg2rad(n))-np.sin(np.deg2rad(s)))

def spectral(x):
    cont=x[:,1]+(x[:,6]-x[:,1])*(950-749)/(1248-749)
    return np.c_[(x[:,1]-x[:,0])/.335/x[:,1],1-x[:,3]/cont,x[:,7]/x[:,1]]

def weighted_summary(x,a):
    w=a/a.sum(); z=np.c_[x,spectral(x)];mu=w@z
    q=[]
    for col in z.T:
        order=np.argsort(col,kind='stable');q.append(np.interp([.1,.25,.5,.75,.9],(np.cumsum(w[order])-.5*w[order])/w.sum(),col[order]))
    return dict(mean=mu,sd=np.sqrt(w@((z-mu)**2)),q=np.array(q))

def tiff_values(path):
    """The pinned CLASS TIFFs: little-endian, float32, 2 interleaved bands, strips."""
    b=Path(path).read_bytes()
    if b[:4]!=b'II*\x00':raise ValueError('CLASS_TIFF_FORMAT')
    start=struct.unpack_from('<I',b,4)[0];tags={}
    for i in range(struct.unpack_from('<H',b,start)[0]):
        p=start+2+12*i;tag,ty,n,offset=struct.unpack_from('<HHII',b,p)
        if ty not in (3,4,12):continue
        fmt,size={3:('H',2),4:('I',4),12:('d',8)}[ty]
        tags[tag]=struct.unpack_from('<'+fmt*n,b,p+8 if size*n<=4 else offset)
    if tags.get(256)!=(2048,) or tags.get(257)!=(1024,) or tags.get(259)!=(1,) or tags.get(277)!=(2,) or tags.get(284)!=(1,) or tags.get(339)!=(3,3):raise ValueError('CLASS_TIFF_LAYOUT_CHANGED')
    if not np.allclose(tags[33550][:2],[.17578125,.17578125]) or not np.allclose(tags[33922][3:5],[-180,90]):raise ValueError('CLASS_GEOMETRY_CHANGED')
    pixels=b''.join(b[o:o+n] for o,n in zip(tags[273],tags[279]))
    return np.frombuffer(pixels,dtype='<f4').reshape(1024,2048,2)

class Inputs:
    def __init__(self, workspace, source_folder, stop=None, notify=None):
        self.ws=workspace;self.source_folder=source_folder;self.stop=stop;self.notify=notify or (lambda x:None)
        self.catalog=read(ASSETS/'map127/sources.json');self.open=OrderedDict();self.classes={};self.provenance={}
        self.order={n:i for i,n in enumerate(self.catalog.get('context_tile_order',[]))};self.verified={}

    def fingerprint(self,path):
        st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
        if key not in self.verified:self.verified[key]=sha(path)
        return self.verified[key]

    def checkpoint(self):
        if self.stop is not None and self.stop.is_set():raise MapPaused()

    def download(self,url,name,max_bytes,expected=None):
        p=self.ws.path(name);p.parent.mkdir(parents=True,exist_ok=True)
        record=read(self.ws.records).get(name,{})
        expected=expected or record.get('sha256')
        if p.exists() and (expected is None or sha(p)==expected):return p
        if record.get('verified') and record.get('remote_id'):
            return self.ws.fetch(name,dict(id=record['remote_id'],size=record['size'],sha256=record['sha256']),self.stop)
        self.ws.reserve(max_bytes+65536,keep=(name,));tmp=p.with_suffix(p.suffix+'.part');tmp.unlink(missing_ok=True)
        self.notify(dict(phase='input_download',current=0,total=max_bytes))
        try:
            with requests.get(url,stream=True,timeout=(15,60)) as r:
                if r.status_code==404:return None
                r.raise_for_status();received=0
                with open(tmp,'wb') as f:
                    for block in r.iter_content(1024*1024):
                        self.checkpoint();received+=len(block)
                        if received>max_bytes:raise ValueError('MAP_SOURCE_OVERSIZE')
                        f.write(block);self.notify(dict(phase='input_download',current=received,total=int(r.headers.get('Content-Length',max_bytes))))
            if expected and sha(tmp)!=expected:raise ValueError('MAP_SOURCE_HASH')
            os.replace(tmp,p);return p
        finally:tmp.unlink(missing_ok=True)

    def class_maps(self):
        if self.classes:return
        for record in self.catalog['class_maps']:
            self.checkpoint();key=record['duration']+'_'+record['ratio'];name='sources/CLASS_'+key+'.tiff'
            local=Path(record.get('local_hint',''))
            if local.is_file() and sha(local)==record['sha256']:p=local
            else:
                p=self.download(record['url'],name,record['bytes'],record['sha256'])
                if p is None:raise MapWait('CLASS 관측 자료를 내려받을 수 없습니다.')
                self.ws.upload(name,self.source_folder,'map-input',stop=self.stop)
            self.classes[key]=tiff_values(p)
            self.provenance[key]=dict(sha256=record['sha256'],url=record['url'],grid='0.17578125 degree; ratio and formal error; not 62m measurements')

    def tile(self,north,west):
        self.checkpoint();west%=360
        if north>85 or north<=-85:return None
        def lat(v):return ('N' if v>=0 else 'S')+f'{abs(v):02d}'
        from .map_source136 import mi_stem
        stem=mi_stem(north,west)
        if stem in self.open:self.open.move_to_end(stem);return self.open[stem]
        rec=self.catalog.get('mi_tiles',{}).get(stem,{})
        local=Path(rec.get('local_hint',''))
        if local.is_file() and local.stat().st_size==IMG_BYTES and self.fingerprint(local)==rec.get('sha256'):
            path=local
        else:
            name='sources/'+stem+'.img';path=self.ws.path(name);gzname='sources/'+stem+'.img.gz'
            known=read(self.ws.records).get(name,{})
            if not path.exists() and known.get('verified') and not known.get('archive'):
                path=self.ws.fetch(name,dict(id=known['remote_id'],size=known['size'],sha256=known['sha256']),self.stop)
            if not (path.exists() and known.get('content_sha256')==sha(path)):
                label_name='sources/'+stem+'.lbl'
                label=self.download(f'{BASE}/lon{west:03d}/data/{stem}.lbl',label_name,150000)
                if label is None:return None
                text=label.read_text('ascii',errors='replace')
                for pattern in [r'BANDS\s*=\s*9',r'SCALING_FACTOR\s*=\s*2\.00000e-05',r'MAP_RESOLUTION\s*=\s*2048',r'IMAGE\s*=\s*\([^\n]*16777217']:
                    if not re.search(pattern,text,re.I):raise ValueError('MI_LABEL_FORMAT_CHANGED')
                self.ws.upload(label_name,self.source_folder,'map-input-label',stop=self.stop)
                url=f'{BASE}/lon{west:03d}/data/{stem}.img'
                gz=self.ws.path(gzname);gr=read(self.ws.records).get(gzname,{})
                if not gz.exists() and gr.get('verified'):
                    gz=self.ws.fetch(gzname,dict(id=gr['remote_id'],size=gr['size'],sha256=gr['sha256']),self.stop)
                if not gz.exists():
                    raw=self.download(url,name,IMG_BYTES,rec.get('sha256'))
                    if raw is None:return None
                    if raw.stat().st_size!=IMG_BYTES:raise ValueError('MI_SOURCE_SIZE')
                    self.ws.reserve(IMG_BYTES+1_000_000,keep=(name,));part=gz.with_suffix('.gz.part')
                    try:
                        with open(raw,'rb') as src,open(part,'wb') as dest,gzip.GzipFile(filename='',mode='wb',fileobj=dest,compresslevel=3,mtime=0) as archive:
                            while chunk:=src.read(1024*1024):self.checkpoint();archive.write(chunk)
                        os.replace(part,gz)
                    finally:part.unlink(missing_ok=True)
                receipt=self.ws.upload(gzname,self.source_folder,'map-input-gzip',stop=self.stop)
                self.ws.reserve(IMG_BYTES+65536,keep=(gzname,name));tmp=path.with_suffix('.img.part');total=0
                try:
                    with gzip.open(gz,'rb') as src,open(tmp,'wb') as dst:
                        while chunk:=src.read(1024*1024):
                            self.checkpoint();total+=len(chunk)
                            if total>IMG_BYTES:raise ValueError('MI_DECOMPRESSED_OVERSIZE')
                            dst.write(chunk)
                    if total!=IMG_BYTES:raise ValueError('MI_SOURCE_SIZE')
                    h=sha(tmp)
                    if rec.get('sha256') and h!=rec['sha256']:raise ValueError('MI_PINNED_HASH_CHANGED')
                    os.replace(tmp,path)
                    # The verified gzip expands to exactly this raw content. Preserve
                    # that proof; do not store a redundant 92 MB raw copy in Drive.
                    self.ws.record(name,content_sha256=h,sha256=h,size=total,archive=receipt,url=url,
                                   archive_content_sha256=h,remote_id=receipt['id'],verified=True)
                finally:tmp.unlink(missing_ok=True)
        im=np.memmap(path,dtype='>i2',mode='r',offset=2048*2048*4,shape=(9,2048,2048))
        self.provenance[stem]=dict(sha256=self.fingerprint(path),url=f'{BASE}/lon{west:03d}/data/{stem}.img')
        self.open[stem]=im
        while len(self.open)>6:
            _,old=self.open.popitem(last=False);old._mmap.close()
        return im

    def clear(self):
        for im in self.open.values():im._mmap.close()
        self.open.clear()

    def reset_provenance(self):
        self.provenance={k:v for k,v in self.provenance.items() if not k.startswith('MI_MAP_')}

    def pixels(self,lon,lat,half_km,centres=False):
        dy=half_km/KM;dx=dy/max(1e-6,math.cos(math.radians(lat)))
        if dx>30 or abs(lat)+dy>85:return None
        w,e,s,n=lon-dx,lon+dx,lat-dy,lat+dy
        # Native centres start at the integer-degree corner. Footprint edges
        # are half a native pixel west/north of it (archived C64 geometry).
        shift=0 if centres else STEP/2
        specs=[(nt,wt) for nt in range(math.floor(s-shift)+1,math.ceil(n-shift)+1) for wt in range(math.floor(w+shift),math.ceil(e+shift))]
        def key(t):
            nt,wt=t
            a=('N' if nt>=0 else 'S')+f'{abs(nt):02d}';v=nt-1;b=('N' if v>=0 else 'S')+f'{abs(v):02d}'
            from .map_source136 import mi_stem
            stem=mi_stem(nt,wt)
            return (self.order.get(stem,100000),nt,wt)
        chunks=[];weights=[];xs=[];ys=[];expected=0
        for nt,wt in sorted(specs,key=key):
            tw,tn=wt-shift,nt+shift
            if centres:
                rr=np.arange(max(0,math.ceil((nt-n)/STEP-1e-9)),min(2047,math.floor((nt-s)/STEP+1e-9))+1)
                cc=np.arange(max(0,math.ceil((w-wt)/STEP-1e-9)),min(2047,math.floor((e-wt)/STEP+1e-9))+1)
            else:
                rr=np.arange(max(0,math.floor((tn-n)/STEP)),min(2048,math.ceil((tn-s)/STEP)))
                cc=np.arange(max(0,math.floor((w-tw)/STEP)),min(2048,math.ceil((e-tw)/STEP)))
            if not len(rr) or not len(cc):continue
            rr,cc=np.meshgrid(rr,cc,indexing='ij');rr=rr.ravel();cc=cc.ravel();expected+=len(rr)
            im=self.tile(nt,wt)
            if im is None:continue
            z=np.column_stack([im[k,rr,cc] for k in IND])
            ww=np.maximum(w,tw+cc*STEP);ee=np.minimum(e,tw+(cc+1)*STEP)
            nn=np.minimum(n,tn-rr*STEP);ss=np.maximum(s,tn-(rr+1)*STEP)
            a=np.ones(len(rr)) if centres else area(ww,ee,ss,nn)
            valid=(z>0).all(1)&(a>0)
            chunks.append(z[valid].astype(float)*2e-5);weights.append(a[valid]);xs.append(((ww+ee)/2)[valid]);ys.append(((ss+nn)/2)[valid])
        if not chunks or not sum(map(len,chunks)):return None
        x=np.concatenate(chunks);a=np.concatenate(weights);coverage=float(a.sum()/(expected if centres else area(w,e,s,n)))
        return dict(x=x,a=a,coverage=min(1.,coverage),lon=np.concatenate(xs),lat=np.concatenate(ys),bounds=(w,e,s,n))

    def context(self,lon,lat,h):
        p=self.pixels(lon,lat,h)
        if p is None or p['coverage']<.95:return None
        x,a=p['x'],p['a'];mu=np.average(x,axis=0,weights=a);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=a))
        w,e,s,n=p['bounds'];block=np.minimum(7,((p['lon']-w)/(e-w)*8).astype(int))+8*np.minimum(7,((p['lat']-s)/(n-s)*8).astype(int))
        rng=np.random.default_rng(6401);sel=[];sw=[]
        for b in np.unique(block):
            ix=np.flatnonzero(block==b);count=min(128,len(ix));sel.extend(rng.choice(ix,count,replace=False));sw.extend([a[ix].sum()/count]*count)
        sx=x[sel];sw=np.array(sw)
        if len(sw)>256:
            index=np.searchsorted(np.cumsum(sw)/sw.sum(),(np.arange(256)+.5)/256);index,count=np.unique(index,return_counts=True);sx=sx[index];sw=count/256
        else:sw=sw/sw.sum()
        return np.r_[mu,sd,p['coverage']],np.pad(sx,((0,256-len(sx)),(0,0)),constant_values=.1),np.pad(sw,(0,256-len(sw)))

    def point(self,lon,lat,names):
        from .alme_features152 import expand_names
        names=expand_names(names)
        if abs(lat)>=85 or lon>=180:return None
        raw=self.pixels(lon,lat,.031,centres=True)
        if raw is None or raw['coverage']<.95:return None
        x=raw['x'].mean(0);out={'raw':x}
        self.class_maps()
        from .coordinate_inputs144 import cell_values
        for duration,field in [('296s','CLASS6'),('96s','CLASS_96s_cell')]:
            out[field]=cell_values(self.classes,lon,lat,duration,RATIOS[:6 if duration=='296s' else 4])
        out['CLASS3']=out['CLASS6'][:3];out['CLASS_296s_cell']=out['CLASS6']
        modes={name.split('|')[-1] for name in names if '|' in name}|set(names)
        if modes&{'M1','M2','M4'}:
            p=self.pixels(lon,lat,.25)
            if p and p['coverage']>=.95:
                a=weighted_summary(p['x'],p['a']);out['M1']=np.array([a['sd'][1],a['sd'][8],a['sd'][9]])
                out['M2']=np.array([a['q'][1,2]-a['mean'][1],a['q'][1,3]-a['q'][1,1],a['q'][9,4]-a['q'][9,0]])
                if 'M4' in modes:
                    p=self.pixels(lon,lat,1.)
                    if p and p['coverage']>=.95:
                        b=weighted_summary(p['x'],p['a']);sx=spectral(x[None])[0];out['M4']=np.array([x[1]-b['mean'][1],sx[0]-b['mean'][8],sx[1]-b['mean'][9],a['sd'][1]/a['mean'][1]-b['sd'][1]/b['mean'][1]])
        for tag,h in [('025',.25),('1',1.),('5',5.)]:
            if any(name in ('B_'+tag,'C_'+tag,'D_'+tag) for name in names):
                q=self.context(lon,lat,h)
                if q is not None:out.update({'broad_'+tag:q[0],'pixels_'+tag:q[1],'weights_'+tag:q[2]})
        return out

def required(name):
    if name.startswith('ALME152:'):
        from .alme_features152 import dependencies
        return dependencies(name)
    from .map_features145 import fields
    extra=fields(name)
    if extra is not None:return extra
    if '|' in name:
        m=name.split('|')[-1];return ['raw']+([m] if m in ('M1','M2','M4') else [])
    if name.startswith(('B_','C_','D_')):
        tag=name.split('_')[1];return ['raw','broad_'+tag]+(['pixels_'+tag,'weights_'+tag] if name.startswith(('C_','D_')) else [])
    if name in ('C3','C6'):return ['raw','CLASS6']
    if 'CLASS' in name:
        if name.startswith('CLASS_CELL96'):return ['raw','CLASS_96s_cell']
        return ['raw','CLASS3' if name=='CLASS3' else 'CLASS6']
    if name in ('M1','M2','M4'):return ['raw',name]
    return ['raw']

def usable(point,names):
    if point is None:return False
    return all(k in point and np.isfinite(point[k]).all() for name in names for k in required(name))
