"""Bounded native-source diagnostics; never substitutes a nearby observation."""
import math,re,json,hashlib,sqlite3,time,tempfile
from pathlib import Path
from collections import OrderedDict,Counter
import numpy as np
import requests
from .common import ASSETS,read,digest,sha
from .map_inputs127 import IND,STEP,KM,area,tiff_values,RATIOS
from .map_source136 import mi_stem
from .adaptive_grid131 import coordinates
from .region_score172 import POLICY
from .map_store127 import MapPaused

BASE='https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-mi-5-map-v3.0'
NATIVE_BYTES=92_274_688

def inspect_native(packet,policy=None):
    p=policy or POLICY
    if packet is None:return dict(code=1,reasons=['OUTSIDE_SOURCE_COVERAGE'],coverage=0.)
    dn=np.asarray(packet['dn']);a=np.asarray(packet['a'],float);nom=float(packet['nominal'])
    if dn.ndim!=2 or dn.shape[1]!=9 or len(a)!=len(dn):raise ValueError('SOURCE_NATIVE_SHAPE')
    # Current product labels explicitly identify these signed invalid values.
    invalid=np.isin(dn,[-32768,-30000,-23000,-22000,-21000,-20000])
    good=(dn[:,IND]>0).all(1)&~invalid[:,IND].any(1)&(dn[:,IND]<32767).all(1)
    coverage=float(a[good].sum()/max(nom,1e-12));reasons=[];warnings=[]
    if coverage<.95:reasons.append('REQUIRED_MI_VALID_AREA_BELOW_95')
    pair=(dn[:,4]>=0)&(dn[:,5]>=0)&~invalid[:,4]&~invalid[:,5]
    pair_area=float(a[pair].sum()/max(nom,1e-12));ratio=None
    if pair.any():
        means=np.average(dn[pair][:,[4,5]],axis=0,weights=a[pair])
        if means[0]>p['source_signal_dn']:
            ratio=float(means[1]/means[0])
            if pair_area>=.8 and not p['source_ratio_min']<=ratio<=p['source_ratio_max']:reasons.append('OVERLAPPING_1000NM_BANDS_DISAGREE')
        else:warnings.append('LOW_SIGNAL_DIAGNOSTIC_ONLY')
    if pair_area<.8:warnings.append('OVERLAP_BAND_CHECK_INCOMPLETE')
    if np.any(dn[:,IND]==-20000):warnings.append('SATURATION_IN_NATIVE_FOOTPRINT')
    if ((dn[:,IND]>0)&(dn[:,IND]<=500)).any():warnings.append('LOW_SIGNAL_PRESENT')
    code=1 if coverage<.95 else 2 if reasons else 4 if warnings else 0
    return dict(code=code,reasons=reasons,warnings=warnings,coverage=min(1.,coverage),overlap_ratio=ratio,
        overlap_coverage=pair_area,invalid_codes={str(k):int((dn[:,IND]==k).sum()) for k in (-32768,-30000,-23000,-22000,-21000,-20000,0,32767)},
        mean9=[float(np.average(dn[:,k][~invalid[:,k]],weights=a[~invalid[:,k]]))*2e-5 if (~invalid[:,k]).any() else None for k in range(9)],
        input_sha256=hashlib.sha256(dn.astype('<i2').tobytes()+a.astype('<f8').tobytes()).hexdigest(),
        scope='Native footprint diagnostic; spectral disagreement is a review flag, not proof of chemistry')

class NativeV3:
    def __init__(self,root,stop=None,local_roots=()):
        self.root=Path(root);self.stop=stop;self.local_roots=[self.root/'sources',*[Path(p) for p in local_roots]]
        self.catalog=read(ASSETS/'map127/sources.json');self.cache=OrderedDict();self.identities={};self.session=requests.Session()
        self.label_cache={};self.bytes_received=0;self.last_sources=set();self.failures={}
    def check(self):
        if self.stop is not None and self.stop.is_set():raise MapPaused()
    def label(self,stem,west):
        if stem in self.label_cache:return self.label_cache[stem]
        local=next((p/(stem+'.lbl') for p in self.local_roots if (p/(stem+'.lbl')).is_file()),None)
        url=f'{BASE}/lon{west:03d}/data/{stem}.lbl'
        if local:data=local.read_bytes()
        else:
            with self.session.get(url,stream=True,timeout=(15,30)) as r:
                r.raise_for_status();data=r.raw.read(200001)
            if len(data)>200000:raise ValueError('MI_LABEL_OVERSIZE')
        text=data.decode('ascii')
        for pat in (r'BANDS\s*=\s*9',r'SCALING_FACTOR\s*=\s*2\.00000e-05',r'MAP_RESOLUTION\s*=\s*2048',r'IMAGE\s*=\s*\([^\n]*16777217',r'INVALID_VALUE\s*=\s*\(-20000\s*,\s*-21000\s*,\s*-22000\s*,\s*-23000\)',r'OUT_OF_IMAGE_BOUNDS_VALUE\s*=\s*-30000'):
            if not re.search(pat,text,re.I):raise ValueError('MI_SOURCE_LABEL_CONTRACT')
        result=dict(url=url,sha256=hashlib.sha256(data).hexdigest(),invalid_values=[-20000,-21000,-22000,-23000,-30000],scale=2e-5)
        self.label_cache[stem]=result;return result
    def tile(self,north,west):
        self.check();west%=360
        if north>85 or north<=-85:return None
        stem=mi_stem(north,west)
        self.last_sources.add(stem)
        if time.monotonic()-self.failures.get(stem,-1e10)<60:raise OSError('SOURCE_READ_BACKOFF_RETRY_LATER')
        if stem in self.cache:self.cache.move_to_end(stem);return self.cache[stem]
        label=self.label(stem,west);record=self.catalog.get('mi_tiles',{}).get(stem,{})
        local=next((p/(stem+'.img') for p in self.local_roots if (p/(stem+'.img')).is_file()),None)
        if local is None:
            hint=Path(record.get('local_hint',''))
            if hint.is_file():local=hint
        url=f'{BASE}/lon{west:03d}/data/{stem}.img'
        if local:
            if local.stat().st_size!=NATIVE_BYTES:raise ValueError('MI_LOCAL_SIZE')
            h=sha(local)
            if record.get('sha256') and h!=record['sha256']:raise ValueError('MI_SOURCE_HASH')
            image=np.memmap(local,dtype='>i2',mode='r',offset=16777216,shape=(9,2048,2048));kind='local_verified_readonly'
        else:
            # The source server can ignore Range. One exact 92 MB product is
            # bounded in memory, never a global mosaic or interpolated input.
            with self.session.get(url,stream=True,timeout=(15,60)) as r:
                r.raise_for_status();parts=[];size=0
                for part in r.iter_content(1024*1024):
                    self.check();size+=len(part)
                    if size>NATIVE_BYTES:raise ValueError('MI_SOURCE_OVERSIZE')
                    parts.append(part)
                raw=b''.join(parts)
            if len(raw)!=NATIVE_BYTES:raise ValueError('MI_SOURCE_SHORT_READ')
            self.bytes_received+=len(raw);h=hashlib.sha256(raw).hexdigest()
            if record.get('sha256') and h!=record['sha256']:raise ValueError('MI_SOURCE_HASH')
            image=np.frombuffer(raw,dtype='>i2',offset=16777216).reshape(9,2048,2048);kind='bounded_original_product_in_memory'
        self.identities[stem]=dict(url=url,sha256=h,bytes=NATIVE_BYTES,label=label,mode=kind)
        self.cache[stem]=image
        while len(self.cache)>4:
            _,old=self.cache.popitem(last=False)
            if isinstance(old,np.memmap):old._mmap.close()
        return image
    def nine(self,lon,lat,width=62.,centres=True):
        self.last_sources=set()
        h=width/2000;dy=h/KM;dx=dy/max(1e-9,math.cos(math.radians(lat)))
        if abs(lat)>=85 or abs(lat)+dy>85 or dx>30:return None
        lon=(lon+180)%360-180;w,e,s,n=lon-dx,lon+dx,lat-dy,lat+dy
        shift=0 if centres else STEP/2;parts=[]
        for nt in range(math.floor(s-shift)+1,math.ceil(n-shift)+1):
            for wt in range(math.floor(w+shift),math.ceil(e+shift)):
                if centres:
                    rr=np.arange(max(0,math.ceil((nt-n)/STEP-1e-9)),min(2047,math.floor((nt-s)/STEP+1e-9))+1)
                    cc=np.arange(max(0,math.ceil((w-wt)/STEP-1e-9)),min(2047,math.floor((e-wt)/STEP+1e-9))+1)
                else:
                    rr=np.arange(max(0,math.floor((nt+shift-n)/STEP)),min(2048,math.ceil((nt+shift-s)/STEP)))
                    cc=np.arange(max(0,math.floor((w-wt+shift)/STEP)),min(2048,math.ceil((e-wt+shift)/STEP)))
                if not len(rr) or not len(cc):continue
                rr,cc=np.meshgrid(rr,cc,indexing='ij');rr=rr.ravel();cc=cc.ravel()
                try:im=self.tile(nt,wt)
                except MapPaused:raise
                except Exception:
                    self.failures[mi_stem(nt,wt%360)]=time.monotonic();raise
                if im is None:continue
                a=np.ones(len(rr)) if centres else area(np.maximum(w,wt-shift+cc*STEP),np.minimum(e,wt-shift+(cc+1)*STEP),np.maximum(s,nt+shift-(rr+1)*STEP),np.minimum(n,nt+shift-rr*STEP))
                parts.append((im[:,rr,cc].T.copy(),a))
        if not parts:return None
        dn=np.concatenate([x[0] for x in parts]);a=np.concatenate([x[1] for x in parts])
        # For centre sampling the number of requested centres is the footprint.
        return dict(dn=dn,a=a,nominal=float(len(a)) if centres else float(area(w,e,s,n)),bounds=(w,e,s,n))
    def close(self):
        for old in self.cache.values():
            if isinstance(old,np.memmap):old._mmap.close()
        self.cache.clear();self.session.close()

class Auditor:
    def __init__(self,job,stop,notify,policy=None,store=None):
        self.job=job;self.stop=stop;self.notify=notify;self.policy=policy or POLICY
        self.v3=NativeV3(job.ws.root,stop);self.v2=None;self.class_cache={}
        path=job.ws.path(job.base+'/source_checks172.sqlite');path.parent.mkdir(parents=True,exist_ok=True)
        self.path=path;self.db=store if store is not None else sqlite3.connect(path,timeout=30)
        self.external_store176=store is not None
        self.db.execute('CREATE TABLE IF NOT EXISTS checks(key TEXT PRIMARY KEY,record TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS sources(key TEXT PRIMARY KEY,record TEXT)');self.db.commit()
        self.model_set=job.manifest()['selection']['model_set']
    def class_check(self,lon,lat,names):
        from .map_validity142 import class_columns
        from .map_inputs127 import required
        from .coordinate_inputs144 import cell_index
        row,col=cell_index(lon,lat);records=[]
        for name in names:
            subset=class_columns(name)
            if subset is None:
                fields=[f for f in required(name) if f.startswith('CLASS')]
                if not fields:continue
                field=fields[0];subset=(field,tuple(range(3 if field=='CLASS3' else 4 if '96' in field else 6)))
            field,columns=subset;duration='96s' if '96' in field else '296s'
            for k in columns:
                key=(duration,RATIOS[k])
                if key not in self.class_cache:
                    meta=next(x for x in self.v3.catalog['class_maps'] if x['duration']==duration and x['ratio']==RATIOS[k])
                    paths=[self.job.ws.path('sources/CLASS_'+duration+'_'+RATIOS[k]+'.tiff'),Path(meta.get('local_hint',''))]
                    path=next((x for x in paths if x.is_file()),None)
                    if path is not None:
                        if sha(path)!=meta['sha256']:raise ValueError('CLASS_SOURCE_HASH')
                        self.class_cache[key]=tiff_values(path)
                    else:
                        # Eviction is not observation absence. Reacquire the
                        # pinned public product with exact size and hash checks.
                        with tempfile.TemporaryDirectory(prefix='class172_',dir=self.path.parent) as tmp:
                            self.job.ws.reserve(meta['bytes']+8192)
                            path=Path(tmp)/'source.tiff';size=0
                            with self.v3.session.get(meta['url'],stream=True,timeout=(15,60)) as response:
                                response.raise_for_status()
                                with path.open('wb') as f:
                                    for block in response.iter_content(1024*1024):
                                        if self.stop.is_set():raise MapPaused()
                                        size+=len(block)
                                        if size>meta['bytes']:raise ValueError('CLASS_SOURCE_OVERSIZE')
                                        f.write(block)
                            if size!=meta['bytes'] or sha(path)!=meta['sha256']:raise ValueError('CLASS_SOURCE_HASH')
                            self.class_cache[key]=tiff_values(path)
                    self.db.execute('INSERT OR IGNORE INTO sources VALUES(?,?)',('CLASS_'+duration+'_'+RATIOS[k],json.dumps(meta)))
                value,error=map(float,self.class_cache[key][row,col]);valid=math.isfinite(value) and value>0 and value!=9999
                records.append(dict(ratio=RATIOS[k],duration=duration,value=value if math.isfinite(value) else None,
                    error=error if math.isfinite(error) else None,usable=valid,row=row,column=col,
                    warning='FORMAL_ERROR_LARGE' if math.isfinite(error) and error>=value>0 else None))
        return records
    def inspect(self,rc,versions,names,oxide,phase):
        from .alme_features152 import expand_names
        names=sorted(expand_names(set(names)));lon,lat=coordinates(rc);codes=[];reasons=Counter();records=[]
        for i,(r,version,x,y) in enumerate(zip(rc,versions,lon,lat)):
            if self.stop.is_set():raise MapPaused()
            key=digest([self.policy,self.model_set,oxide,r.tolist(),int(version),names])
            previous=self.db.execute('SELECT record FROM checks WHERE key=?',(key,)).fetchone()
            if previous:record=json.loads(previous[0])
            else:
                try:
                    if int(version)==2:
                        if self.v2 is None:
                            from .map_v2_inputs147 import V2Inputs
                            self.v2=V2Inputs(self.job.ws,self.stop)
                        packet=self.v2.nine(float(x),float(y))
                    else:packet=self.v3.nine(float(x),float(y))
                    record=inspect_native(packet,self.policy);sources=set(self.v3.last_sources) if int(version)!=2 else set()
                    from .map_inputs127 import required
                    fields={f for n in names for f in required(n)};widths=set()
                    feature_checks=[]
                    for width in (62,250):
                        if any(f.startswith(('F0R_','F1_','F2_','F3_','F4_','F5_')) and f.endswith('_'+str(width)) for f in fields):
                            fp=self.v2.nine(float(x),float(y),width) if int(version)==2 else self.v3.nine(float(x),float(y),width)
                            sources.update(self.v3.last_sources if int(version)!=2 else ())
                            if fp is None:fc=dict(width_m=width,coverage=0.,paired_mask=False)
                            else:
                                dn=fp['dn'];good=(dn>500).all(1)&(dn<32767).all(1);eight=(dn[:,IND]>500).all(1)&(dn[:,IND]<32767).all(1)
                                fc=dict(width_m=width,coverage=float(fp['a'][good].sum()/max(fp['nominal'],1e-12)),paired_mask=bool(np.array_equal(good,eight)))
                            feature_checks.append(fc)
                            if fc['coverage']<.8 or (int(version)!=2 and not fc['paired_mask']):record.update(code=1,reasons=record['reasons']+['REQUIRED_F_OBSERVATION_UNAVAILABLE'])
                    if fields&{'M1','M2','M4'}:widths.add(500)
                    if 'M4' in fields:widths.add(2000)
                    for tag,width in [('025',500),('1',2000),('5',10000)]:
                        if fields&{'broad_'+tag,'pixels_'+tag,'weights_'+tag}:widths.add(width)
                    context=[]
                    for width in sorted(widths):
                        context_packet=self.v2.nine(float(x),float(y),width) if int(version)==2 else self.v3.nine(float(x),float(y),width,centres=False)
                        ctx=inspect_native(context_packet,self.policy);ctx['width_m']=width;context.append(ctx);sources.update(self.v3.last_sources if int(version)!=2 else ())
                        if ctx['code'] in (1,2):record['code']=1 if record['code']==1 or ctx['code']==1 else 2;record['reasons']+=['CONTEXT_'+reason for reason in ctx['reasons']]
                    classes=self.class_check(float(x),float(y),names)
                    if any(not c['usable'] for c in classes):record.update(code=1,reasons=record['reasons']+['REQUIRED_CLASS_SOURCE_INVALID'])
                    for stem in sources:self.db.execute('INSERT OR IGNORE INTO sources VALUES(?,?)',(stem,json.dumps(self.v3.identities[stem])))
                    supported={'raw','M1','M2','M4','CLASS3','CLASS6','CLASS_96s_cell','CLASS_96s_site','CLASS_296s_cell','coordinates'}
                    unknown=[f for f in fields if f not in supported and not f.startswith(('broad_','pixels_','weights_','F0R_','F1_','F2_','F3_','F4_','F5_'))]
                    if unknown and record['code'] not in (1,2):record.update(code=3,reasons=record['reasons']+['ADDITIONAL_SOURCE_CONTRACT_NOT_AUDITED'])
                    record.update(rc=r.tolist(),version=int(version),classes=classes,context=context,feature_checks=feature_checks,unaudited_fields=unknown,source_ids=sorted(sources),source_policy='selected saved MI version; no lower-value substitution')
                    if int(version)==2:record['v2_files']=[dict(url=u,size=v[0],etag=v[1],last_modified=v[2]) for u,v in self.v2.transport.identities.items()]
                    self.db.execute('INSERT OR REPLACE INTO checks VALUES(?,?)',(key,json.dumps(record,allow_nan=False)))
                except MapPaused:raise
                except Exception as e:
                    # Transient failures are not cached as permanent no-data.
                    record=dict(code=3,reasons=['SOURCE_READ_UNVERIFIED'],error=type(e).__name__+': '+str(e),rc=r.tolist(),version=int(version))
            codes.append(record['code']);reasons.update(record.get('reasons',[]));reasons.update(record.get('warnings',[]))
            if record['code'] in (1,2,3) and len(records)<32:records.append(record)
            if i%64==0:
                self.job.ws.reserve(2_000_000)
                self.db.commit();self.notify(dict(phase=phase,current=i+1,total=len(rc),operation='native_source_quality',oxide=oxide,progress_scope='item'))
        self.db.commit()
        return dict(codes=np.asarray(codes,np.uint8),reasons=dict(reasons),examples=records,
            scope='Every usable saved prediction footprint checked in its selected MI version; CLASS dependencies only; source discrepancy is diagnostic')
    def close(self):
        self.db.commit();self.db.close();self.v3.close()
        if self.v2:self.v2.close()
        name=self.path.relative_to(self.job.ws.root).as_posix() if self.path.is_relative_to(self.job.ws.root) else str(self.path)
        if not self.external_store176:self.job.ws.record(name,'result')
