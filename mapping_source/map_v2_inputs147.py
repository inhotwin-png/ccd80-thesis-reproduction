"""Native 512 ppd MI V2 observations at the same geographic support as V3."""
import math,hashlib,uuid
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .common import ASSETS,read,sha
from .map_inputs127 import Inputs,IND,KM,area,weighted_summary,spectral,RATIOS,tiff_values
from .coordinate_inputs144 import cell_values
from .map_v2_range147 import Transport,RangeFile,RangeError

BASE='https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_MI_multispectral_maps'
BANDS=('MV1','MV2','MV3','MV4','MV5','MN1','MN2','MN3','MN4')
WAVELENGTHS=(414,749,901,950,1001,1000,1049,1248,1548)
STEP=1/512

class V2Inputs:
    def __init__(self,ws,stop=None,transport=None,classes=None,band_workers=1,decoded_cache_bytes=32*1024*1024):
        self.ws=ws;self.stop=stop;self.transport=transport or Transport(stop);self.classes=classes or {};self.datasets={}
        self.token=uuid.uuid4().hex;self.last=None;self.windows=[]
        self.band_pools={};self.band_transports={}
        self.decoded162=OrderedDict();self.decoded_bytes162=0;self.decoded_limit162=max(0,int(decoded_cache_bytes))
        if band_workers>1:
            from concurrent.futures import ThreadPoolExecutor
            # Rasterio opener registries are thread-local. Every band's open,
            # reads and close must stay on its one owning thread.
            self.band_pools={k:ThreadPoolExecutor(max_workers=1,thread_name_prefix='V2Band'+str(k)) for k in range(9)}
            self.band_transports={k:Transport(stop,cache_bytes=4*1024*1024) for k in range(9)}
    def checkpoint(self):self.transport.checkpoint()
    def dataset(self,k):
        self.checkpoint()
        if k not in self.datasets:
            import rasterio
            name=f'Lunar_Kaguya_MIMap_Band{k+1}_{BANDS[k]}_{WAVELENGTHS[k]}nm_65N65S_512ppd.tif';url=BASE+'/'+name
            virtual=self.token+'_'+name
            def opener(path,mode='rb'):
                if path!=virtual or mode not in ('r','rb'):raise FileNotFoundError(path)
                return RangeFile(url,self.band_transports.get(k,self.transport))
            ds=rasterio.open(virtual,opener=opener,sharing=False)
            if ds.width!=184320 or ds.height!=66560 or ds.count!=1 or ds.dtypes!=('int16',) or ds.nodata!=-32768 or ds.scales!=(2e-5,) or ds.offsets!=(0.,) or tuple(ds.transform)!=(STEP,0.,-180.,0.,-STEP,65.,0.,0.,1.):
                ds.close();raise RangeError('V2_GEOTIFF_CONTRACT_CHANGED')
            # No Earth CRS transformation is performed. Native Moon degrees only.
            self.datasets[k]=ds
        return self.datasets[k]
    def read_bands(self,window):
        """Reuse exact native pixels, keeping all raster reads on band owners.

        Coarse points repeatedly hit the same small native pixel blocks. A
        bounded decoded cache avoids nine executor/GDAL round trips per point;
        slicing and the subsequent geographic area weights remain exact.
        """
        self.checkpoint()
        if window is None or not self.decoded_limit162:return self._read_bands162(window)
        from rasterio.windows import Window
        c0,r0,w,h=map(int,(window.col_off,window.row_off,window.width,window.height))
        if (c0,r0,w,h)!=(window.col_off,window.row_off,window.width,window.height):
            return self._read_bands162(window)
        if w<=0 or h<=0 or c0<0 or r0<0 or c0+w>184320 or r0+h>66560:
            return self._read_bands162(window)
        out=np.empty((h,w,9),np.int16);side=64
        for row in range(r0//side*side,r0+h,side):
            for col in range(c0//side*side,c0+w,side):
                self.checkpoint();key=(row,col);entry=self.decoded162.get(key)
                if entry is None:
                    height=min(side,66560-row);width=min(side,184320-col)
                    raw=self._read_bands162(Window(col,row,width,height))
                    block=raw.reshape(height,width,9)
                    # As with GDAL's decoded cache, wire ranges describe reads
                    # actually performed. Every point still records exact
                    # windows, source identities and its feature SHA-256.
                    entry=block;self.decoded162[key]=entry
                    self.decoded_bytes162+=block.nbytes
                    while self.decoded_bytes162>self.decoded_limit162 and self.decoded162:
                        _,old=self.decoded162.popitem(last=False);self.decoded_bytes162-=old.nbytes
                else:self.decoded162.move_to_end(key)
                block=entry
                a=max(row,r0);b=min(row+len(block),r0+h);x=max(col,c0);y=min(col+block.shape[1],c0+w)
                out[a-r0:b-r0,x-c0:y-c0]=block[a-row:b-row,x-col:y-col]
        return out.reshape(-1,9)
    def _read_bands162(self,window):
        def read_band(k):
            transport=self.band_transports.get(k,self.transport);transport.callback_error=None
            try:
                value=self.dataset(k).read(1,window=window).ravel()
                if transport.callback_error is not None:raise transport.callback_error
                return value
            except Exception:
                dataset=self.datasets.pop(k,None)
                if dataset is not None:dataset.close()
                if transport.callback_error is not None:raise transport.callback_error
                raise
        if not self.band_pools:return np.column_stack([read_band(k) for k in range(9)])
        futures=[self.band_pools[k].submit(read_band,k) for k in range(9)]
        # Await every band before reusing a dataset, even if one request failed.
        from concurrent.futures import wait
        wait(futures)
        try:return np.column_stack([f.result() for f in futures])
        finally:
            for k in range(9):
                t=self.band_transports[k];self.transport.identities.update(t.identities);self.transport.trace.update(t.trace)
    def nine(self,lon,lat,width=62):
        from rasterio.windows import Window
        self.checkpoint();h=width/2000;dy=h/KM;dx=dy/max(1e-6,math.cos(math.radians(lat)));lon=(lon+180)%360-180
        w,e,s,n=lon-dx,lon+dx,lat-dy,lat+dy
        if s< -65 or n>65 or e-w>=360:return None
        r0=max(0,int(math.floor((65-n)/STEP)));r1=min(66560,int(math.ceil((65-s)/STEP)))
        c0=int(math.floor((w+180)/STEP));c1=int(math.ceil((e+180)/STEP))
        if r0>=r1 or c0>=c1:return None
        if (r1-r0)*(c1-c0)>1_000_000:raise RangeError('V2_WINDOW_TOO_LARGE')
        parts=[];cursor=c0
        while cursor<c1:
            start=cursor%184320;count=min(c1-cursor,184320-start)
            dn=self.read_bands(Window(start,r0,count,r1-r0))
            rr,cc=np.meshgrid(np.arange(r0,r1),np.arange(cursor,cursor+count),indexing='ij');rr=rr.ravel();cc=cc.ravel()
            ww=np.maximum(w,-180+cc*STEP);ee=np.minimum(e,-180+(cc+1)*STEP);nn=np.minimum(n,65-rr*STEP);ss=np.maximum(s,65-(rr+1)*STEP)
            parts.append((dn,area(ww,ee,ss,nn),(ww+ee)/2,(ss+nn)/2))
            self.windows.append(dict(width_m=width,row=r0,column=start,height=r1-r0,width=count));cursor+=count
        dn,a,x,y=(np.concatenate([p[i] for p in parts]) for i in range(4))
        return dict(dn=dn,a=a,lon=x,lat=y,bounds=(w,e,s,n),nominal=float(area(w,e,s,n)))
    def pixels(self,lon,lat,h,centres=False):
        p=self.nine(lon,lat,h*2000)
        if p is None:return None
        valid=(p['dn'][:,IND]>0).all(1)&(p['dn'][:,IND]<32767).all(1)&(p['a']>0)
        if not valid.any():return None
        return dict(x=p['dn'][valid][:,IND]*2e-5,a=p['a'][valid],coverage=float(p['a'][valid].sum()/p['nominal']),lon=p['lon'][valid],lat=p['lat'][valid],bounds=p['bounds'])
    context=Inputs.context
    def class_maps(self,names=None):
        from .map_sources174 import acquire
        acquire(self.ws,self.classes,names,self.stop)
        if len(self.classes)!=len(getattr(self,'class_identities174',{})):
            self.class_identities174={r['duration']+'_'+r['ratio']:r['sha256'] for r in read(ASSETS/'map127/sources.json')['class_maps'] if r['duration']+'_'+r['ratio'] in self.classes}
    def point(self,lon,lat,names):
        from .alme_features152 import expand_names
        names=expand_names(names)
        self.windows=[];self.transport.trace={};names=set(names)
        for transport in self.band_transports.values():transport.trace={}
        self.last=dict(version='MI_MAP_02_USGS_512ppd',source_base=BASE,longitude=lon,latitude=lat,status='input_unavailable',windows=self.windows)
        if not (-65<lat<65):
            self.last['status']='outside_v2_coverage';return None
        p=self.pixels(lon,lat,.031)
        if p is None or p['coverage']<.95:
            self.last.update(status='native_mi_valid_area_below_95',coverage62=0. if p is None else p['coverage']);return None
        coverage62=p['coverage'];x=np.average(p['x'],axis=0,weights=p['a']);out={'raw':x}
        if any('CLASS' in n or n in ('C3','C6') for n in names):
            self.class_maps(names)
            from .map_sources174 import fields
            out.update(fields(self.classes,lon,lat))
        modes={n.split('|')[-1] for n in names}|names
        if modes&{'M1','M2','M4'}:
            p=self.pixels(lon,lat,.25)
            if p is not None and p['coverage']>=.95:
                z=weighted_summary(p['x'],p['a']);out['M1']=np.array([z['sd'][1],z['sd'][8],z['sd'][9]])
                out['M2']=np.array([z['q'][1,2]-z['mean'][1],z['q'][1,3]-z['q'][1,1],z['q'][9,4]-z['q'][9,0]])
                if 'M4' in modes:
                    large=self.pixels(lon,lat,1.)
                    if large is not None and large['coverage']>=.95:
                        b=weighted_summary(large['x'],large['a']);sx=spectral(x[None])[0]
                        out['M4']=np.array([x[1]-b['mean'][1],sx[0]-b['mean'][8],sx[1]-b['mean'][9],z['sd'][1]/z['mean'][1]-b['sd'][1]/b['mean'][1]])
        for tag,h in [('025',.25),('1',1.),('5',5.)]:
            if names&{'B_'+tag,'C_'+tag,'D_'+tag,'B'+tag,'S'+tag}:
                c=self.context(lon,lat,h)
                if c is not None:out.update({'broad_'+tag:c[0],'pixels_'+tag:c[1],'weights_'+tag:c[2]})
        primary={n.split('|')[0] for n in names}
        for width in (62,250):
            if primary&{'F0R_'+str(width),'F4_'+str(width)}:
                nine=self.nine(lon,lat,width)
                if nine is not None:
                    good=(nine['dn']>500).all(1)&(nine['dn']<32767).all(1)
                    if nine['a'][good].sum()/nine['nominal']>=.8:
                        mean=np.average(nine['dn'][good]*2e-5,axis=0,weights=nine['a'][good]);out['F4_'+str(width)]=mean;out['F0R_'+str(width)]=mean[IND]
        if names&{'COORD2','MI_COORD2','GEO3'}:out['coordinates']=np.array([lat,(lon+180)%360-180])
        # Unsupported spatial/WAC/archived contracts remain absent. The caller
        # retains V3 for that oxide instead of mixing a V3-derived feature.
        h=hashlib.sha256()
        for k in sorted(out):h.update(k.encode());h.update(np.asarray(out[k],dtype='<f8').tobytes())
        self.last=dict(version='MI_MAP_02_USGS_512ppd',input_sha256=h.hexdigest(),coverage62=coverage62,windows=self.windows,files=[dict(url=url,size=v[0],etag=v[1],last_modified=v[2]) for url,v in self.transport.identities.items()],ranges=list(self.transport.trace.values()),scale=2e-5,nodata=-32768,spatial_policy='native pixel intersection-area weighting; no V3 MI bands or interpolation')
        if getattr(self,'class_identities174',None):self.last['class_sources174']=self.class_identities174.copy()
        return out
    @staticmethod
    def pixels_coverage(p):return None if p is None else p.get('coverage')
    def close(self):
        if self.band_pools:
            futures=[self.band_pools[k].submit(ds.close) for k,ds in self.datasets.items()]
            try:
                for f in futures:f.result()
            finally:
                for pool in self.band_pools.values():pool.shutdown(wait=True)
                self.band_pools.clear()
        else:
            for ds in self.datasets.values():ds.close()
        for transport in self.band_transports.values():transport.close()
        self.band_transports.clear()
        self.datasets.clear();self.transport.close();self.decoded162.clear();self.decoded_bytes162=0
