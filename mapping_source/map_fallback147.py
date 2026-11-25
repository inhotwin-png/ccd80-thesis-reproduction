"""Per-oxide V3 >100 fallback; original predictions and MI identity stay auditable."""
import json,time
import numpy as np
from .common import OXIDES
from .map_store127 import MapPaused

POLICY=dict(version=147,trigger='V3 prediction > 100 wt%',resolved='use finite V2 prediction <= 100',unresolved='use the finite prediction closest to 100; ties retain V3',unavailable='retain V3',scope='only triggered oxide; unchanged frozen model and CLASS; existing model zero floor retained',accuracy='fallback is not independently validated chemical accuracy')
REASONS={0:'V3_NOT_TRIGGERED',1:'V2_RESOLVED_GT100',2:'V2_CLOSER_TO100',3:'V3_CLOSER_OR_TIED',4:'V2_NO_DATA',5:'V2_REQUIRED_INPUT_UNAVAILABLE',6:'V2_READ_ERROR',7:'V2_MODEL_ERROR'}
POLICY151=dict(POLICY,version=151,trigger='V3 prediction > 100 wt% or required V3 input/prediction unavailable',missing='use finite V2 prediction if available; otherwise remain unavailable')
REASONS[8]='V2_RESOLVED_MISSING_V3'

def choose(v3,v2):
    if v3 is None or not np.isfinite(v3):
        return (float(v2),2,8) if v2 is not None and np.isfinite(v2) else (float('nan'),0,4)
    if not np.isfinite(v3) or v3<=100:return float(v3),3 if np.isfinite(v3) else 0,0
    if v2 is None or not np.isfinite(v2):return float(v3),3,4
    if v2<=100:return float(v2),2,1
    if abs(v2-100)<abs(v3-100):return float(v2),2,2
    return float(v3),3,3

class Fallback:
    def __init__(self,ws,selection,stop=None,models=None,provider=None,backend='cpu'):
        self.ws=ws;self.selection=selection;self.stop=stop;self.models=models;self.provider=provider;self.retry_at=0.;self.subsets_ready=False;self.remote156=None
        self.backend156=backend
    def checkpoint(self):
        if self.stop is not None and self.stop.is_set():raise MapPaused()
        if hasattr(self,'heartbeat156'):self.heartbeat156()
    def apply(self,values,lon,lat,retry_missing=False,allowed=None):
        out=np.asarray(values,np.float32).copy();events=[]
        triggers=np.isfinite(out)&(out>100)
        if retry_missing:triggers|=~np.isfinite(out)
        if allowed is not None:triggers&=np.asarray(allowed,bool)
        if not triggers.any():return out,events
        if self.provider is None:
            from .map_v2_worker156 import RemoteFallback
            if self.remote156 is None:self.remote156=RemoteFallback(self.ws,self.selection,self.stop,backend=self.backend156)
            return self.remote156.apply(values,lon,lat,retry_missing,allowed)
        for i in np.flatnonzero(triggers.any(1)):
            self.checkpoint();names={n for j in np.flatnonzero(triggers[i]) for n in self.selection['models'][OXIDES[j]]['inputs']}
            point=None;error=None;code=4;source=None;attempted=False;error_chain=[]
            try:
                if time.monotonic()<self.retry_at:raise OSError('V2_TEMPORARY_READ_BACKOFF')
                if self.provider is None:
                    from .map_v2_inputs147 import V2Inputs
                    self.provider=V2Inputs(self.ws,self.stop)
                attempted=True;point=self.provider.point(float(lon[i]),float(lat[i]),names)
                source=getattr(self.provider,'last',None) if attempted else None
            except MapPaused:raise
            except Exception as e:
                self.checkpoint();error=type(e).__name__+': '+str(e);code=6
                cause=e;seen=set()
                while cause is not None and id(cause) not in seen:
                    seen.add(id(cause));error_chain.append(type(cause).__name__+': '+str(cause));cause=cause.__cause__ or cause.__context__
                # A skipped point must not extend the original failure's
                # backoff indefinitely while new batches keep arriving.
                if attempted:self.retry_at=time.monotonic()+60
                source=getattr(self.provider,'last',None) if attempted else None
            for j in np.flatnonzero(triggers[i]):
                oxide=OXIDES[j];v3=float(out[i,j]) if np.isfinite(out[i,j]) else None;v2=None;reason=code;detail=error
                if point is not None:
                    from .map_validity142 import usable
                    if not usable(point,self.selection['models'][oxide]['inputs']):reason=5
                    else:
                        try:
                            self.checkpoint()
                            if self.models is None:
                                from .map_models127 import load_models
                                self.models=load_models(self.ws,self.selection)
                            if not self.subsets_ready:
                                from .map_validity142 import optimize
                                optimize(self.models);self.subsets_ready=True
                            packet=self.models[oxide];batch={k:np.asarray(v)[None] for k,v in point.items()}
                            pred=np.asarray(packet['model'].predict(batch['raw'] if packet.get('engine_version') is None else batch),float)
                            if pred.shape!=(1,) or not np.isfinite(pred[0]):raise ValueError('V2_NONFINITE_MODEL_OUTPUT')
                            v2=float(pred[0]);_,_,reason=choose(v3,v2)
                        except MapPaused:raise
                        except Exception as e:reason=7;detail=type(e).__name__+': '+str(e)
                value,version,_=choose(v3,v2);out[i,j]=value
                events.append(dict(row=int(i),oxide=int(j),oxide_name=oxide,longitude=float(lon[i]),latitude=float(lat[i]),v3=v3,v2=v2,selected_mi=version,selected_value=float(out[i,j]) if np.isfinite(out[i,j]) else None,reason=reason,reason_text=REASONS[reason],error=detail,model_sha256=self.selection['models'][oxide].get('sha256'),source=source))
                events[-1].update(attempted=attempted,error_chain=error_chain,retry_pending=reason in (6,7))
        self.checkpoint();return out,events
    def close(self):
        if self.remote156 is not None:self.remote156.close()
        if self.provider is not None:self.provider.close()

def pack(values,events):
    values=np.asarray(values);n=len(values)
    if values.shape!=(n,6):raise ValueError('MI_PROVENANCE_SHAPE')
    versions=np.where(np.isfinite(values),3,0).astype(np.uint8);seen=set()
    for event in events:
        row,j=event['row'],event['oxide'];key=(row,j)
        if key in seen or not(0<=row<n and 0<=j<6) or (event['v3'] is not None and (not np.isfinite(event['v3']) or not event['v3']>100)):raise ValueError('MI_PROVENANCE_IDENTITY')
        seen.add(key);value,version,reason=choose(event['v3'],event['v2'])
        if version!=event['selected_mi'] or not np.array_equal(np.float32(value),values[row,j],equal_nan=True) or event['reason'] not in REASONS:raise ValueError('MI_PROVENANCE_PREDICTION')
        selected=float(values[row,j]) if np.isfinite(values[row,j]) else None
        if event['selected_value']!=selected or event['reason_text']!=REASONS[event['reason']]:raise ValueError('MI_PROVENANCE_DECISION')
        if event['v2'] is not None and (not np.isfinite(event['v2']) or event['reason']!=reason):raise ValueError('MI_PROVENANCE_REASON')
        if event['v2'] is None and event['reason'] not in (4,5,6,7):raise ValueError('MI_PROVENANCE_REASON')
        versions[row,j]=version
    # Deduplicate shared file identities and same-point sources. UTF-8 bytes
    # avoid NumPy Unicode's four-byte expansion for large map audit records.
    files=[];file_index={};sources=[];source_index={};compact=[]
    def intern(value,items,index):
        key=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        if key not in index:index[key]=len(items);items.append(value)
        return index[key]
    for event in events:
        row=dict(event);source=row.pop('source',None)
        if source is not None:
            source=dict(source)
            if 'files' in source:source['file_ids']=[intern(f,files,file_index) for f in source.pop('files')]
            row['source_index']=intern(source,sources,source_index)
        else:row['source_index']=None
        compact.append(row)
    meta=dict(policy=POLICY151 if any(e['v3'] is None for e in events) else POLICY,reason_codes=REASONS,events=compact,sources=sources,files=files,encoding='shared_sources_utf8')
    encoded=json.dumps(meta,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8')
    return dict(mi_version=versions,mi_fallback_rows=np.array([e['row'] for e in events],np.int32),mi_fallback_oxides=np.array([e['oxide'] for e in events],np.uint8),mi_v3_prediction=np.array([e['v3'] for e in events],np.float64),mi_v2_prediction=np.array([np.nan if e['v2'] is None else e['v2'] for e in events],np.float64),mi_fallback_reason=np.array([e['reason'] for e in events],np.uint8),mi_fallback_metadata=np.frombuffer(encoded,np.uint8))

def read_audit(archive,values):
    """Old chunks are V3; new chunks carry explicit per-oxide source arrays."""
    if 'mi_version' not in archive.files:return np.where(np.isfinite(values),3,0).astype(np.uint8),[]
    versions=archive['mi_version'];encoded=archive['mi_fallback_metadata'];meta=json.loads(encoded.tobytes().decode('utf8') if encoded.dtype==np.uint8 else str(encoded));events=meta['events']
    if meta.get('encoding')=='shared_sources_utf8':
        sources=[]
        for value in meta['sources']:
            source=dict(value)
            if 'file_ids' in source:source['files']=[meta['files'][i] for i in source.pop('file_ids')]
            sources.append(source)
        events=[dict({k:v for k,v in e.items() if k!='source_index'},source=sources[e['source_index']] if e['source_index'] is not None else None) for e in events]
    if meta.get('policy') not in (POLICY,POLICY151):raise ValueError('MI_PROVENANCE_POLICY')
    expected=pack(np.asarray(values).reshape(-1,6),events)
    if versions.dtype!=np.uint8 or not np.array_equal(versions,expected['mi_version'].reshape(np.shape(values))):raise ValueError('MI_PROVENANCE_VERSION')
    for key in ('mi_fallback_rows','mi_fallback_oxides','mi_v3_prediction','mi_v2_prediction','mi_fallback_reason'):
        if key not in archive.files or archive[key].dtype!=expected[key].dtype or not np.array_equal(archive[key],expected[key],equal_nan=True):raise ValueError('MI_PROVENANCE_ARRAY:'+key)
    return versions,events

def original_v3(archive,values):
    """Reuse V3 baselines, then apply the requested fallback once in the target job."""
    out=values.copy()
    if 'mi_version' in archive.files:
        _,events=read_audit(archive,values)
        for e in events:out[e['row'],e['oxide']]=e['v3']
    return out
