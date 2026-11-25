"""One spatial CLASS contract for training and map inference; legacy packs stay frozen."""
from functools import lru_cache
import numpy as np
from .common import ASSETS,read,sha,digest,OXIDES

SCHEMA='CLASS_coordinate_cell_v1_144'

def cell_index(lon,lat):
    return min(1023,max(0,int((90-lat)/.17578125))),int(((lon+180)%360)/.17578125)

def cell_values(classes,lon,lat,duration,ratios):
    row,col=cell_index(lon,lat)
    v=np.array([classes[duration+'_'+r][row,col,0] for r in ratios],float)
    v[(v<=0)|(v==9999)|~np.isfinite(v)]=np.nan
    return v

@lru_cache(maxsize=1)
def contract():
    doc=read(ASSETS/'class_grid144/contract.json')
    body={k:v for k,v in doc.items() if k!='hash'}
    if doc.get('schema')!=SCHEMA or digest(body)!=doc.get('hash'):raise ValueError('CLASS_GRID_CONTRACT_HASH')
    if sha(ASSETS/'class_grid144/inputs.npz')!=doc['observations_sha256']:raise ValueError('CLASS_GRID_INPUT_HASH')
    if sha(ASSETS/'map127/sources.json')!=doc['map_source_catalog_sha256']:raise ValueError('CLASS_GRID_SOURCE_CATALOG_CHANGED')
    return doc

def batch(dataset,semantics):
    from .ext_inputs import load_observations
    from .reevaluation_pipeline import observations
    doc=contract()
    if doc['dataset_sha256']!=dataset.hash:raise ValueError('CLASS_GRID_DATASET_MISMATCH')
    b=observations(dataset) if semantics=='fixed_archived_six_v105' else load_observations(dataset)
    with np.load(ASSETS/'class_grid144/inputs.npz',allow_pickle=False) as z:
        if list(z['ids'])!=dataset.ids:raise ValueError('CLASS_GRID_AREA_ORDER')
        b=dict(b,CLASS6=z['CLASS6'].copy(),CLASS3=z['CLASS6'][:,:3].copy(),CLASS_96s_site=z['CLASS_96s_cell'].copy())
        if not np.array_equal(b['CLASS6'],b['CLASS_296s_cell']) or not np.array_equal(b['CLASS_96s_site'],b['CLASS_96s_cell']):raise ValueError('CLASS_GRID_CELL_INPUT_MISMATCH')
    return b

def full_training(dataset,recipe):
    return dict(mode='full_available_area_refit',excluded_sites=[],area_ids=dataset.ids,sites=dataset.sites,
        valid_target_ids={o:[dataset.ids[i] for i in np.flatnonzero(np.isfinite(dataset.y[:,j]))] for j,o in enumerate(OXIDES)},
        recipe_sha256=digest(recipe),dataset_sha256=dataset.hash,input_contract=contract()['hash'])

def validate_packet(packet,selected=None):
    from .data import Dataset
    d=Dataset();expected=full_training(d,packet['recipe'])
    if packet.get('input_contract')!=contract()['hash'] or packet.get('full_training')!=expected or packet.get('training_ids')!=d.ids:raise ValueError('MAP_REQUIRES_FULL_COORDINATE_TRAINING')
    model=packet['model'];pipe=getattr(model,'pipeline',None)
    if pipe is not None and hasattr(pipe,'training_sites') and list(pipe.training_sites)!=d.sites:raise ValueError('MAP_REJECTS_HELDOUT_MODEL')
    if selected and selected.get('input_contract')!=contract()['hash']:raise ValueError('MAP_TRAINING_INPUT_CONTRACT_MISMATCH')
    return packet
