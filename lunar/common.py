"""Portable paths and constants used by the app."""
from pathlib import Path
import hashlib, json
ASSETS=Path(__file__).resolve().parent.parent/"assets"
OXIDES=("SiO2","Al2O3","FeO","MgO","CaO","TiO2")
BANDS=("MV1","MV2","MV3","MV4","MV5","MN2","MN3","MN4")

def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
