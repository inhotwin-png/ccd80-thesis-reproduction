"""Retrieve one named orbital input and check it against the archived SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda:data.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    manifest=Path(__file__).resolve().parent/"data/orbital_sources.json"
    catalog=json.loads(manifest.read_bytes())
    entries=dict(catalog["mi_tiles"])
    entries.update({r["duration"]+"/"+r["ratio"]:r for r in catalog["class_maps"]})
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("product",nargs="?",help="e.g. 296s/CaKa_Si")
    p.add_argument("--cache",type=Path,default=Path("orbital-cache"))
    args=p.parse_args()
    if args.product is None:
        print("\n".join(entries));return
    if args.product not in entries:p.error("Unknown product. Run without arguments to list IDs.")
    record=entries[args.product];expected=record["sha256"]
    args.cache.mkdir(parents=True,exist_ok=True)
    target=args.cache/(expected+Path(record["url"]).suffix)
    if target.is_file() and digest(target)==expected:
        print(target);return
    request=Request(record["url"],headers={"User-Agent":"CCD80-thesis-reproduction/1.0"})
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=args.cache,prefix=".partial-",delete=False) as output:
            temp=Path(output.name)
            with urlopen(request,timeout=60) as response:
                for chunk in iter(lambda:response.read(1048576),b""):
                    output.write(chunk)
        if digest(temp)!=expected:raise ValueError("The downloaded bytes differ from the recorded version.")
        os.replace(temp,target)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
    print(target)

if __name__=="__main__":
    main()
