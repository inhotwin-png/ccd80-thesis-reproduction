"""Apply the twelve frozen full-fit models to the measured input windows."""
import argparse
import csv
import hashlib
from pathlib import Path
import json
import joblib
import numpy as np
from refit import site_arrays, ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results/full_fit_windows.csv"))
    args=parser.parse_args()
    ds,features,_=site_arrays()
    specs=json.loads((ROOT/"config/selected_models.json").read_bytes())
    records=[]
    for entry in specs:
        oxide=entry["oxide"];goal=entry["selection_goal"]
        path=ROOT/"models"/goal/(oxide+".joblib")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["model_sha256"]:
            raise ValueError("Unexpected frozen model file: "+path.name)
        packet=joblib.load(path)
        assert packet["dataset"]==ds.hash and packet["oxide"]==oxide
        assert packet["training_ids"]==ds.ids and packet["full_training"]["excluded_sites"]==[]
        values=np.asarray(packet["model"].predict(features)).reshape(-1)
        assert len(values)==len(ds.ids) and np.isfinite(values).all()
        for area,value in zip(ds.ids,values):
            records.append(dict(oxide=oxide,model_goal=goal,window=area,full_fit_prediction_wt_pct=float(value)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,list(records[0]));writer.writeheader();writer.writerows(records)
    print(f"{len(records)} full-fit predictions written to {args.out}. Validation uses held-out predictions in reproduce.py.")

if __name__=="__main__":
    main()
