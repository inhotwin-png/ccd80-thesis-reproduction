"""Refit the selected CCD80 models and the two neural comparators on the published site splits."""
import argparse
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
import time
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parent
OXIDES = ("SiO2", "Al2O3", "FeO", "MgO", "CaO", "TiO2")

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def site_arrays():
    from lunar.data import Dataset
    from lunar.ext_inputs import load_observations
    from lunar.ext_evidence import load_support
    dataset = Dataset()
    batch = load_observations(dataset)
    with np.load(ROOT / "assets/class_grid144/inputs.npz") as values:
        assert values["ids"].tolist() == dataset.ids
        batch.update(CLASS6=values["CLASS6"], CLASS3=values["CLASS6"][:, :3],
                     CLASS_96s_site=values["CLASS_96s_cell"])
    return dataset, batch, load_support(dataset)

def fit_ccd80(recipe, batch, dataset, support, train, test, index, cache):
    from lunar.ext_inputs import subset
    if recipe["semantics"] == "sequential_residual_v1":
        from lunar.models import Pipeline
        valid = train[np.isfinite(dataset.y[train, index])]
        model = Pipeline(recipe).fit(dataset.x[valid], dataset.y[valid, index])
        return model.predict(dataset.x[test])
    from lunar.performance_pipeline import fit_cached as historical
    from lunar.combination_pipeline114 import fit_cached as mixed
    from lunar.logdiff151 import fit_cached as logdiff
    factory = {"fixed_six_crossfit_v104": historical, "unified_fixed_six_v114": mixed,
               "unified_fixed_six_v151": logdiff}[recipe["semantics"]]
    model = factory(recipe, subset(batch, train), dataset.y[train], dataset.site[train], support[train], cache)
    return model.predict(subset(batch, test))[:, index]

def fit_yang(recipe, x, y, test_x):
    from lunar.ext_estimators import Regressor
    model = Regressor("YangCNN", recipe["stages"][0]["params"]).fit(x, y)
    return np.maximum(0., model.predict(test_x))

def fit_shi(oxide, x, y, test_x):
    import math
    import random
    import torch
    from third_party.shi.model_resdcnn import ResDCNN
    def design(raw):
        if oxide not in ("TiO2", "FeO"):
            return raw
        extra = [[-math.atan((d/b-1.25)/(b-.037)), math.atan((a/b-.208)/(b+.108))]
                 for a,b,_,d,*_rest in raw.tolist()]
        return np.column_stack((raw, extra))
    x, test_x = design(x), design(test_x)
    xm, xs, ym, ys = x.mean(0), x.std(0), y.mean(), y.std()
    assert np.all(xs > 0) and ys > 0
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    train_x = tensor((x-xm)/xs).unsqueeze(1)
    train_y = tensor((y-ym)/ys).reshape(-1, 1)
    held_x = tensor((test_x-xm)/xs).unsqueeze(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    net = ResDCNN().train()
    opt = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(train_x), train_y)
        loss.backward()
        opt.step()
    net.eval()
    with torch.inference_mode():
        return net(held_x).ravel().double().numpy()*ys+ym

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("ccd80","yang","shi"), default="ccd80")
    p.add_argument("--oxide", choices=(*OXIDES,"all"), default="FeO")
    p.add_argument("--goal", choices=("MAE","RANK"), default="MAE")
    p.add_argument("--two-sites", action="store_true", help="also fit all 28 two-site splits")
    p.add_argument("--out", type=Path, default=Path("results/refit"))
    args=p.parse_args()
    if args.family!="ccd80" and (args.goal!="MAE" or args.two_sites):
        p.error("The published CNN comparisons use one-site folds and the MAE goal.")
    if args.family in ("shi","yang"):
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    from lunar.data import Dataset
    ds=Dataset()
    if args.family=="ccd80":
        from lunar.performance_pipeline import Cache
        ds,batch,support=site_arrays()
    split_list=[f for f in read("data/folds.json") if len(f["excluded_sites"])==1 or args.two_sites]
    config={(s["oxide"],s["selection_goal"]):s for s in read("config/selected_models.json")}
    signature = hashlib.sha256()
    source_files = [Path(__file__), *sorted((ROOT/"lunar").glob("*.py")), *sorted((ROOT/"assets").rglob("*.*")), *sorted((ROOT/"config").glob("*.json")), ROOT/"third_party/shi/model_resdcnn.py"]
    for path in source_files:
        signature.update(path.read_bytes())
    packages = ["numpy"] + (["scipy","scikit-learn"] if args.family != "shi" else []) + (["torch"] if args.family != "ccd80" else [])
    versions = {name:version(name) for name in packages}
    signature.update(json.dumps([sys.version,platform.platform(),platform.processor(),versions],sort_keys=True).encode())
    fingerprint = signature.hexdigest()
    for oxide in (OXIDES if args.oxide=="all" else [args.oxide]):
        index=OXIDES.index(oxide)
        cache=Cache(limit=256) if args.family=="ccd80" else None
        errors=[]
        folder=args.out / args.family / args.goal / oxide
        folder.mkdir(parents=True,exist_ok=True)
        for split in split_list:
            sites=split["excluded_sites"]
            mask=np.isin(ds.site,sites)
            train, test=np.flatnonzero(~mask),np.flatnonzero(mask)
            assert [ds.ids[i] for i in train]==split["train"] and [ds.ids[i] for i in test]==split["test"]
            start=time.perf_counter()
            dest=folder/("+".join(sites)+".json")
            if dest.is_file():
                prior=json.loads(dest.read_text(encoding="utf-8"))
                if prior.get("fingerprint")!=fingerprint or prior["excluded_sites"]!=sites or [r["area_id"] for r in prior["predictions"]]!=split["test"]:
                    raise ValueError("A different run already uses this output directory. Choose a new --out directory.")
                pred=[r["prediction_wt_pct"] for r in prior["predictions"]]
            else:
                if args.family=="ccd80":
                    pred=fit_ccd80(config[oxide,args.goal]["recipe"],batch,ds,support,train,test,index,cache)
                else:
                    finite_train=train[np.isfinite(ds.y[train,index])]
                    x,y,test_x=ds.x[finite_train],ds.y[finite_train,index],ds.x[test]
                    if args.family=="yang":pred=fit_yang(read("config/yang.json")["recipe"],x,y,test_x)
                    else:pred=fit_shi(oxide,x,y,test_x)
                assert np.isfinite(pred).all()
                # Each fold is committed independently. No incomplete JSON can be read as a completed fold.
                rows=[dict(area_id=ds.ids[j],prediction_wt_pct=float(value)) for j,value in zip(test,pred)]
                record=dict(oxide=oxide,family=args.family,goal=args.goal,excluded_sites=sites,predictions=rows,fingerprint=fingerprint)
                tmp=dest.with_suffix(".tmp")
                tmp.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
                tmp.replace(dest)
            if len(sites)==1:
                errors.extend(abs(float(v)-ds.y[j,index]) for j,v in zip(test,pred) if np.isfinite(ds.y[j,index]))
            print(f"{oxide:<6} {'+'.join(sites):<22} {time.perf_counter()-start:.2f} s",flush=True)
        print(f"{args.family} {oxide} pooled MAE = {np.mean(errors):.9f} wt%p ({len(errors)} windows)",flush=True)

if __name__=="__main__":
    main()
