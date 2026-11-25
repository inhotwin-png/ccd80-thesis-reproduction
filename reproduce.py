"""Recreate the thesis tables and verify the native-grid candidate regions."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np
from lunar.data import Dataset
from lunar.region_score172 import statistics, score
from lunar.adaptive_grid131 import coordinates
from lunar.common import OXIDES

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def write_csv(folder, name, rows):
    with (folder / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def fixed_equation(oxide, raw):
    blue, near, _, red = raw[:4]
    if oxide == "FeO":
        angle = -math.atan((red / near - 1.39) / (near - .04))
        return 1.0708 * .0656 * math.exp(3.6681 * angle) - .3986
    angle = math.atan((blue / near - .208) / (near + .108))
    return .72 * angle**14.964

def validation(out):
    ds = Dataset()
    frozen = read("data/ccd80_fold_predictions.json")
    archived = {}
    for record in frozen:
        key = (record["oxide"], record["selection_goal"])
        excluded = tuple(record["excluded_sites"])
        pred = np.array(record["predictions"], dtype=float)
        assert np.array_equal(np.isfinite(pred), np.isin(ds.site, excluded))
        archived.setdefault(key,{})[excluded] = pred
    shi = {(r["oxide"],r["area_id"]):r["prediction_wt_pct"] for r in read("data/shi_heldout.json")}
    yang = {(r["oxide"],r["area_id"]):r["prediction_wt_pct"] for r in read("data/yang_original_heldout.json")}
    replay = {(r["oxide"],r["area_id"]):r["prediction_wt_pct"] for r in read("data/yang_replay.json")}
    table6, table8, windows = [], [], []
    for j, oxide in enumerate(OXIDES):
        valid = np.flatnonzero(np.isfinite(ds.y[:,j]))
        actual = ds.y[valid,j]
        row = {"oxide":oxide}
        for goal in ("MAE","RANK"):
            fold = archived[oxide,goal]
            assert len(fold)==36
            pred = np.array([fold[(ds.site[i],)][i] for i in valid])
            m = ds.metrics(oxide, fold)
            row[goal+"_LOLO_MAE_wt_pp"] = m["mae_area"]
            row[goal+"_LOLO_R2"] = 1-float(np.sum((actual-pred)**2)/np.sum((actual-actual.mean())**2))
            row[goal+"_pair_accuracy_pct"] = 100*m["rank_accuracy"]
            row[goal+"_correct_pairs"] = int(m["rank_correct"])
            row[goal+"_evaluated_pairs"] = int(m["rank_total"])
        table6.append(row)
        fits = []
        for i in valid:
            area = ds.ids[i]
            obs = ds.y[i,j]
            fits.append(dict(oxide=oxide, area_id=area, site=ds.site[i], reference_wt_pct=obs,
                ccd80_prediction=archived[oxide,"MAE"][(ds.site[i],)][i],
                shi_prediction=shi[oxide,area], yang_published_prediction=yang[oxide,area], yang_replay_prediction=replay[oxide,area],
                fixed_prediction=fixed_equation(oxide,ds.x[i]) if oxide in ("FeO","TiO2") else ""))
        windows.extend(fits)
        def error(field):
            return fmean(abs(r[field]-r["reference_wt_pct"]) for r in fits)
        table8.append(dict(oxide=oxide,valid_windows=len(valid),CCD80_MAE=error("ccd80_prediction"),
                Yang_published_MAE=error("yang_published_prediction"),Shi_MAE=error("shi_prediction"),
                fixed_equation_MAE=error("fixed_prediction") if oxide in ("FeO","TiO2") else "",
                Yang_release_replay_MAE=error("yang_replay_prediction")))
    write_csv(out,"table6.csv",table6)
    published = read("data/published_tables.json")["table8"]
    for r in table8:
        values=[r[k] for k in ("valid_windows","CCD80_MAE","Yang_published_MAE","Shi_MAE","fixed_equation_MAE")]
        assert [None if value=="" else round(value,3) for value in values]==published[r["oxide"]]
    write_csv(out,"table8.csv",table8)
    write_csv(out,"heldout_windows.csv",windows)
    print("Table 8 / MAE in wt percentage points")
    for r in table8:
        values = ["-" if r[k]=="" else f"{r[k]:.3f}" for k in ("CCD80_MAE","Yang_published_MAE","Shi_MAE","fixed_equation_MAE","Yang_release_replay_MAE")]
        print(r["oxide"], *values)
    return table8

def candidates(out):
    regions = read("data/representative_regions.json")
    policy = read("config/selection_policy.json")
    with (ROOT/"data/candidate_statistics.csv").open(encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    full, first = [], []
    with np.load(ROOT/"data/coarse_reference.npz") as refs:
        for oxide in OXIDES:
            entries=[]
            ref={"edges":refs[oxide+"_edges"],"cdf":refs[oxide+"_cdf"]}
            for item in (r for r in rows if r["oxide"]==oxide):
                s={name:float(item[name]) for name in ("nominal_area_km2","valid_fraction","stability","source_checked_fraction","source_suspect_fraction")}
                s.update(trimmed_mean=float(item["trimmed_mean_wt_pct"]),area_km2=float(item["connected_area_km2"]))
                scored=score(s,ref,final=True,policy=policy)
                if not scored["admissible"] or abs(scored["score"]-float(item["published_score"]))>1e-9:
                    raise ValueError("Candidate no longer agrees with the recorded source review: "+item["candidate_id"])
                entries.append((scored["score"],item))
            entries.sort(key=lambda entry:-entry[0])
            assert len(entries)==20
            for rank,(v,item) in enumerate(entries,1):
                assert rank==int(item["published_rank"])
                full.append(dict(oxide=oxide,rank=rank,latitude_n=float(item["latitude_n"]),
                     longitude_e=float(item["longitude_e"]),trimmed_mean_wt_pct=float(item["trimmed_mean_wt_pct"]),
                     valid_pct=100*float(item["valid_fraction"]),priority_score=v))
            _, top=entries[0]; region=regions[oxide]
            assert top["candidate_id"]==region["candidate_id"]
            # All five native-grid cutouts have complete archived source checks and zero rejected area.
            assert [float(top[k]) for k in ("source_checked_fraction","source_invalid_fraction","source_suspect_fraction")]==[1.,0.,0.]
            with np.load(ROOT/region["file"]) as cut:
                j=region["oxide_column"]; assert str(cut["oxide_order"][j])==oxide
                centre=cut["centre_rc"]; rc=cut["rc"]; values=cut["prediction"][:,j]; quality=cut["quality"][:,j]
                recalculated=statistics(rc,values,quality,centre,float(cut["radius_m"]),1,float(top["high_threshold"]),
                    policy,source={"codes":np.zeros(len(rc),dtype=np.uint8),"scope":"archived complete source check; zero invalid or suspect area"})
                priority=score(recalculated,ref,final=True,policy=policy)
                lon,lat=coordinates(centre)
                row=dict(oxide=oxide,rank=1,latitude_n=float(lat),longitude_e=float(lon),
                    trimmed_mean_wt_pct=recalculated["trimmed_mean"],valid_pct=100*recalculated["valid_fraction"],
                    priority_score=priority["score"])
                for k in row:
                    if k not in ("oxide","rank") and not math.isclose(row[k],full[-20][k],abs_tol=2e-8,rel_tol=0):
                        raise ValueError(f"{oxide} {k}: native {row[k]} vs archived {full[-20][k]}")
                first.append(row)
    published=read("data/published_tables.json")["table7"]
    for r in first:
        assert [round(r[k],3) for k in ("latitude_n","longitude_e","trimmed_mean_wt_pct","valid_pct","priority_score")]==published[r["oxide"]]
    write_csv(out,"table7.csv",first);write_csv(out,"appendix_C.csv",full)
    print("Table 7 / independently re-scored native cutouts")
    for r in first:print(r["oxide"],*(f"{r[k]:.3f}" for k in ("latitude_n","longitude_e","trimmed_mean_wt_pct","valid_pct","priority_score")))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results"))
    parser.add_argument("--maps",action="store_true",help="also draw Figures 2 and 4-7")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for filename,expected in read("SHA256.json").items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError("File integrity check failed for "+filename)
    from select_models import select
    write_csv(args.out,"selected_models.csv",select())
    validation(args.out)
    candidates(args.out)
    if args.maps:
        from figures import draw
        draw(args.out)
    print("Saved:",args.out.resolve())

if __name__=="__main__":
    main()
