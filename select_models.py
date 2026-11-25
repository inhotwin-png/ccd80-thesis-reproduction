"""Re-select the six mapping models and six ranking models from the archived search scores."""
import csv
import gzip
import json
from pathlib import Path
from lunar.common import OXIDES

ROOT=Path(__file__).resolve().parent

def select():
    with gzip.open(ROOT/"data/search_scores.csv.gz","rt",encoding="utf-8",newline="") as stream:
        scores=list(csv.DictReader(stream))
    seed=json.loads((ROOT/"config/seed_targets.json").read_text())
    designated={c for group in seed.values() for c in group.values()}
    frozen={(m["oxide"],m["selection_goal"]):m for m in json.loads((ROOT/"config/selected_models.json").read_text())}
    choices=[]
    for oxide in OXIDES:
        for goal in ("MAE","RANK"):
            pool=[r for r in scores if r["oxide"]==oxide and r["map_eligible"]=="1"
                  and (r["candidate"] not in designated or r["candidate"]==seed[goal].get(oxide))
                  and (goal=="MAE" or (r["status"]=="complete" and r["pair_accuracy"]!=""))]
            winner=min(pool,key=lambda r:((0 if goal=="MAE" else -float(r["pair_accuracy"])),
                                        float(r["mae_wt_pp"]),r["candidate"]))
            recorded=frozen[oxide,goal]
            if (winner["candidate"],winner["model_sha256"])!=(recorded["candidate"],recorded["model_sha256"]):
                raise ValueError("Selection differs from the frozen job: "+oxide+" / "+goal)
            choices.append(dict(oxide=oxide,goal=goal,eligible_count=len(pool),candidate=winner["candidate"],
                        mae_wt_pp=float(winner["mae_wt_pp"]),pair_accuracy=winner["pair_accuracy"]))
    return choices

if __name__=="__main__":
    for row in select():
        print(row["oxide"],row["goal"],row["candidate"],"eligible:",row["eligible_count"])
