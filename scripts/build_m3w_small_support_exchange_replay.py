"""Recompute small support-exchange statistics from anonymous source scalars."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_incumbent_direction_replay as parent

base = parent.base
SOURCE = parent.SOURCE.parent / "small_support_exchange_v1"
DEFAULT = base.ROOT / "data/stage_cvpr2027_experiments/moving_small_support_exchange_replay_v1/anonymous_exchange.zip"


def inputs():
    parent.inputs()
    reg = SOURCE / "registration.json"
    a = json.loads((SOURCE / "analysis.json").read_text())
    r = json.loads(reg.read_text())
    if a["registration"] != base.sha(reg) or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require the frozen original evaluation")
    for p in [r["code"], r["model"], *r["parents"]]:
        if base.sha(base.ROOT / p["path"]) != p["sha256"]:
            raise ValueError("Changed support exchange lineage")
    return a


def build(out):
    a = inputs()
    modes = tuple(a["intervals"])
    groups = []
    for _, source in sorted(a["by_site"].items()):
        group = []
        for z in sorted(source, key=lambda r: r["seed"]):
            r = z["records"]
            counts = dict(views=len(r), undefined=sum(x["undefined"] for x in r),
                structural=sum(not x["full_easy"] for x in r),
                undefined_with_options=sum(x["undefined"] and x["legal_options"] > 0 for x in r),
                options=sum(x["legal_options"] for x in r), strong_ols=sum(x["strong_ols"] for x in r))
            metrics, effects = {}, {}
            for k in modes:
                rows = [(x, x["outcomes"][k]) for x in r]
                status = Counter(y["status"] for _, y in rows)
                metrics[k] = dict(witnesses=sum("exchanges" in y for _, y in rows),
                    exchanged=sum(y.get("exchanges", 0) for _, y in rows),
                    risk_rescues=sum(y.get("risk_rescue", False) for _, y in rows),
                    undefined_rescues=sum(x["undefined"] and y.get("risk_rescue", False) for x, y in rows),
                    strictly_joint=sum(y.get("strictly_joint", False) for _, y in rows),
                    strong_joint=sum(x["strong_ols"] and y.get("strictly_joint", False) for x, y in rows))
                metrics[k].update({f"status_{s}": status[s] for s in a["counts"][k]["status"]})
                effects[k] = 100*sum(y.get("lower", [0.])[0] for _, y in rows)/sum(x["reference"] for x in r)
            group.append(dict(counts=counts, metrics=metrics, effects=effects, capacity={}, bridge={}))
        groups.append(group)
    metrics = {k: {f: v for f, v in a["counts"][k].items() if f != "status"}
               | {f"status_{s}": v for s, v in a["counts"][k]["status"].items()} for k in modes}
    base.package(out, dict(groups=groups), dict(metrics=metrics, effects=a["intervals"],
        counts={k: v for k, v in a["counts"].items() if k not in modes and v}, capacity={}, bridge={}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
