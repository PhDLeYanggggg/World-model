"""Recompute AV2 fixed-count and calibrated outcomes without the research repository."""

from hashlib import sha256
from itertools import product
import json
from pathlib import Path


def read(name):
    return json.loads(Path(name).read_bytes())


def percentile(x, p):
    at = p*(len(x)-1)
    i = int(at)
    return x[i]+(at-i)*(x[min(i+1, len(x)-1)]-x[i])


def interval(values):
    n = len(values)
    draws = sorted(sum(v)/n for v in product(values, repeat=n))
    return dict(point_pp=sum(values)/n, exact_ci95_pp=[percentile(draws, p) for p in (.025, .975)], draws=len(draws))


def verify(evidence):
    cities = evidence["anonymous_source_cells"]
    arms = evidence["arms"]
    assert len(cities) == 6 and all(c["folds"] == 3 for c in cities)
    assert sum(c["views"] for c in cities) == evidence["views"]
    fields = ("qualified", "risk_fail", "floor_fail", "easy_over")
    compare = {}
    for arm in arms:
        compare[arm] = dict(top5={k: sum(c["top5"][arm][k] for c in cities) for k in fields},
            calibrated={k: sum(c["calibrated"][arm][k] for c in cities) for k in ("released", *fields)})
    contrasts = {}
    for arm in ("uniform_one", "all_legal", "transferred_cost"):
        contrasts[arm] = {k: interval([100.*(c["top5"]["causal_selected"][k]-c["top5"][arm][k])/c["budget"]
                                      for c in cities]) for k in ("qualified", "risk_fail")}
    return dict(views=evidence["views"], distinct_development_scenarios=evidence["scenarios"],
        top5_budget=sum(c["budget"] for c in cities), counts=compare, selected_minus=contrasts)


if __name__ == "__main__":
    manifest = read("manifest.json")
    for name, digest in manifest["members"].items():
        if sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    print(json.dumps(verify(read("evidence.json")), indent=2))
