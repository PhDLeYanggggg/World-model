"""Recompute fixed-budget counts and city-bootstrap intervals, offline."""

import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
ARMS = ("timing", "direction", "union")
FIELDS = ("anchor_eligible", *(a+"_"+k for a in ARMS
    for k in ("options", "standalone", "available", "top", "release", "pass")))


def quantile(x, p):
    x = sorted(x)
    i = p*(len(x)-1)
    j = int(i)
    return x[j]+(i-j)*(x[min(j+1, len(x)-1)]-x[j])


def verify(data):
    cities, draws = data["cities"], data["draws"]
    if len(cities) != 6 or len(draws) != 3000 or any(len(d) != len(cities) or min(d) < 0 or max(d) >= len(cities) for d in draws):
        raise ValueError("Unexpected resampling units")
    if any(len(s) != 3 for city in cities for s in city):
        raise ValueError("Require all three seeds for each scenario")
    rows = [r for city in cities for s in city for r in s]
    counts = lambda z: {k: sum(r[k] for r in z) for k in FIELDS}
    out = dict(views=len(rows), scenarios=sum(map(len, cities)), counts=counts(rows),
               complete_counts=counts([r for r in rows if r["complete"]]), paired={})
    for a in ARMS[1:]:
        for m in ("available", "standalone", "top"):
            z = [mean(mean(r[a+"_"+m]-r["timing_"+m] for r in s) for s in city) for city in cities]
            x = [mean(z[i] for i in d) for d in draws]
            out["paired"][a+"_minus_timing_"+m] = [mean(z), quantile(x, .025), quantile(x, .975)]
    return out


def main():
    m = json.loads((ROOT/"manifest.json").read_bytes())
    for k, v in m["members"].items():
        if hashlib.sha256((ROOT/k).read_bytes()).hexdigest() != v:
            raise ValueError("Archive checksum mismatch")
    print(json.dumps(verify(json.loads((ROOT/"evidence.json").read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
