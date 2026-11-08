"""Anonymous scalar replay of OpenDD timing and recipient studies."""

import hashlib
import json
from pathlib import Path

import numpy as np


def aggregate(studies):
    out = {}
    for name, z in studies.items():
        sites = z["sites"]
        if len(sites) != 7 or any(len(s) != 3 for s in sites):
            raise ValueError("Expected seven sites, each with three seed views")
        counts = {k: {f: sum(c["counts"][k][f] for s in sites for c in s) for f in fields}
                  for k, fields in sites[0][0]["counts"].items()}
        rng = np.random.default_rng(817)
        effects = {}
        for k in z["effect_order"]:
            if not z["sequential_draws"]:
                rng = np.random.default_rng(817)
            v = np.asarray([np.mean([c["effects"][k] for c in s]) for s in sites])
            if not np.isfinite(v).all():
                raise ValueError("Expected all seven site effects")
            draw = rng.integers(len(sites), size=(3000, len(sites)))
            effects[k] = dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
        out[name] = dict(counts=counts, effects=effects)
    return out


def verify(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Study roster changed")
    for k, a in actual.items():
        b = expected[k]
        if a["counts"] != b["counts"] or a["effects"].keys() != b["effects"].keys():
            raise ValueError("Count or interval roster changed")
        for name, x in a["effects"].items():
            for col in ("point", "interval95"):
                np.testing.assert_allclose(x[col], b["effects"][name][col], atol=1e-12, rtol=0.)


def main():
    root = Path(__file__).resolve().parent
    read = lambda n: json.loads((root/n).read_text())
    manifest = read("manifest.json")
    if set(manifest) != {"statistics.json", "expected.json", "README.md", "requirements.txt", "replay.py"}:
        raise ValueError("Unexpected reviewer package files")
    for k, digest in manifest.items():
        if hashlib.sha256((root/k).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reviewer file changed: {k}")
    p = read("statistics.json")
    if p["independent_confirmation"] or p["original_risk_cap"] != .02:
        raise ValueError("Development results with the unchanged 2% rule only")
    summary = aggregate(p["studies"])
    verify(summary, read("expected.json"))
    print(json.dumps(dict(verified=True, studies=len(summary),
        count_tables=sum(len(s["counts"]) for s in summary.values()),
        intervals=sum(len(s["effects"]) for s in summary.values())), sort_keys=True))


if __name__ == "__main__":
    main()
