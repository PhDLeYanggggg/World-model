"""Standalone scalar replay of matched OpenDD development comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def recompute(packet):
    if packet["independent_confirmation"] or packet["risk_cap"] != .02:
        raise ValueError("This replay covers development and the original risk rule only")
    out = {}
    for name, sites in packet["studies"].items():
        if len(sites) != 7 or any(len(s) != 3 for s in sites):
            raise ValueError(f"Expected seven sites and three paired seeds for {name}")
        rows = [r for site in sites for r in site]
        if any(set(r) != {"effects", "counts"} for r in rows):
            raise ValueError("Unexpected replay fields")
        first = rows[0]
        counts, effects = {}, {}
        for k, c in first["counts"].items():
            counts[k] = {f: sum(r["counts"][k][f] for r in rows) for f in c}
        draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
        for k in first["effects"]:
            v = np.asarray([np.mean([r["effects"][k] for r in site]) for site in sites])
            if not np.isfinite(v).all():
                raise ValueError("Expected complete scalar contrast roster")
            effects[k] = dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
        out[name] = dict(counts=counts, effects=effects)
    return out


def check(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Replay study roster differs")
    for k, v in actual.items():
        e = expected[k]
        if v["counts"] != e["counts"] or v["effects"].keys() != e["effects"].keys():
            raise ValueError(f"Replay counts or contrast roster changed: {k}")
        for name, a in v["effects"].items():
            b = e["effects"][name]
            np.testing.assert_allclose(a["point"], b["point"], rtol=0., atol=1e-12)
            np.testing.assert_allclose(a["interval95"], b["interval95"], rtol=0., atol=1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    manifest = json.loads((a.root/"manifest.json").read_text())
    if set(manifest) != {"README.md", "requirements.txt", "replay.py", "statistics.json", "expected.json"}:
        raise ValueError("Unexpected reviewer package roster")
    for name, digest in manifest.items():
        if hashlib.sha256((a.root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reviewer file changed: {name}")
    z = recompute(json.loads((a.root/"statistics.json").read_text()))
    check(z, json.loads((a.root/"expected.json").read_text()))
    print(json.dumps(dict(studies=len(z), intervals=sum(len(r["effects"]) for r in z.values()),
                          policy_count_tables=sum(len(r["counts"]) for r in z.values()), verified=True), sort_keys=True))


if __name__ == "__main__":
    main()
