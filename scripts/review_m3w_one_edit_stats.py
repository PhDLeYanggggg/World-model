"""Verify the one-edit statistical readout using only anonymous scalar receipts."""

import argparse
from itertools import combinations
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np


def verify(bundle):
    p, g, heads = (bundle[k] for k in ("primary", "guard", "head_roles"))
    sites = tuple(p["by_site"])
    seeds = (17, 29, 43)
    expected = {(e, seed) for n in (1, 2) for e in combinations(sites, n) for seed in seeds}
    actual = {(tuple(r["excluded"]), r["seed"]) for r in heads}
    assert actual == expected and len(heads) == len(expected)
    for r in heads:
        assert set(r["fit"]) == set(sites)-set(r["excluded"])
    for site, v in p["selected_by_donors"].items():
        stop, fixed = [max(v[k], key=lambda r: v[k][r]) for k in ("stop_grid", "fixed_grid")]
        assert v["chosen"] == {"learned_stop": stop, "fixed_delay": fixed}
        assert v["donor_cells"] == (len(sites)-1)*len(seeds)
        grid = v["stop_grid"]
        allowed = [k for k, x in grid.items() if x[0] >= grid["threshold_0"][0] and x[1] >= grid["threshold_0"][1]]
        assert g["by_site"][site]["chosen"] == max(allowed, key=lambda k: grid[k])

    cells = [c for v in p["by_site"].values() for c in v]
    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    for scope in ("metrics", "short"):
        for arm, saved in p["summary"][scope].items():
            assert {k: sum(c[scope][arm][k] for c in cells if arm in c[scope]) for k in fields} == saved
    for ref in ("immediate", "fixed_delay"):
        assert sum(c["common"].get(ref, 0) for c in cells) == p["summary"]["common"][ref]
        for scope in ("full", "matched"):
            z = [[r["effects"][ref][scope] for r in rows if r["effects"][ref][scope] is not None]
                 for rows in p["by_site"].values()]
            x = np.array([np.mean(v) for v in z if v])
            draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
            saved = p["summary"]["effects"][ref][scope]
            assert saved["supported_locations"] == len(x)
            np.testing.assert_allclose(saved["point"], x.mean(), atol=1e-12, rtol=0.)
            np.testing.assert_allclose(saved["interval95"], np.quantile(x[draw].mean(1), [.025, .975]), atol=1e-12, rtol=0.)
    rows = [r for s in g["by_site"].values() for r in s["rows"]]
    for k in ("edits", "views", "undefined", "risk_pass", "dual_positive"):
        assert g["totals"][k] == sum(r[k] for r in rows)
    assert g["totals"]["fallback_sites"] == sum(s["chosen"] == "threshold_0" for s in g["by_site"].values())
    x = np.array([np.mean([r["lower_pp"] for r in s["rows"]]) for s in g["by_site"].values()])
    draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
    np.testing.assert_allclose(g["lower_vs_immediate_pp"]["point"], x.mean(), atol=1e-12, rtol=0.)
    np.testing.assert_allclose(g["lower_vs_immediate_pp"]["interval95"], np.quantile(x[draw].mean(1), [.025, .975]),
                               atol=1e-12, rtol=0.)
    return dict(sites=len(sites), seeds=len(seeds), expected_fit_roles=len(heads), primary_intervals=4, secondary_intervals=1,
                matched_views=p["summary"]["common"], full_metrics=p["summary"]["metrics"],
                guard_fallback_sites=g["totals"]["fallback_sites"])


def package(source, output):
    p, g, h = [json.loads((source/k).read_text()) for k in ("analysis.json", "donor_guard/analysis.json", "head_freeze.json")]
    assert not p["confirmation_access"] and not g["confirmation_access"] and p["original_2pct"] and g["original_2pct"]
    bundle = dict(primary={k: p[k] for k in ("by_site", "summary", "selected_by_donors")},
                  guard={k: g[k] for k in ("by_site", "totals", "lower_vs_immediate_pp")},
                  head_roles=[{k: v["identity"][k] for k in ("excluded", "fit", "seed")} for v in h["heads"].values()])
    receipt = verify(bundle)
    files = {"replay.py": Path(__file__).read_bytes(), "data.json": json.dumps(bundle, sort_keys=True, indent=2).encode()+b"\n",
             "README.txt": b"Anonymous scalar replay. Requires Python and NumPy. Run: python replay.py data.json.\n"
                           b"No model training, raw data, weights, independent confirmation or deployment claim.\n"}
    with ZipFile(output, "w") as z:
        for name, payload in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            z.writestr(info, payload, compress_type=ZIP_DEFLATED)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Scalar bundle JSON or directory of verified study results")
    p.add_argument("--package", type=Path)
    a = p.parse_args()
    v = package(a.path, a.package) if a.package else verify(json.loads(a.path.read_text()))
    print(json.dumps(v, sort_keys=True, indent=2))
