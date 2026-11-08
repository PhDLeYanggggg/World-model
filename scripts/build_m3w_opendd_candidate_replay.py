"""Build an anonymous, deterministic archive of scalar OpenDD readouts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.reproduction import m3w_opendd_paired_replay as replay


REPORTS = ROOT/"outputs/publication_readiness_2026_09"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/opendd_candidate_replay_v1/anonymous_opendd_replay.zip"
FIELDS = ("edits", "views", "undefined", "risk_pass", "dual_positive")
README = """# Matched OpenDD Development Comparisons

Run `python replay.py` in an environment with the version of NumPy in
`requirements.txt`. The command checks the files and recomputes the totals
and paired intervals from scalar values. Sites and seeds are indexed by
position; three seed views are averaged before each of the 3,000 seven-site
bootstrap draws. The recording-seed views are not independent observations.

The packet includes cross-site shape generation, learned shape choice,
earlier-error recipient choice, and the fixed-head EqMotion comparison.
Matched comparisons keep query counts and slot radii fixed; whole-roster
forecast effects are kept separate. Original risk checks require support
as well as the 2% limits; undefined views never count as passes.

This is an anonymous scalar replay, not full model training, trajectory
release, independent confirmation or a deployment result. It contains
no trajectories, weight files, site names, or workstation paths.
"""


def encoded(x):
    return (json.dumps(x, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(name):
    folder = REPORTS/name
    p = folder/"analysis.json"
    a = json.loads(p.read_text())
    if a["registration"] != sha(folder/"registration.json") or not a["original_2pct"] or a["confirmation_access"]:
        raise ValueError("Not a bound exposed development readout")
    if "heads" in a and a["heads"] != sha(folder/"head_freeze.json"):
        raise ValueError("Training freeze digest changed")
    return a


def pack(a, cells, per_cell, expected_effects, expected_counts):
    sites = []
    for s, rows in sorted(a["by_site"].items()):
        sites.append([per_cell(r) for r in sorted(cells(rows), key=lambda r: r["seed"])])
    expected = dict(counts={k: {f: v[f] for f in FIELDS} for k, v in expected_counts.items()},
        effects={k: {f: v[f] for f in ("point", "interval95")} for k, v in expected_effects.items()})
    return sites, expected


def row(effects, policies):
    return dict(effects=effects, counts={k: {f: v[f] for f in FIELDS} for k, v in policies.items()})


def collect():
    studies, expected = {}, {}
    for name in ("donor_motion_shape_v1", "donor_shape_choice_v1", "opendd_feedback_recipient_v1"):
        a = read(name)
        archive = "cells" in next(iter(a["by_site"].values()))
        label = "metrics" if name == "opendd_feedback_recipient_v1" else "policies"
        studies[name], expected[name] = pack(a, lambda r: r["cells"] if archive else r,
            lambda r: row(r["effects"], r[label]), a["summary"]["effects"], a["summary"]["policies"])
    name = "opendd_comoving_eqmotion_v1"
    a = read(name)
    s = a["summary"]
    effects = {**s["raw_effects"], **{f"matched_{k}": v for k, v in s["matched_effects"].items()}}
    for k, v in s["raw"].items():
        for ref, value in v["lower"].items():
            effects[f"raw_{k}_vs_{ref}"] = value

    def eq_row(r):
        raw, matched = r["raw"], r["matched"]
        effect = {**raw["effects"], **{f"matched_{k}": v for k, v in matched["effects_vs_anchor"].items()}}
        effect.update({f"raw_{k}_vs_{ref}": value for k, v in raw["arms"].items() for ref, value in v["lower"].items()})
        return row(effect, matched["policies"])

    studies[name], expected[name] = pack(a, lambda r: r, eq_row, effects, s["matched"])
    return dict(studies=studies, risk_cap=.02, independent_confirmation=False), expected


def build(out):
    packet, expected = collect()
    replay.check(replay.recompute(packet), expected)
    files = {"replay.py": Path(replay.__file__).read_bytes(), "README.md": README.encode(),
             "requirements.txt": f"numpy=={np.__version__}\n".encode(),
             "statistics.json": encoded(packet), "expected.json": encoded(expected)}
    files["manifest.json"] = encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pending")
    with ZipFile(tmp, "w") as z:
        for k, v in sorted(files.items()):
            zi = ZipInfo(k, date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o100644 << 16
            z.writestr(zi, v, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(out)
    return sha(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    print(build(a.out), flush=True)
