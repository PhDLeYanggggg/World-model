"""Build the anonymous, offline AV2 selected-projection evidence archive."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_av2_selected_projection_risk as run
from scripts.package_m3w_av2_selection_evidence import digest, encode

TEMPLATE = ROOT/"reproducibility/av2_selected_projection"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/submission_admin/av2_selected_projection_evidence.zip"
FIELDS = ("qualified", "risk_fail", "floor_fail", "easy_over")


def payload():
    key, _, _, _, _, _, _, _, parent, _, _, _ = run.register()
    path = run.OUT/"summary.json"
    study = run.study
    summary = study.read(path)
    freeze = study.read(study.checked(summary["freeze"]))
    cal = study.read(study.checked(summary["calibration"]))
    if (freeze["registration"] != key or freeze["held_labels_used"] or cal["held_receipts_used"]
            or summary["independent_confirmation"] or summary["reserved_outcomes_opened"]):
        raise ValueError("Wrong development receipt")
    cities = {}
    for name, ref in freeze["folds"].items():
        z = study.read(study.checked(ref))
        cell = cities.setdefault(z["identity"]["held"], dict(folds=0, budget=0, views=0,
            top5={a: dict.fromkeys(FIELDS, 0.) for a in run.ARMS},
            calibrated={a: dict.fromkeys(("released", *FIELDS), 0) for a in run.ARMS}))
        rows = list(run.old.prior.role_rows(z, parent["folds"][name], name, "held"))
        cell["folds"] += 1
        cell["budget"] += min(5, len(rows))
        cell["views"] += len(rows)
        for a in run.ARMS:
            for k, v in run.counts(rows, a, 5).items():
                cell["top5"][a][k] += v
            threshold = cal["rules"][name][a]["threshold"]
            for r in rows:
                if threshold is None or r["scores"][a] < threshold:
                    continue
                m = r["result"]
                b = cell["calibrated"][a]
                b["released"] += 1
                b["qualified"] += m["eligible"]
                b["risk_fail"] += not m["risk"]
                b["floor_fail"] += m["gain"] is None or min(m["gain"]) <= 0.
                b["easy_over"] += m["easy_over"]
    return dict(views=summary["views"], scenarios=summary["scenarios"], arms=run.ARMS,
                anonymous_source_cells=[cities[k] for k in sorted(cities)]), summary


def package(path):
    data, source = payload()
    items = {k: (TEMPLATE/k).read_bytes() for k in ("README.md", "verify.py")}
    items["evidence.json"] = encode(data)
    items["manifest.json"] = encode(dict(members={k: digest(v) for k, v in items.items()}))
    for raw in items.values():
        if re.search(rb"/Users/|/home/|/cephfs/|github\.com|kcl\.ac\.uk|yangyue|PhDLeYang", raw, re.I):
            raise ValueError("Identity or private path in archive")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.zip")
    with ZipFile(pending, "w", compression=ZIP_DEFLATED) as z:
        for name, raw in sorted(items.items()):
            z.writestr(ZipInfo(name, (1980, 1, 1, 0, 0, 0)), raw, compress_type=ZIP_DEFLATED)
    with TemporaryDirectory() as tmp, ZipFile(pending) as z:
        z.extractall(tmp)
        result = subprocess.run([sys.executable, "-I", "-B", "-S", "verify.py"], cwd=tmp, check=True,
                                capture_output=True, text=True, timeout=20)
    replay = json.loads(result.stdout)
    top = source["matched"]["5"]
    if replay["top5_budget"] != top["n"]:
        raise ValueError("Intervention count changed")
    for a, c in replay["counts"].items():
        if c["top5"] != top["counts"][a] or c["calibrated"] != {k: source["held"][a][k] for k in ("released", *FIELDS)}:
            raise ValueError("Archive did not replay the observed counts")
    for a, metrics in replay["selected_minus"].items():
        for k, v in metrics.items():
            if abs(v["point_pp"]-top["selected_minus"][a][k]["site_mean"]) > 1e-10:
                raise ValueError("Archive paired point estimate did not replay")
    pending.replace(path)
    meta = dict(archive=path.name, sha256=digest(path.read_bytes()), size=path.stat().st_size,
        offline_verifier_passed=True, result=replay, independent_confirmation=False)
    run.base.immutable(run.OUT/"anonymous_replay.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    package(p.parse_args().output)
