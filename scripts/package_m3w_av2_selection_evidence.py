"""Create an offline, anonymous scalar replay for the AV2 appendix."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/"outputs/publication_readiness_2026_09"
TEMPLATE = ROOT/"reproducibility/av2_selection_evidence"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/submission_admin/av2_selection_evidence.zip"


def digest(p):
    return sha256(p).hexdigest()


def encode(v):
    return (json.dumps(v, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def payload():
    p = REPORT/"av2_harm_quantile_v1/postselection.json"
    r = json.loads(p.read_bytes())
    v = json.loads((REPORT/"av2_selected_veto_v1/summary.json").read_bytes())
    if (r["reserved_outcomes_opened"] or v["reserved_outcomes_opened"] or v["independent_confirmation"]
            or not v["same_forced_paths_and_budget"] or r["two_slot_seed_views"] != v["counts"]["two_slots"]):
        raise ValueError("Wrong scope for the anonymous development contrast")
    clusters = [a for _, a in sorted(r["by_city"].items())]
    views = [a["views"] for a in clusters]
    diff = [[round(b*a["views"]) for b in a["coverage_delta"]] for a in clusters]
    if any(abs(c/a["views"]-b) > 1e-12 for a, row in zip(clusters, diff, strict=True)
           for b, c in zip(a["coverage_delta"], row, strict=True)):
        raise ValueError("Counts did not recover the paired contrast")
    frozen, selected = [[round(a*sum(views)) for a in row] for row in r["pooled_coverage"]]
    if [sum(row[k] for row in diff) for k in range(len(r["terms"]))] != [b-a for a, b in zip(frozen, selected, strict=True)]:
        raise ValueError("Coverage totals do not reconcile")
    veto = {m: dict(released=v["counts"][f"{m}_released"], eligible=v["counts"][f"{m}_released_eligible"],
                   risk_fail=v["counts"][f"{m}_risk_fail"]) for m in ("self", "uniform", "selected")}
    return dict(cluster_views=views, coverage_count_changes=diff, views=sum(views), scenarios=r["two_slot_scenarios"],
        frozen_covered=frozen, selected_covered=selected, terms=r["terms"], held_veto=veto), r


def package(destination):
    data, parent = payload()
    items = {k: (TEMPLATE/k).read_bytes() for k in ("README.md", "verify.py")}
    items["evidence.json"] = encode(data)
    items["manifest.json"] = encode(dict(members={k: digest(v) for k, v in items.items()},
        scope="anonymous scalar replay of development results"))
    for raw in items.values():
        if re.search(rb"/Users/|/home/|/cephfs/|github\.com|kcl\.ac\.uk|yangyue|PhDLeYang", raw, re.I):
            raise ValueError("Identity or source path in archive")
    with TemporaryDirectory() as tmp:
        p = Path(tmp)
        for k, v in items.items():
            (p/k).write_bytes(v)
        result = subprocess.run([sys.executable, "-I", "-B", "-S", "verify.py"], cwd=p,
            check=True, capture_output=True, text=True, timeout=20)
    replay = json.loads(result.stdout)
    for got, expected in zip(replay["terms"], parent["paired_delta"]["site_mean"], strict=True):
        if abs(got["delta_pp"]-100*expected) > 1e-10 or got["exact_ci95_pp"][1] >= 0:
            raise ValueError("Independent scalar replay did not verify the coverage drop")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending.zip")
    with ZipFile(pending, "w", compression=ZIP_DEFLATED) as z:
        for name, raw in sorted(items.items()):
            z.writestr(ZipInfo(name, (1980, 1, 1, 0, 0, 0)), raw, compress_type=ZIP_DEFLATED)
    pending.replace(destination)
    print(json.dumps(dict(archive_sha256=digest(destination.read_bytes()), result=replay), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    package(p.parse_args().output)
