"""Create and test the identity-minimized supplementary replay bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_aggregate_replay_package import screen
from scripts.build_m3w_review_replay import write_zip
from src.reproduction import m3w_deferral_review as review

import numpy as np

HOME = ROOT/"outputs/publication_readiness_2026_09"
PUBLIC = HOME/"deferral_review_replay_v1"
PRIVATE = ROOT/"data/stage_cvpr2027_experiments"/PUBLIC.name
STUDIES = ("slot_timing_v1", "ledger_wait_v2", "wait_peer_value_v1", "expiry_pricing_v1")
README = """# Supplementary Deferral Evidence Replay

Run `python replay.py` after installing the pinned NumPy dependency.
No network access, repository checkout, raw data or trained models are
needed to replay. The archive contains anonymized location and seed
indices and aggregate scalars, not trajectories or checkpoints.

This recomputes recording-seed counts and paired confidence intervals:
average over three seed views in each of seven exposed development
locations, then use 3,000 paired location bootstrap draws. The 606
recording-seed views repeat 202 recordings, not 606 independent samples.
All undefined easy-risk counts and negative effects are retained.

This is a numerical evidence replay, not a reproduction of training,
raw-label evaluation, independence, or deployment qualification. A
mechanical identifier check does not certify anonymity.
"""


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def data():
    source, expected, parents = {}, {}, {}
    for i, name in enumerate(STUDIES, 1):
        home = HOME/name
        reg = (home/"registration.json").read_bytes()
        raw = (home/"analysis.json").read_bytes()
        doc = json.loads(raw)
        if doc["registration"] != digest(reg) or doc["confirmation_access"] or not doc["original_2pct"]:
            raise ValueError("Only bound development evidence can enter the bundle")
        if doc.get("heads") and doc["heads"] != digest((home/"head_freeze.json").read_bytes()):
            raise ValueError("Missing frozen head binding")
        key = f"study_{i}"
        cells = []
        for _, rows in sorted(doc["by_site"].items()):
            group = []
            for j, r in enumerate(sorted(rows, key=lambda z: z["seed"]), 1):
                group.append(dict(seed=j, direct=r["direct"],
                    metrics={k: {f: v[f] for f in review.FIELDS} for k, v in r["metrics"].items()},
                    **({"oracle": r["oracle"]} if i == 1 else {})))
            cells.append(group)
        s = doc["summary"]
        target = dict(arms={k: {f: v[f] for f in review.FIELDS} for k, v in s["arms"].items()},
            direct={k: [v["point"], *v["interval95"]] for k, v in s["direct"].items()},
            oracle=s["oracle"] if i == 1 else {})
        review.match(review.replay(cells), target)
        source[key], expected[key], parents[name] = cells, target, digest(raw)
    return source, expected, parents


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    source, expected, parents = data()
    files = {"README.md": README.encode(), "replay.py": Path(review.__file__).read_bytes(),
             "statistics.json": encode(source), "expected.json": encode(expected),
             "requirements.txt": f"numpy=={np.__version__}\n".encode()}
    files["manifest.json"] = encode({k: digest(v) for k, v in files.items()})
    screen(files)
    dest = PRIVATE/"review_bundle.zip"
    receipt = PUBLIC/"verification.json"
    with tempfile.TemporaryDirectory(prefix="deferral-review-") as folder:
        root = Path(folder)
        archive = root/"review_bundle.zip"
        write_zip(archive, files)
        with ZipFile(archive) as z:
            if set(z.namelist()) != set(files):
                raise ValueError("Unexpected archive member")
            z.extractall(root)
        result = subprocess.run([sys.executable, "-I", str(root/"replay.py")], cwd=root,
                                check=True, capture_output=True, text=True)
        v = json.loads(result.stdout)
        raw = archive.read_bytes()
    report = dict(**v, archive_bytes=len(raw), archive_sha256=digest(raw), source_analyses=parents,
        num_members=len(files), isolated_python=True, no_data_or_checkpoints=True,
        anonymity_certified=False, training_reproduced=False, submission_ready=False)
    if a.check:
        if dest.read_bytes() != raw or json.loads(receipt.read_text()) != report:
            raise ValueError("Isolated replay receipt or archive changed")
    else:
        write_zip(dest, files)
        PUBLIC.mkdir(parents=True, exist_ok=True)
        if receipt.exists() and receipt.read_bytes() != encode(report):
            raise ValueError("Cannot replace immutable review receipt")
        receipt.write_bytes(encode(report))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
