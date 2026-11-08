"""Export only anonymous source/seed scalars from the pedestrian experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_m3w_opendd_candidate_replay import REPORTS, ROOT, encoded, sha
from src.reproduction import m3w_sparse_support_replay as replay

SOURCE = REPORTS/"moving_record_joint_actions_v1/causal_ledger/sparse_support_v1"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/moving_sparse_support_replay_v1/anonymous_sparse_support.zip"


def inputs():
    read = lambda k: json.loads((SOURCE/(k+".json")).read_text())
    docs = [read(k) for k in ("analysis", "capacity", "bridge")]
    for a, r in zip(docs, ("registration", "capacity_registration", "bridge_registration"), strict=True):
        if a["registration"] != sha(SOURCE/(r+".json")) or a["confirmation_access"] or not a["original_2pct"]:
            raise ValueError("Unbound or disallowed source evidence")
    if docs[0]["heads"] != sha(SOURCE/"head_freeze.json"):
        raise ValueError("Changed sparse support weights receipt")
    for k in ("capacity_registration", "bridge_registration"):
        for parent in read(k)["parents"]:
            if sha(ROOT/parent["path"]) != parent["sha256"]:
                raise ValueError("Changed capacity parent")
    return docs


def build(out):
    a, b, c = inputs()
    cells = [{(s, r["seed"]): r for s, rows in z["by_site"].items() for r in rows} for z in (a, b, c)]
    if set(cells[0]) != set(cells[1]) or set(cells[0]) != set(cells[2]):
        raise ValueError("Require identical source/seed rosters")
    groups = []
    for s in sorted(a["by_site"]):
        g = []
        for r in sorted(a["by_site"][s], key=lambda z: z["seed"]):
            k = s, r["seed"]
            g.append(dict(metrics=r["metrics"], effects=r["effects"], counts=r["counts"],
                          capacity=cells[1][k]["counts"], bridge=cells[2][k]["counts"]))
        groups.append(g)
    data = dict(groups=groups)
    expected = dict(**a["summary"], capacity=b["summary"], bridge=c["summary"])
    replay.verify(replay.aggregate(data), expected)
    files = {
        "README.md": b"Run python replay.py with the pinned NumPy version. This reconstructs counts "
                     b"and intervals from anonymous source scalars, not training or independent outcomes.\n",
        "replay.py": Path(replay.__file__).read_bytes(), "requirements.txt": f"numpy=={np.__version__}\n".encode(),
        "expected.json": encoded(expected), "statistics.json": encoded(data),
    }
    files["manifest.json"] = encoded({k: hashlib.sha256(x).hexdigest() for k, x in files.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".pending")
    with ZipFile(temp, "w") as z:
        for k, x in sorted(files.items()):
            info = ZipInfo(k, (1980, 1, 1, 0, 0, 0))
            z.writestr(info, x, compress_type=ZIP_DEFLATED, compresslevel=9)
    temp.replace(out)
    print(out, sha(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
