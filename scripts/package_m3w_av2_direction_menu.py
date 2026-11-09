"""Package an anonymous standard-library replay of the AV2 direction study."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from package_m3w_av2_selection_evidence import digest, encode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_av2_direction_menu as stage


TEMPLATE = ROOT/"reproducibility/av2_direction_menu"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/submission_admin/av2_direction_menu.zip"


def package(destination):
    result = stage.study.read(stage.OUT/"summary.json")
    seal = stage.study.read(stage.study.checked(result["freeze"]))
    if result["reserved_outcomes_opened"] or result["independent_confirmation"] or seal["held_labels_used"]:
        raise ValueError("Require the development-only score")
    fields = tuple(result["counts"])+("complete",)
    grouped = defaultdict(lambda: defaultdict(list))
    for name, f in seal["folds"].items():
        r = stage.study.read(stage.LOCAL/f"{name}.score.json")
        if r["freeze"] != f:
            raise ValueError("Unsealed scalar rows")
        for z in r["rows"]:
            grouped[z["city"]][z["clip"]].append({k: z[k] for k in fields})
    cities = [[v for _, v in sorted(grouped[c].items())] for c in sorted(grouped)]
    payload = dict(cities=cities, draws=stage.np.random.default_rng(3183).integers(len(cities), size=(3000, len(cities))).tolist())
    items = {k: (TEMPLATE/k).read_bytes() for k in ("verify.py", "README.md")}
    items["evidence.json"] = encode(payload)
    items["manifest.json"] = encode(dict(members={k: digest(v) for k, v in items.items()}))
    for v in items.values():
        if re.search(rb"/Users/|/home/|/cephfs/|github.com|kcl.ac.uk|yangyue|PhDLeYang", v, re.I):
            raise ValueError("Non-anonymous artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending.zip")
    with ZipFile(pending, "w", compression=ZIP_DEFLATED) as z:
        for k, v in sorted(items.items()):
            z.writestr(ZipInfo(k, (1980, 1, 1, 0, 0, 0)), v, compress_type=ZIP_DEFLATED)
    with TemporaryDirectory() as tmp:
        with ZipFile(pending) as z:
            z.extractall(tmp)
        raw = subprocess.run([sys.executable, "-I", "-B", "-S", "verify.py"], cwd=tmp, check=True,
                             capture_output=True, text=True, timeout=20).stdout
        actual = stage.study.json.loads(raw)
    for k in ("views", "scenarios", "counts", "complete_counts"):
        if actual[k] != result[k]:
            raise ValueError("Anonymous counts did not match the experiment")
    for k, v in actual["paired"].items():
        z = result["paired"][k]
        stage.np.testing.assert_allclose(v, [z["site_mean"], *z["ci95"]], atol=1e-12, rtol=0.)
    pending.replace(destination)
    print(stage.study.json.dumps(dict(sha256=digest(destination.read_bytes()), views=actual["views"],
                                      scenarios=actual["scenarios"], paired_intervals=len(actual["paired"])), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    package(p.parse_args().output)
