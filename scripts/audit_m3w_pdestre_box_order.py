"""Replay identical camera boxes with the two documented/plausible size orders."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_pdestre_camera as camera

source, np = camera.source, camera.np


def swap_size(frames):
    return [{k: np.r_[b[:2], b[:2]+(b[2:]-b[:2])[::-1]] for k, b in t.items()} for t in frames]


def main():
    d, key = camera.register()
    p = camera.OUT/"analysis.json"
    primary = source.load(p)
    f = source.load(camera.OUT/"freeze.json")
    if (primary["registration"] != key or f["registration"] != key
            or primary["freeze_sha256"] != source.digest((camera.OUT/"freeze.json").read_bytes())
            or f["label_geometry_used"] or primary["independent_confirmation"] or primary["reserved_outcomes_opened"]
            or not (set(d["windows"]) == set(f["receipts"]) == set(primary["rows"]))):
        raise ValueError("Need exact development-only camera freeze and score")
    rows, total, aspect = {}, {a: Counter() for a in camera.ARMS}, Counter()
    for day, item in d["windows"].items():
        p0 = camera.LOCAL/f"{day}.json"
        if source.digest(p0.read_bytes()) != f["receipts"][day]:
            raise ValueError("Frozen camera receipt changed")
        e = item["annotations"]
        b = source.byte_range(e["offset"], e["bytes"])
        if source.digest(b) != item["annotation_sha256"]:
            raise ValueError("Publisher label bytes changed")
        t, z = camera.labels(b, item["start"]), source.load(p0)
        if {a: camera.metrics(z, t, a) for a in camera.ARMS} != primary["rows"][day]:
            raise ValueError("Published size order did not replay")
        sizes = np.asarray([v[2:]-v[:2] for obs in t[:8] for v in obs.values()])
        aspect.update(past_boxes=len(sizes), wider_in_documented_order=int((sizes[:, 0] > sizes[:, 1]).sum()),
                      taller_in_documented_order=int((sizes[:, 0] < sizes[:, 1]).sum()))
        alt = swap_size(t)
        rows[day] = {a: camera.metrics(z, alt, a) for a in camera.ARMS}
        for a, m in rows[day].items():
            total[a].update({k: int(v) for k, v in m.items() if isinstance(v, int)})
    out = dict(input_sha256=source.digest(p.read_bytes()), code_sha256=source.digest(Path(__file__).read_bytes()),
               windows=len(rows), aspect=aspect, declared_order="x,y,height,width",
               alternative_order="x,y,width,height", primary_unchanged=True, posthoc_geometry_diagnostic=True,
               used_for_model_selection=False, independent_confirmation=False,
               reserved_outcomes_opened=False, rows=rows, totals=total)
    source.save(camera.OUT/"box_order_diagnostic.json", out)


if __name__ == "__main__":
    main()
