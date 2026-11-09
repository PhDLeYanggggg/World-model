"""Test cross-detector history agreement at an equal retained track budget."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_pdestre_camera as camera
from scripts.audit_m3w_pdestre_box_order import swap_size

source, np = camera.source, camera.np
OUT = camera.OUT.parent/"pdestre_track_agreement_v1"
LOCAL = camera.LOCAL.parent/"pdestre_track_agreement_v1"
ARMS = ("all", "confidence_matched", "frame_agreement_matched", "consistent_agreement")
MIN_HITS = 6


def binding(path):
    return source.digest(path.read_bytes())


def parent():
    d, key = camera.register()
    p = camera.OUT/"freeze.json"
    f = source.load(p)
    if (f["registration"] != key or f["label_geometry_used"] or f["reserved_outcomes_opened"]
            or set(f["receipts"]) != set(d["windows"])):
        raise ValueError("Require frozen past-only camera outputs")
    r = OUT/"registration.json"
    source.save(r, dict(parent_freeze_sha256=binding(p), code_sha256=binding(Path(__file__)),
        base="RAPiD from previous development coverage; forecast paths unchanged by filtering",
        agreement="one-to-one per-frame detector IoU >= 0.5; same partner in >=6 of 8 frames including last",
        controls="same per-window count by mean detector confidence or per-frame match count ignoring partner ID",
        fusion="on retained tracks only, average the two centers on frames with the same partner; no label in inference",
        original_roster_retained_for_coverage=True, original_2pct_unchanged=True,
        fit=False, independent_confirmation=False, reserved_outcomes_opened=False))
    return d, f, binding(r)


def center(b):
    return (b[1:3]+b[3:5])*.5


def ols4(h):
    last = h[-1]
    slope = np.array([-1.5, -.5, .5, 1.5]) @ (h[-4:]-last) / 5.
    return last + slope*np.arange(1., 13.)[:, None]


def proposals(z):
    a, b, matches = [], [], []
    for f in z["frames"]:
        x = np.asarray(f["boxes"]["rapid"]).reshape(-1, 7)
        y = {int(r[0]): np.asarray(r) for r in f["boxes"]["mobilenet"]}
        a.append({int(r[0]): r for r in x})
        b.append(y)
        matches.append(dict(camera.associate(x, {i: r[1:5] for i, r in y.items()})))
    if len(a) != 8:
        raise ValueError("Expected exactly eight frozen camera observations")
    paths = {k: o for k, _, o in z["paths"]["rapid"]}
    base = sorted(paths)
    scored, keep, fusion = {}, [], {}
    for k in base:
        h = np.stack([center(t[k]) for t in a])
        np.testing.assert_array_equal(ols4(h), paths[k])
        partner = matches[-1].get(k)
        same = [partner is not None and t.get(k) == partner for t in matches]
        hits = sum(k in t for t in matches)
        confidence = float(np.mean([t[k][5] for t in a]))
        scored[k] = confidence, hits
        if sum(same) >= MIN_HITS:
            keep.append(k)
            avg = np.stack([.5*(x+center(y[partner])) if ok else x
                            for x, y, ok in zip(h, b, same, strict=True)])
            fusion[k] = ols4(avg).tolist()
    n = len(keep)
    conf = sorted(base, key=lambda k: (-scored[k][0], k))[:n]
    frame = sorted(base, key=lambda k: (-scored[k][1], -scored[k][0], k))[:n]
    return dict(arms=dict(zip(ARMS, (base, conf, frame, keep), strict=True)), fusion=fusion)


def parent_row(day, f):
    p = camera.LOCAL/f"{day}.json"
    if binding(p) != f["receipts"][day]:
        raise ValueError("Frozen camera output changed")
    return source.load(p)


def freeze(d, f, key):
    refs = {}
    for day in d["windows"]:
        q = LOCAL/f"{day}.json"
        source.save(q, dict(registration=key, parent=f["receipts"][day], **proposals(parent_row(day, f))))
        refs[day] = binding(q)
    source.save(OUT/"freeze.json", dict(registration=key, selection_receipts=refs,
                                       score_labels_read=False, reserved_outcomes_opened=False))


def availability(ids, stable, future):
    support = [sum(stable.get(k) in t for t in future) for k in ids]
    return dict(kept=len(ids), stable=sum(k in stable for k in ids),
                complete=support.count(12), absent=support.count(0),
                partial=sum(0 < i < 12 for i in support),
                full_selected=len(ids) >= 2 and all(i == 12 for i in support),
                complete_pair=support.count(12) >= 2)


def fused_accuracy(frozen, paths, stable, truth):
    ds, fs, yy = [], [], []
    for k in frozen["arms"]["consistent_agreement"]:
        i = stable.get(k)
        if all(i in t for t in truth):
            y = np.stack([.5*(t[i][:2]+t[i][2:]) for t in truth])
            ds.append(paths[k])
            fs.append(frozen["fusion"][str(k)])
            yy.append(y)
    n = len(yy)
    if not n:
        return dict(n=0, base_ade=None, fused_ade=None, base_fde=None, fused_fde=None, pairs=0,
                    base_pair_ade=None, fused_pair_ade=None)
    y = np.stack(yy)
    a, b = map(np.stack, (ds, fs))
    ea, eb = (np.linalg.norm(p-y, axis=-1) for p in (a, b))
    ix, iy = np.triu_indices(n, 1)
    pair = lambda p: float(np.linalg.norm((p-y)[ix]-(p-y)[iy], axis=-1).mean()) if len(ix) else None
    return dict(n=n, base_ade=float(ea.mean()), fused_ade=float(eb.mean()),
                base_fde=float(ea[:, -1].mean()), fused_fde=float(eb[:, -1].mean()),
                pairs=len(ix), base_pair_ade=pair(a), fused_pair_ade=pair(b))


def score(d, f, key):
    s = source.load(OUT/"freeze.json")
    if (s["registration"] != key or s["score_labels_read"] or s["reserved_outcomes_opened"]
            or set(s["selection_receipts"]) != set(d["windows"])):
        raise ValueError("Selection must be frozen before scoring")
    p = camera.OUT/"box_order_diagnostic.json"
    prior = source.load(p)
    if not prior["posthoc_geometry_diagnostic"] or prior["used_for_model_selection"] or prior["independent_confirmation"]:
        raise ValueError("Require alternate-order development diagnostic")
    rows, totals = {}, {k: Counter() for k in ARMS}
    for day, item in d["windows"].items():
        q = LOCAL/f"{day}.json"
        if binding(q) != s["selection_receipts"][day]:
            raise ValueError("Frozen selection changed")
        z, a = parent_row(day, f), source.load(q)
        e = item["annotations"]
        buf = source.byte_range(e["offset"], e["bytes"])
        if source.digest(buf) != item["annotation_sha256"]:
            raise ValueError("Publisher annotation changed")
        t = swap_size(camera.labels(buf, item["start"]))
        if camera.metrics(z, t, "rapid") != prior["rows"][day]["rapid"]:
            raise ValueError("Parent alternate-order result did not replay")
        matches = [camera.associate(np.asarray(w["boxes"]["rapid"]).reshape(-1, 7), t0)
                   for w, t0 in zip(z["frames"], t[:8], strict=True)]
        stable = dict(set.intersection(*matches))
        values = {k: availability(a["arms"][k], stable, t[8:]) for k in ARMS}
        for k, v in values.items():
            totals[k].update({m: int(n) for m, n in v.items()})
        if len({values[k]["kept"] for k in ARMS[1:]}) != 1:
            raise ValueError("Matched track count changed")
        paths = {k: o for k, _, o in z["paths"]["rapid"]}
        rows[day] = dict(arms=values, matched_center_fusion=fused_accuracy(a, paths, stable, t[8:]))
    baseline = prior["totals"]["rapid"]
    if any(totals["all"][k] != baseline[v] for k, v in (("kept", "online"), ("complete", "complete"), ("absent", "absent"))):
        raise ValueError("Original full online denominator changed")
    paired = [r["matched_center_fusion"] for r in rows.values()]
    def pool(k, n):
        total = sum(r[n] for r in paired)
        return sum(r[k]*r[n] for r in paired if r[n])/total if total else None

    out = dict(registration=key, freeze_sha256=binding(OUT/"freeze.json"), parent_box_order_sha256=binding(p),
        windows=len(rows), linked_sources=1, rows=rows, totals=totals,
        fusion=dict(complete=sum(r["n"] for r in paired), pairs=sum(r["pairs"] for r in paired),
            **{k: pool(k, "pairs" if "pair" in k else "n") for k in (
                "base_ade", "fused_ade", "base_fde", "fused_fde", "base_pair_ade", "fused_pair_ade")}),
        only_development=True, geometry_interpretation="alternate width/height readout",
        original_2pct_unchanged=True, risk_scored=False, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False)
    source.save(OUT/"analysis.json", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if phase == "freeze" else score)(*parent())


if __name__ == "__main__":
    main()
