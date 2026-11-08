"""Independently reconcile frozen development targets with the released MOT rows."""

import csv
import io
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_weighted_velocity as study

job, run, np = study.job, study.run, study.np
OUT = study.OUT.parent / "moving_future_support_v1"
PRIVATE = study.PRIVATE.parent / OUT.name
HOME = OUT.parent
MISSING = ("past_record_detection_end", "past_track_detection_end",
           "interior_other_agents_visible", "interior_no_agent_visible")
UNKNOWN = ("no_later_same_id", "later_by_24", "later_by_48", "later_after_48")


def parse_csv(data):
    """Use a separate reader; no window, trajectory converter or interpolation."""
    boxes, frames = {}, {}
    for row in csv.reader(io.StringIO(data.decode("utf-8-sig"))):
        if not row or not any(c.strip() for c in row):
            continue
        if len(row) != 10:
            raise ValueError("Expected MOT-style row")
        a = np.asarray(row, dtype=float)
        f, person = map(int, a[:2])
        if (not np.isfinite(a).all() or (a[:2] != [f, person]).any() or f < 0 or person < 0
                or (a[4:6] <= 0.).any() or (person, f) in boxes):
            raise ValueError("Invalid or duplicate released observation")
        boxes[person, f] = a[2:4]+a[4:6]/2
        frames.setdefault(person, []).append(f)
    if not boxes:
        raise ValueError("No released rows")
    return boxes, {i: sorted(v) for i, v in frames.items()}


def keys_and_categories(boxes, frames, nframes):
    if not boxes or max(f for _, f in boxes) >= nframes:
        raise ValueError("Annotation exceeds image inventory")
    record_frames = {f for _, f in boxes}
    record_last = max(record_frames)
    queried = {}
    for agent, ticks in frames.items():
        contiguous = 0
        last = -2
        for p, t in enumerate(ticks):
            contiguous = contiguous+1 if last+1 == t else 1
            last = t
            if contiguous >= 8 and t+12 < nframes:
                queried.setdefault(t, []).append((agent, ticks[p+1] if p+1 < len(ticks) else None))
    for q, agents in sorted(queried.items()):
        for agent, later in sorted(agents):
            labels = []
            for f in range(q+1, q+13):
                if (agent, f) in boxes:
                    labels.append(None)
                elif f > record_last:
                    labels.append(MISSING[0])
                elif f > frames[agent][-1]:
                    labels.append(MISSING[1])
                else:
                    labels.append(MISSING[2] if f in record_frames else MISSING[3])
            kind = None
            if all(k is not None for k in labels):
                delta = later-q if later is not None else None
                kind = UNKNOWN[0] if delta is None else UNKNOWN[1] if delta <= 24 else UNKNOWN[2] if delta <= 48 else UNKNOWN[3]
            yield q, agent, labels, kind


def audit(boxes, frames, nframes, target, valid, history, cv, ols, weight):
    keys = list(keys_and_categories(boxes, frames, nframes))
    if len(keys) != len(target):
        raise ValueError("Raw query roster differs from frozen fitting data")
    mask, relative = np.zeros_like(valid), np.zeros_like(target)
    steps = dict.fromkeys(MISSING, 0)
    distinct, missing_by_query = set(), set()
    unknown_count, unknown_debt = dict.fromkeys(UNKNOWN, 0), dict.fromkeys(UNKNOWN, 0.)
    hist_mismatch = np.zeros(len(keys), bool)
    debit = np.linalg.norm(ols-cv, axis=2).max(1)*weight
    for i, (q, agent, labels, kind) in enumerate(keys):
        anchor = boxes[agent, q]
        hist = np.stack([boxes[agent, t]-anchor for t in range(q-7, q+1)]).astype(history.dtype)
        hist_mismatch[i] = np.any(hist != history[i])
        for j, group in enumerate(labels, 1):
            if group is None:
                mask[i, j-1] = True
                relative[i, j-1] = boxes[agent, q+j]-anchor
            else:
                distinct.add((agent, q+j))
                missing_by_query.add((agent, q))
                steps[group] += 1
        if kind:
            unknown_count[kind] += 1
            unknown_debt[kind] += debit[i]
    ref = job.errors(cv, target, valid)[0]
    known = valid.any(1)
    return dict(rows=len(keys), requested_steps=int(mask.size), raw_missing_steps=int((~mask).sum()),
        known_steps=int(mask.sum()), complete_rows=int(mask.all(1).sum()), missing_steps=steps,
        unique_missing_track_frame_keys=len(distinct), rows_with_missing=len(missing_by_query),
        mask_mismatches=int((mask != valid).sum()),
        exact_recoverable_steps=int((mask & ~valid).sum()),
        false_positive_steps=int((valid & ~mask).sum()),
        coordinate_mismatches=int((np.any(relative != target, axis=2) & mask & valid).sum()),
        past_geometry_row_mismatches=int(hist_mismatch.sum()), wholly_unknown=unknown_count,
        missing_reference_debt=unknown_debt, reference_mass=float(weight[known]@ref[known]))


def main():
    key, identity, _, _ = study.register()
    training = job.prior.prior.prior.base.train_source
    frozen = training.PUBLIC / "fit_population.json"
    intake = HOME / "movingdronecrowd_intake_v1"
    admitted = HOME / "moving_development_admission_v1" / "admission.json"
    manifest = intake / "source_manifest.json"
    prior = study.OUT / "debt.json"
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(parent=key, debt=run.artifact(prior), frozen=run.artifact(frozen),
        admission=run.artifact(admitted), inventory=run.artifact(manifest), code=run.artifact(Path(__file__)),
        original_fitting_only=True, reconstruction_or_imputation=False, original_rule_unchanged=True,
        confirmation_access=False)))
    start = time.monotonic()
    config = run.read(training.source.CONFIG)
    admission = training.prep.load_admission(training.ROOT, training.ROOT/config["admission"])
    clips = {r["clip"]: r for r in run.read(intake/"analysis.json")["recordings"]}
    data = training.fitting_only(identity, lambda **_: None)
    old = job.prior.prior.old
    outputs, lo = [], 0
    for rec, group, h, cv, ols, _, target, valid, _, _, w in old.blocks(identity, lambda **_: None):
        meta = admission["recordings"][rec]
        path = training.ROOT/meta["source_annotation_path"]
        training.prep.require_fit_record(admission, rec, "fit", path)
        boxes, frames = parse_csv(path.read_bytes())
        hi = lo+len(h)
        np.testing.assert_array_equal(data["valid"][lo:hi], valid)
        np.testing.assert_array_equal(data["target"][lo:hi], target)
        value = audit(boxes, frames, clips[meta["clip"]]["image_frames"], target, valid, h, cv, ols, w)
        outputs.append(dict(recording=rec, source=group, annotation_sha256=run.sha(path), **value))
        lo = hi
        run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", dict(recording=rec,
            rows=lo, recordings=len(outputs), seconds=round(time.monotonic()-start, 2)))
    if len(outputs) != 24 or lo != 69746:
        raise ValueError("Require the original 24 fitting recordings")
    counts = ("rows", "requested_steps", "raw_missing_steps", "known_steps", "complete_rows",
              "unique_missing_track_frame_keys", "rows_with_missing", "mask_mismatches",
              "exact_recoverable_steps", "false_positive_steps", "coordinate_mismatches", "past_geometry_row_mismatches")
    summary = {k: sum(r[k] for r in outputs) for k in counts}
    summary["missing_steps"] = {k: sum(r["missing_steps"][k] for r in outputs) for k in MISSING}
    summary["wholly_unknown"] = {k: sum(r["wholly_unknown"][k] for r in outputs) for k in UNKNOWN}
    sources = sorted({r["source"] for r in outputs})
    debt = {}
    for kind in UNKNOWN:
        a = [[r for r in outputs if r["source"] == s] for s in sources]
        debt[kind] = float(np.mean([100*sum(r["missing_reference_debt"][kind] for r in z)
                                   / sum(r["reference_mass"] for r in z) for z in a]))
    if not np.isclose(sum(debt.values()), run.read(prior)["sum_debit_floor_pp"], rtol=1e-8, atol=1e-8):
        raise ValueError("Independent support classification did not recover the reference debit")
    run.once(OUT / "analysis.json", run.stable(dict(registration=run.sha(reg), summary=summary,
        reference_debt_pp_by_wholly_unknown_type=debt, recordings=outputs, source_groups=len(sources),
        no_annotation_or_mask_changes=True, only_original_fitting_rows=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
