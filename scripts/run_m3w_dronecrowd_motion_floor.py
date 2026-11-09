"""Freeze and score causal full-scene motion controls on the approved calibration clips."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_dronecrowd_perfect_switch_ceiling as parent
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses, merge_comparisons
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_dronecrowd_motion_floor as model


cal, run, np = parent.cal, parent.run, parent.np
PUBLIC = parent.PUBLIC.parent/"dronecrowd_motion_floor_v1"
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
DRAWS, SEED = 3000, 73107


def register():
    binding, manifest, old, roster, _ = parent.freeze()
    p = parent.PUBLIC/"analysis.json"
    baseline = run.read(p)
    if (baseline["registration_sha256"] != binding or baseline["confirmation_access"]
            or not baseline["original_ols4_readout_reproduced"]):
        raise ValueError("Require the completed original-cap diagnostic")
    r = PUBLIC/"registration.json"
    run.once(r, run.stable(dict(
        source=run.artifact(p), parent_forecast=run.artifact(cal.PUBLIC/"forecast_freeze.json"),
        code=[run.artifact(Path(k)) for k in (__file__, model.__file__)],
        role=run.artifact(cal.roles.DEST), arms=model.ARMS, scope="approved_15_calibration_groups_only",
        observation_steps=8, forecast_steps=12, coordinate="image-space head-center pixels",
        experiment="full-scene forecast diagnostic, not a replacement of the original two-edit protocol",
        features="complete exported prefix only; robust coordinatewise median of six recent pair slopes; "
                 "last-point vs fitted intercept vs query-median intercept correction",
        selection="four one-step observed-prefix probes; lowest mean with OLS4 tie preference; "
                  "guarded version also requires every observed probe to stay within 2 percent of CV",
        evaluation="unchanged CV-referenced per-recording risk plus paired accuracy versus CV and OLS4, native units",
        sampling="all original calibration queries and labels, no new exclusions",
        bootstrap_draws=DRAWS, bootstrap_seed=SEED, no_training=True, risk_rule_changed=False,
        confirmation_access=False, independent_confirmation=False)))
    return run.sha(r), manifest, old, roster, baseline


def context(roster):
    inventory = run.read(cal.CACHE)
    meta = {x["sequence_id"]: x for x in inventory["recordings"] if x["sequence_id"] in roster}
    if set(meta) != set(roster):
        raise ValueError("Incomplete approved metadata roster")
    return meta, inventory["provenance"], run.read(cal.REG)["cut"]


def heartbeat(phase, name, done, total):
    v = dict(pid=os.getpid(), epoch=time.time(), phase=phase, recording=name, completed=done, total=total)
    run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
    if done % 5 == 0:
        print(v, flush=True)


def old_arrays(cell):
    with np.load(run.checked(cell["artifact"]), allow_pickle=False) as z:
        return {k: z[k] for k in ("frames", "offsets", "agents", "origin", "cv", "ols")}


def new_coefficients(source, old):
    np.testing.assert_array_equal(source.query_frames, old["frames"])
    parts = []
    for frame, lo, hi in zip(old["frames"], old["offsets"][:-1], old["offsets"][1:], strict=True):
        x = source.get_inputs(int(frame))
        at = x["history_complete"]
        h = x["positions"][at]
        np.testing.assert_array_equal(x["agent_ids"][at], old["agents"][lo:hi])
        np.testing.assert_array_equal(h[:, -1], old["origin"][lo:hi])
        coeff = model.coefficients(h)
        np.testing.assert_allclose(model.paths(coeff[:, :2]), np.stack((old["cv"][lo:hi], old["ols"][lo:hi]), axis=1),
                                   atol=1e-8, rtol=1e-10)
        parts.append(coeff[:, 2:])
    return np.concatenate(parts)


def save_array(path, coeff):
    tmp = path.with_suffix(".pending.npy")
    np.save(tmp, coeff, allow_pickle=False)
    tmp.replace(path)
    return run.artifact(path)


def freeze(key, manifest, old, roster):
    meta, provenance, _ = context(roster)
    records = []
    for i, cell in enumerate(old["clips"], 1):
        rec = cell["sequence"]
        p = PRIVATE/f"{rec}.forecast.json"
        if not p.exists():
            source = cal.reader(manifest, rec, meta, provenance)
            coeff = new_coefficients(source, old_arrays(cell))
            a = save_array(PRIVATE/f"{rec}.npy", coeff)
            run.once(p, run.stable(dict(registration_sha256=key, recording=rec, rows=len(coeff),
                                         parent=cell["artifact"], coefficients=a)))
        f = run.read(p)
        if f["registration_sha256"] != key or f["parent"] != cell["artifact"] or f["recording"] != rec:
            raise ValueError("Incorrect frozen recording")
        run.checked(f["coefficients"])
        records.append(f)
        heartbeat("freeze", rec, i, len(roster))
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration_sha256=key, recordings=records,
                                                   labels_used=False, confirmation_access=False)))


def score_recording(manifest, cell, receipt, meta, provenance, cut):
    rec = cell["sequence"]
    source = cal.reader(manifest, rec, meta, provenance)
    frozen = old_arrays(cell)
    y, valid = cal.label_arrays(source, frozen)
    cv, ols = frozen["cv"], frozen["ols"]
    sizes = np.diff(frozen["offsets"])
    weights = np.repeat(1./sizes, sizes)
    groups = np.full(len(y), 0, dtype=int)
    d = {}
    coeff = np.load(run.checked(receipt["coefficients"]), mmap_mode="r", allow_pickle=False)
    if coeff.shape != (len(y), len(model.ARMS)-2, 2, 2):
        raise ValueError("Coefficient/label roster mismatch")
    for k, mode in enumerate(model.ARMS[1:], 1):
        pred = ols if k == 1 else model.paths(coeff[:, k-2])
        m = policy(pred, cv, cv, y, valid, groups, easy_cut=cut["easy"], hard_cut=cut["hard"])
        risk = m["recordings"]["0"]
        d[mode] = dict(risk=risk, ade=m["known_ADE_pixel"], fde=m["available_endpoint_FDE_diagnostic_pixel"],
            tail=m["p95_known_ADE_pixel"], hard=m["subsets"]["hard"]["gain_percent"],
            cv=comparison_masses(cv, pred, cv, y, valid, weights),
            ols4=comparison_masses(ols, pred, cv, y, valid, weights))
    return dict(recording=rec, rows=len(y), queries=len(sizes), labelled=int(valid.any(1).sum()), arms=d)


def summarize(rows, roster, baseline):
    keys = sorted(set(roster.values()))
    draws = np.random.default_rng(SEED).integers(len(keys), size=(DRAWS, len(keys)))
    by_group = {k: {} for k in keys}
    for g in keys:
        group = [r for r in rows if roster[r["recording"]] == g]
        for k in model.ARMS[1:]:
            by_group[g][k] = {b: merge_comparisons([r["arms"][k][b] for r in group])["completion_lower_pp"]
                              for b in ("cv", "ols4")}
    metrics = {}
    for k in model.ARMS[1:]:
        rows_k = [r["arms"][k] for r in rows]
        risk = [r["risk"] for r in rows_k]
        stats = dict(edits=sum(x["selected"] for x in risk), support=sum(x["supported"] for x in risk),
            undefined_easy=sum(x["completion_easy_risk_upper"] is None for x in risk),
            both_floors_and_risk=sum(x["risk"]["supported"] and all(
                merge_comparisons([x[b]])["completion_lower_pp"] > 0. for b in ("cv", "ols4")) for x in rows_k),
            mean_clip_ade=float(np.mean([x["ade"] for x in rows_k if x["ade"] is not None])),
            mean_clip_fde=float(np.mean([x["fde"] for x in rows_k if x["fde"] is not None])),
            mean_clip_hard_gain_vs_cv_pp=float(np.mean([x["hard"] for x in rows_k if x["hard"] is not None])),
            max_clip_p95_ade=max(x["tail"] for x in rows_k if x["tail"] is not None),
            max_signed_easy_degradation=max(x["known_easy_net_degradation"] for x in risk
                                            if x["known_easy_net_degradation"] is not None))
        for ref in ("cv", "ols4"):
            v = np.asarray([by_group[g][k][ref] for g in keys])
            stats[f"lower_gain_vs_{ref}_pp"] = dict(point=float(v.mean()),
                ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(), positive_groups=int((v > 0.).sum()))
        metrics[k] = stats
    x = metrics["ols4"]["lower_gain_vs_cv_pp"]
    y = baseline["metrics"]["ols4_lower_pp"]
    np.testing.assert_allclose([x["point"], *x["ci95"]], [y["point"], *y["ci95"]], atol=1e-9, rtol=1e-10)
    return dict(rows=sum(r["rows"] for r in rows), queries=sum(r["queries"] for r in rows),
                labelled_rows=sum(r["labelled"] for r in rows), recordings=len(rows), source_groups=len(keys),
                by_group=by_group, metrics=metrics, original_ols4_replayed=True)


def score(key, manifest, old, roster, baseline, replay):
    p = PUBLIC/"freeze.json"
    seal = run.read(p)
    if (seal["registration_sha256"] != key or seal["labels_used"] or seal["confirmation_access"]
            or [r["recording"] for r in seal["recordings"]] != [r["sequence"] for r in old["clips"]]):
        raise ValueError("Require all approved forecast coefficients sealed")
    meta, provenance, cut = context(roster)
    rows = []
    for i, (cell, receipt) in enumerate(zip(old["clips"], seal["recordings"], strict=True), 1):
        rec = cell["sequence"]
        path = PRIVATE/f"{rec}.score.json"
        if not path.exists():
            if replay:
                raise ValueError("Replay cannot reopen labels")
            row = score_recording(manifest, cell, receipt, meta, provenance, cut)
            run.once(path, run.stable(dict(registration_sha256=key, coefficients=receipt["coefficients"], row=row)))
        r = run.read(path)
        if (r["registration_sha256"] != key or r["coefficients"] != receipt["coefficients"]
                or r["row"]["recording"] != rec or r["row"]["rows"] != receipt["rows"]):
            raise ValueError("Invalid recorded score")
        rows.append(r["row"])
        heartbeat("replay" if replay else "score", rec, i, len(roster))
    out = dict(registration_sha256=key, freeze=run.artifact(p), **summarize(rows, roster, baseline),
        result_source="fresh_run", confirmation_access=False, independent_confirmation=False,
        original_risk_rule=True, no_new_fits=True, deployment_changed=False, matched_intervention_claim=False)
    run.once(PUBLIC/"analysis.json", run.stable(out))
    print(out["metrics"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "freeze", "score", "replay"))
    phase = p.parse_args().phase
    for d in (PUBLIC, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, manifest, old, roster, baseline = register()
        if phase == "freeze":
            freeze(key, manifest, old, roster)
        elif phase in ("score", "replay"):
            score(key, manifest, old, roster, baseline, replay=phase == "replay")


if __name__ == "__main__":
    main()
