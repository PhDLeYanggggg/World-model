"""Select a standalone hindcast policy in fitting sources, then diagnose transfer."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dense_motion_calibration as base
from src.evaluation.m3w_recording_forecast_comparison import merge_comparisons
from src.world_model import m3w_hindcast_policy as method


run, cal, np = base.run, base.cal, base.np
OUT = base.PUBLIC.parent/"standalone_hindcast_v1"
PRIVATE = base.PRIVATE.parent/OUT.name
FIXED = "fixed_half"
MATCH = "matched_latest_origin"
ARM = "standalone"
PAIRS = {f"{a}_vs_{'cv' if b == cal.BASE else b}": (b, a, cal.BASE)
         for a, b in ((ARM, cal.BASE), (MATCH, cal.BASE), (FIXED, cal.BASE), (cal.OLS, cal.BASE),
                      (ARM, MATCH), (ARM, FIXED), (ARM, cal.OLS))}


def register():
    train = base.train_source
    identity = train.registration()[0]["identity"]
    cut = run.read(cal.REG)["cut"]
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), method=run.artifact(Path(method.__file__)),
        parent_calibration_protocol=run.artifact(cal.REG), fitting_population=run.artifact(train.PUBLIC/"fit_population.json"),
        fitting_ids_sha256=identity["train_ids_sha256"], menu=method.MENU,
        selection="max worst group risk-pass fraction, then mean pass fraction, then mean conservative gain",
        validation="leave one source group out; select from the other eight groups",
        transfer="one fitted policy, original opened calibration roster, no outcome-based threshold updates",
        matched="same query and amplitude counts, rank by latest-origin hindcast gain at the selected amplitude",
        confirmation_access=False, risk_rule_changed=False, fit_train_seeds=0)))
    return run.sha(reg), identity, cut


def fit(binding, identity, cut, beat):
    train = base.train_source
    data = train.fitting_only(identity, beat)
    if len(data["groups"]) != 9 or len(data["ids"]) != 69746:
        raise ValueError("Require the original nine fitting groups")
    prep = train.prep
    cfg = run.read(train.source.CONFIG)
    admission = prep.load_admission(ROOT, ROOT/cfg["admission"])
    support = run.read(train.bank.old.HOME/"movingdronecrowd_intake_v1/analysis.json")
    sizes = {r["clip"]: r["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
             for r in support["recordings"]}
    group_rows = {}
    lo = 0
    for rec in data["recordings"]:
        entry = admission["recordings"][rec]
        hi = lo+sizes[entry["clip"]]
        g = entry["exclusion_group"]
        if not np.all(data["sites"][lo:hi] == g):
            raise ValueError("Fitting recording alignment mismatch")
        h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
        cv = data["geometry"][lo:hi, 332:356].reshape(-1, 12, 2).astype(float)
        ols = base.motion.slope(h[:, -4:])[:, None]*np.arange(1., 13.)[None, :, None]
        paths = {cal.BASE: cv}
        paths.update({k: base.motion.make_path(cv, ols, v) for k, v in method.menu(
            method.backtests(h), np.any(cv != ols, axis=(1, 2))).items()})
        # Fitting uses row-weighted gain only as a tie-break. Transfer retains
        # the original query-weighted scoring protocol.
        result = base.score_recording(paths, {k: (cal.BASE, k, cal.BASE) for k in method.MENU},
            data["target"][lo:hi], data["valid"][lo:hi], np.arange(hi-lo), rec,
            easy_cut=cut["easy"], hard_cut=cut["hard"])
        group_rows.setdefault(g, []).append(result)
        lo = hi
    if lo != len(data["ids"]) or set(group_rows) != set(data["sites"]):
        raise ValueError("Fitting roster not fully scored")
    scores = {k: {} for k in method.MENU}
    for k in method.MENU:
        for g, rows in group_rows.items():
            scores[k][g] = dict(recordings=len(rows), passed=sum(r["metrics"][k]["risk"]["supported"] for r in rows),
                gain_pp=merge_comparisons([r["comparisons"][k] for r in rows])["completion_lower_pp"])
    groups = sorted(group_rows)
    oof = {g: method.select(scores, [other for other in groups if other != g]) for g in groups}
    run.once(OUT/"fitting.json", run.stable(dict(registration=binding, scores=scores,
        chosen=method.select(scores, groups), left_out={g: dict(arm=k, **scores[k][g]) for g, k in oof.items()},
        fitting_rows=lo, fitting_groups=groups, opened_calibration_outcomes_read=False)))


def target_context():
    freeze_path = cal.PUBLIC/"forecast_freeze.json"
    base.verified.verify_receipt(ROOT, run.read(cal.PUBLIC/"infer_execution.json"), freeze_path)
    manifest = run.read(cal.roles.DEST)
    parent = run.read(freeze_path)
    roster = {r: g for r, g in run.read(cal.roles.GROUPS)["recording_group"].items()
              if manifest["recording_role_index"][r] == "calibration"}
    if (len(roster) != 40 or len(set(roster.values())) != 15
            or set(roster) != {x["sequence"] for x in parent["clips"]}
            or set(roster) & set(manifest["assignments"]["confirmation"]["recordings"])
            or parent["future_labels_read"] or parent["registration"] != run.sha(cal.REG)):
        raise ValueError("Use the 40 opened clips only")
    inv = run.read(cal.CACHE)
    meta = {r["sequence_id"]: r for r in inv["recordings"] if r["sequence_id"] in roster}
    return manifest, parent, roster, inv, meta


def infer(binding, beat):
    fit = run.read(OUT/"fitting.json")
    if fit["registration"] != binding or fit["opened_calibration_outcomes_read"]:
        raise ValueError("Require fitting-only frozen choice")
    manifest, parent, roster, inv, meta = target_context()
    clips = []
    for i, clip in enumerate(parent["clips"], 1):
        rec = clip["sequence"]
        marker = PRIVATE/f"choice_{rec}.json"
        if not marker.exists():
            f = base.frozen_paths(clip)
            src = cal.reader(manifest, rec, meta, inv["provenance"])
            gains = method.backtests(base.history(src, f))
            legal = np.any(f["cv"] != f["ols"], axis=(1, 2))
            action = method.menu(gains, legal)[fit["chosen"]]
            level = int(fit["chosen"].rsplit("_", 1)[1])
            matched = base.motion.matched_hindcast(action, gains[:, -1, level-1], f["offsets"], legal)
            def counts(a):
                return np.diff(np.r_[0, np.cumsum(a > 0)][f["offsets"]])

            np.testing.assert_array_equal(counts(action), counts(matched))
            p = marker.with_suffix(".npz")
            tmp = p.with_suffix(".pending.npz")
            np.savez_compressed(tmp, action=action, matched=matched)
            tmp.replace(p)
            run.once(marker, run.stable(dict(registration=binding, fitting=run.artifact(OUT/"fitting.json"),
                parent=clip["artifact"], recording=rec, choices=run.artifact(p),
                matched_queries=len(f["frames"]), switches=int((action > 0).sum()))))
        r = run.read(marker)
        if r["parent"] != clip["artifact"] or r["registration"] != binding or r["fitting"] != run.artifact(OUT/"fitting.json"):
            raise ValueError("Stale choice cache")
        run.checked(r["choices"])
        clips.append(r)
        if i % 5 == 0:
            beat(finished=i, total=len(roster))
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, clips=clips,
        calibration_labels_used=False, chosen=fit["chosen"])))


def score(binding, cut, beat, replay):
    manifest, parent, roster, inv, meta = target_context()
    freeze = run.read(OUT/"freeze.json")
    if freeze["registration"] != binding or freeze["calibration_labels_used"]:
        raise ValueError("Require frozen causal predictions")
    original, _ = base.verified.verify_artifacts(ROOT, cal.PUBLIC, cal.roles.DEST)
    rows = []
    for i, (clip, choice) in enumerate(zip(parent["clips"], freeze["clips"], strict=True), 1):
        rec = clip["sequence"]
        if choice["recording"] != rec or choice["parent"] != clip["artifact"]:
            raise ValueError("Frozen recording alignment changed")
        cache = PRIVATE/f"score_{rec}.json"
        if not cache.exists():
            if replay:
                raise ValueError("Replay cannot open outcome arrays")
            f = base.frozen_paths(clip)
            src = cal.reader(manifest, rec, meta, inv["provenance"])
            target, valid = cal.label_arrays(src, f)
            with np.load(run.checked(choice["choices"]), allow_pickle=False) as z:
                action, matched = z["action"], z["matched"]
            paths = {cal.BASE: f["cv"], cal.OLS: f["ols"], FIXED: .5*(f["cv"]+f["ols"]),
                     ARM: base.motion.make_path(f["cv"], f["ols"], action),
                     MATCH: base.motion.make_path(f["cv"], f["ols"], matched)}
            result = base.score_recording(paths, PAIRS, target, valid,
                np.repeat(f["frames"], np.diff(f["offsets"])), rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
            run.once(cache, run.stable(dict(choices=choice["choices"], result=result)))
        r = run.read(cache)
        if r["choices"] != choice["choices"]:
            raise ValueError("Stale score cache")
        rows.append(dict(r["result"], source=roster[rec], seed=0))
        if i % 5 == 0:
            beat(finished=i, total=len(roster))
    summary = base.grouped_readout(rows, roster, (0,))
    ols = f"{cal.OLS}_vs_cv"
    for k, v in summary["effects"][ols].items():
        ref = original["summary"]["effects"][ols][k]
        np.testing.assert_allclose([v["point"], *v["interval95"]], [ref["point"], *ref["interval95"]], atol=1e-10)
    if summary["risk"][ARM]["switches"] != summary["risk"][MATCH]["switches"]:
        raise ValueError("Actual intervention counts do not match")
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, freeze=run.artifact(OUT/"freeze.json"),
        summary=summary, chosen=freeze["chosen"], matched_queries=sum(x["matched_queries"] for x in freeze["clips"]),
        deterministic_seed_count=0, independent_confirmation=False, risk_rule_changed=False,
        calibration_outcomes_used_to_select=False, deployable=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "fit", "infer", "score", "replay"))
    a = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut = register()

        def beat(**v):
            v.update(pid=os.getpid(), phase=a.phase)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            print(v, flush=True)

        if a.phase == "fit":
            fit(binding, identity, cut, beat)
        elif a.phase == "infer":
            infer(binding, beat)
        elif a.phase in ("score", "replay"):
            score(binding, cut, beat, a.phase == "replay")


if __name__ == "__main__":
    main()
