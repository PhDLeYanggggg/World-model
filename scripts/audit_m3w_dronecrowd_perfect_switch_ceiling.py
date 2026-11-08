"""Separate action-budget limits from prediction quality on opened calibration clips."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_calibration as cal
from scripts import report_m3w_dronecrowd_calibration as official
from src.evaluation.m3w_perfect_switch_ceiling import capped_masses, effective_caps
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_selection_forecast_readout import errors, policy


run, np = cal.run, cal.np
PUBLIC = cal.PUBLIC.parent/"dronecrowd_perfect_switch_ceiling_v1"
PRIVATE = cal.PRIVATE.parent/PUBLIC.name
K = (2, 4, 8, 16, 32, 64, 128)
SEEDS = cal.SEEDS


def freeze():
    baseline, bindings = official.verify_artifacts(ROOT, cal.PUBLIC, cal.roles.DEST)
    if official.summarize(baseline)["selected_arm"] is not None:
        raise ValueError("Expected the original unsuccessful calibration")
    manifest = run.read(cal.roles.DEST)
    parent = run.read(cal.PUBLIC/"forecast_freeze.json")
    roster = {r:g for r,g in run.read(cal.roles.GROUPS)["recording_group"].items()
              if manifest["recording_role_index"][r] == "calibration"}
    if len(roster) != 40 or len(set(roster.values())) != 15 or set(roster) != {c["sequence"] for c in parent["clips"]}:
        raise ValueError("Require exactly the approved calibration recordings")
    reg = PUBLIC/"registration.json"
    run.once(reg, run.stable(dict(
        parent=bindings, code=run.artifact(Path(__file__)),
        method=run.artifact(Path(capped_masses.__code__.co_filename)), seeds=SEEDS,
        original_cap="effective joint changes per query and seed", hypothetical_caps=K,
        oracle="zero ADE on selected known rows; omit all risk constraints for the upper bound",
        zero_harm_menu="positive available-label gain, CV for all unknown rows, from OLS4 and frozen sparse actions",
        aggregation="original query weights; combine recording masses per group; mean across seeds and groups",
        draws=3000, bootstrap_seed=73107, risk_rule_changed=False,
        scope="outcome_aware_calibration_diagnostic_not_policy_selection", confirmation_access=False,
    )))
    return run.sha(reg), manifest, parent, roster, baseline


def read_recording(cell, manifest, meta, inventory, cut):
    rec = cell["sequence"]
    source = cal.reader(manifest, rec, meta, inventory["provenance"])
    with np.load(run.checked(cell["artifact"]), allow_pickle=False) as loaded:
        frozen = {k:loaded[k] for k in loaded.files}
    np.testing.assert_array_equal(frozen["frames"], source.query_frames)
    y, valid = cal.label_arrays(source, frozen)
    base, ols, offsets = (frozen[k] for k in ("cv", "ols", "offsets"))
    sizes = np.diff(offsets)
    weights = np.repeat(1./sizes, sizes)
    known = valid.any(1)
    b, _ = errors(base, y, valid)
    o, _ = errors(ols, y, valid)
    perfect = np.where(known, b, 0.)
    ols_gain = np.where(known, np.maximum(b-o, 0.), 0.)
    mass = comparison_masses(base, ols, base, y, valid, weights)
    cap_const = np.broadcast_to(np.asarray(K), (len(sizes), len(K)))
    cap_mass = capped_masses(perfect, offsets, cap_const).tolist()
    joint_index = cal.model.ARMS.index("joint")
    rows = []
    for seed in SEEDS:
        slots, paths = (frozen[f"{k}_{seed}"] for k in ("index", "paths"))
        cap = effective_caps(base, offsets, slots[joint_index], paths[joint_index])
        if cap.max(initial=0) > 2:
            raise ValueError("Unexpected registered joint budget")
        gain = ols_gain.copy()
        witness = base.copy()
        witness[gain>0] = ols[gain>0]
        for ids, path in zip(slots, paths, strict=True):
            effective_caps(base, offsets, ids, path)
            use = ids >= 0
            ix = ids[use]
            error, _ = errors(path[use], y[ix], valid[ix])
            better = known[ix] & ((b[ix]-error) > gain[ix])
            gain[ix[better]] = b[ix[better]]-error[better]
            witness[ix[better]] = path[use][better]
        if np.any(witness[~known] != base[~known]):
            raise ValueError("Zero-harm witness must fall back on wholly unknown rows")
        risk = policy(witness, base, base, y, valid, np.zeros(len(y), dtype=int),
                      easy_cut=cut["easy"], hard_cut=cut["hard"])["recordings"]["0"]
        if any(risk[k] is not None and risk[k] > 1e-12 for k in (
                "completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation")):
            raise ValueError("Observed zero-harm construction violates original risk")
        upper = float(capped_masses(perfect, offsets, cap[:, None])[0])
        witness_cap = float(capped_masses(gain, offsets, cap[:, None])[0])
        if witness_cap > upper+1e-9:
            raise ValueError("Candidate subset exceeds the perfect-switch ceiling")
        rows.append(dict(seed=seed, effective_joint_switches=int(cap.sum()),
                         capped_perfect_mass=upper, capped_zero_harm_menu_gain_mass=witness_cap,
                         unrestricted_zero_harm_menu_gain_mass=float(weights@gain),
                         unrestricted_witness_switches=risk["selected"], unrestricted_risk=risk,
                         capped_perfect_upper_vs_ols4_mass=upper-mass["known_delta"]))
    return dict(recording=rec, queries=len(sizes), rows=len(y), known=int(known.sum()), mass=mass,
                perfect_at_uniform_cap_mass=cap_mass, views=rows)


def summarize(records, roster, baseline):
    fields = ("capped_perfect_mass", "capped_zero_harm_menu_gain_mass",
              "unrestricted_zero_harm_menu_gain_mass", "capped_perfect_upper_vs_ols4_mass")
    by_group = {}
    for group in sorted(set(roster.values())):
        cells = [r for r in records if roster[r["recording"]] == group]
        mass = sum(r["mass"]["reference"] for r in cells)
        if mass <= 0:
            raise ValueError("Required CV mass missing")
        if any(tuple(v["seed"] for v in r["views"]) != SEEDS for r in cells):
            raise ValueError("Expected all registered seeds")
        out = {k:100*sum(v[k] for r in cells for v in r["views"])/(len(SEEDS)*mass) for k in fields}
        out["ols4_lower_pp"] = 100*sum(r["mass"]["known_delta"]-r["mass"]["unknown_radius"] for r in cells)/mass
        out["ols4_observed_pp"] = 100*sum(r["mass"]["known_delta"] for r in cells)/mass
        out.update({f"perfect_k{k}_pp":100*sum(r["perfect_at_uniform_cap_mass"][j] for r in cells)/mass
                    for j,k in enumerate(K)})
        by_group[group] = out
    draws = np.random.default_rng(73107).integers(len(by_group),size=(3000,len(by_group)))
    metrics = {}
    for key in next(iter(by_group.values())):
        z = np.asarray([r[key] for r in by_group.values()])
        metrics[key] = dict(point=float(z.mean()),ci95=np.quantile(z[draws].mean(1),[.025,.975]).tolist())
    expected = baseline["summary"]["effects"][f"{cal.OLS}_vs_cv"]["completion_lower_pp"]
    np.testing.assert_allclose(
        [metrics["ols4_lower_pp"]["point"], *metrics["ols4_lower_pp"]["ci95"]],
        [expected["point"], *expected["interval95"]], rtol=1e-10, atol=1e-10,
    )
    actual = sum(v["effective_joint_switches"] for r in records for v in r["views"])
    if actual != baseline["summary"]["risk"]["joint"]["switches"]:
        raise ValueError("Original joint switch count did not reproduce")
    risks = [v["unrestricted_risk"] for r in records for v in r["views"]]
    return dict(metrics=metrics, groups=by_group, original_joint_switches=actual, views=len(risks),
                unrestricted_witness_supported_views=sum(r["supported"] for r in risks),
                unrestricted_witness_undefined_easy_views=sum(r["completion_easy_risk_upper"] is None for r in risks),
                original_ols4_readout_reproduced=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "score", "replay"))
    args = cli.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, manifest, parent, roster, baseline = freeze()
        if args.phase == "register":
            return
        inventory = run.read(cal.CACHE)
        meta = {r["sequence_id"]:r for r in inventory["recordings"] if r["sequence_id"] in roster}
        cut = run.read(cal.REG)["cut"]
        records = []
        for i, cell in enumerate(parent["clips"], 1):
            cached = PRIVATE/f'{cell["sequence"]}.json'
            if cached.exists():
                v = run.read(cached)
                if v["registration_sha256"] != binding or v["clip"] != cell["artifact"]:
                    raise ValueError("Checkpoint does not match the registered recording")
            elif args.phase == "score":
                v = dict(registration_sha256=binding, clip=cell["artifact"],
                         record=read_recording(cell, manifest, meta, inventory, cut))
                run.once(cached, run.stable(v))
            else:
                raise ValueError("Replay cannot open recording labels")
            records.append(v["record"])
            beat = dict(pid=os.getpid(), completed=i, total=len(roster), phase=args.phase)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", beat)
            if i % 5 == 0:
                print(beat, flush=True)
        run.once(PUBLIC/"analysis.json", run.stable(dict(
            registration_sha256=binding, recordings=records, **summarize(records, roster, baseline),
            results_use_calibration_labels_only=True, no_new_training=True, risk_rule_changed=False,
            confirmation_access=False, deployment=False, stage5c_executed=False, smc_enabled=False,
        )))
        receipt = PUBLIC/f"{args.phase}_execution.json"
        if not receipt.exists():
            run.once(receipt, dict(pid=os.getpid(), finished_unix=time.time(),
                                   analysis_sha256=run.sha(PUBLIC/"analysis.json"),
                                   calibration_labels_reopened=args.phase=="score"))


if __name__ == "__main__":
    main()
