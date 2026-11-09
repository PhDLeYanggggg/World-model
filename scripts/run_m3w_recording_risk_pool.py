"""Freeze equal-count forecasts, then cross-fit recording-level risk offsets."""

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_ols_relative_gain as parent
from src.world_model import m3w_recording_risk_pool as study


prior, np, torch, run, cal, target, model = parent.prior, parent.np, parent.torch, parent.run, parent.cal, parent.target, parent.model
PUBLIC = parent.PUBLIC.parent/"dronecrowd_recording_risk_pool_v1"
PRIVATE = parent.PRIVATE.parent/PUBLIC.name


def beat(phase, completed, total):
    d = dict(pid=prior.os.getpid(), epoch=prior.time.time(), phase=phase, completed=completed, total=total)
    run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", d)
    if completed % 5 == 0:
        print(d, flush=True)


def register():
    k, manifest, old, roster, _ = parent.register()
    source = parent.PUBLIC/"analysis.json"
    results = run.read(source)
    freeze = run.read(run.checked(results["freeze"]))
    if (results["registration"] != k or freeze["registration"] != k or freeze["labels_used"]
            or freeze["confirmation_access"] or results["confirmation_access"] or len(set(roster.values())) != 15):
        raise ValueError("Require the completed calibration-only parent")
    p = PUBLIC/"registration.json"
    run.once(p, run.stable(dict(parent=run.artifact(source), code=[run.artifact(Path(x)) for x in (__file__, study.__file__)],
        arms=study.ARMS, prices=study.PRICES, rules=study.RULES, seeds=parent.SEEDS,
        counts="same actual OLS4 replacements per query and seed as frozen CV-target independent choice",
        calibration="leave out the entire source, fit per-arm additive risk corrections on the other 14 sources; "
                    "compare raw, median of within-source medians, and maximum observed residual envelope",
        selection="largest positive predicted OLS4 gain among arms below both projected risk limits, or abstain",
        original_cv_2pct=True, independent_confirmation=False, original_sparse_protocol_unchanged=True,
        confirmation_access=False, no_new_training=True)))
    return run.sha(p), manifest, old, roster, freeze, results


@torch.inference_mode()
def forecast(src, d, x, nets):
    np.testing.assert_array_equal(src.query_frames, d["frames"])
    blocks, totals = [], np.zeros((len(nets), len(study.ARMS), 6))
    for f, lo, hi in zip(d["frames"], d["offsets"][:-1], d["offsets"][1:], strict=True):
        q = src.get_inputs(int(f))
        at = q["history_complete"]
        h = q["positions"][at]
        np.testing.assert_array_equal(q["agent_ids"][at], d["agents"][lo:hi])
        np.testing.assert_array_equal(h[:, -1], d["origin"][lo:hi])
        a, o, bank, scale = model.features(h)
        bank[:, 0], bank[:, 1] = d["cv"][lo:hi], d["ols"][lo:hi]
        aa, oo = torch.from_numpy(a), torch.from_numpy(o)
        choices = []
        for s, net in enumerate(nets):
            v, ref = net(aa, oo)
            z, m = study.query_menu(bank, v.numpy()*scale[:, None, None], ref.numpy()*scale[:, None], x[lo:hi, s, 0])
            choices.append(z)
            totals[s] += m
        blocks.append(np.stack(choices, axis=1))
    return np.concatenate(blocks), totals.tolist()


def freeze(key, manifest, old, roster, seal):
    nets = prior.load_nets(run.sha(prior.PUBLIC/"registration.json"))
    meta, provenance, _ = target.context(roster)
    rows = []
    for i, (cell, f) in enumerate(zip(old["clips"], seal["records"], strict=True), 1):
        rec = cell["sequence"]
        if rec != f["recording"]:
            raise ValueError("Wrong frozen control")
        p = PRIVATE/f"{rec}.freeze.json"
        if not p.exists():
            src = cal.reader(manifest, rec, meta, provenance)
            d = target.old_arrays(cell)
            x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
            z, totals = forecast(src, d, x, nets)
            c = target.save_array(PRIVATE/f"{rec}.choices.npy", z)
            run.once(p, run.stable(dict(registration=key, recording=rec, rows=len(z), parent=f,
                                       choices=c, predicted_recording_masses=totals)))
        row = run.read(p)
        if row["registration"] != key or row["recording"] != rec or row["parent"] != f:
            raise ValueError("Invalid forecast receipt")
        run.checked(row["choices"])
        rows.append(row)
        beat("freeze", i, len(roster))
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration=key, recordings=rows, outcome_free=True,
                                                  confirmation_access=False)))


def observed(manifest, cell, f, meta, provenance, cut):
    rec = cell["sequence"]
    src = cal.reader(manifest, rec, meta, provenance)
    d = target.old_arrays(cell)
    y, v = cal.label_arrays(src, d)
    cv, ols, at = (d[k] for k in ("cv", "ols", "offsets"))
    idx, group = np.arange(len(cv)), np.zeros(len(cv), dtype=int)
    w = np.repeat(1./np.diff(at), np.diff(at))
    x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
    co = np.load(run.checked(f["parent"]["coefficients"]), mmap_mode="r", allow_pickle=False)
    bank = np.concatenate((ols[:, None], model.floor.paths(co[:, :3])), axis=1)
    if x.shape != (len(cv), len(parent.SEEDS), len(study.ARMS)) or x.max(initial=0) > 3:
        raise ValueError("Misaligned choice bank")
    seeds = []
    for s in range(len(parent.SEEDS)):
        anchor = bank[idx, x[:, s, 0]]
        c = np.add.reduceat((anchor != ols).any((1, 2)).astype(int), at[:-1])
        view = []
        for j in range(len(study.ARMS)):
            path = bank[idx, x[:, s, j]]
            changes = (path != ols).any((1, 2))
            np.testing.assert_array_equal(np.add.reduceat(changes.astype(int), at[:-1]), c)
            m = prior.policy(path, cv, cv, y, v, group, easy_cut=cut["easy"], hard_cut=cut["hard"])
            comps = {k: prior.comparison_masses(p, path, cv, y, v, w) for k, p in (("ols4", ols), ("cv", cv), ("independent", anchor))}
            actual = {k: prior.merge_comparisons([r])["completion_lower_pp"] for k, r in comps.items()}
            r = m["recordings"]["0"]
            risk = [r["completion_all_risk_upper"], r["completion_easy_risk_upper"]]
            view.append(dict(edits=int(changes.sum()), risk=risk, risk_pass=r["supported"],
                dual_pass=r["supported"] and all(actual[k] > 0. for k in ("ols4", "cv")), actual=actual, masses=comps,
                ade=m["known_ADE_pixel"], fde=m["available_endpoint_FDE_diagnostic_pixel"], p95=m["p95_known_ADE_pixel"]))
        seeds.append(view)
    return seeds


def summarize(rows, roster, parent_metrics):
    by_source = {}
    for r in rows:
        for pred, actual in zip(r["predicted_recording_masses"], r["observed"], strict=True):
            risk, gain = study.prediction(pred)
            if not np.isfinite(risk).all() or not np.isfinite(gain).all():
                raise ValueError("Forecast needs positive predicted support")
            actual_risk = np.asarray([[1. if x is None else x for x in v["risk"]] for v in actual])
            by_source.setdefault(roster[r["recording"]], []).append((risk, gain, actual_risk-risk, actual))
    residuals = {g: [x[2] for x in v] for g, v in by_source.items()}
    cv_control = parent_metrics["cv_gain"]
    select = {k: [] for k in study.RULES}
    for g, views in sorted(by_source.items()):
        offsets = study.group_offsets(residuals, g)
        for risk, gain, _, actual in views:
            for k, offset in zip(study.RULES, offsets, strict=True):
                idx = study.select(risk, gain, offset)
                if idx is not None:
                    select[k].append((g, study.ARMS[idx], actual[idx]))
    chosen = {}
    for k, x in select.items():
        chosen[k] = dict(released=len(x), source_groups=len({g for g, _, _ in x}),
            passed_risk=sum(v["risk_pass"] for _, _, v in x), passed_both_and_risk=sum(v["dual_pass"] for _, _, v in x),
            arms={a: sum(j == a for _, j, _ in x) for a in study.ARMS})
    boot = np.random.default_rng(target.SEED).integers(len(by_source), size=(target.DRAWS, len(by_source)))
    menu = {}
    for j, arm in enumerate(study.ARMS):
        full = [t[3][j] for z in by_source.values() for t in z]
        out = dict(risk_pass=sum(r["risk_pass"] for r in full), dual_pass=sum(r["dual_pass"] for r in full),
            edits=sum(r["edits"] for r in full), ade=float(np.mean([r["ade"] for r in full])),
            fde=float(np.mean([r["fde"] for r in full])), worst_p95=max(r["p95"] for r in full),
            easy_over_2=sum(r["risk"][1] is not None and r["risk"][1] > .02 for r in full))
        for ref in ("independent", "ols4"):
            z = np.asarray([prior.merge_comparisons([t[3][j]["masses"][ref] for t in by_source[g]])["completion_lower_pp"]
                            for g in sorted(by_source)])
            out[f"vs_{ref}"] = dict(point=float(z.mean()), ci95=np.quantile(z[boot].mean(1), [.025, .975]).tolist(),
                                    positive_groups=int((z > 0.).sum()))
        if out["edits"] != cv_control["edits_over_ols"]:
            raise ValueError("Matched edit totals changed")
        menu[arm] = out
    a = menu["independent"]
    if a["risk_pass"] != cv_control["risk_pass"] or a["dual_pass"] != cv_control["dual_pass"]:
        raise ValueError("Original control risk changed")
    np.testing.assert_allclose(a["vs_ols4"]["point"], cv_control["vs_ols4"]["lower_pp"], atol=1e-12)
    return dict(menu=menu, held_source_selection=chosen, source_groups=len(by_source), views=sum(map(len, by_source.values())),
        hindsight_available_joint=sum(any(v["dual_pass"] for v in t[3]) for z in by_source.values() for t in z))


def score(key, manifest, old, roster, parent_result, replay):
    p = PUBLIC/"freeze.json"
    freeze = run.read(p)
    if freeze["registration"] != key or not freeze["outcome_free"] or freeze["confirmation_access"]:
        raise ValueError("Require a sealed forecast menu")
    meta, provenance, cut = target.context(roster)
    rows = []
    for i, (cell, f) in enumerate(zip(old["clips"], freeze["recordings"], strict=True), 1):
        rec = cell["sequence"]
        if f["recording"] != rec:
            raise ValueError("Wrong scoring roster")
        path = PRIVATE/f"{rec}.score.json"
        if not path.exists():
            if replay:
                raise ValueError("Replay cannot read labels")
            r = observed(manifest, cell, f, meta, provenance, cut)
            run.once(path, run.stable(dict(freeze=f, observed=r)))
        z = run.read(path)
        if z["freeze"] != f:
            raise ValueError("Wrong score receipt")
        rows.append(dict(recording=rec, predicted_recording_masses=f["predicted_recording_masses"], observed=z["observed"]))
        beat("replay" if replay else "score", i, len(roster))
    result = summarize(rows, roster, parent_result["metrics"])
    run.once(PUBLIC/"analysis.json", run.stable(dict(**result, freeze=run.artifact(p), result_source="fresh_run",
        original_cv_2pct=True, independent_confirmation=False, deployment_changed=False, confirmation_access=False)))
    print(result, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (PUBLIC, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, manifest, old, roster, seal, result = register()
        if phase == "freeze":
            freeze(key, manifest, old, roster, seal)
        else:
            score(key, manifest, old, roster, result, replay=phase == "replay")


if __name__ == "__main__":
    main()
