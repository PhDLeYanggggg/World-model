"""Train long-horizon analytic-bank costs on Moving A, then transfer to calibration."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_motion_floor as target
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses, merge_comparisons
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_full_scene_harm_transfer as model


cal, run, np, torch = target.cal, target.run, target.np, target.cal.torch
PUBLIC, PRIVATE = target.PUBLIC.parent/"dronecrowd_full_scene_transfer_v1", target.PRIVATE.parent/"dronecrowd_full_scene_transfer_v1"
DONORS = run.old.prior.PARENT/"population.json"
SCREEN = target.PUBLIC.parent/"dronecrowd_controller_coverage_v1"/"analysis.json"
STEPS, SAVE, BATCH, SEEDS = 600, 100, 192, cal.SEEDS
ARMS = model.METHODS


def beat(phase, **fields):
    v = dict(pid=os.getpid(), epoch=time.time(), phase=phase, **fields)
    run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
    if fields.get("completed", 1) % 5 == 0 or phase == "train":
        print(v, flush=True)


def register():
    parent_key, manifest, old, roster, _ = target.register()
    path = target.PUBLIC/"analysis.json"
    result = run.read(path)
    seal = run.read(run.checked(result["freeze"]))
    screened = run.read(SCREEN)
    for p, expected in screened["sources"].items():
        if run.sha(ROOT/p) != expected:
            raise ValueError("Changed source screen")
    if (seal["registration_sha256"] != parent_key or seal["labels_used"] or result["confirmation_access"]
            or screened["possible_fit_recording_union"] != 34
            or screened["sources"][str(DONORS.relative_to(ROOT))] != run.sha(DONORS)):
        raise ValueError("Require screened donors and the sealed target roster")
    d = run.read(DONORS)
    if d["role"] != "parent_A_only_out_of_fold_training_donors" or d["outer_readout"] or not d["labels_never_inputs"]:
        raise ValueError("Donor role changed")
    p = PUBLIC/"registration.json"
    run.once(p, run.stable(dict(code=[run.artifact(Path(v)) for v in (__file__, model.__file__)],
        parents=[run.artifact(v) for v in (path, SCREEN, DONORS, cal.roles.DEST)],
        scope="existing_Moving_A_train_to_15_approved_DroneCrowd_calibration_groups",
        observations=8, forecast=12, source_groups=14, seed=SEEDS, steps=STEPS, checkpoint=SAVE, batch=BATCH,
        proposal="fixed CV, OLS4, median slope, fitted origin and pooled origin; same as prior diagnostic",
        objective="signed 12-step available-ADE gain with unknown debit, positive harm and CV mass, "
                  "conditional easy harm and easy CV mass; actor history scale only; supervised on donor outcomes",
        controllers="positive gain; individual predicted 2% gate; query-aggregate predicted 2% prefix; "
                    "gain-only control inherits the exact number of interventions per query",
        calibration_labels_used_for_training_or_thresholds=False, original_2pct=True, independent_confirmation=False,
        full_scene_diagnostic=True, original_two_edit_protocol_changed=False,
        confirmation_access=False, torch_threads=4, num_workers=0)))
    return run.sha(p), manifest, old, roster, seal, result


def donor_data():
    d = run.population()
    z = run.old.prior.arrays(("target_eval", "valid"))
    files = run.read(DONORS)["files"]
    for k in ("target_eval", "valid"):
        run.checked(files[k])
    ids = d["global_ids"]
    if not np.array_equal(ids, np.sort(ids)) or len(ids) != len(z["valid"]):
        raise ValueError("Donor alignment changed")
    cut = run.read(cal.REG)["cut"]["easy"]
    parts = {}
    for q in d["groups"]:
        site = str(d["sites"][q[0]])
        a, o, paths, scale = model.features(d["history"][q])
        y = z["target_eval"][q]-d["origin"][q, None]
        v = z["valid"][q]
        m, r = model.targets(paths, y, v, scale, cut)
        parts.setdefault(site, []).append((a, o, m, r))
    result = {k: tuple(torch.from_numpy(np.concatenate(x)) for x in zip(*v, strict=True)) for k, v in sorted(parts.items())}
    if len(result) != 14 or sum(len(v[0]) for v in result.values()) != 125540:
        raise ValueError("Incomplete donor roster")
    return result


def fit(key, seed, samples):
    rows = list(samples.values())
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1867)
    widths = [rows[0][j].shape[-1] for j in range(2)]
    net = model.HarmHead(*widths)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    ident = dict(registration=key, seed=seed, widths=widths, donors={k: len(v[0]) for k, v in samples.items()})
    path = PRIVATE/f"seed_{seed}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Invalid full-scene head checkpoint")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, trace = ck["step"], ck["trace"]
    for step in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(rows), 1./len(rows)))
        picks = [(r, rng.integers(len(r[0]), size=n)) for r, n in zip(rows, counts, strict=True) if n]
        a, o, m, r = [torch.cat([d[k][i] for d, i in picks]) for k in range(4)]
        loss = model.loss(*net(a, o), m, r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=net.state_dict(),
                            optimizer=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("train", seed=seed, step=step, total=STEPS)
    return dict(identity=ident, checkpoint=run.artifact(path), trace=trace)


def train(key):
    data = donor_data()
    run.once(PUBLIC/"training.json", run.stable(dict(registration=key, fits=[fit(key, k, data) for k in SEEDS],
        donor_rows=sum(len(v[0]) for v in data.values()), calibration_labels_used=False)))


def load_nets(key):
    p = PUBLIC/"training.json"
    fits = run.read(p)
    if fits["registration"] != key or fits["calibration_labels_used"] or tuple(r["identity"]["seed"] for r in fits["fits"]) != SEEDS:
        raise ValueError("Require the frozen donor models")
    nets = []
    for r in fits["fits"]:
        ck = torch.load(run.checked(r["checkpoint"]), weights_only=True, map_location="cpu")
        if ck["identity"] != r["identity"] or ck["step"] != STEPS:
            raise ValueError("Incorrect donor checkpoint")
        net = model.HarmHead(*ck["identity"]["widths"]).eval()
        net.load_state_dict(ck["net"])
        nets.append(net)
    return nets


@torch.inference_mode()
def selections(source, prev, base, nets):
    np.testing.assert_array_equal(prev["frames"], source.query_frames)
    out = []
    for frame, lo, hi in zip(prev["frames"], prev["offsets"][:-1], prev["offsets"][1:], strict=True):
        r = source.get_inputs(int(frame))
        h = r["positions"][r["history_complete"]]
        np.testing.assert_array_equal(r["agent_ids"][r["history_complete"]], prev["agents"][lo:hi])
        np.testing.assert_array_equal(h[:, -1], prev["origin"][lo:hi])
        a, x, bank, scale = model.features(h)
        np.testing.assert_allclose(bank[:, :2], np.stack((prev["cv"][lo:hi], prev["ols"][lo:hi]), axis=1),
                                   atol=1e-8, rtol=1e-10)
        np.testing.assert_array_equal(bank[:, 2:], model.floor.paths(base[lo:hi, :3]))
        c, o = prev["cv"][lo:hi], prev["ols"][lo:hi]
        legal = (np.concatenate((o[:, None], bank[:, 2:]), axis=1) != c[:, None]).any((2, 3))
        block = []
        for net in nets:
            m, ref = net(torch.from_numpy(a), torch.from_numpy(x))
            decision = model.choose(m.numpy()*scale[:, None, None], ref.numpy()*scale[:, None], legal)
            block.append(decision)
        out.append(np.stack(block, axis=1))
    return np.concatenate(out)


def freeze(key, manifest, old, roster, parent):
    nets = load_nets(key)
    meta, provenance, _ = target.context(roster)
    rows = []
    for i, (cell, f) in enumerate(zip(old["clips"], parent["recordings"], strict=True), 1):
        rec = cell["sequence"]
        if f["recording"] != rec:
            raise ValueError("Wrong target coefficients")
        p = PRIVATE/f"{rec}.choices.json"
        if not p.exists():
            source = cal.reader(manifest, rec, meta, provenance)
            coeff = np.load(run.checked(f["coefficients"]), allow_pickle=False, mmap_mode="r")
            choices = selections(source, target.old_arrays(cell), coeff, nets)
            bank = target.save_array(PRIVATE/f"{rec}.choices.npy", choices)
            run.once(p, run.stable(dict(registration=key, recording=rec, choices=bank, rows=len(choices), parent=f)))
        r = run.read(p)
        if r["registration"] != key or r["parent"] != f or r["recording"] != rec:
            raise ValueError("Invalid choice receipt")
        run.checked(r["choices"])
        rows.append(r)
        beat("freeze", recording=rec, completed=i, total=len(roster))
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration=key, records=rows, training=run.artifact(PUBLIC/"training.json"),
                                                   held_labels_used=False, confirmation_access=False)))


def score_recording(manifest, cell, f, meta, provenance, cut):
    rec = cell["sequence"]
    source = cal.reader(manifest, rec, meta, provenance)
    old = target.old_arrays(cell)
    y, v = cal.label_arrays(source, old)
    cv, ols, offsets = (old[k] for k in ("cv", "ols", "offsets"))
    rows = np.arange(len(cv))
    sizes = np.diff(offsets)
    weights = np.repeat(1./sizes, sizes)
    coef = np.load(run.checked(f["parent"]["coefficients"]), mmap_mode="r", allow_pickle=False)
    idx = np.load(run.checked(f["choices"]), allow_pickle=False)
    if idx.shape != (len(cv), len(SEEDS), len(ARMS)) or idx.max(initial=0) > 4:
        raise ValueError("Invalid frozen choices")
    bank = np.concatenate((cv[:, None], ols[:, None], model.floor.paths(coef[:, :3])), axis=1)
    group = np.zeros(len(cv), dtype=int)
    cells = []
    for i, seed in enumerate(SEEDS):
        paths = {m: bank[rows, idx[:, i, j]] for j, m in enumerate(ARMS)}
        actual = {m: (p != cv).any((1, 2)) for m, p in paths.items()}
        for j, m in enumerate(ARMS):
            np.testing.assert_array_equal(actual[m], idx[:, i, j] != 0)
        np.testing.assert_array_equal(np.add.reduceat(actual["scene_budget"].astype(int), offsets[:-1]),
                                      np.add.reduceat(actual["matched_gain"].astype(int), offsets[:-1]))
        d = {}
        for m, p in paths.items():
            t = policy(p, cv, cv, y, v, group, easy_cut=cut["easy"], hard_cut=cut["hard"])
            d[m] = dict(risk=t["recordings"]["0"], ade=t["known_ADE_pixel"], fde=t["available_endpoint_FDE_diagnostic_pixel"],
                hard=t["subsets"]["hard"]["gain_percent"], p95=t["p95_known_ADE_pixel"],
                cv=comparison_masses(cv, p, cv, y, v, weights), ols4=comparison_masses(ols, p, cv, y, v, weights))
        cells.append(dict(seed=seed, modes=d, direct=comparison_masses(
            paths["matched_gain"], paths["scene_budget"], cv, y, v, weights)))
    return dict(recording=rec, rows=len(cv), queries=len(sizes), cells=cells)


def summarize(records, roster):
    groups = sorted(set(roster.values()))
    draws = np.random.default_rng(target.SEED).integers(len(groups), size=(target.DRAWS, len(groups)))
    by_group = {}
    for g in groups:
        cells = [c for r in records if roster[r["recording"]] == g for c in r["cells"]]
        modes = {m: {ref: merge_comparisons([c["modes"][m][ref] for c in cells])["completion_lower_pp"]
                     for ref in ("cv", "ols4")} for m in ARMS}
        by_group[g] = dict(modes=modes, direct=merge_comparisons([c["direct"] for c in cells])["completion_lower_pp"])

    def interval(v):
        a = np.asarray(v)
        return dict(point=float(a.mean()), ci95=np.quantile(a[draws].mean(1), [.025, .975]).tolist(),
                    positive_groups=int((a > 0.).sum()))

    metrics = {}
    for m in ARMS:
        views = [c["modes"][m] for r in records for c in r["cells"]]
        risk = [v["risk"] for v in views]
        signed = [v["known_easy_net_degradation"] for v in risk if v["known_easy_net_degradation"] is not None]
        metrics[m] = dict(views=len(risk), edits=sum(r["selected"] for r in risk),
            supported=sum(r["supported"] for r in risk),
            undefined_easy=sum(r["completion_easy_risk_upper"] is None for r in risk),
            easy_positive_over_2=sum(r["completion_easy_risk_upper"] is not None and r["completion_easy_risk_upper"] > .02 for r in risk),
            both_floors_and_risk=sum(v["risk"]["supported"] and all(merge_comparisons([v[f]])["completion_lower_pp"] > 0
                                                                    for f in ("cv", "ols4")) for v in views),
            worst_signed_easy=max(signed) if signed else None,
            mean_ade=float(np.mean([v["ade"] for v in views if v["ade"] is not None])),
            mean_fde=float(np.mean([v["fde"] for v in views if v["fde"] is not None])),
            mean_hard_gain_vs_cv=float(np.mean([v["hard"] for v in views if v["hard"] is not None])),
            max_p95=max(v["p95"] for v in views if v["p95"] is not None),
            **{f"lower_gain_vs_{f}_pp": interval([r["modes"][m][f] for r in by_group.values()]) for f in ("cv", "ols4")})
    if metrics["scene_budget"]["edits"] != metrics["matched_gain"]["edits"]:
        raise ValueError("Matched budget totals changed")
    return dict(by_group=by_group, metrics=metrics,
                matched_trajectory_lower_pp=interval([r["direct"] for r in by_group.values()]))


def score(key, manifest, old, roster, original, replay):
    p = PUBLIC/"freeze.json"
    seal = run.read(p)
    if (seal["registration"] != key or seal["held_labels_used"] or seal["confirmation_access"]
            or [r["recording"] for r in seal["records"]] != [r["sequence"] for r in old["clips"]]):
        raise ValueError("Expected all choices frozen before calibration label access")
    run.checked(seal["training"])
    meta, provenance, cut = target.context(roster)
    records = []
    for i, (c, f) in enumerate(zip(old["clips"], seal["records"], strict=True), 1):
        rec = c["sequence"]
        mark = PRIVATE/f"{rec}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay cannot open labels")
            row = score_recording(manifest, c, f, meta, provenance, cut)
            run.once(mark, run.stable(dict(registration=key, row=row, parent=f)))
        d = run.read(mark)
        if d["registration"] != key or d["parent"] != f or d["row"]["recording"] != rec:
            raise ValueError("Wrong score checkpoint")
        records.append(d["row"])
        beat("replay" if replay else "score", recording=rec, completed=i, total=len(roster))
    results = summarize(records, roster)
    n = sum(r["rows"] for r in records)
    if n != original["rows"] or len(records) != original["recordings"]:
        raise ValueError("Original target roster changed")
    out = dict(registration=key, freeze=run.artifact(p), rows=n, recordings=len(records), groups=len(set(roster.values())),
        **results, original_2pct=True, confirmation_access=False, independent_confirmation=False,
        matched_count_only=True, thresholds_tuned=False, result_source="fresh_run", deployment_changed=False)
    run.once(PUBLIC/"analysis.json", run.stable(out))
    print(results["metrics"], results["matched_trajectory_lower_pp"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("train", "freeze", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (PUBLIC, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, manifest, old, roster, parent, original = register()
        if phase == "train":
            train(key)
        elif phase == "freeze":
            freeze(key, manifest, old, roster, parent)
        else:
            score(key, manifest, old, roster, original, replay=phase == "replay")


if __name__ == "__main__":
    main()
