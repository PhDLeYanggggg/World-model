"""Frozen OpenDD selection heads on nongenerated Okutama keyframes."""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_okutama_label_histories as source

np, base, study, turn = source.np, source.base, source.study, source.turn
slate, risk, shared, torch = source.optional.slate, source.optional.risk, source.optional.shared, source.optional.torch
LOCAL = source.LOCAL/"transfer"
OUT = source.OUT
ARMS = ("anchor", "additive", "risk_joint", "relative_joint")
KEY = "nongenerated_10"


def register():
    key, manifest = source.register()
    p = OUT/"freeze.json"
    labels = study.read(p)
    if labels["registration"] != key or labels["future_features_used"] or not labels["forecast_scoring_not_run"]:
        raise ValueError("Require frozen clean annotation prefixes")
    _, _, parent, prior, _, _ = risk.register()
    costs, relative = parent["fits"], prior["fits"]
    path = OUT/"transfer_registration.json"
    base.immutable(path, dict(source=study.artifact(p), code=study.artifact(Path(__file__)),
        cost_checkpoints={k: v["checkpoint"] for k, v in costs.items()},
        relative_checkpoints={k: v["checkpoint"] for k, v in relative.items()},
        scope=KEY, original_2pct=True, seeds=slate.SEEDS, roles=ARMS,
        seed_ensemble="mean of all seven frozen OpenDD location folds for each seed",
        release="at most two OLS4 radii nominated by positive past CV-vs-OLS hindcast; leave one CV recipient",
        menu="OLS4 and tapered turn; identical legal slate menus and peak slots",
        easy_reference="other publisher-train scenario groups, positive complete CV ADE quartile; readout only",
        past_only=True, no_fit_or_threshold_search=True, independent_confirmation=False))
    return base.sha(path), manifest, labels, costs, relative


def load(seed, records, builder, steps):
    nets = []
    for site in slate.SITES:
        net = builder().eval()
        net.load_state_dict(risk.state(records[f"{site}_{seed}"], site, steps))
        nets.append(net)
    return nets


@torch.inference_mode()
def predict(h, q, cv, ols, turn_path, costs, relatives):
    outputs = {k: cv.copy() for k in ARMS}
    bank = np.stack((ols, turn_path), axis=1)
    _, _, _, past = shared.costs.features(h, cv, ols, bank)
    peak = source.motion.peak(ols-cv)
    for ix in slate.chunks(q):
        c, o, b, hist = (a[ix] for a in (cv, ols, bank, h))
        eligible = np.flatnonzero((past[ix] > 0.) & (peak[ix] > 1e-9))
        keep = eligible[np.argsort(-past[ix][eligible], kind="stable")[:min(2, len(c)-1)]]
        if not len(keep):
            continue
        original = c.copy()
        original[keep] = o[keep]
        outputs["anchor"][ix] = original
        paths, row, arm, opts, group = source.optional.options(c, b, peak[ix][keep])
        opts = opts[group == (2 if len(keep) == 2 else 0)]
        x, scale, valid, _ = shared.costs.features(hist, c, o, paths)
        if not len(opts) or not valid[row, arm].all():
            raise ValueError("An analytic anchor has no exact matched choice")
        inputs = torch.from_numpy(x[row, arm])
        pred = torch.stack([net(inputs) for net in costs]).mean(0).numpy()*scale[row, None]
        additive, joint = turn.refit.model.ranks(pred, opts)
        a = source.optional.paint(c, paths, row, arm, opts[int(additive.argmax())])
        outputs["additive"][ix] = a
        outputs["risk_joint"][ix] = source.optional.paint(c, paths, row, arm, opts[int(joint.argmax())])
        if len(keep) == 1:
            outputs["relative_joint"][ix] = a
        else:
            best = (-np.inf, None)
            for k in range(0, len(opts), 128):
                pick = opts[k:k+128]
                trial = slate.assemble(c, paths, row, arm, pick)
                pair = np.mean([slate.gain(n, hist, c, o, trial, a)[:, 2] for n in relatives], axis=0)
                score = joint[k:k+len(pick)]+slate.WEIGHT*pair
                j = int(score.argmax())
                if score[j] > best[0]:
                    best = (score[j], trial[j])
            outputs["relative_joint"][ix] = best[1]
        for p in outputs.values():
            rr = source.motion.peak(p[ix]-c)
            np.testing.assert_allclose(np.sort(rr[rr > 0.]), np.sort(peak[ix][keep]), atol=1e-8, rtol=1e-8)
    return outputs


def freeze(key, labels, costs, relative):
    paths = {}
    for seed in slate.SEEDS:
        c = load(seed, costs, turn.refit.model.CostHead, turn.refit.STEPS)
        r = load(seed, relative, slate.model.RelativeGain, slate.STEPS)
        for clip, refs in sorted(labels["clips"].items()):
            with np.load(study.checked(refs[KEY]), allow_pickle=False) as z:
                h, t, cv, ols = (z[k] for k in ("h", "start", "cv", "ols4"))
                q = np.unique(t, return_inverse=True)[1]
                out = predict(h, q, cv, ols, z["turn"], c, r) if len(h) else {k: cv.copy() for k in ARMS}
            name = f"{clip}_{seed}"
            paths[name] = turn.exp.previous.save_bank(LOCAL/f"{name}.npz", out)
            source.beat("transfer_freeze", clip=clip, seed=seed, done=len(paths), total=99)
    base.immutable(OUT/"transfer_freeze.json", dict(registration=key, forecasts=paths,
        outcomes_used=False, complete_before_scoring=True))


def score(key, m, labels):
    p = OUT/"transfer_freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["outcomes_used"] or not f["complete_before_scoring"] or len(f["forecasts"]) != 99:
        raise ValueError("Require all external predictions before label scoring")
    data = {}
    for k, frames in sorted(m["frames"].items()):
        points = source.trajectories(source.parse((source.SOURCE/f"{k}.txt").read_bytes(), frames), len(frames), "nongenerated")
        with np.load(study.checked(labels["clips"][k][KEY]), allow_pickle=False) as z:
            h, cv, ols, t, ids = (z[name] for name in ("h", "cv", "ols4", "start", "ids"))
        y = points[t[:, None]+10*np.arange(8, 20), ids[:, None]]-h[:, -1:]
        v = np.isfinite(y).all(-1)
        y = np.where(v[..., None], y, 0.)
        e = np.linalg.norm(cv[v.all(1)]-y[v.all(1)], axis=-1).mean(1)
        data[k] = dict(cv=cv, ols=ols, y=y, v=v, q=np.unique(t, return_inverse=True)[1], e=e[e > 0.],
                       scenario=k.split(".", 1)[1])
    cases, pair = [], []
    for k, d in data.items():
        if not len(d["cv"]):
            continue
        cut = np.quantile(np.concatenate([x["e"] for x in data.values() if x["scenario"] != d["scenario"]]), [.25, .75])
        cv, ols, y, v, q = (d[x] for x in ("cv", "ols", "y", "v", "q"))
        for seed in slate.SEEDS:
            a = {}
            with np.load(study.checked(f["forecasts"][f"{k}_{seed}"]), allow_pickle=False) as forecasts:
                paths = {arm: forecasts[arm] for arm in ARMS}
            for arm, path in paths.items():
                report = study.policy(path, cv, cv, y, v, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])
                row = report["recordings"]["r"]
                gain = optional_gain(cv, ols, path, y, v, q)
                a[arm] = dict(risk=row, gain=gain, ade=report["known_ADE_pixel"], fde=report["available_endpoint_FDE_diagnostic_pixel"])
            c = turn.expanded.actual_contrast(cv, ols, paths["relative_joint"], paths["additive"], y, v, q, cut)
            if c is not None:
                pair.append(dict(scenario=d["scenario"], clip=k, seed=seed, **c))
            cases.append(dict(scenario=d["scenario"], clip=k, seed=seed, arms=a))
        source.beat("transfer_score", clip=k)
    scenarios = sorted({a["scenario"] for a in cases})
    totals = {}
    for arm in ARMS:
        a = [x["arms"][arm] for x in cases]
        edited = [r for r in a if r["risk"]["selected"]]
        effect = scenario_interval(cases, scenarios, lambda x: x["arms"][arm]["gain"])
        totals[arm] = dict(edits=sum(x["risk"]["selected"] for x in edited), edited_views=len(edited),
            risk_pass=sum(x["risk"]["supported"] for x in edited),
            undefined_easy=sum(x["risk"]["completion_easy_risk_upper"] is None for x in edited),
            signed_easy_over_2pct=sum((x["risk"]["known_easy_net_degradation"] or 0.) > .02 for x in a),
            gains_pp=effect)
    if len({r["edits"] for r in totals.values()}) != 1:
        raise ValueError("Frozen transfer changed the matched budget")
    pair_summary = dict(edited_estimable=len(pair), qualified=sum(x["qualified"] for x in pair),
        strict_pair_wins=sum(x["qualified"] and x["pair_native"][1] > 1e-10 for x in pair),
        interval_native=scenario_interval(pair, scenarios, lambda x: x["pair_native"]))
    base.immutable(OUT/"transfer_readout.json", dict(registration=key, freeze=study.artifact(p), totals=totals,
        matched_pair=pair_summary, by_clip=cases, result_source="fresh_run", independent_confirmation=False,
        no_fine_tuning=True, original_2pct=True, all_views_from_same_facility=True))
    print(study.json.dumps(dict(totals=totals, matched_pair=pair_summary), indent=2))


def optional_gain(cv, ols, path, y, v, q):
    w = 1./np.bincount(q)[q]
    mass = float(w @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    return (turn.expanded.original.paired_utility(path, cv, ols, cv, y, v, w, mass)[:2].tolist()
            if mass > 0. else None)


def scenario_interval(rows, scenarios, extract):
    values = []
    for scenario in scenarios:
        keep = [(a["clip"], extract(a)) for a in rows if a["scenario"] == scenario and extract(a) is not None]
        values.append(turn.effects.site_means(keep))
    result = turn.effects.interval(values)
    if result is not None:
        result["scenario_groups"] = result.pop("sites")
        result["resampling_unit"] = "paired camera scenario, not independent site"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, m, labels, costs, relative = register()
        if phase == "freeze":
            freeze(key, labels, costs, relative)
        else:
            score(key, m, labels)


if __name__ == "__main__":
    main()
