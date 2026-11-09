"""Rerank the frozen AV2 history policy with explicit relative error costs."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_av2_history_rank as audit
from scripts import run_m3w_av2_history_rank as prior
from src.world_model import m3w_av2_forecast_cost as cost

av, prev, eq, np, torch, study, base, old = (
    prior.av, prior.prev, prior.eq, prior.np, prior.torch, prior.study, prior.base, prior.old)
OUT, LOCAL = prior.OUT.parent/"av2_forecast_cost_v2", prior.LOCAL.parent/"av2_forecast_cost_v2"
F, STEPS, SAVE, BATCH = prior.F, prior.STEPS, prior.SAVE, prior.BATCH
MODES = cost.MODES
PRIMARY = "debit_easy"


def beat(phase, **fields):
    path = LOCAL/"heartbeat.pending"
    path.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    path.replace(LOCAL/"heartbeat.json")


def register():
    parent, data, src, roles, inner, outer, control, _ = prior.register()
    path = prior.OUT/"summary.json"
    report = study.read(path)
    freeze = study.read(study.checked(report["freeze"]))
    if report["registration"] != parent or freeze["registration"] != parent or freeze["held_labels_used"] or report["reserved_outcomes_opened"]:
        raise ValueError("Require the sealed history experiment")
    p = OUT/"registration.json"
    base.immutable(p, dict(history=study.artifact(path), history_freeze=report["freeze"],
        code=[study.artifact(Path(v)) for v in (__file__, cost.__file__, audit.__file__)],
        primary=PRIMARY, ablations=list(MODES), inputs="unchanged exact two-slot menu, frozen history score and causal relative geometry",
        targets=list(cost.COMPONENTS), scaling="all trajectory/interaction steps count in denominators; train on target/bound",
        crossfit="three donor cities for each nested additive model, EqMotion features and action cutoff; exclude recipient donor city, outer held and calibration",
        evaluation_reference="exact unchanged frozen outer additive predictions; 896 reused development scenarios",
        decision="multiply the fixed history score by sigmoid(4*normalized gain) for trajectory and pair, and sigmoid(-4*normalized easy excess) in primary",
        calibration="original both-arm qualification and exact nonempty prefixes; never select the primary by held results",
        seeds=eq.SEEDS, steps=STEPS, checkpoint=SAVE, batch=BATCH, workers=0, torch_threads=4,
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), data, src, roles, inner, outer, control, freeze, report


def fit(identity, path, factory, loss_fn, examples):
    torch.manual_seed(identity["seed"])
    rng = np.random.default_rng(identity["seed"]+7283)
    net = factory()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != identity or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Invalid resume checkpoint")
        step, trace = ck["step"], ck["loss"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
    if not examples:
        raise ValueError("Empty training roster")
    for t in range(step+1, STEPS+1):
        sizes = rng.multinomial(BATCH, np.full(len(examples), 1./len(examples)))
        rows = [(z, rng.integers(len(z[0]), size=n)) for z, n in zip(examples, sizes, strict=True) if n]
        batch = [torch.cat([z[k][ix] for z, ix in rows]) for k in range(len(examples[0]))]
        loss = loss_fn(net, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % SAVE == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=t, loss=trace, net=net.state_dict(),
                            opt=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", model=path.stem, step=t, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(path), loss=trace, donor_queries=len(examples))


def nested_reference(identity, frozen, data, dy, name):
    ck = torch.load(study.checked(frozen["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    donors = identity["donors"]
    if (ck["identity"] != frozen["fit"]["identity"] or ck["identity"]["donors"] != donors
            or ck["identity"]["seed"] != identity["seed"] or ck["step"] != eq.STEPS
            or set(donors) & {identity["held"], identity["calibration"], identity["recipient"]}):
        raise ValueError("Nested reference producer has an invalid city lineage")
    forecaster = eq.eq.Forecast(eq.VARIANT).eval()
    forecaster.load_state_dict(ck["net"])
    names = [n for n in dy if data[n]["city"] in donors]
    raw, offsets = eq.predict(forecaster, data, names)
    examples = []
    for n, lo, hi in zip(names, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        if (m := old.action.menu(d, identity["seed"], identity["action_cut"])) is not None:
            examples.append(old.action.action_targets(d, m, *dy[n], identity["action_cut"]))
    return fit(identity, LOCAL/f"{name}.pt", old.action.Action, old.action.action_loss, examples)


@torch.inference_mode()
def slate(d, seed, cut, action, action_cut=None):
    m = old.action.menu(d, seed, cut if action_cut is None else action_cut)
    if m is None:
        return None
    paths, rows, x, g, _, _ = old.options(d, m, cut)
    action_value = action(torch.from_numpy(m["x"])).numpy()
    _, order, _ = old.action.rank(action_value, m["opts"], m["unit"][m["row"]])
    anchor = paths[order[0]]
    h = prior.history.features(d["h"], d["cv"], paths, rows, cut[0])
    aux, bounds, unit = cost.describe(d["h"], d["cv"], d["ols4"], paths, anchor, cut[0])
    return paths, anchor, x, g, h, aux, bounds, unit


def label(d, prepared, truth, cut):
    paths, anchor, _, _, _, _, bounds, unit = prepared
    target = cost.targets(paths, anchor, d["cv"], *truth, cut[0], unit)
    if (not np.isfinite(target).all() or (np.abs(target) > bounds+1e-5).any()
            or (target[:, [1, 3]] < 0.).any()):
        raise ValueError("A supervision term exceeds its causal geometric bound")
    return target


def cost_loss(net, batch):
    x, b, y = batch
    mask = b > 1e-7
    ratio = y/b.clamp_min(1e-7)
    error = F.smooth_l1_loss(net(x), ratio, beta=.1, reduction="none")
    return ((error*mask).sum(0)/mask.sum(0).clamp_min(1)).mean()


def history_model(frozen):
    ck = torch.load(study.checked(frozen["fits"]["history"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != dict(frozen["identity"], method="history") or ck["step"] != prior.STEPS:
        raise ValueError("Wrong fixed history scorer")
    net = prior.network().eval()
    net.load_state_dict(ck["net"])
    return net


@torch.inference_mode()
def rank(net, history, prepared, singleton):
    if prepared is None:
        return singleton, dict.fromkeys(MODES, singleton), dict.fromkeys(MODES, -1.), None
    p, a, x, g, h, aux, bounds, _ = prepared
    z = history(prior.with_history(torch.from_numpy(x), torch.from_numpy(h), "history"))
    ratio = net(torch.from_numpy(aux))
    b = torch.from_numpy(bounds)
    pred, confidence = {}, {}
    for method, s in cost.scores(prior.expected(z, torch.from_numpy(g)), ratio, b).items():
        i = int(s.argmax())
        confidence[method] = float(s[i]) if s[i] > 0. else -1.
        pred[method] = p[i] if s[i] > 0. else a
    return a, pred, confidence, (ratio*b).numpy()


def calibrate(net, history, action, forecaster, data, truth, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(truth))
    rows = {m: [] for m in MODES}
    error, zero, count = np.zeros(len(cost.COMPONENTS)), np.zeros(len(cost.COMPONENTS)), 0
    for n, lo, hi in zip(truth, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        prepared = slate(d, seed, cut, action)
        if prepared is None:
            continue
        a, paths, conf, estimate = rank(net, history, prepared, d["cv"])
        y, v = truth[n]
        target = label(d, prepared, (y, v), cut)
        error += np.abs(estimate-target).mean(0)
        zero += np.abs(target).mean(0)
        count += 1
        ok = eq.ceiling.eligible_additive(prev.metrics(a, d, y, v, cut))
        for m in MODES:
            if conf[m] > 0.:
                _, q, _ = prev.label_terms(d, paths[m][None], a, y, v, cut)
                rows[m].append(dict(score=conf[m], strict=bool(q[0]), add_ok=ok))
    return ({m: old.action.prior.calibration(rows[m]) for m in MODES},
            dict(queries=count, abs_error_sum=error.tolist(), zero_error_sum=zero.tolist()))


def freeze(key, data, source, roles, inner, outer, control, history):
    receipts = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            control_fold, hfold = control["folds"][name], history["folds"][name]
            cut = hfold["identity"]["cut"]
            identity = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if any(identity[k] != hfold["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut")):
                raise ValueError("History and reranker roles disagree")
            receipt = LOCAL/f"{name}.json"
            if not receipt.exists():
                nested = old.donor_forecasts(inner, data, held, role, seed)
                examples, references = [], {}
                for city, producer in role["inner"].items():
                    others = [s for s in donors if s != city]
                    inner_cut = prev.donor_cut(data, {n: t for n, t in dy.items() if data[n]["city"] in others}).tolist()
                    inner_id = dict(identity, recipient=city, donors=others, action_cut=inner_cut)
                    action, references[city] = nested_reference(inner_id, inner["models"][f"{producer}_{seed}"],
                                                                 data, dy, f"{name}_ref_{city}")
                    for n in dy:
                        if data[n]["city"] != city:
                            continue
                        d = dict(data[n], bank=eq.expanded(data[n], nested[n]))
                        if (prepared := slate(d, seed, cut, action, action_cut=inner_cut)) is not None:
                            examples.append(tuple(torch.from_numpy(a) for a in
                                (*prepared[5:7], label(d, prepared, dy[n], cut))))
                net, fitted = fit(identity, LOCAL/f"{name}_cost.pt",
                                  lambda: cost.CostHead(examples[0][0].shape[1]), cost_loss, examples)
                action, h = prior.load_action(control_fold), history_model(hfold)
                item = outer["folds"][name]
                table, errors = calibrate(net, h, action, old.load_outer(item), data, cy, seed, cut)
                if table["history"] != hfold["calibration"]["history"]:
                    raise ValueError("Frozen history calibration did not replay")
                with (np.load(study.checked(item["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(control_fold["forecast"]), allow_pickle=False) as fixed,
                      np.load(study.checked(hfold["forecast"]), allow_pickle=False) as hist):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], fixed[k])
                        np.testing.assert_array_equal(raw[k], hist[k])
                    paths, scores = {m: [] for m in MODES}, {m: [] for m in MODES}
                    for i, (n, lo, hi) in enumerate(zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True)):
                        d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                        a, p, s, _ = rank(net, h, slate(d, seed, cut, action), fixed["additive"][lo:hi])
                        np.testing.assert_array_equal(a, fixed["additive"][lo:hi])
                        np.testing.assert_array_equal(p["history"], hist["history"][lo:hi])
                        np.testing.assert_array_equal(s["history"], hist["score_history"][i])
                        for m in MODES:
                            paths[m].append(p[m])
                            scores[m].append(s[m])
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=raw["names"],
                        offsets=raw["offsets"], registration=key, **{m: np.concatenate(v) for m, v in paths.items()},
                        **{f"score_{m}": np.asarray(v) for m, v in scores.items()}))
                base.immutable(receipt, dict(identity=identity, fit=fitted, references=references,
                    calibration=table, component_errors=errors, forecast=bank))
            saved = study.read(receipt)
            if saved["identity"] != identity or set(saved["references"]) != set(donors):
                raise ValueError("Incorrect role or reference receipt")
            for ref in (saved["forecast"], saved["fit"]["checkpoint"],
                        *(v["checkpoint"] for v in saved["references"].values())):
                study.checked(ref)
            receipts[name] = saved
            beat("freeze", fold=name, completed=len(receipts), total=len(history["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=receipts, held_labels_used=False))


def score(key, data, source, control, history, report):
    path = OUT/"freeze.json"
    seal = study.read(path)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(history["folds"]):
        raise ValueError("Freeze all folds before scoring")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    arms = ("cv", "ols4", "additive", *MODES, "released_additive", "released_joint")
    rows, counts = [], Counter()
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as fixed):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], fixed[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Incorrect held data")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                c, a = d["cv"], fixed["additive"][lo:hi]
                p = dict(cv=c, ols4=d["ols4"], additive=a, **{m: z[m][lo:hi] for m in MODES})
                active = fold["calibration"][PRIMARY]["threshold"]
                active = active is not None and z[f"score_{PRIMARY}"][i] >= active
                p.update(released_additive=a if active else c, released_joint=p[PRIMARY] if active else c)
                metrics = {m: prev.metrics(pp, d, y, v, cut) for m, pp in p.items()}
                paired = eq.ceiling.eligible_additive(metrics["additive"])
                if len(eq.radii(d, seed)) == 2:
                    _, q, _ = prev.label_terms(d, np.stack([p[m] for m in MODES]), a, y, v, cut)
                    for m, strict in zip(MODES, q, strict=True):
                        counts.update({f"{m}_strict": int(strict), f"{m}_both": int(paired and strict),
                            f"{m}_same_as_additive": int(np.array_equal(p[m], a))})
                    if active:
                        counts.update(release_views=1, release_both=int(paired and q[-1]))
                direct = {m: {ref: dict(pair=av.frontier.pair_reduction(p[m][None], p[ref], y, v)[0].tolist(),
                    trajectory=audit.direct_trajectory(p[m], p[ref], c, y, v))
                    for ref in ("history", "additive")} for m in MODES[1:]}
                rows.append(dict(city=held, clip=str(n), seed=seed, complete=bool(v.all()), metrics=metrics, direct=direct))
        beat("score", fold=name, views=len(rows), total=len(truth)*len(eq.SEEDS))
    metrics = {}
    for m in arms:
        risk = [r["metrics"][m]["risk"] for r in rows if r["metrics"][m]["risk"]["selected"]]
        metrics[m] = dict(edits=sum(r["selected"] for r in risk), edited=len(risk), risk_pass=sum(r["supported"] for r in risk),
            dual_risk_pass=sum(eq.ceiling.eligible_additive(r["metrics"][m]) for r in rows),
            easy_undefined=sum(r["completion_easy_risk_upper"] is None for r in risk),
            easy_over_2=sum((r["known_easy_net_degradation"] or 0.) > .02 for r in risk))
        metrics[m].update({q: av.interval(rows, lambda r: r["metrics"][m][q]) for q in ("ade", "fde", "gain", "hard", "tail")})
    for m in ("cv", "ols4", "additive", "history"):
        if metrics[m] != report["metrics"][m]:
            raise ValueError("Fixed comparison failed to replay")
    if len({metrics[m]["edits"] for m in ("additive", *MODES)}) != 1:
        raise ValueError("Edit budgets differ")
    subsets = {"all": rows, "complete": [r for r in rows if r["complete"]]}
    terms = {m: {ref: {kind: {k: av.interval(sub, lambda r: r["direct"][m][ref][kind])
        for k, sub in subsets.items()} for kind in ("pair", "trajectory")}
        for ref in ("history", "additive")} for m in MODES[1:]}
    cal = {m: {k: sum(z["calibration"][m][k] for z in seal["folds"].values())
               for k in ("tested", "feasible", "local_strict", "strict_with_add", "accepted")} for m in MODES}
    er = [f["component_errors"] for f in seal["folds"].values()]
    count = sum(v["queries"] for v in er)
    errors = {k: (np.sum([v[k] for v in er], axis=0)/count).tolist() for k in ("abs_error_sum", "zero_error_sum")}
    out = dict(result_source="fresh_run", registration=key, freeze=study.artifact(path), primary=PRIMARY,
        metrics=metrics, counts=dict(counts), calibration=cal, contrasts=terms, held_scenarios=len(truth),
        new_nested_references=sum(len(z["references"]) for z in seal["folds"].values()),
        new_cost_fits=len(seal["folds"]), original_2pct=True, calibration_components=list(cost.COMPONENTS),
        calibration_error=dict(queries=count, model_mae=errors["abs_error_sum"], zero_mae=errors["zero_error_sum"]),
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration", "calibration_error")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, source, roles, inner, outer, control, history, report = register()
        if phase == "fit":
            freeze(key, data, source, roles, inner, outer, control, history)
        else:
            score(key, data, source, control, history, report)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
