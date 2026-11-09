"""Compare point and upper-tail harm estimates with fixed gain and support heads."""

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
from scripts import audit_m3w_av2_harm_margins as audit
from src.world_model import m3w_av2_harm_quantile as model

prior, np, torch, av, eq, prev, study, base, old = (
    audit.prior, audit.np, audit.torch, audit.av, audit.eq, audit.prev, audit.study, audit.base, audit.old)
OUT, LOCAL = prior.OUT.parent/"av2_harm_quantile_v1", prior.LOCAL.parent/"av2_harm_quantile_v1"
MODES, PRIMARY = ("prior", "point", "upper", "frozen_upper"), "upper"
STEPS, SAVE, SCENES, OPTIONS = 320, 40, 8, 16
F = torch.nn.functional


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    key, data, src, roles, inner, outer, control, _ = prior.register()
    path = prior.OUT/"summary.json"
    result = study.read(path)
    seal = study.read(study.checked(result["freeze"]))
    audit_path = prior.OUT/"harm_margin_audit.json"
    a = study.read(audit_path)
    if (result["registration"] != key or seal["registration"] != key or seal["held_labels_used"]
            or a["input"] != study.artifact(path) or a["reserved_outcomes_opened"] or result["reserved_outcomes_opened"]):
        raise ValueError("Require the prior exact replay")
    p = OUT/"registration.json"
    base.immutable(p, dict(parent=study.artifact(path), parent_freeze=result["freeze"], harm_audit=study.artifact(audit_path),
        code=[study.artifact(Path(v)) for v in (__file__, model.__file__)], primary=PRIMARY, modes=list(MODES),
        donor_targets="exact all/easy positive excess and signed easy excess of a complete slate; corrected missing actors",
        training="uniform donor scenes and uniform legal options; Huber point head; nonnegative increment fit with detached point and pinball loss",
        fixed="frozen gain and support heads; same option menu, exact two slot radii, same fallback",
        calibration="standalone original dual-floor and three 2% risk checks, every distinct nonempty prefix",
        quantile=model.QUANTILE, steps=STEPS, checkpoint=SAVE, scenes_per_step=SCENES, options_per_scene=OPTIONS,
        seeds=eq.SEEDS, torch_threads=4, num_workers=0, original_2pct=True,
        independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), data, src, roles, inner, outer, control, seal, result


def fit(ident, name, data):
    data = [(x, ctx, opts, row, u, -prior.model.margins(a, b, opts, row, u)[:, 2:])
            for x, ctx, opts, row, u, a, b, _ in data]
    if not data:
        raise ValueError("Empty donor menu")
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+1909)
    net = model.Harm(data[0][0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Incorrect resume checkpoint")
        start, trace = ck["step"], ck["trace"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    for step in range(start+1, STEPS+1):
        estimates, tails, labels = [], [], []
        for i in rng.integers(len(data), size=SCENES):
            x, ctx, opts, row, u, target = data[i]
            j = rng.integers(len(opts), size=OPTIONS)
            a, b = net(x, ctx, opts[j], row, u)
            estimates.append(a)
            tails.append(b)
            labels.append(target[j])
        y = torch.cat(labels)
        point_loss = F.smooth_l1_loss(torch.cat(estimates), y, beta=.03)
        tail_loss = model.quantile_loss(torch.cat(tails), y)
        opt.zero_grad(set_to_none=True)
        (point_loss+tail_loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(v.detach()) for v in (point_loss, tail_loss)])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(path), trace=trace, donor_queries=len(data))


@torch.inference_mode()
def rank(net, fixed, d, m, cut):
    x, ctx, opts, row = prior.features(d, m, cut)
    u = torch.from_numpy(m["unit"]).float()
    a, b = fixed(x, ctx)
    margins = prior.model.margins(a, b, opts, row, u)
    point, upper = net(x, ctx, opts, row, u)

    def scores(risk):
        return prior.model.slate_score(torch.column_stack((margins[:, :2], -risk)), b[:, 2], opts, row)

    values = dict(prior=scores(-margins[:, 2:]), point=scores(point), upper=scores(upper))
    values["frozen_upper"] = values["upper"]
    ix = {k: int(v.argmax()) for k, v in values.items()}
    ix["frozen_upper"] = ix["prior"]
    paths = av.slate.assemble(d["cv"], m["bank"], m["row"], m["col"], m["opts"][list(ix.values())])
    return dict(zip(ix, paths, strict=True)), {m: float(values[m][i].exp()) for m, i in ix.items()}, ix, {
        m: np.stack((point[i].numpy(), upper[i].numpy())) for m, i in ix.items()}


def calibrate(net, fixed, forecaster, data, truth, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(truth))
    rows = {k: [] for k in MODES}
    for n, lo, hi in zip(truth, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        if (m := old.action.menu(d, seed, cut)) is None:
            continue
        paths, scores, _, _ = rank(net, fixed, d, m, cut)
        for k, p in paths.items():
            ok = eq.ceiling.eligible_additive(prev.metrics(p, d, *truth[n], cut))
            rows[k].append(dict(score=scores[k], strict=ok, add_ok=True))
    return {m: old.action.prior.calibration(rows[m]) for m in MODES}


def freeze(key, data, src, roles, inner, outer, control, seal):
    receipts = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            fold = seal["folds"][name]
            cut = fold["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if fold["identity"] != dict(ident, registration=seal["registration"]):
                raise ValueError("Roles differ from the fixed assignment")
            path = LOCAL/f"{name}.json"
            if not path.exists():
                net, fitted = fit(ident, name, prior.examples(inner, data, dy, role, ident))
                fixed = audit.load(fold)
                raw_ref = outer["folds"][name]
                table = calibrate(net, fixed, old.load_outer(raw_ref), data, cy, seed, cut)
                if table["prior"] != fold["calibration"]["supervised"]:
                    raise ValueError("Fixed calibration does not replay")
                with (np.load(study.checked(raw_ref["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(fold["forecast"]), allow_pickle=False) as p):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], p[k])
                    paths, values, indexes, estimates = ({m: [] for m in MODES} for _ in range(4))
                    for i, (n, lo, hi) in enumerate(zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True)):
                        d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                        if (m := old.action.menu(d, seed, cut)) is None:
                            pred = dict.fromkeys(MODES, p["supervised"][lo:hi])
                            scores, ids, harms = (dict.fromkeys(MODES, v) for v in (-1., -1, np.zeros((2, 3))))
                        else:
                            pred, scores, ids, harms = rank(net, fixed, d, m, cut)
                        np.testing.assert_array_equal(pred["prior"], p["supervised"][lo:hi])
                        np.testing.assert_array_equal(scores["prior"], p["score_supervised"][i])
                        for k in MODES:
                            paths[k].append(pred[k])
                            values[k].append(scores[k])
                            indexes[k].append(ids[k])
                            estimates[k].append(harms[k])
                    forecast = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=raw["names"],
                        offsets=raw["offsets"], registration=key, **{m: np.concatenate(v) for m, v in paths.items()},
                        **{f"score_{m}": np.asarray(v) for m, v in values.items()},
                        **{f"ix_{m}": np.asarray(v) for m, v in indexes.items()},
                        **{f"harm_{m}": np.asarray(v) for m, v in estimates.items()}))
                base.immutable(path, dict(identity=ident, fit=fitted, forecast=forecast, calibration=table))
            r = study.read(path)
            if r["identity"] != ident:
                raise ValueError("Wrong freeze identity")
            study.checked(r["fit"]["checkpoint"])
            study.checked(r["forecast"])
            receipts[name] = r
            beat("freeze", fold=name, completed=len(receipts), total=len(seal["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=receipts, held_labels_used=False))


def score(key, data, src, outer, control, parent, result):
    path = OUT/"freeze.json"
    seal = study.read(path)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(parent["folds"]):
        raise ValueError("Require sealed held forecasts")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    rows, tally = [], Counter()
    per_city = {v["identity"]["held"]: Counter() for v in seal["folds"].values()}
    comparisons = {m: [] for m in MODES}
    for name, f in seal["folds"].items():
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as control_bank):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], raw[k])
                np.testing.assert_array_equal(z[k], control_bank[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Held roster changed")
            np.testing.assert_array_equal(z["prior"], z["frozen_upper"])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                cv = d["cv"]
                pp = {m: z[m][lo:hi] for m in MODES}
                pp.update(cv=cv, fixed=control_bank["additive"][lo:hi], uniform=control_bank["uniform"][lo:hi])
                metrics = {m: prev.metrics(p, d, y, v, cut) for m, p in pp.items()}
                counts = Counter()
                released = dict.fromkeys(MODES, 0.)
                if z["ix_prior"][i] >= 0:
                    counts.update(two_slots=1)
                    dd = dict(d, bank=eq.expanded(d, raw["raw_eq"][lo:hi]))
                    menu = old.action.menu(dd, seed, cut)
                    for m in MODES:
                        r, ok = metrics[m]["risk"], eq.ceiling.eligible_additive(metrics[m])
                        t = f["calibration"][m]["threshold"]
                        active = t is not None and z[f"score_{m}"][i] >= t
                        counts.update({f"{m}_eligible": int(ok),
                            f"{m}_released": int(active), f"{m}_released_eligible": int(active and ok),
                            f"{m}_released_risk_fail": int(active and not r["supported"])})
                        h = audit.actual(dd, menu, y, v, cut, int(z[f"ix_{m}"][i]))
                        point, tail = z[f"harm_{m}"][i]
                        comparisons[m].append((h, point, tail, active))
                        if active:
                            g = prior.cost.audit.direct_trajectory(pp[m], cv, cv, y, v)
                            released[m] = None if g is None else g[2]
                tally.update(counts)
                per_city[held].update(counts)
                pair = {m: av.frontier.pair_reduction(pp[m][None], pp["fixed"], y, v)[0].tolist() for m in MODES}
                rows.append(dict(city=held, clip=str(n), seed=seed, counts=dict(counts), metrics=metrics,
                                 pair=pair, released=released))
        beat("score", fold=name, views=len(rows), total=len(truth)*len(eq.SEEDS))
    metrics = {}
    for m in (*MODES, "cv", "fixed", "uniform"):
        risk = [r["metrics"][m]["risk"] for r in rows if r["metrics"][m]["risk"]["selected"]]
        metrics[m] = dict(edits=sum(r["selected"] for r in risk), risk_pass=sum(r["supported"] for r in risk),
            easy_over_2=sum((r["known_easy_net_degradation"] or 0.) > .02 for r in risk),
            undefined_easy=sum(r["completion_easy_risk_upper"] is None for r in risk))
        metrics[m].update({q: av.interval(rows, lambda r: r["metrics"][m][q]) for q in ("ade", "fde", "gain", "hard", "tail")})
    if (len({metrics[m]["edits"] for m in (*MODES, "fixed", "uniform")}) != 1
            or tally["prior_eligible"] != result["counts"]["supervised_eligible"]
            or tally["prior_released"] != result["counts"]["supervised_released"]):
        raise ValueError("Fixed paths or matched edit budgets changed")
    harm = {}
    for m, val in comparisons.items():
        a, p, u = (np.asarray([v[k] for v in val]) for k in range(3))
        harm[m] = dict(points=len(a), point_coverage=(a <= p).mean(0).tolist(), upper_coverage=(a <= u).mean(0).tolist(),
            point_false_safe=((p <= 0.) & (a > 0.)).sum(0).tolist(), upper_false_safe=((u <= 0.) & (a > 0.)).sum(0).tolist(),
            point_median_residual=np.median(a-p, axis=0).tolist(), upper_median_residual=np.median(a-u, axis=0).tolist())
    cal = {m: {k: sum(f["calibration"][m][k] for f in seal["folds"].values())
               for k in ("tested", "feasible", "accepted", "local_strict")} for m in MODES}
    twos = [r for r in rows if r["counts"].get("two_slots")]
    out = dict(registration=key, freeze=study.artifact(path), primary=PRIMARY, result_source="fresh_run",
        held_scenarios=len(truth), fitted_models=len(seal["folds"]), metrics=metrics, counts=dict(tally),
        calibration=cal, by_city={k: dict(v) for k, v in per_city.items()}, harm_terms=list(audit.TERMS), harm=harm,
        qualification_delta_vs_prior={m: av.interval(twos, lambda r: r["counts"][f"{m}_eligible"]-r["counts"]["prior_eligible"])
                                      for m in MODES[1:]},
        pair_gain_vs_fixed={m: av.interval(rows, lambda r: r["pair"][m]) for m in MODES},
        released_gain_vs_cv={m: av.interval(rows, lambda r: r["released"][m]) for m in MODES},
        original_2pct=True, reference_diagnostic_only=True, independent_confirmation=False,
        deployment_changed=False, reserved_outcomes_opened=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration", "harm")}, indent=2))


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
        key, data, src, roles, inner, outer, control, seal, result = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, seal)
        else:
            score(key, data, src, outer, control, seal, result)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
