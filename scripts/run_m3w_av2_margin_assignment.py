"""Learn independent action margins, then assign the existing two AV2 slots."""

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
from scripts import run_m3w_av2_observability as obs
from src.world_model import m3w_av2_margin_assignment as model

av, prev, eq, np, torch, study, base, old, prior, cost = (
    obs.av, obs.prev, obs.eq, obs.np, obs.torch, obs.study, obs.base, obs.old, obs.prior, obs.cost)
OUT, LOCAL = obs.OUT.parent/"av2_margin_assignment_v2", obs.LOCAL.parent/"av2_margin_assignment_v2"
MODES = ("fixed", "point", "heuristic", "margins", "supervised")
PRIMARY, STEPS, SAVE, BATCH, PAIRS = "supervised", 320, 40, 32, 8
F = torch.nn.functional


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    parent, data, src, roles, inner, outer, control, _, _ = obs.register()
    path = obs.OUT/"summary.json"
    report = study.read(path)
    seal = study.read(study.checked(report["freeze"]))
    cap_path = obs.OUT/"reference_capacity.json"
    cap = study.read(cap_path)
    if (report["registration"] != parent or seal["registration"] != parent or seal["held_labels_used"]
            or cap["input"] != study.artifact(path) or report["reserved_outcomes_opened"]):
        raise ValueError("Need the sealed AV2 menu and reference capacity audit")
    r = OUT/"registration.json"
    base.immutable(r, dict(code=[study.artifact(Path(p)) for p in (__file__, model.__file__)],
        parent=study.artifact(path), capacity=study.artifact(cap_path), primary=PRIMARY, modes=list(MODES),
        causal_input="same old per-actor history and path features, no scene metadata or future masks",
        action_terms="CV and OLS full-grid lower gain increments, all/easy positive-harm excess, signed easy change",
        context_terms="OLS fallback intercept, easy cap contribution and probability of edited-easy support",
        missing_actor_correction="zero known error for wholly unlabelled actors, charge disagreement only if edited; replay scalar evaluator and legacy ceiling",
        training="matched seeds, queries, regression and support loss; supervised adds donor exact-label conjunction and same-query ranking losses",
        decoding="same two recipients and peak budget constraint; point, heuristic or soft conjunction of exact-form margins",
        calibration="standalone reference diagnostic only: original dual-floor and all three 2% risk checks at every nonempty prefix",
        bootstrap="3000 city draws after averaging seeds within scenario", seeds=eq.SEEDS,
        temperature=model.TEMPERATURE, steps=STEPS, checkpoint=SAVE, batch=BATCH, rank_pairs=PAIRS,
        torch_threads=4, num_workers=0, original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(r), data, src, roles, inner, outer, control, cap


def features(d, m, cut):
    raw, unit, _, _ = old.action.cost.features(d["h"], d["cv"], d["ols4"], m["bank"])
    np.testing.assert_array_equal(unit, m["unit"])
    ctx = np.column_stack((raw[:, 0, :20], np.log1p(cut[0]/unit))).astype(np.float32)
    return torch.from_numpy(m["x"]), torch.from_numpy(ctx), torch.from_numpy(m["opts"]), torch.from_numpy(m["row"])


def labels(d, m, y, v, cut):
    a, b = model.components(d["cv"], d["ols4"], m["bank"], y, v, cut[0], m["row"], m["col"], m["unit"])
    a, b, opts, row, unit = map(torch.from_numpy, (a, b, m["opts"], m["row"], m["unit"]))
    z = model.margins(a, b, opts, row, unit)
    label = model.qualification(z, b, opts, row)
    return a.float(), b.float(), label


def examples(inner, data, truth, role, ident):
    held, seed, cut = (ident[k] for k in ("held", "seed", "cut"))
    raw = old.donor_forecasts(inner, data, held, role, seed)
    out = []
    for n, (y, v) in truth.items():
        d = dict(data[n], bank=eq.expanded(data[n], raw[n]))
        m = old.action.menu(d, seed, cut)
        if m is not None:
            x, ctx, opts, row = features(d, m, cut)
            a, b, label = labels(d, m, y, v, cut)
            out.append((x, ctx, opts, row, torch.from_numpy(m["unit"]).float(), a, b, label))
    return out


def regression(net, batch):
    x, ctx, ta, tb = batch
    a, b = net(x, ctx)
    return (F.smooth_l1_loss(a, ta, beta=.05) + F.smooth_l1_loss(b[:, :2], tb[:, :2], beta=.02)
            + .25*F.binary_cross_entropy_with_logits(b[:, 2], tb[:, 2]))


def constraint_loss(net, pairs):
    scores = []
    for (x, ctx, opts, row, unit, *rest), pos, neg in pairs:
        a, b = net(x, ctx)
        choices = opts[[pos, neg]]
        z = model.margins(a, b, choices, row, unit)
        scores.append(model.slate_score(z, b[:, 2], choices, row))
    v = torch.stack(scores)
    label = torch.tensor([True, False]).expand_as(v)
    return model.supervised_loss(v, label)+.5*F.softplus(v[:, 1]-v[:, 0]).mean()


def fit(ident, name, data):
    if not data:
        raise ValueError("No donor queries")
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+21647)
    nets = torch.nn.ModuleDict({k: model.Assignment(data[0][0].shape[1]) for k in ("plain", "supervised")})
    nets["supervised"].load_state_dict(nets["plain"].state_dict())
    opt = torch.optim.AdamW(nets.parameters(), lr=.001)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Mismatched checkpoint")
        start, trace = ck["step"], ck["trace"]
        nets.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    qualified = [(d, p, n) for d in data if len(p := np.flatnonzero(d[-1])) and len(n := np.flatnonzero(~d[-1]))]
    if not qualified:
        raise ValueError("No mixed donor menus for ranking")
    for step in range(start+1, STEPS+1):
        queries = [data[i] for i in rng.integers(len(data), size=BATCH)]
        picks = [(d, rng.integers(len(d[0]), size=4), rng.integers(len(d[1]), size=4)) for d in queries]
        batch = tuple(torch.cat([d[k][ia if k in (0, 5) else ib] for d, ia, ib in picks]) for k in (0, 1, 5, 6))
        q = [qualified[i] for i in rng.integers(len(qualified), size=PAIRS)]
        pairs = [(d, int(rng.choice(a)), int(rng.choice(b))) for d, a, b in q]
        losses = [regression(net, batch) for net in nets.values()]
        decision = constraint_loss(nets["supervised"], pairs)
        opt.zero_grad(set_to_none=True)
        (sum(losses)+.3*decision).backward()
        for net in nets.values():
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(v.detach()) for v in (*losses, decision)])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=nets.state_dict(),
                optimizer=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return nets.eval(), dict(checkpoint=study.artifact(path), trace=trace, donor_queries=len(data),
                             mixed_donor_queries=len(qualified), qualifying_options=sum(int(r[-1].sum()) for r in data))


@torch.inference_mode()
def rank(nets, action, d, seed, cut, singleton):
    m = old.action.menu(d, seed, cut)
    if m is None:
        return dict.fromkeys(MODES, singleton), dict.fromkeys(MODES, -1.), {}
    x, ctx, opts, row = features(d, m, cut)
    paths = av.slate.assemble(d["cv"], m["bank"], m["row"], m["col"], m["opts"])
    unit = torch.from_numpy(m["unit"]).float()
    z = action(x).numpy()
    _, order, confidence = old.action.rank(z, m["opts"], m["unit"][m["row"]])
    tops = {"fixed": order[:5]}
    values = {"fixed": confidence}
    for k, net in nets.items():
        a, b = net(x, ctx)
        pred = model.margins(a, b, opts, row, unit)
        logp = model.slate_score(pred, b[:, 2], opts, row)
        scores = {"supervised" if k == "supervised" else "margins": logp}
        if k == "plain":
            aux = torch.column_stack((a[:, :4], a[:, 4]-b[row, 1], b[row, 2]))
            point, heuristic, c = old.action.rank(aux.numpy(), m["opts"], m["unit"][m["row"]])
            point_score = (a[:, :2].mean(1)*unit[row])[opts].sum(1)
            tops["point"] = np.argsort(-point_score.numpy(), kind="stable")[:5]
            if tops["point"][0] != point:
                # Match the unstructured point definition even at numerical ties.
                tops["point"] = np.r_[point, tops["point"][tops["point"] != point]][:5]
            tops["heuristic"] = heuristic[:5]
            values["point"], values["heuristic"] = float(point_score[point].sigmoid()), c
        for mode, value in scores.items():
            take = np.argsort(-value.numpy(), kind="stable")[:5]
            tops[mode], values[mode] = take, float(value[take[0]].exp())
    return {k: paths[ix[0]] for k, ix in tops.items()}, values, tops


def calibrate(nets, action, forecaster, data, truth, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(truth))
    rows = {m: [] for m in MODES}
    for n, lo, hi in zip(truth, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        p, conf, tops = rank(nets, action, d, seed, cut, d["cv"])
        if tops:
            for m, path in p.items():
                ok = eq.ceiling.eligible_additive(prev.metrics(path, d, *truth[n], cut))
                rows[m].append(dict(score=conf[m], strict=ok, add_ok=True))
    return {m: old.action.prior.calibration(rows[m]) for m in MODES}


def freeze(key, data, source, roles, inner, outer, control):
    receipts = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            anchor, item = control["folds"][name], outer["folds"][name]
            cut = anchor["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if any(anchor["identity"][k] != v for k, v in ident.items() if k != "registration"):
                raise ValueError("Control roles changed")
            path = LOCAL/f"{name}.json"
            if not path.exists():
                nets, fitted = fit(ident, name, examples(inner, data, dy, role, ident))
                action = prior.load_action(anchor)
                table = calibrate(nets, action, old.load_outer(item), data, cy, seed, cut)
                with (np.load(study.checked(item["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(anchor["forecast"]), allow_pickle=False) as fixed):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], fixed[k])
                    paths, scores, top = ({m: [] for m in MODES} for _ in range(3))
                    for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                        d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                        p, conf, ix = rank(nets, action, d, seed, cut, fixed["additive"][lo:hi])
                        np.testing.assert_array_equal(p["fixed"], fixed["additive"][lo:hi])
                        for m in MODES:
                            paths[m].append(p[m])
                            scores[m].append(conf[m])
                            ids = ix.get(m, np.array([], dtype=int))
                            top[m].append(np.pad(ids, (0, 5-len(ids)), constant_values=-1))
                    forecast = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=raw["names"],
                        offsets=raw["offsets"], registration=key, **{m: np.concatenate(v) for m, v in paths.items()},
                        **{f"score_{m}": np.asarray(v) for m, v in scores.items()},
                        **{f"top_{m}": np.asarray(v) for m, v in top.items()}))
                base.immutable(path, dict(identity=ident, fit=fitted, calibration=table, forecast=forecast))
            r = study.read(path)
            if r["identity"] != ident:
                raise ValueError("Invalid freeze receipt")
            study.checked(r["fit"]["checkpoint"])
            study.checked(r["forecast"])
            receipts[name] = r
            beat("freeze", fold=name, completed=len(receipts), total=len(control["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=receipts, held_labels_used=False))


def pair_exists(qualified, anchor, y, valid, cv_mass):
    if len(qualified) < 2 or cv_mass <= 0.:
        return False
    pair = av.frontier.pair_reduction(qualified, anchor, y, valid)[:, 1] > 1e-10
    if not pair.any():
        return False
    p = qualified[pair]
    err = np.linalg.norm(anchor-y, axis=-1)-np.linalg.norm(p-y, axis=-1)
    missing = np.linalg.norm(p-anchor, axis=-1)
    return np.where(valid, err, -missing).sum((1, 2)).max()*100./cv_mass >= -1e-8


def score(key, data, source, outer, control, cap):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(control["folds"]):
        raise ValueError("Require the full freeze before held readout")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    counts, rows, unique = Counter(), [], {m: set() for m in MODES}
    by_city = {z["identity"]["held"]: Counter() for z in seal["folds"].values()}
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as fixed,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], fixed[k])
                np.testing.assert_array_equal(z[k], raw[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Wrong held roster or freeze")
            np.testing.assert_array_equal(z["fixed"], fixed["additive"])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                c = d["cv"]
                r = {m: z[m][lo:hi] for m in MODES}
                r.update(cv=c, ols4=d["ols4"], raw_eq=raw["raw_eq"][lo:hi], uniform=fixed["uniform"][lo:hi])
                metrics = {m: prev.metrics(f, d, y, v, cut) for m, f in r.items()}
                stat = Counter()
                gains = dict.fromkeys(MODES, 0.)
                if len(eq.radii(d, seed)) == 2:
                    dd = dict(d, bank=eq.expanded(d, r["raw_eq"]))
                    menu = old.action.menu(dd, seed, cut)
                    _, _, qualified = labels(dd, menu, y, v, cut)
                    pool = av.slate.assemble(c, menu["bank"], menu["row"], menu["col"], menu["opts"])
                    safe = pool[qualified.numpy()]
                    legacy = v.any(1).all()
                    mass = np.where(v, np.linalg.norm(c-y, axis=-1), 0.).sum()
                    any_pair = any(pair_exists(safe, a, y, v, mass) for a in safe)
                    stat.update(two_slots=1, any_qualified=int(len(safe) > 0), any_pair=int(any_pair),
                        wholly_missing_actor=int(not legacy), legacy_any_qualified=int(legacy and len(safe) > 0),
                        legacy_any_pair=int(legacy and any_pair))
                    for m in MODES:
                        ix = z[f"top_{m}"][i]
                        ix = ix[ix >= 0]
                        np.testing.assert_array_equal(pool[ix[0]], r[m])
                        ok = eq.ceiling.eligible_additive(metrics[m])
                        if ok != bool(qualified[ix[0]]):
                            raise ValueError("Additive margins disagree with the original evaluation")
                        support = metrics[m]["risk"]["completion_easy_risk_upper"] is not None
                        has_pair = ok and pair_exists(safe, r[m], y, v, mass)
                        if ok:
                            if has_pair:
                                unique[m].add(str(n))
                        if m == "fixed":
                            stat.update(legacy_fixed_pair=int(legacy and has_pair))
                        stat.update({f"{m}_support": int(support), f"{m}_eligible": int(ok),
                                     f"{m}_pair_capacity": int(has_pair), f"{m}_top5": int(qualified[ix].any())})
                        t = fold["calibration"][m]["threshold"]
                        if t is not None and z[f"score_{m}"][i] >= t:
                            stat.update({f"{m}_released": 1, f"{m}_released_eligible": int(ok),
                                f"{m}_released_risk_fail": int(not metrics[m]["risk"]["supported"])})
                            g = cost.audit.direct_trajectory(r[m], c, c, y, v)
                            gains[m] = None if g is None else g[2]
                counts.update(stat)
                by_city[held].update(stat)
                pair = {m: av.frontier.pair_reduction(r[m][None], r["fixed"], y, v)[0].tolist()
                        for m in MODES[1:]}
                rows.append(dict(city=held, clip=str(n), seed=seed, metrics=metrics, stats=dict(stat), pair=pair, gain=gains))
        beat("score", fold=name, views=len(rows), total=len(truth)*len(eq.SEEDS))
    summary = {}
    for m in (*MODES, "cv", "ols4", "raw_eq", "uniform"):
        risk = [r["metrics"][m]["risk"] for r in rows if r["metrics"][m]["risk"]["selected"]]
        summary[m] = dict(edits=sum(r["selected"] for r in risk), edited=len(risk),
            risk_pass=sum(r["supported"] for r in risk), easy_undefined=sum(r["completion_easy_risk_upper"] is None for r in risk),
            easy_over_2=sum((r["known_easy_net_degradation"] or 0.) > .02 for r in risk),
            worst_easy_net=max((r["known_easy_net_degradation"] for r in risk if r["known_easy_net_degradation"] is not None), default=None))
        summary[m].update({q: av.interval(rows, lambda r: r["metrics"][m][q]) for q in ("ade", "fde", "gain", "hard", "tail")})
    if (counts["legacy_any_qualified"] != cap["totals"]["any_reference"]
            or counts["legacy_any_pair"] != cap["totals"]["any_pair"]
            or counts["fixed_eligible"] != cap["totals"]["fixed_reference"]
            or counts["legacy_fixed_pair"] != cap["totals"]["fixed_pair"]
            or len({summary[m]["edits"] for m in (*MODES, "uniform")}) != 1):
        raise ValueError("Reference capacity or matched budget changed")
    twos = [r for r in rows if r["stats"].get("two_slots")]
    qualification_delta = {m: av.interval(twos, lambda r: r["stats"][f"{m}_eligible"]-r["stats"]["fixed_eligible"])
                           for m in MODES[1:]}
    calibration = {m: {k: sum(v["calibration"][m][k] for v in seal["folds"].values())
        for k in ("tested", "feasible", "local_strict", "accepted")} for m in MODES}
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run", primary=PRIMARY,
        metrics=summary, counts=dict(counts), by_city={k: dict(v) for k, v in by_city.items()},
        unique_pair_capacity={k: len(v) for k, v in unique.items()}, calibration=calibration,
        qualification_delta_vs_fixed=qualification_delta,
        pair_gain_vs_fixed={m: av.interval(rows, lambda r: r["pair"][m]) for m in MODES[1:]},
        released_trajectory_gain_vs_cv={m: av.interval(rows, lambda r: r["gain"][m]) for m in MODES},
        held_scenarios=len(truth), fitted_models=2*len(seal["folds"]), original_2pct=True,
        reference_diagnostic_only=True, joint_policy_trained=False, deployment_changed=False,
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration", "qualification_delta_vs_fixed")}, indent=2))


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
        key, data, src, roles, inner, outer, control, cap = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control)
        else:
            score(key, data, src, outer, control, cap)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
