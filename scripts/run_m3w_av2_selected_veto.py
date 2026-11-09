"""Fit an independent harm veto on options selected out of city."""

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
from scripts import run_m3w_av2_harm_quantile as upper

prior, np, torch, av, eq, prev, study, base, old = (
    upper.prior, upper.np, upper.torch, upper.av, upper.eq, upper.prev, upper.study, upper.base, upper.old)
OUT, LOCAL = upper.OUT.parent/"av2_selected_veto_v1", upper.LOCAL.parent/"av2_selected_veto_v1"
MODES, PRIMARY = ("self", "uniform", "selected"), "selected"
STEPS, SAVE, BATCH = 320, 40, 32
F = torch.nn.functional


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    key, data, src, roles, inner, outer, _, margin, _ = upper.register()
    path = upper.OUT/"summary.json"
    report = study.read(path)
    parent = study.read(study.checked(report["freeze"]))
    paired_path = upper.OUT/"postselection.json"
    paired = study.read(paired_path)
    if (parent["registration"] != key or report["registration"] != key or parent["held_labels_used"]
            or paired["input"] != study.artifact(path) or report["reserved_outcomes_opened"]):
        raise ValueError("Require the completed upper-harm contrast")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=study.artifact(Path(__file__)), parent=study.artifact(path),
        parent_freeze=report["freeze"], postselection=study.artifact(paired_path), primary=PRIMARY, modes=list(MODES),
        nested="ranker, EqMotion producer and ranker cutoff exclude recipient donor city, outer calibration and held; "
               "inner ranker training uses that same three-city producer, in-sample for its own three cities",
        veto="same detached-point upper risk architecture, seed, donor scene draws and updates; selected uses one fixed inner argmax, uniform draws one menu option",
        inference="fixed outer upper choices and gain/support heads; veto may reorder scenes but cannot change selected paths",
        calibration="original standalone exact qualification and 2 percent checks on independent calibration city",
        steps=STEPS, checkpoint=SAVE, batch=BATCH, seeds=eq.SEEDS, torch_threads=4, num_workers=0,
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, src, roles, inner, outer, margin, parent, report


def load_or_init(ident, path, nets, optimizers, samplers):
    if not path.exists():
        return 0, []
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
        raise ValueError("Resume identity or step changed")
    nets.load_state_dict(ck["net"])
    for opt, state in zip(optimizers, ck["optimizers"], strict=True):
        opt.load_state_dict(state)
    for rng, state in zip(samplers, ck["rng"], strict=True):
        rng.bit_generator.state = state
    return ck["step"], ck["trace"]


def checkpoint(ident, path, nets, optimizers, samplers, step, trace):
    p = path.with_suffix(".pending")
    torch.save(dict(identity=ident, step=step, trace=trace, net=nets.state_dict(),
        optimizers=[o.state_dict() for o in optimizers], rng=[r.bit_generator.state for r in samplers]), p)
    p.replace(path)
    beat("fit", model=path.stem, step=step, total=STEPS)


def example(d, m, y, v, cut):
    x, ctx, opts, row = prior.features(d, m, cut)
    a, b, label = prior.labels(d, m, y, v, cut)
    return x, ctx, opts, row, torch.from_numpy(m["unit"]).float(), a, b, label


def harm_data(rows):
    return [(x, ctx, opts, row, u, -prior.model.margins(a, b, opts, row, u)[:, 2:])
            for x, ctx, opts, row, u, a, b, _ in rows]


def loss_harm(net, samples):
    p, q, y = [], [], []
    for (x, ctx, opts, row, unit, target), j in samples:
        a, b = net(x, ctx, opts[j], row, unit)
        p.append(a)
        q.append(b)
        y.append(target[j])
    target = torch.cat(y)
    return F.smooth_l1_loss(torch.cat(p), target, beta=.03)+upper.model.quantile_loss(torch.cat(q), target)


def fit_ranker(ident, path, data):
    seed = ident["seed"]
    torch.manual_seed(seed)
    chooser = prior.model.Assignment(data[0][0].shape[1])
    torch.manual_seed(seed)
    harm = upper.model.Harm(data[0][0].shape[1])
    nets = torch.nn.ModuleDict(dict(chooser=chooser, harm=harm))
    optimizers = tuple(torch.optim.AdamW(n.parameters(), lr=.001) for n in nets.values())
    a, b = samplers = (np.random.default_rng(seed+21647), np.random.default_rng(seed+1909))
    start, trace = load_or_init(ident, path, nets, optimizers, samplers)
    positives = [(d, p, n) for d in data if len(p := np.flatnonzero(d[-1])) and len(n := np.flatnonzero(~d[-1]))]
    if not positives:
        raise ValueError("Nested ranker has no mixed training menu")
    hr = harm_data(data)
    for step in range(start+1, STEPS+1):
        draw = [data[i] for i in a.integers(len(data), size=prior.BATCH)]
        pick = [(d, a.integers(len(d[0]), size=4), a.integers(len(d[1]), size=4)) for d in draw]
        batch = tuple(torch.cat([d[k][ia if k in (0, 5) else ib] for d, ia, ib in pick]) for k in (0, 1, 5, 6))
        pp = [positives[i] for i in a.integers(len(positives), size=prior.PAIRS)]
        pairs = [(d, int(a.choice(p)), int(a.choice(n))) for d, p, n in pp]
        loss = prior.regression(chooser, batch)+.3*prior.constraint_loss(chooser, pairs)
        harm_loss = loss_harm(harm, [(hr[i], b.integers(len(hr[i][2]), size=upper.OPTIONS))
            for i in b.integers(len(hr), size=upper.SCENES)])
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        (loss+harm_loss).backward()
        for net, opt in zip(nets.values(), optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            opt.step()
        if step % SAVE == 0:
            trace.append([float(loss.detach()), float(harm_loss.detach())])
            checkpoint(ident, path, nets, optimizers, samplers, step, trace)
    return nets.eval(), dict(identity=ident, checkpoint=study.artifact(path), trace=trace, donor_queries=len(data))


def ranker_donors(ident, city):
    donor = [s for s in ident["donors"] if s != city]
    if len(donor) != 3 or len(set(donor)) != 3 or city not in ident["donors"] or set(donor) & {ident["held"], ident["calibration"]}:
        raise ValueError("Recipient or excluded city enters the ranker")
    return donor


def nested(donor_truth, data, raw, role, inner, ident):
    rows, receipts = [], {}
    for city, group in role["inner"].items():
        donor = ranker_donors(ident, city)
        forecast = inner["models"][f"{group}_{ident['seed']}"]["fit"]
        ck = torch.load(study.checked(forecast["checkpoint"]), weights_only=True, map_location="cpu")
        if ck["identity"] != forecast["identity"] or ck["identity"]["donors"] != donor or ck["step"] != eq.STEPS:
            raise ValueError("Nested candidate producer lineage differs")
        tr = {n: t for n, t in donor_truth.items() if data[n]["city"] in donor}
        names = list(tr)
        cut = prev.donor_cut(data, tr).tolist()
        rank_identity = dict(ident, donors=donor, recipient=city, cut=cut, candidate_checkpoint=forecast["checkpoint"])
        forecaster = eq.eq.Forecast(eq.VARIANT).eval()
        forecaster.load_state_dict(ck["net"])
        pred, offsets = eq.predict(forecaster, data, names)
        examples = []
        for n, lo, hi in zip(names, offsets[:-1], offsets[1:], strict=True):
            d = dict(data[n], bank=eq.expanded(data[n], pred[lo:hi]))
            if (m := old.action.menu(d, ident["seed"], cut)) is not None:
                examples.append(example(d, m, *tr[n], cut))
        ranker, receipt = fit_ranker(rank_identity, LOCAL/f"{ident['held']}_{ident['seed']}_{city}.pt", examples)
        count = 0
        for n in donor_truth:
            if data[n]["city"] != city:
                continue
            d = dict(data[n], bank=eq.expanded(data[n], raw[n]))
            m = old.action.menu(d, ident["seed"], cut)
            if m is None:
                continue
            _, _, ix, _ = upper.rank(ranker["harm"], ranker["chooser"], d, m, cut)
            # Ranker cutoff is three-city; supervision is defined on the outer donor cutoff.
            outer_menu = old.action.menu(d, ident["seed"], ident["cut"])
            for k in ("opts", "row", "col"):
                np.testing.assert_array_equal(m[k], outer_menu[k])
            row = harm_data([example(d, outer_menu, *donor_truth[n], ident["cut"])])[0]
            rows.append((row, ix["upper"]))
            count += 1
        receipts[city] = dict(receipt, selected_queries=count)
        beat("inner_complete", fold=f"{ident['held']}_{ident['seed']}", city=city, queries=count)
    return rows, receipts


def fit_veto(ident, name, rows):
    torch.manual_seed(ident["seed"])
    nets = torch.nn.ModuleDict({m: upper.model.Harm(rows[0][0][0].shape[1]) for m in MODES[1:]})
    nets["selected"].load_state_dict(nets["uniform"].state_dict())
    opt = torch.optim.AdamW(nets.parameters(), lr=.001)
    rng = np.random.default_rng(ident["seed"]+17339)
    path = LOCAL/f"{name}_veto.pt"
    start, trace = load_or_init(ident, path, nets, (opt,), (rng,))
    for step in range(start+1, STEPS+1):
        take = [rows[i] for i in rng.integers(len(rows), size=BATCH)]
        uniform = [(r, [rng.integers(len(r[2]))]) for r, _ in take]
        selected = [(r, [k]) for r, k in take]
        losses = [loss_harm(nets[k], sample) for k, sample in zip(MODES[1:], (uniform, selected), strict=True)]
        opt.zero_grad(set_to_none=True)
        sum(losses).backward()
        for net in nets.values():
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(v.detach()) for v in losses])
            checkpoint(ident, path, nets, (opt,), (rng,), step, trace)
    return nets.eval(), dict(checkpoint=study.artifact(path), donor_queries=len(rows), trace=trace)


def load_harm(fold):
    ck = torch.load(study.checked(fold["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != fold["identity"] or ck["step"] != upper.STEPS:
        raise ValueError("Wrong fixed upper model")
    net = upper.model.Harm(ck["net"]["point.0.weight"].shape[1]).eval()
    net.load_state_dict(ck["net"])
    return net


@torch.inference_mode()
def rank(nets, self_harm, fixed, d, m, cut):
    x, ctx, opts, row = prior.features(d, m, cut)
    u = torch.from_numpy(m["unit"]).float()
    a, b = fixed(x, ctx)
    margin = prior.model.margins(a, b, opts, row, u)
    _, risk = self_harm(x, ctx, opts, row, u)
    logp = prior.model.slate_score(torch.column_stack((margin[:, :2], -risk)), b[:, 2], opts, row)
    i = int(logp.argmax())
    path = av.slate.assemble(d["cv"], m["bank"], m["row"], m["col"], m["opts"][i:i+1])[0]
    scores, harm = {"self": float(logp[i].exp())}, {"self": risk[i].numpy()}
    for k, net in nets.items():
        _, q = net(x, ctx, opts[i:i+1], row, u)
        value = torch.column_stack((margin[i:i+1, :2], -q))
        scores[k] = float(prior.model.slate_score(value, b[:, 2], opts[i:i+1], row)[0].exp())
        harm[k] = q[0].numpy()
    return path, i, scores, harm


def calibrate(nets, self_harm, fixed, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {k: [] for k in MODES}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        if (m := old.action.menu(d, seed, cut)) is None:
            continue
        path, _, scores, _ = rank(nets, self_harm, fixed, d, m, cut)
        ok = eq.ceiling.eligible_additive(prev.metrics(path, d, *cy[n], cut))
        for k, s in scores.items():
            rows[k].append(dict(score=s, strict=ok, add_ok=True))
    return {k: old.action.prior.calibration(rows[k]) for k in MODES}


def freeze(key, data, src, roles, inner, outer, margin, parent):
    receipts = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            fixed, producer, q = margin["folds"][name], outer["folds"][name], parent["folds"][name]
            cut = fixed["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if q["identity"] != dict(ident, registration=parent["registration"]):
                raise ValueError("Mismatched outer parent")
            receipt = LOCAL/f"{name}.json"
            if not receipt.exists():
                raw = old.donor_forecasts(inner, data, held, role, seed)
                samples, rankers = nested(dy, data, raw, role, inner, ident)
                nets, fitted = fit_veto(ident, name, samples)
                action, self_harm = upper.audit.load(fixed), load_harm(q)
                table = calibrate(nets, self_harm, action, old.load_outer(producer), data, cy, seed, cut)
                if table["self"] != q["calibration"]["upper"]:
                    raise ValueError("Existing calibration did not replay")
                with (np.load(study.checked(producer["forecast"]), allow_pickle=False) as f,
                      np.load(study.checked(q["forecast"]), allow_pickle=False) as p):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(f[k], p[k])
                    scores, harm = {m: [] for m in MODES}, {m: [] for m in MODES}
                    for i, (n, lo, hi) in enumerate(zip(f["names"], f["offsets"][:-1], f["offsets"][1:], strict=True)):
                        d = dict(data[n], bank=eq.expanded(data[n], f["raw_eq"][lo:hi]))
                        if (m := old.action.menu(d, seed, cut)) is None:
                            values = dict.fromkeys(MODES, -1.)
                            h = dict.fromkeys(MODES, np.zeros(3))
                        else:
                            path, j, values, h = rank(nets, self_harm, action, d, m, cut)
                            np.testing.assert_array_equal(path, p["upper"][lo:hi])
                            np.testing.assert_array_equal(j, p["ix_upper"][i])
                            np.testing.assert_array_equal(values["self"], p["score_upper"][i])
                            np.testing.assert_array_equal(h["self"], p["harm_upper"][i, 1])
                        for k in MODES:
                            scores[k].append(values[k])
                            harm[k].append(h[k])
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=f["names"], offsets=f["offsets"],
                        registration=key, **{f"score_{m}": np.asarray(v) for m, v in scores.items()},
                        **{f"harm_{m}": np.asarray(v) for m, v in harm.items()}))
                base.immutable(receipt, dict(identity=ident, fit=fitted, nested=rankers, forecast=bank, calibration=table))
            r = study.read(receipt)
            if r["identity"] != ident or set(r["nested"]) != set(donors):
                raise ValueError("Mismatched local receipt")
            for city, ranker in r["nested"].items():
                d = ranker["identity"]
                if d["donors"] != ranker_donors(ident, city) or d["recipient"] != city:
                    raise ValueError("Invalid receipt donor list")
                study.checked(ranker["checkpoint"])
                study.checked(d["candidate_checkpoint"])
            study.checked(r["fit"]["checkpoint"])
            study.checked(r["forecast"])
            receipts[name] = r
            beat("freeze", fold=name, completed=len(receipts), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=receipts, held_labels_used=False))


def score(key, data, src, outer, parent, result):
    path = OUT/"freeze.json"
    seal = study.read(path)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(parent["folds"]):
        raise ValueError("Require the sealed folds")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    rows, gains = [], []
    total = Counter()
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(parent["folds"][name]["forecast"]), allow_pickle=False) as p,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as f):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], p[k])
                np.testing.assert_array_equal(z[k], f[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Wrong frozen roster")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                s = dict(city=held, clip=str(n), seed=seed)
                release_gains = dict.fromkeys(MODES, 0.)
                if p["ix_upper"][i] >= 0:
                    d, (y, v) = data[n], truth[n]
                    m = old.action.menu(dict(d, bank=eq.expanded(d, f["raw_eq"][lo:hi])), seed, cut)
                    a = upper.audit.actual(d, m, y, v, cut, int(p["ix_upper"][i]))
                    trajectory = p["upper"][lo:hi]
                    met = prev.metrics(trajectory, d, y, v, cut)
                    ok = eq.ceiling.eligible_additive(met)
                    total.update(two_slots=1, eligible=int(ok))
                    predicted = np.stack([z[f"harm_{m}"][i] for m in MODES])
                    cover = (a <= predicted).astype(int)
                    false_safe = (predicted <= 0.) & (a > 0.)
                    for m in MODES:
                        t = fold["calibration"][m]["threshold"]
                        active = t is not None and z[f"score_{m}"][i] >= t
                        total.update({f"{m}_released": int(active), f"{m}_released_eligible": int(active and ok),
                            f"{m}_risk_fail": int(active and not met["risk"]["supported"])})
                        if active:
                            gain = prior.cost.audit.direct_trajectory(trajectory, d["cv"], d["cv"], y, v)
                            release_gains[m] = None if gain is None else gain[2]
                    rows.append(dict(s, coverage=cover.tolist(), false_safe=false_safe.tolist()))
                gains.append(dict(s, values=release_gains))
        beat("score", fold=name, views=len(gains), total=len(truth)*len(eq.SEEDS))
    cover = np.asarray([r["coverage"] for r in rows])
    if total["eligible"] != result["counts"]["upper_eligible"] or total["two_slots"] != result["counts"]["two_slots"]:
        raise ValueError("Forced qualification changed")
    np.testing.assert_allclose(cover[:, 0].mean(0), result["harm"]["upper"]["upper_coverage"], atol=1e-15, rtol=0.)
    cal = {m: {k: sum(f["calibration"][m][k] for f in seal["folds"].values())
               for k in ("tested", "feasible", "accepted", "local_strict")} for m in MODES}
    out = dict(registration=key, freeze=study.artifact(path), result_source="fresh_run", primary=PRIMARY,
        held_scenarios=len(truth), unique_two_slot_scenarios=len({r["clip"] for r in rows}),
        nested_ranker_fits=sum(len(f["nested"]) for f in seal["folds"].values()), veto_fits=2*len(seal["folds"]),
        counts=dict(total), calibration=cal, harm_terms=result["harm_terms"],
        coverage=dict(zip(MODES, cover.mean(0).tolist(), strict=True)),
        false_safe=dict(zip(MODES, np.asarray([r["false_safe"] for r in rows]).sum(0).tolist(), strict=True)),
        paired_selected_minus_uniform=av.interval(rows, lambda r: (np.asarray(r["coverage"][2])-r["coverage"][1]).tolist()),
        paired_selected_minus_self=av.interval(rows, lambda r: (np.asarray(r["coverage"][2])-r["coverage"][0]).tolist()),
        release_gain={m: av.interval(gains, lambda r: r["values"][m]) for m in MODES},
        same_forced_paths_and_budget=True, original_2pct=True, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration", "coverage", "false_safe",
        "paired_selected_minus_uniform", "paired_selected_minus_self")}, indent=2))


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
        key, data, src, roles, inner, outer, margin, parent, result = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, margin, parent)
        else:
            score(key, data, src, outer, parent, result)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
