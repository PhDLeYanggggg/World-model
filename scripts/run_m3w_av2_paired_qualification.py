"""Fit and evaluate a two-stage decision against the fixed AV2 additive control."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts import run_m3w_av2_forecast_cost as cost
from src.world_model import m3w_av2_paired_qualification as model

av, prev, eq, np, torch, study, base, old, prior = (
    cost.av, cost.prev, cost.eq, cost.np, cost.torch, cost.study, cost.base, cost.old, cost.prior)
OUT, LOCAL = cost.OUT.parent/"av2_paired_qualification_v1", cost.LOCAL.parent/"av2_paired_qualification_v1"
MODES, PRIMARY = model.MODES, "conditional"
STEPS, SAVE, BATCH, PAIRS = 320, 40, 64, 16
CONTROLS = ("cv", "ols4", "raw_eq", "uniform", "additive", "history", "debit_easy")


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    parent, data, src, roles, inner, outer, control, _, _ = cost.register()
    p = cost.OUT/"summary.json"
    r = study.read(p)
    frozen = study.read(study.checked(r["freeze"]))
    audit = cost.OUT/"calibration_capacity.json"
    a = study.read(audit)
    if (r["registration"] != parent or frozen["registration"] != parent or frozen["held_labels_used"]
            or r["reserved_outcomes_opened"] or a["input"] != study.artifact(p)
            or a["verified_inner_checkpoint_cutoffs"] != 72):
        raise ValueError("Require the audited crossfit references")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(v)) for v in (__file__, model.__file__)],
        parent=study.artifact(p), parent_freeze=r["freeze"], cutoff_audit=study.artifact(audit),
        primary=PRIMARY, ablations=list(MODES), seeds=eq.SEEDS, steps=STEPS, batch=BATCH,
        rank_pairs=PAIRS, checkpoint=SAVE, torch_threads=4, num_workers=0,
        causal_features="past history, geometry and comparison to the fixed additive path",
        targets="exact reference eligibility and candidate relative qualification, all donor-only",
        conditioning="fit one relative head on all donor queries and one on eligible donors only",
        ranking="same-query positive-negative logistic term plus weighted classification",
        restriction="an exact copy of the reference cannot yield strict pair gain; exclude it from new ranks",
        release="unchanged per-view exact both-arm calibration, every nonempty prefix",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, src, roles, inner, outer, control, frozen, r, a


def features(prepared):
    p, a, x, g, h, aux, _, _ = prepared
    same = np.all(p == a, axis=(1, 2, 3))
    if not same.any() or not (~same).any():
        raise ValueError("Require the reference and at least one distinct lawful option")
    x = torch.from_numpy(np.column_stack((x, h, aux, np.log1p(g))).astype(np.float32))
    return x, x[np.flatnonzero(same)[0]], np.flatnonzero(~same)


def truth_terms(d, p, a, y, v, cut):
    _, label, _ = prev.label_terms(d, p, a, y, v, cut)
    eligible = eq.ceiling.eligible_additive(prev.metrics(a, d, y, v, cut))
    return label, eligible


def inner_action(ref, parent_identity, city):
    ck = torch.load(study.checked(ref["checkpoint"]), weights_only=True, map_location="cpu")
    ident = ck["identity"]
    expect = dict(parent_identity, donors=[s for s in parent_identity["donors"] if s != city],
                  recipient=city, action_cut=ident["action_cut"])
    if ident != expect or ck["step"] != cost.STEPS or city not in parent_identity["donors"]:
        raise ValueError("Out-of-city reference identity mismatch")
    net = old.action.Action().eval()
    net.load_state_dict(ck["net"])
    return net, ident["action_cut"]


def samples(inner, data, y, role, ident, parent):
    held, seed, cut = (ident[k] for k in ("held", "seed", "cut"))
    nested = old.donor_forecasts(inner, data, held, role, seed)
    references = {city: inner_action(ref, parent["identity"], city)
                  for city, ref in parent["references"].items()}
    out = []
    for n, (target, mask) in y.items():
        d = dict(data[n], bank=eq.expanded(data[n], nested[n]))
        net, ac = references[d["city"]]
        if (s := cost.slate(d, seed, cut, net, action_cut=ac)) is not None:
            x, a, idx = features(s)
            label, eligible = truth_terms(d, s[0], s[1], target, mask, cut)
            out.append((x[idx], a, torch.tensor(label[idx], dtype=torch.float32), float(eligible)))
    return out


def fit(ident, name, rows):
    seed = ident["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+12619)
    net = model.Qualification(rows[0][0].shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Incorrect resume checkpoint")
        start, trace = ck["step"], ck["trace"]
        net.load_state_dict(ck["net"])
        optimizer.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    eligible = [r for r in rows if r[3]]
    pools = (rows, eligible)
    positive = [[(r[0], a, b) for r in pool if len(a := np.flatnonzero(r[2])) and
                 len(b := np.flatnonzero(r[2] == 0.))] for pool in pools]
    if not all(positive):
        raise ValueError("A donor objective has no same-query positive/negative examples")
    for step in range(start+1, STEPS+1):
        rr = [rows[i] for i in rng.integers(len(rows), size=BATCH)]
        q = net.reference(torch.stack([r[1] for r in rr])).flatten()
        losses = [model.binary(q, torch.tensor([r[3] for r in rr]))]
        for h, pool in enumerate(pools):
            draw = rr if h == 0 else [pool[i] for i in rng.integers(len(pool), size=BATCH)]
            idx = [rng.integers(len(r[0])) for r in draw]
            x = torch.stack([r[0][j] for r, j in zip(draw, idx, strict=True)])
            y = torch.stack([r[2][j] for r, j in zip(draw, idx, strict=True)])
            pair = [positive[h][j] for j in rng.integers(len(positive[h]), size=PAIRS)]
            pos, neg = (torch.stack([r[0][rng.choice(r[k])] for r in pair]) for k in (1, 2))
            z = net.candidate(torch.cat((x, pos, neg)))
            losses.append(model.binary(z[:BATCH, h], y)+.5*model.ranking(z[BATCH:], h))
        loss = sum(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % SAVE == 0:
            trace.append([float(v.detach()) for v in losses])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=net.state_dict(),
                optimizer=optimizer.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(path), loss=trace, donors=len(rows),
                            eligible_donors=len(eligible), ranked_queries=[len(p) for p in positive])


@torch.inference_mode()
def rank(net, prepared, singleton):
    if prepared is None:
        return singleton, dict.fromkeys(MODES, singleton), dict.fromkeys(MODES, -1.), {}, -1.
    p, a = prepared[:2]
    x, anchor, idx = features(prepared)
    z, q = net(x[idx], anchor)
    scores, q = model.probabilities(z, q)
    tops, paths, value = {}, {}, {}
    for m, v in scores.items():
        s = v.numpy()
        # Sorting the unweighted logits keeps ties identical across direct/query-weighted.
        k = np.argsort(-z[:, int(m == PRIMARY)].numpy(), kind="stable")[:5]
        tops[m] = np.pad(idx[k], (0, 5-len(k)), constant_values=-1)
        paths[m], value[m] = p[idx[k[0]]], float(s[k[0]])
    return a, paths, value, tops, float(q)


def diagnostics(rows):
    y, p = np.asarray(rows, dtype=float).T
    return dict(queries=len(y), positives=int(y.sum()), brier=float(np.mean((p-y)**2)),
        prevalence=float(y.mean()), auc=float(roc_auc_score(y, p)) if np.unique(y).size > 1 else None,
        ap=float(average_precision_score(y, p)) if y.any() else None)


def calibrate(net, action, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    tally, rows, eligibility = Counter(), {m: [] for m in MODES}, []
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        if (s := cost.slate(d, seed, cut, action)) is None:
            continue
        a, _, scores, tops, q = rank(net, s, d["cv"])
        label, ok = truth_terms(d, s[0], a, *cy[n], cut)
        eligibility.append((int(ok), q))
        tally.update(two_slots=1, paired_available=int(ok and label.any()))
        for m, idx in tops.items():
            pick = bool(label[idx[0]])
            tally.update({f"{m}_paired_top5": int(ok and label[idx[idx >= 0]].any())})
            rows[m].append(dict(score=scores[m], strict=pick, add_ok=ok))
    return dict(counts=dict(tally), query=diagnostics(eligibility),
        modes={m: old.action.prior.calibration(rows[m]) for m in MODES})


def freeze(key, data, source, roles, inner, outer, control, parent, audit):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            fold, raw_fold = parent["folds"][name], outer["folds"][name]
            cut = fold["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if fold["identity"] != dict(ident, registration=parent["registration"]):
                raise ValueError("Roles differ from frozen comparator")
            path = LOCAL/f"{name}.json"
            if not path.exists():
                rows = samples(inner, data, dy, role, ident, fold)
                net, fitted = fit(ident, name, rows)
                action = prior.load_action(control["folds"][name])
                table = calibrate(net, action, old.load_outer(raw_fold), data, cy, seed, cut)
                if any(table["counts"][k] != audit["by_fold"][name][k] for k in ("two_slots", "paired_available")):
                    raise ValueError("Calibration opportunity set changed")
                with (np.load(study.checked(raw_fold["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as c):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], c[k])
                    paths, scores, tops, qs = {m: [] for m in MODES}, {m: [] for m in MODES}, {m: [] for m in MODES}, []
                    for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                        d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                        a, p, v, t, q = rank(net, cost.slate(d, seed, cut, action), c["additive"][lo:hi])
                        np.testing.assert_array_equal(a, c["additive"][lo:hi])
                        qs.append(q)
                        for m in MODES:
                            paths[m].append(p[m])
                            scores[m].append(v[m])
                            tops[m].append(t.get(m, np.full(5, -1)))
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=raw["names"],
                        offsets=raw["offsets"], registration=key, eligible=qs,
                        **{m: np.concatenate(v) for m, v in paths.items()},
                        **{f"score_{m}": np.asarray(v) for m, v in scores.items()},
                        **{f"top_{m}": np.asarray(v) for m, v in tops.items()}))
                base.immutable(path, dict(identity=ident, fit=fitted, forecast=bank, calibration=table))
            saved = study.read(path)
            if saved["identity"] != ident:
                raise ValueError("Mismatched freeze identity")
            study.checked(saved["fit"]["checkpoint"])
            study.checked(saved["forecast"])
            done[name] = saved
            beat("freeze", fold=name, completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, data, source, outer, control, parent, report):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(parent["folds"]):
        raise ValueError("Require the complete freeze before held scoring")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    arms = (*CONTROLS, *MODES, "released_additive", "released_joint")
    counts, rows, query = Counter(), [], []
    by_city = {z["identity"]["held"]: Counter() for z in seal["folds"].values()}
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as fixed,
              np.load(study.checked(parent["folds"][name]["forecast"]), allow_pickle=False) as previous,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw):
            for k in ("names", "offsets"):
                for reference in (fixed, previous, raw):
                    np.testing.assert_array_equal(z[k], reference[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Invalid held readout")
            np.testing.assert_array_equal(z["direct"], z["query_weighted"])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                c, a = d["cv"], fixed["additive"][lo:hi]
                pred = dict(cv=c, ols4=d["ols4"], raw_eq=raw["raw_eq"][lo:hi], additive=a,
                    uniform=fixed["uniform"][lo:hi], **{m: previous[m][lo:hi] for m in ("history", "debit_easy")},
                    **{m: z[m][lo:hi] for m in MODES})
                t = fold["calibration"]["modes"][PRIMARY]["threshold"]
                active = t is not None and z[f"score_{PRIMARY}"][i] >= t
                pred.update(released_additive=a if active else c, released_joint=pred[PRIMARY] if active else c)
                metrics = {m: prev.metrics(f, d, y, v, cut) for m, f in pred.items()}
                stat = Counter()
                if len(rr := eq.radii(d, seed)) == 2:
                    pool = eq.slates(d, rr, eq.expanded(d, pred["raw_eq"]))
                    label, eligible = truth_terms(d, pool, a, y, v, cut)
                    both = bool(eligible and label.any())
                    query.append((int(eligible), float(z["eligible"][i])))
                    stat.update(two_slots=1, additive_eligible=int(eligible), paired_ceiling=int(both))
                    for m in MODES:
                        ix = z[f"top_{m}"][i]
                        ix = ix[ix >= 0]
                        np.testing.assert_array_equal(pool[ix[0]], pred[m])
                        stat.update({f"{m}_strict": int(label[ix[0]]),
                            f"{m}_both": int(eligible and label[ix[0]]), f"{m}_top5": int(both and label[ix].any())})
                    if active:
                        stat.update(released=1, released_both=int(eligible and label[z[f"top_{PRIMARY}"][i, 0]]))
                counts.update(stat)
                by_city[held].update(stat)
                direct = {m: {ref: dict(pair=av.frontier.pair_reduction(pred[m][None], pred[ref], y, v)[0].tolist(),
                    trajectory=cost.audit.direct_trajectory(pred[m], pred[ref], c, y, v))
                    for ref in ("additive", "history", "debit_easy")}
                    for m in ("direct", PRIMARY, "released_joint")}
                rows.append(dict(clip=str(n), city=held, complete=bool(v.all()), seed=seed, metrics=metrics, direct=direct))
        beat("score", fold=name, views=len(rows), total=len(truth)*len(eq.SEEDS))
    metrics = {}
    for m in arms:
        risk = [r["metrics"][m]["risk"] for r in rows if r["metrics"][m]["risk"]["selected"]]
        metrics[m] = dict(edits=sum(r["selected"] for r in risk), edited=len(risk), risk_pass=sum(r["supported"] for r in risk),
            dual_risk_pass=sum(eq.ceiling.eligible_additive(r["metrics"][m]) for r in rows),
            easy_undefined=sum(r["completion_easy_risk_upper"] is None for r in risk),
            easy_over_2=sum((r["known_easy_net_degradation"] or 0.) > .02 for r in risk))
        metrics[m].update({q: av.interval(rows, lambda r: r["metrics"][m][q]) for q in ("ade", "fde", "gain", "hard", "tail")})
        if m in report["metrics"] and not m.startswith("released_") and metrics[m] != report["metrics"][m]:
            raise ValueError(f"Fixed result changed: {m}")
    if len({metrics[m]["edits"] for m in ("uniform", "additive", "history", "debit_easy", *MODES)}) != 1:
        raise ValueError("Matched edit budgets changed")
    subsets = {"all": rows, "complete": [r for r in rows if r["complete"]]}
    terms = {m: {ref: {kind: {k: av.interval(sub, lambda r: r["direct"][m][ref][kind])
        for k, sub in subsets.items()} for kind in ("pair", "trajectory")}
        for ref in ("additive", "history", "debit_easy")} for m in ("direct", PRIMARY)}
    cal = {m: {k: sum(f["calibration"]["modes"][m][k] for f in seal["folds"].values())
               for k in ("tested", "feasible", "strict_with_add", "accepted")} for m in MODES}
    cal.update({m: dict(report["calibration"][m]) for m in ("history", "debit_easy")})
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run", primary=PRIMARY,
        held_scenarios=len(truth), counts=dict(counts), by_city={k: dict(v) for k, v in by_city.items()},
        calibration=cal, calibration_top5={m: sum(f["calibration"]["counts"][f"{m}_paired_top5"]
            for f in seal["folds"].values()) for m in MODES},
        held_eligibility=diagnostics(query), metrics=metrics, contrasts=terms,
        new_fits=len(seal["folds"]), reused_inner_additive=72, original_2pct=True,
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration", "calibration_top5", "held_eligibility")}, indent=2))


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
        key, data, src, roles, inner, outer, control, parent, report, audit = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, parent, audit)
        else:
            score(key, data, src, outer, control, parent, report)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
