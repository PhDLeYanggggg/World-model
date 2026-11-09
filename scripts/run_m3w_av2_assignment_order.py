"""Compare pointwise and listwise costs at the same fixed AV2 edit budget."""

import argparse
from collections import defaultdict
import copy
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_assignment_constraints as prior
from src.world_model import m3w_av2_assignment_order as model


np, torch, study, base, prev, eq, av = prior.np, prior.torch, prior.study, prior.base, prior.prev, prior.eq, prior.av
old, timing, bank = prior.old, prior.timing, prior.bank
OUT, LOCAL = prior.OUT.parent/"av2_assignment_order_v2", prior.LOCAL.parent/"av2_assignment_order_v2"
STEPS, SAVE, BATCH = 400, 50, 16
ARMS = ("soft", "pointwise", "listwise")


def beat(phase, **fields):
    r = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **fields)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(r)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(r, flush=True)


def register():
    key, data, src, roles, inner, outer, control, refs, parent, soft = prior.register()
    previous = prior.OUT/"summary.json"
    s = study.read(previous)
    f = study.read(study.checked(s["freeze"]))
    if f["registration"] != key or f["held_labels_used"] or s["reserved_outcomes_opened"]:
        raise ValueError("Require scored development-only parent")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(previous), parent_soft=study.artifact(old.OUT/"summary.json"),
        arms=ARMS, steps=STEPS, checkpoint_every=SAVE, queries_per_batch=BATCH, seeds=eq.SEEDS,
        target="exact support, two-floor and original risk conjunction takes priority; bounded gain and risk margins rank within each class",
        fit="matched network, initialization and city-balanced batches of whole donor queries; only listwise loss differs",
        temperature=model.TEMPERATURE, rank_weight=model.RANK_WEIGHT,
        choice="argmax score; positive monotone calibration score without probability floor",
        calibration="original nonempty all-qualified prefix on separate city, CV otherwise",
        primary="paired held development eligibility, utility and release vs pointwise and frozen soft",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, workers=0, threads=4))
    return base.sha(reg), data, src, roles, inner, outer, control, refs, parent, soft


def examples(data, dy, role, inner, ref, seed, cut):
    raw = timing.old.donor_forecasts(inner, data, ref["identity"]["held"], role, seed)
    if set(raw) != set(dy):
        raise ValueError("Incorrect donor roster")
    actions = {k: timing.rel.pair.inner_action(v, ref["identity"], k) for k, v in ref["references"].items()}
    groups = defaultdict(list)
    for n, (y, v) in dy.items():
        d = data[n]
        action, ac = actions[d["city"]]
        a = timing.incumbent(d, seed, cut, raw[n], action, action_cut=ac)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, _, _ = bank.candidate(d, a)
        c, _, z, scale = old.model.inputs(d, p, opts, cut[0])
        label, ok = model.targets(d, p, opts, y, v, cut[0], scale)
        want, _ = bank.model.eligibility(d["cv"], d["ols4"], p, y, v, cut[0], opts)
        if not np.array_equal(ok, want) or not np.array_equal(label > 0., want):
            raise ValueError("Priority does not match exact evaluator")
        groups[d["city"]].append((prior.model.features(c, z), label))
    if set(groups) != set(role["donors"]):
        raise ValueError("Need four donor cities")
    return dict(sorted(groups.items()))


def batch(groups, rng, width):
    picked = [v[i] for v in groups.values() for i in rng.integers(len(v), size=BATCH//len(groups))]
    n = max(len(y) for _, y in picked)
    x, y, valid = np.zeros((len(picked), n, width), np.float32), np.zeros((len(picked), n), np.float32), np.zeros((len(picked), n), bool)
    for i, (xx, yy) in enumerate(picked):
        x[i, :len(yy)], y[i, :len(yy)], valid[i, :len(yy)] = xx, yy, True
    return tuple(torch.from_numpy(a) for a in (x, y, valid))


def fit(ident, name, groups):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+5137)
    width = next(iter(groups.values()))[0][0].shape[1]
    pnet = model.head(width)
    nets = (pnet, copy.deepcopy(pnet))
    opt = [torch.optim.AdamW(n.parameters(), lr=.001) for n in nets]
    p = LOCAL/f"{name}.pt"
    step0, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong resume identity")
        for n, o, s, t in zip(nets, opt, ck["nets"], ck["optimizers"], strict=True):
            n.load_state_dict(s)
            o.load_state_dict(t)
        rng.bit_generator.state = ck["rng"]
        step0, trace = ck["step"], ck["loss"]
    for step in range(step0+1, STEPS+1):
        x, y, valid = batch(groups, rng, width)
        row = []
        for n, o, weight in zip(nets, opt, (0., model.RANK_WEIGHT), strict=True):
            point, rank = model.losses(n(x)[:, :, 0], y, valid)
            loss = point+weight*rank
            o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n.parameters(), 5., error_if_nonfinite=True)
            o.step()
            row.append([float(point.detach()), float(rank.detach())])
        if step % SAVE == 0:
            trace.append(row)
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, nets=[n.state_dict() for n in nets],
                optimizers=[o.state_dict() for o in opt], rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return tuple(n.eval() for n in nets), dict(checkpoint=study.artifact(p), width=width, loss=trace,
        donor_queries=sum(map(len, groups.values())), donor_options=sum(len(y) for g in groups.values() for _, y in g))


@torch.inference_mode()
def predict(nets, soft, d, p, opts, cut):
    c, x, z, _ = old.model.inputs(d, p, opts, cut[0])
    ix, conf = old.model.choose(soft.context(torch.from_numpy(c)).numpy(), soft.action(torch.from_numpy(x)).numpy(),
                                np.zeros(len(opts)), opts)
    xx = torch.from_numpy(prior.model.features(c, z))
    return [(int(ix[0]), float(conf[0])), *(model.choose(n(xx)[:, 0].numpy()) for n in nets)]


def calibrate(nets, soft, cf, of, data, cy, seed, cut):
    action, forecaster = timing.rel.pair.prior.load_action(cf), timing.old.load_outer(of)
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {k: [] for k in ARMS}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = data[n]
        a = timing.incumbent(d, seed, cut, raw[lo:hi], action)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, _, _ = bank.candidate(d, a)
        y, v = cy[n]
        for k, (i, conf) in zip(ARMS, predict(nets, soft, d, p, opts, cut), strict=True):
            path = bank.model.render(d["cv"], p, opts[i])
            rows[k].append(dict(score=conf, strict=eq.ceiling.eligible_additive(prev.metrics(path, d, y, v, cut)), add_ok=True))
    return {k: timing.old.action.prior.calibration(v) for k, v in rows.items()}


def freeze(key, data, src, roles, inner, outer, control, refs, parent):
    done = {}
    for name, r in parent["folds"].items():
        f = study.read(study.checked(r))
        held, seed, cal, donors, cut = (f["identity"][k] for k in ("held", "seed", "calibration", "donors", "cut"))
        ident = dict(f["identity"], registration=key, parent=r)
        role = roles[held]
        if cal != role["calibration"] or donors != role["donors"] or cal in donors or held in [cal, *donors]:
            raise ValueError("Overlapping label roles")
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
            nets, receipt = fit(ident, name, examples(data, dy, role, inner, refs["folds"][name], seed, cut))
            soft = prior.load_soft(f)
            cf = control["folds"][name]
            rules = calibrate(nets, soft, cf, outer["folds"][name], data, cy, seed, cut)
            if rules["soft"] != f["rules"]["standalone_additive"]:
                raise ValueError("Original calibration changed")
            rows = []
            for n, d, p, opts, q in prior.views(data, cf, held, seed, f):
                a = predict(nets, soft, d, p, opts, cut)
                if a[0] != (q["choice"][0], q["confidence"][0]):
                    raise ValueError("Original prediction changed")
                rows.append(dict(clip=str(n), hash=q["hash"], choices=a))
            base.immutable(mark, dict(identity=ident, fit=receipt, rules=rules, rows=rows))
        z = study.read(mark)
        if z["identity"] != ident or len(z["rows"]) != len(f["rows"]):
            raise ValueError("Wrong frozen receipt")
        study.checked(z["fit"]["checkpoint"])
        done[name] = study.artifact(mark)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False, reserved_outcomes_opened=False))


def fold_score(data, truth, f, cf):
    held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
    parent = study.read(study.checked(f["identity"]["parent"]))
    rows = []
    for (n, d, p, opts, q), z in zip(prior.views(data, cf, held, seed, parent), f["rows"], strict=True):
        if z["clip"] != n or z["hash"] != q["hash"]:
            raise ValueError("Wrong frozen choice")
        y, v = truth[n]
        cv = d["cv"]
        paths = [bank.model.render(cv, p, opts[i]) for i, _ in z["choices"]]
        exact, _ = bank.model.eligibility(cv, d["ols4"], p, y, v, cut[0], opts)
        row = dict(city=held, clip=str(n), seed=seed, complete=bool(v.all()), opportunity=bool(exact.any()), arms={})
        for k, path, (i, conf) in zip(ARMS, paths, z["choices"], strict=True):
            m = prev.metrics(path, d, y, v, cut)
            ok = eq.ceiling.eligible_additive(m)
            if ok != bool(exact[i]):
                raise ValueError("Scalar and bank eligibility disagree")
            threshold = f["rules"][k]["threshold"]
            release = threshold is not None and conf >= threshold
            risk = m["risk"]
            row["arms"][k] = dict(eligible=ok, released=release, release_pass=release and ok,
                risk=risk["supported"], easy_defined=risk["completion_easy_risk_upper"] is not None,
                easy_over=(risk["known_easy_net_degradation"] or 0.) > .02, ade=m["ade"], fde=m["fde"],
                tail=m["tail"], hard=m["hard"], gain=m["gain"], vs_soft=prior.trajectory(path, paths[0], d, y, v),
                vs_pointwise=prior.trajectory(path, paths[1], d, y, v),
                pair_vs_soft=float(av.frontier.pair_reduction(path[None], paths[0], y, v)[0, 1]),
                pair_vs_pointwise=float(av.frontier.pair_reduction(path[None], paths[1], y, v)[0, 1]),
                deployed_vs_cv=prior.trajectory(path if release else cv, cv, d, y, v))
        rows.append(row)
    return rows


def summarize(rows, parent, folds):
    arms = {}
    for k in ARMS:
        a = [r["arms"][k] for r in rows]
        arms[k] = dict(**{n: sum(z[n] for z in a) for n in ("eligible", "released", "release_pass", "risk", "easy_defined", "easy_over")},
            worst_p95=max(z["tail"] for z in a if z["tail"] is not None),
            complete_eligible=sum(r["complete"] and r["arms"][k]["eligible"] for r in rows),
            **{n: av.interval(rows, lambda r: r["arms"][k][n]) for n in ("ade", "fde", "hard", "gain",
                "vs_soft", "vs_pointwise", "pair_vs_soft", "pair_vs_pointwise", "deployed_vs_cv")},
            **{f"eligible_vs_{cmp}": av.interval(rows, lambda r: int(r["arms"][k]["eligible"])-int(r["arms"][cmp]["eligible"]))
                for cmp in ("soft", "pointwise")},
            released_same_set_vs_soft=[z["vs_soft"] for z in a if z["released"]],
            released_same_set_soft_pass=sum(r["arms"][k]["released"] and r["arms"]["soft"]["eligible"] for r in rows),
            calibration_pass=sum(f["rules"][k]["accepted"] for f in folds))
    c, expected = arms["soft"], parent["releases"]["standalone_additive"]
    if len(rows) != parent["views"] or (c["eligible"], c["released"], c["release_pass"]) != (
            parent["arms"]["additive"]["eligible"], expected["held_selected"], expected["held_pass"]):
        raise ValueError("Frozen soft control changed")
    return dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows),
        opportunities=sum(r["opportunity"] for r in rows), arms=arms,
        by_city={city: {k: {n: sum(r["arms"][k][n] for r in rows if r["city"] == city)
            for n in ("eligible", "released", "release_pass")} for k in ARMS} for city in sorted({r["city"] for r in rows})})


def score(key, data, src, control, result, replay):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or seal["reserved_outcomes_opened"]:
        raise ValueError("Require complete freeze")
    truth, rows, folds = None, [], []
    for name, ref in seal["folds"].items():
        f = study.read(study.checked(ref))
        folds.append(f)
        mark = LOCAL/f"{name}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay may not read outcomes")
            if truth is None:
                truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
            base.immutable(mark, dict(freeze=ref, rows=fold_score(data, truth, f, control["folds"][name])))
        z = study.read(mark)
        if z["freeze"] != ref:
            raise ValueError("Wrong score receipt")
        rows.extend(z["rows"])
    summary = summarize(rows, result, folds)
    base.immutable(OUT/"summary.json", dict(**summary, freeze=study.artifact(p), result_source="fresh_run",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False))
    print(study.json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score", "replay"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, inner, outer, control, refs, parent, result = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, refs, parent)
        else:
            score(key, data, src, control, result, replay=phase == "replay")


if __name__ == "__main__":
    main()
