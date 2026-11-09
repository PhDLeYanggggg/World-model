"""Train and compare separate assignment constraints on the frozen AV2 bank."""

import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_assignment_cost as old
from src.world_model import m3w_av2_assignment_constraints as model


np, torch, study, base, prev, eq, av = old.np, old.torch, old.study, old.base, old.prev, old.eq, old.av
timing, bank = old.timing, old.prior
OUT, LOCAL = old.OUT.parent/"av2_assignment_constraints_v1", old.LOCAL.parent/"av2_assignment_constraints_v1"
STEPS, SAVE, BATCH = 400, 50, 128


def beat(phase, **fields):
    row = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **fields)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(row)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(row, flush=True)


def register():
    key, data, src, roles, inner, outer, control, refs, _, _ = old.register()
    parent = old.OUT/"summary.json"
    result = study.read(parent)
    frozen = study.read(study.checked(result["freeze"]))
    if frozen["registration"] != key or frozen["held_labels_used"] or result["reserved_outcomes_opened"]:
        raise ValueError("Require the sealed matched cost study")
    p = OUT/"registration.json"
    base.immutable(p, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(parent), arms=model.ARMS, steps=STEPS, save_every=SAVE, batch=BATCH, seeds=eq.SEEDS,
        probability_cut=model.PROB_CUT, no_proposal="forced readout uses soft choice; release abstains to CV",
        primary="learned feasibility constraints vs frozen soft cost, including same-release-set utility and failures",
        analysis="analytic and learned restrict the same bank, choose maximum predicted minimum two-floor gain",
        training="learn P(support), P(risk|support), P(utility|support,risk) with observed antecedent masking",
        threshold="unchanged original nonempty all-qualified confidence prefix on calibration city",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, threads=4, workers=0))
    return base.sha(p), data, src, roles, inner, outer, control, refs, frozen, result


def examples(data, truth, role, inner, ref, seed, cut):
    raw = timing.old.donor_forecasts(inner, data, ref["identity"]["held"], role, seed)
    if set(raw) != set(truth):
        raise ValueError("Incorrect donor roster")
    actions = {k: timing.rel.pair.inner_action(v, ref["identity"], k) for k, v in ref["references"].items()}
    groups = defaultdict(list)
    for n, (y, v) in truth.items():
        d = data[n]
        action, ac = actions[d["city"]]
        a = timing.incumbent(d, seed, cut, raw[n], action, action_cut=ac)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, _, _ = bank.candidate(d, a)
        c, _, x, _ = old.model.inputs(d, p, opts, cut[0])
        label = model.labels(d, p, opts, y, v, cut[0])
        expected, support = bank.model.eligibility(d["cv"], d["ols4"], p, y, v, cut[0], opts)
        if not (np.array_equal(label[:, 0], support) and np.array_equal(label.all(1), expected)):
            raise ValueError("Training labels diverge from exact bank evaluator")
        groups[d["city"]].append((model.features(c, x), label))
    if set(groups) != set(role["donors"]):
        raise ValueError("Need all four donor cities")
    return {k: tuple(torch.from_numpy(np.concatenate(v)) for v in zip(*z, strict=True)) for k, z in sorted(groups.items())}


def fit(ident, name, groups):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+11281)
    width = next(iter(groups.values()))[0].shape[1]
    net = model.head(width)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    step0, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong resume identity")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        step0, trace = ck["step"], ck["loss"]
    for step in range(step0+1, STEPS+1):
        samples = [(x, y, rng.integers(len(y), size=BATCH//len(groups))) for x, y in groups.values()]
        x, y = (torch.cat([a[k][a[2]] for a in samples]) for k in (0, 1))
        loss = model.loss(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append(float(loss.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), width=width, loss=trace,
                            donor_options=sum(len(y) for _, y in groups.values()))


def load_soft(parent):
    ck = torch.load(study.checked(parent["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != parent["identity"] or ck["step"] != old.STEPS:
        raise ValueError("Wrong frozen soft comparator")
    net = old.model.Heads(parent["fit"]["widths"])
    net.load_state_dict(ck["net"])
    return net.eval()


@torch.inference_mode()
def predict(net, soft, d, p, opts, cut):
    c, x, z, _ = old.model.inputs(d, p, opts, cut[0])
    prob = net(torch.from_numpy(model.features(c, z))).sigmoid().numpy()
    return model.select(soft.context(torch.from_numpy(c)).numpy(), soft.action(torch.from_numpy(x)).numpy(), prob, opts)


def calibrate(net, soft, control, outer, data, cy, seed, cut):
    action, forecaster = timing.rel.pair.prior.load_action(control), timing.old.load_outer(outer)
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {k: [] for k in model.ARMS}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = data[n]
        a = timing.incumbent(d, seed, cut, raw[lo:hi], action)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, _, _ = bank.candidate(d, a)
        y, v = cy[n]
        for k, (i, score, _) in zip(model.ARMS, predict(net, soft, d, p, opts, cut), strict=True):
            if score is None:
                continue
            path = bank.model.render(d["cv"], p, opts[i])
            rows[k].append(dict(score=score, strict=eq.ceiling.eligible_additive(prev.metrics(path, d, y, v, cut)), add_ok=True))
    return {k: timing.old.action.prior.calibration(v) for k, v in rows.items()}


def views(data, control, held, seed, parent):
    with np.load(study.checked(control["forecast"]), allow_pickle=False) as a:
        k = 0
        for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
            d = data[n]
            if d["city"] != held:
                raise ValueError("Wrong held source")
            if len(eq.radii(d, seed)) != 2:
                continue
            r = parent["rows"][k]
            k += 1
            p, _, opts, _, digest = bank.candidate(d, a["additive"][lo:hi])
            if n != r["clip"] or digest != r["hash"]:
                raise ValueError("Held bank changed")
            yield n, d, p, opts, r
        if k != len(parent["rows"]):
            raise ValueError("Held roster changed")


def freeze(key, data, src, roles, inner, outer, control, refs, parent):
    done = {}
    for name, r in parent["folds"].items():
        f = study.read(study.checked(r))
        held, seed, cal, donors, cut = (f["identity"][k] for k in ("held", "seed", "calibration", "donors", "cut"))
        role = roles[held]
        ident = dict(f["identity"], registration=key, parent=r)
        if cal != role["calibration"] or donors != role["donors"] or cal in donors or held in [cal, *donors]:
            raise ValueError("Overlapping label roles")
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
            net, receipt = fit(ident, name, examples(data, dy, role, inner, refs["folds"][name], seed, cut))
            soft = load_soft(f)
            cf = control["folds"][name]
            rules = calibrate(net, soft, cf, outer["folds"][name], data, cy, seed, cut)
            if rules["soft"] != f["rules"]["standalone_additive"]:
                raise ValueError("Original calibration changed")
            rows = []
            for n, d, p, opts, q in views(data, cf, held, seed, f):
                a = predict(net, soft, d, p, opts, cut)
                if a[0][:2] != (q["choice"][0], q["confidence"][0]):
                    raise ValueError("Original prediction changed")
                rows.append(dict(clip=str(n), hash=q["hash"], proposals=a))
            base.immutable(mark, dict(identity=ident, fit=receipt, rules=rules, rows=rows))
        z = study.read(mark)
        if z["identity"] != ident or len(z["rows"]) != len(f["rows"]):
            raise ValueError("Wrong frozen receipt")
        study.checked(z["fit"]["checkpoint"])
        done[name] = study.artifact(mark)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False, reserved_outcomes_opened=False))


def trajectory(p, a, d, y, v):
    z = timing.cost.audit.direct_trajectory(p, a, d["cv"], y, v)
    return None if z is None else z[2]


def fold_score(data, truth, f, cf):
    held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
    parent = study.read(study.checked(f["identity"]["parent"]))
    rows = []
    for n, d, p, opts, q in views(data, cf, held, seed, parent):
        seal = f["rows"][len(rows)]
        if seal["clip"] != n or seal["hash"] != q["hash"]:
            raise ValueError("Wrong frozen proposal")
        y, v = truth[n]
        choices = seal["proposals"]
        paths = [bank.model.render(d["cv"], p, opts[i]) for i, _, _ in choices]
        soft = paths[0]
        row = dict(clip=str(n), city=held, seed=seed, complete=bool(v.all()), arms={})
        for k, path, (_, score, count) in zip(model.ARMS, paths, choices, strict=True):
            m = prev.metrics(path, d, y, v, cut)
            ok = eq.ceiling.eligible_additive(m)
            risk = m["risk"]
            threshold = f["rules"][k]["threshold"]
            release = score is not None and threshold is not None and score >= threshold
            row["arms"][k] = dict(eligible=ok, options=count, proposal=bool(count), released=release, release_pass=release and ok,
                risk=risk["supported"], easy_defined=risk["completion_easy_risk_upper"] is not None,
                easy_over=(risk["known_easy_net_degradation"] or 0.) > .02,
                ade=m["ade"], fde=m["fde"], tail=m["tail"], gain=m["gain"],
                vs_soft=trajectory(path, soft, d, y, v),
                pair_vs_soft=float(av.frontier.pair_reduction(path[None], soft, y, v)[0, 1]),
                policy_vs_cv=trajectory(path if release else d["cv"], d["cv"], d, y, v))
        rows.append(row)
    return rows


def summarize(rows, result):
    arms = {}
    for k in model.ARMS:
        a = [r["arms"][k] for r in rows]
        arms[k] = dict(**{n: sum(z[n] for z in a) for n in ("eligible", "proposal", "options", "released", "release_pass",
                    "risk", "easy_defined", "easy_over")}, worst_p95=max(z["tail"] for z in a if z["tail"] is not None),
            **{n: av.interval(rows, lambda r: r["arms"][k][n]) for n in ("ade", "fde", "gain", "vs_soft", "pair_vs_soft", "policy_vs_cv")},
            eligible_vs_soft=av.interval(rows, lambda r: int(r["arms"][k]["eligible"])-int(r["arms"]["soft"]["eligible"])),
            released_same_set_vs_soft=[z["vs_soft"] for z in a if z["released"]],
            released_same_set_soft_pass=sum(r["arms"][k]["released"] and r["arms"]["soft"]["eligible"] for r in rows))
    c, expected = arms["soft"], result["releases"]["standalone_additive"]
    if len(rows) != result["views"] or (c["eligible"], c["released"], c["release_pass"]) != (
            result["arms"]["additive"]["eligible"], expected["held_selected"], expected["held_pass"]):
        raise ValueError("Frozen soft comparator did not replay")
    cities = sorted({r["city"] for r in rows})
    return dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows), arms=arms,
        by_city={city: {k: {n: sum(r["arms"][k][n] for r in rows if r["city"] == city)
            for n in ("eligible", "proposal", "released", "release_pass")} for k in model.ARMS} for city in cities})


def score(key, data, src, control, result, replay):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or seal["reserved_outcomes_opened"]:
        raise ValueError("Require the complete freeze")
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
    s = summarize(rows, result)
    accepted = {k: sum(f["rules"][k]["accepted"] for f in folds) for k in model.ARMS}
    base.immutable(OUT/"summary.json", dict(**s, freeze=study.artifact(p), calibration_accepted=accepted, result_source="fresh_run",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False))
    print(study.json.dumps(s, indent=2))


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
