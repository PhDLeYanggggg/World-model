"""Fit additive gain/risk and pair costs on the same fixed AV2 assignment bank."""

import argparse
from collections import Counter, defaultdict
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_av2_recipient_reassignment as prior
from src.world_model import m3w_av2_assignment_cost as model


np, torch, study, base, prev, av, eq = prior.np, prior.torch, prior.study, prior.base, prior.prev, prior.av, prior.eq
timing = prior.stage.prior
OUT, LOCAL = prior.OUT.parent/"av2_assignment_cost_v1", prior.LOCAL.parent/"av2_assignment_cost_v1"
STEPS, SAVE, BATCH = 400, 50, 128
ARMS = ("anchor", "history_support", "history_joint", *model.ARMS)
RULES = ("standalone_additive", "standalone_joint", "matched_joint")


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    d = dict(pid=os.getpid(), epoch=time.time(), phase=phase, **fields)
    p.write_text(study.json.dumps(d)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(d, flush=True)


def register():
    pkey, _, _, _, _, _, _ = prior.register()
    p = prior.OUT/"summary.json"
    report = study.read(p)
    freeze = study.read(study.checked(report["freeze"]))
    if freeze["registration"] != pkey or freeze["held_labels_used"] or report["reserved_outcomes_opened"]:
        raise ValueError("Require the frozen assignment experiment")
    _, data, src, roles, inner, outer, control, refs, _ = timing.register()
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)], parent=study.artifact(p),
        arms=ARMS, rules=RULES, steps=STEPS, checkpoint_every=SAVE, batch=BATCH, seeds=eq.SEEDS,
        bank="same two transported per-step radial profiles as prior; no extra options or changed support rule",
        objective="donor-only dense two-floor gains, all/easy positive-harm excess, signed easy change, easy probability and reference mass; "
                  "separate residual on the one relative edge touched by both edits",
        comparison="both learned policies share every additive cost, support estimate and candidate; only joint uses predicted pair residual",
        pair_weight=model.PAIR_WEIGHT, support_weight=model.SUPPORT_WEIGHT,
        calibration="largest original nonempty strict prefix on separate city, standalone vs both floors and matched joint vs additive reported separately",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, num_workers=0, torch_threads=4))
    return base.sha(reg), data, src, roles, inner, outer, control, refs, freeze, report


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
        p, _, opts, _, _ = prior.candidate(d, a)
        c, x, z, scale = model.inputs(d, p, opts, cut[0])
        cy, xy, zy = model.targets(d, p, opts, y, v, cut[0], scale)
        groups[d["city"]].append((c, cy, x.reshape(-1, x.shape[-1]), xy.reshape(-1, xy.shape[-1]), z, zy))
    if set(groups) != set(role["donors"]):
        raise ValueError("Need all four donor cities")
    return {k: tuple(torch.from_numpy(np.concatenate(v)) for v in zip(*z, strict=True)) for k, z in sorted(groups.items())}, sum(map(len, groups.values()))


def fit(ident, name, data, queries):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+39403)
    widths = [next(iter(data.values()))[i].shape[-1] for i in (0, 2, 4)]
    net = model.Heads(widths)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    step0, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Invalid assignment checkpoint")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        step0, trace = ck["step"], ck["loss"]
    f = torch.nn.functional
    for step in range(step0+1, STEPS+1):
        batch = []
        for i in (0, 2, 4):
            picks = [(r, rng.integers(len(r[i]), size=BATCH//len(data))) for r in data.values()]
            batch.extend(torch.cat([r[j][at] for r, at in picks]) for j in (i, i+1))
        c, cy, x, xy, z, zy = batch
        cv = net.context(c)
        losses = [f.binary_cross_entropy_with_logits(cv[:, 0], cy[:, 0]), f.smooth_l1_loss(cv[:, 1:], cy[:, 1:], beta=.05),
                  f.smooth_l1_loss(net.action(x), xy, beta=.05), f.smooth_l1_loss(net.pair(z)[:, 0], zy, beta=.05)]
        opt.zero_grad(set_to_none=True)
        sum(losses).backward()
        # A pair-head gradient must not rescale the additive update.
        for head in (net.context, net.action, net.pair):
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(v.detach()) for v in losses])
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), donor_queries=queries, widths=widths, loss=trace,
        donor_actions=sum(len(x[2]) for x in data.values()), donor_pairs=sum(len(x[4]) for x in data.values()))


@torch.inference_mode()
def predict(net, d, p, opts, cut):
    c, x, z, _ = model.inputs(d, p, opts, cut[0])
    return model.choose(net.context(torch.from_numpy(c)).numpy(), net.action(torch.from_numpy(x)).numpy(),
                        net.pair(torch.from_numpy(z)).numpy()[:, 0], opts)


def matched(d, a, j, y, v, cut, eligible):
    direct = timing.cost.audit.direct_trajectory(j, a, d["cv"], y, v)
    pair = float(av.frontier.pair_reduction(j[None], a, y, v)[0, 1])
    ok = bool(all(eligible) and direct is not None and direct[2] >= -1e-8 and pair > 1e-10)
    return ok, direct[2] if direct is not None else None, pair


def calibrate(net, action, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {k: [] for k in RULES}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = data[n]
        a = timing.incumbent(d, seed, cut, raw[lo:hi], action)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, _, _ = prior.candidate(d, a)
        choice, confidence = predict(net, d, p, opts, cut)
        paths = [prior.model.render(d["cv"], p, opts[i]) for i in choice]
        y, v = cy[n]
        e = [eq.ceiling.eligible_additive(prev.metrics(z, d, y, v, cut)) for z in paths]
        for k, s, ok in zip(RULES[:2], confidence, e, strict=True):
            rows[k].append(dict(score=float(s), strict=bool(ok), add_ok=True))
        ok, _, _ = matched(d, *paths, y, v, cut, e)
        rows["matched_joint"].append(dict(score=float(confidence[1]), strict=ok, add_ok=bool(e[0])))
    return {k: timing.old.action.prior.calibration(v) for k, v in rows.items()}


def freeze(key, data, src, roles, inner, outer, control, refs, parent):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        if held in donors or cal in donors or cal == held:
            raise ValueError("Overlapping source roles")
        dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            cf, of, ref = control["folds"][name], outer["folds"][name], refs["folds"][name]
            cut = ref["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            for r in (cf, ref, of):
                if any(ident[k] != r["identity"][k] for k in ("held", "calibration", "donors", "seed")):
                    raise ValueError("Incorrect donor/calibration lineage")
            mark = LOCAL/f"{name}.json"
            frozen = study.read(study.checked(parent["folds"][name]))
            if not mark.exists():
                net, fit_ref = fit(ident, name, *examples(data, dy, role, inner, ref, seed, cut))
                rules = calibrate(net, timing.rel.pair.prior.load_action(cf), timing.old.load_outer(of), data, cy, seed, cut)
                rows = []
                with np.load(study.checked(cf["forecast"]), allow_pickle=False) as a:
                    for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                        d = data[n]
                        if d["city"] != held:
                            raise ValueError("Wrong held city")
                        if len(eq.radii(d, seed)) != 2:
                            continue
                        p, _, opts, _, digest = prior.candidate(d, a["additive"][lo:hi])
                        q = frozen["rows"][len(rows)]
                        if str(n) != q["clip"] or digest != q["hash"]:
                            raise ValueError("Candidate bank differs from the heuristic control")
                        ix, conf = predict(net, d, p, opts, cut)
                        rows.append(dict(clip=str(n), hash=digest, choice=ix.tolist(), confidence=conf.tolist()))
                base.immutable(mark, dict(identity=ident, fit=fit_ref, rules=rules, rows=rows))
            r = study.read(mark)
            if r["identity"] != ident or len(r["rows"]) != len(frozen["rows"]):
                raise ValueError("Frozen roster changed")
            study.checked(r["fit"]["checkpoint"])
            done[name] = study.artifact(mark)
            beat("freeze", completed=len(done), total=len(refs["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False, reserved_outcomes_opened=False))


def fold_score(data, truth, f, cf, old):
    held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
    rows = []
    with np.load(study.checked(cf["forecast"]), allow_pickle=False) as a:
        for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
            d, (y, v) = data[n], truth[n]
            if d["city"] != held:
                raise ValueError("Wrong score source")
            if len(eq.radii(d, seed)) != 2:
                continue
            q, h = f["rows"][len(rows)], old["rows"][len(rows)]
            anchor = a["additive"][lo:hi]
            p, _, opts, idx, digest = prior.candidate(d, anchor)
            if q["clip"] != str(n) or h["clip"] != str(n) or digest != q["hash"] or digest != h["hash"] or h["choices"][0] != idx:
                raise ValueError("Scoring bank differs from freeze")
            paths = [prior.model.render(d["cv"], p, opts[i]) for i in (idx, *h["choices"][-2:], *q["choice"])]
            row = dict(clip=str(n), city=held, seed=seed, complete=bool(v.all()), arms={}, releases={})
            for k, z in zip(ARMS, paths, strict=True):
                m = prev.metrics(z, d, y, v, cut)
                risk = m["risk"]
                direct = timing.cost.audit.direct_trajectory(z, anchor, d["cv"], y, v)
                row["arms"][k] = dict(eligible=eq.ceiling.eligible_additive(m), risk=risk["supported"],
                    easy_defined=risk["completion_easy_risk_upper"] is not None, easy_over=(risk["known_easy_net_degradation"] or 0.) > .02,
                    gain=m["gain"], ade=m["ade"], fde=m["fde"], p95=m["tail"],
                    lower_vs_anchor=direct[2] if direct is not None else None)
            eligible = [row["arms"][k]["eligible"] for k in model.ARMS]
            win, traj, pair = matched(d, *paths[-2:], y, v, cut, eligible)
            row.update(matched=win, joint_vs_additive_utility=traj, joint_vs_additive_pair=pair)
            for k, s, ok in zip(RULES, [*q["confidence"], q["confidence"][1]], [*eligible, win], strict=True):
                threshold = f["rules"][k]["threshold"]
                active = threshold is not None and s >= threshold
                row["releases"][k] = [bool(active), bool(active and ok)]
            rows.append(row)
    return rows


def summarize(rows, freeze, report):
    arms = {}
    for k in ARMS:
        a = [r["arms"][k] for r in rows]
        arms[k] = dict(**{n: sum(z[n] for z in a) for n in ("eligible", "risk", "easy_defined", "easy_over")},
            complete_eligible=sum(r["arms"][k]["eligible"] for r in rows if r["complete"]),
            worst_p95=max(z["p95"] for z in a if z["p95"] is not None),
            **{n: av.interval(rows, lambda r: r["arms"][k][n]) for n in ("gain", "ade", "fde", "lower_vs_anchor")})
    if len(rows) != report["views"] or any(arms[k]["eligible"] != report["arms"][k]["eligible"] for k in ARMS[:3]):
        raise ValueError("Previous fixed-bank controls did not replay")
    gates = {}
    for k in RULES:
        gates[k] = dict(calibration=sum(f["rules"][k]["accepted"] for f in freeze),
            held_selected=sum(r["releases"][k][0] for r in rows), held_pass=sum(r["releases"][k][1] for r in rows))
    return dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows),
        arms=arms, releases=gates, forced_matched_joint=sum(r["matched"] for r in rows),
        paired={k: av.interval(rows, lambda r: r[k]) for k in ("joint_vs_additive_utility", "joint_vs_additive_pair")},
        eligibility_joint_minus_additive=av.interval(rows, lambda r: int(r["arms"]["joint"]["eligible"])-int(r["arms"]["additive"]["eligible"])),
        by_city={s: {k: sum(r["arms"][k]["eligible"] for r in rows if r["city"] == s) for k in ARMS} for s in sorted({r["city"] for r in rows})})


def score(key, data, src, control, parent, report, replay):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["held_labels_used"] or f["reserved_outcomes_opened"]:
        raise ValueError("Require frozen assignment costs")
    rows, folds, truth = [], [], None
    for name, ref in f["folds"].items():
        r = study.read(study.checked(ref))
        folds.append(r)
        mark = LOCAL/f"{name}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay cannot read held labels")
            if truth is None:
                truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
            z = fold_score(data, truth, r, control["folds"][name], study.read(study.checked(parent["folds"][name])))
            base.immutable(mark, dict(freeze=ref, rows=z))
        z = study.read(mark)
        if z["freeze"] != ref:
            raise ValueError("Incorrect score receipt")
        rows.extend(z["rows"])
    result = summarize(rows, folds, report)
    base.immutable(OUT/"summary.json", dict(**result, freeze=study.artifact(p), result_source="fresh_run", fitted_heads=3*len(folds),
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False))
    print(study.json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, inner, outer, control, refs, parent, report = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, refs, parent)
        else:
            score(key, data, src, control, parent, report, replay=phase == "replay")


if __name__ == "__main__":
    main()
