"""Replace only the support term of the frozen AV2 confidence score."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_position_release as prior
from scripts.audit_m3w_av2_position_release import ranks
from scripts.audit_m3w_av2_projection_risk_matched import weights
from src.evaluation.m3w_calibration_failure_taxonomy import classify
from src.world_model import m3w_av2_projection_support as model

np, torch, study, base, av = prior.np, prior.torch, prior.study, prior.base, prior.av
rel, cost = prior.prior, prior.cost
OUT, LOCAL = prior.OUT.parent/"av2_projection_support_v1", prior.LOCAL.parent/"av2_projection_support_v1"
ARMS = ("transferred_cost", "without_support", "actor_support")
STEPS, SAVE, BATCH = 400, 50, 96


def beat(phase, **kw):
    r = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **kw)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(r)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(r, flush=True)


def register():
    k, data, outer, control, parent, previous = prior.register()
    path = prior.OUT/"summary.json"
    s = study.read(path)
    f = study.read(study.checked(s["freeze"]))
    if f["registration"] != k or f["outcomes_read"] or s["reserved_outcomes_opened"]:
        raise ValueError("Require the frozen position and confidence scores")
    _, _, src, roles, _, _, _, _, _, _ = rel.register()
    p = OUT/"registration.json"
    base.immutable(p, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)], parent=study.artifact(path),
        arms=ARMS, seeds=rel.eq.SEEDS, steps=STEPS, save_every=SAVE, batch=BATCH,
        support="all eligible donor actor histories; CV error in (0,easy cutoff]; no projection or selected-action target",
        score="same frozen transferred cost risk/gain term, remove or replace only probability that either edited actor has support",
        original_2pct=True, calibration="original nonempty all-qualified prefix, separate calibration city",
        diagnostic="held top-five with equal count, risk cause decomposition and known-support oracle, not used for release",
        independent_confirmation=False, reserved_outcomes_opened=False, threads=4, workers=0))
    return base.sha(p), data, src, roles, outer, control, parent, f, s, previous


def fit(ident, name, data, truth):
    rows = defaultdict(list)
    cutoff = ident["cut"][0]
    for n, (y, v) in truth.items():
        d = data[n]
        if d["city"] not in ident["donors"]:
            raise ValueError("Fit attempted to open a non-donor")
        if len(d["h"]) and "cv" in d:
            rows[d["city"]].append((model.inputs(d, cutoff), model.labels(d["cv"], y, v, cutoff)))
    if set(rows) != set(ident["donors"]):
        raise ValueError("Need all four donors")
    groups = [tuple(torch.from_numpy(np.concatenate(z)) for z in zip(*rows[c], strict=True)) for c in sorted(rows)]
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+6449)
    width = groups[0][0].shape[1]
    net = model.head(width)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong resume identity")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, trace = ck["step"], ck["trace"]
    for step in range(start+1, STEPS+1):
        b = [(r, rng.integers(len(r[0]), size=BATCH//len(groups))) for r in groups]
        x, y = (torch.cat([r[k][ix] for r, ix in b]) for k in range(2))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], y.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append(float(loss.detach()))
            q = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), q)
            q.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), actors=sum(len(y) for _, y in groups),
                            easy_actors=sum(int(y.sum()) for _, y in groups), trace=trace)


@torch.inference_mode()
def scores(net, legacy, d, path, ix, cut, expected):
    c, a, _, _ = cost.inputs(d, np.repeat(path[:, None], 2, axis=1), ix[None], cut)
    context, action = legacy.context(torch.from_numpy(c)).numpy(), legacy.action(torch.from_numpy(a)).numpy()
    _, prob = cost.choose(context, action, np.zeros(1), ix[None])
    if float(prob[0]) != expected:
        raise ValueError("Legacy confidence changed")
    context[:, 0] = 50.
    _, gain_risk = cost.choose(context, action, np.zeros(1), ix[None])
    est = float(model.either(net(torch.from_numpy(model.inputs(d, cut)[ix]))[:, 0].sigmoid().numpy()))
    g = float(gain_risk[0])
    return dict(transferred_cost=expected, without_support=g, actor_support=g*est), dict(old=expected/g, new=est)


def frozen_paths(data, f, cf, of, role):
    p = study.read(study.checked(f["identity"]["parent"]))
    for (n, d, _, ix, digest, _), q in zip(rel.inputs(data, p, cf, of, role), f["rows"][role], strict=True):
        if n != q["clip"] or digest != q["hash"] or ix.tolist() != q["edited"]:
            raise ValueError("Wrong frozen position")
        path = d["cv"].copy()
        path[ix] = q["paths"][0]
        yield n, d, ix, path


def freeze(key, data, src, roles, outer, control, parent, baseline):
    done = {}
    for name, ref in parent["folds"].items():
        f = study.read(study.checked(ref))
        old = study.read(study.checked(baseline["folds"][name]))
        held, cal, donors = (f["identity"][k] for k in ("held", "calibration", "donors"))
        role = roles[held]
        if cal != role["calibration"] or donors != role["donors"] or held in [cal, *donors] or cal in donors or old["identity"]["parent"] != ref:
            raise ValueError("Overlapping source roles")
        ident = dict(f["identity"], registration=key, source=ref)
        p = LOCAL/f"{name}.json"
        if not p.exists():
            dy = rel.prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            net, r = fit(ident, name, data, dy)
            projected = study.read(study.checked(f["identity"]["parent"]))
            calref = study.read(study.checked(projected["identity"]["calibration_parent"]))
            legacy = prior.load_soft(study.read(study.checked(calref["identity"]["parent"])))
            rows = {}
            for side in ("calibration", "held"):
                rows[side] = []
                for (n, d, ix, path), q in zip(frozen_paths(data, f, control["folds"][name], outer["folds"][name], side),
                                               old["rows"][side], strict=True):
                    if n != q["clip"]:
                        raise ValueError("Wrong legacy confidence row")
                    s, a = scores(net, legacy, d, path, ix, ident["cut"][0], q["score"]["transferred_cost"])
                    rows[side].append(dict(clip=str(n), city=d["city"], score=s, probability=a))
            base.immutable(p, dict(identity=ident, rows=rows, fit=r))
        z = study.read(p)
        if z["identity"] != ident:
            raise ValueError("Wrong support freeze")
        study.checked(z["fit"]["checkpoint"])
        done[name] = study.artifact(p)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, calibration_labels_used=False,
        held_labels_used=False, reserved_outcomes_opened=False))


def annotate(data, src, f, z, parent_ref, name, cf, of):
    mark = LOCAL/f"{name}.support.json"
    if not mark.exists():
        names = [r["clip"] for r in z["rows"]["held"]]
        held = f["identity"]["held"]
        if any(data[n]["city"] != held for n in names) or held in [f["identity"]["calibration"], *f["identity"]["donors"]]:
            raise ValueError("Wrong held outcome role")
        truth = rel.prev.load_targets(src, data, names)
        cells = []
        original = list(prior.role_rows(z, parent_ref, name, "held"))
        for (n, d, ix, path), q in zip(frozen_paths(data, f, cf, of, "held"), original, strict=True):
            if n != q["clip"]:
                raise ValueError("Wrong held annotation")
            y, valid = truth[n]
            cut = f["identity"]["cut"]
            risk = rel.prev.metrics(path, d, y, valid, cut)["risk"]
            support = bool(model.labels(d["cv"], y, valid, cut[0])[ix].any())
            if support != (risk["completion_easy_risk_upper"] is not None) or risk["supported"] != q["result"]["risk"]:
                raise ValueError("Support or risk did not replay")
            cells.append(dict(clip=str(n), support=support, cause=classify(risk)["category"]))
        base.immutable(mark, dict(parent=parent_ref, cells=cells))
    q = study.read(mark)
    if q["parent"] != parent_ref:
        raise ValueError("Wrong support outcome receipt")
    return q["cells"]


def matched(rows, k):
    out = {}
    for arm in (*ARMS, "known_support_oracle"):
        score = np.asarray([r["scores"][arm] if arm in ARMS else
            (r["scores"]["without_support"] if r["support"] else -np.inf) for r in rows])
        w = weights(score, k)
        a = np.array([(r["result"]["eligible"], not r["result"]["risk"], not r["support"]) for r in rows])
        out[arm] = dict(zip(("qualified", "risk_fail", "unsupported"), map(float, w@a), strict=True))
    return out


def score(key, data, src, outer, control, parent, baseline, previous):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or any(f[k] for k in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require predictions frozen before outcome access")
    rules = {}
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        rows = list(prior.role_rows(z, parent["folds"][name], name, "calibration"))
        rules[name] = {a: rel.timing.old.action.prior.calibration([dict(score=r["scores"][a], strict=r["result"]["eligible"],
            add_ok=True) for r in rows]) for a in ARMS}
    cp = OUT/"calibration.json"
    base.immutable(cp, dict(freeze=study.artifact(p), rules=rules, held_receipts_used=False))
    all_rows, paired, ap, actors = [], [], defaultdict(list), Counter()
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        actors.update({k: z["fit"][k] for k in ("actors", "easy_actors")})
        pf = parent["folds"][name]
        zz = study.read(study.checked(pf))
        rows = list(prior.role_rows(z, pf, name, "held"))
        cells = annotate(data, src, zz, z, pf, name, control["folds"][name], outer["folds"][name])
        for r, q, c in zip(rows, z["rows"]["held"], cells, strict=True):
            if r["clip"] != c["clip"]:
                raise ValueError("Wrong support outcome row")
            r.update(support=c["support"], cause=c["cause"], prob=q["probability"])
            r["release"] = {a: rules[name][a]["threshold"] is not None and r["scores"][a] >= rules[name][a]["threshold"] for a in ARMS}
        for a in ("old", "new"):
            temp = [dict(scores={"p": r["prob"][a]}, result={"eligible": r["support"]}) for r in rows]
            ap[a].append(ranks(temp, "p")["ap"])
        n = min(5, sum(r["support"] for r in rows))
        if n:
            paired.append(dict(city=zz["identity"]["held"], clip="same_count", n=n, arms=matched(rows, n)))
        all_rows.extend(rows)
    h = {a: prior.held_metrics(all_rows, a) for a in ARMS}
    if h["transferred_cost"] != baseline["held"]["transferred_cost"] or len(all_rows) != previous["held"]["views"]:
        raise ValueError("Frozen reference changed")
    fixed = {a: {k: sum(r["arms"][a][k] for r in paired) for k in ("qualified", "risk_fail", "unsupported")}
             for a in (*ARMS, "known_support_oracle")}
    delta = {a: {k: av.interval(paired, lambda r: 100.*(r["arms"][a][k]-r["arms"]["transferred_cost"][k])/r["n"])
                 for k in fixed[a]} for a in (*ARMS[1:], "known_support_oracle")}
    brier = {a: av.interval(all_rows, lambda r: (r["prob"][a]-r["support"])**2) for a in ("old", "new")}
    result = dict(views=len(all_rows), scenarios=len({r["clip"] for r in all_rows}), donor_actors=actors,
        support_views=sum(r["support"] for r in all_rows), risk_taxonomy=Counter(r["cause"] for r in all_rows),
        support_brier=brier, support_brier_new_minus_old=av.interval(all_rows,
            lambda r: (r["prob"]["new"]-r["support"])**2-(r["prob"]["old"]-r["support"])**2),
        support_ap={a: float(np.mean([s for s in z if s is not None])) for a, z in ap.items()},
        same_count=sum(r["n"] for r in paired), matched=fixed, matched_minus_legacy_pp=delta,
        held=h, release_causes={a: Counter(r["cause"] for r in all_rows if r["release"][a]) for a in ARMS},
        calibration_pass={a: sum(r[a]["accepted"] for r in rules.values()) for a in ARMS},
        freeze=study.artifact(p), calibration=study.artifact(cp), result_source="fresh_run", original_2pct=True,
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, outer, control, parent, baseline, baseline_summary, previous = register()
        if phase == "fit":
            freeze(key, data, src, roles, outer, control, parent, baseline)
        else:
            score(key, data, src, outer, control, parent, baseline_summary, previous)


if __name__ == "__main__":
    main()
