"""Train source-excluded gain and risk scores for the unchanged AV2 position model."""

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
from scripts import run_m3w_av2_position_release as prior
from scripts.audit_m3w_av2_position_release import ranks
from src.world_model import m3w_av2_projection_risk as model

np, torch, study, base, av = prior.np, prior.torch, prior.study, prior.base, prior.av
rel, timing, prev, bank, proj = prior.prior, prior.prior.timing, prior.prior.prev, prior.prior.bank, prior.prior.proj
OUT, LOCAL = prior.OUT.parent/"av2_projection_risk_v1", prior.LOCAL.parent/"av2_projection_risk_v1"
DIR_STEPS, RISK_STEPS, SAVE, BATCH = 250, 400, 50, 96


def beat(phase, **kw):
    z = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **kw)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(z)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(z, flush=True)


def register():
    k, data, outer, control, parent, old = prior.register()
    summary = prior.OUT/"summary.json"
    s = study.read(summary)
    f = study.read(study.checked(s["freeze"]))
    if f["registration"] != k or f["outcomes_read"] or s["reserved_outcomes_opened"] or s["raw_future_opened"]:
        raise ValueError("Require the frozen position confidence study")
    _, _, src, roles, inner, _, _, refs, _, _ = rel.register()
    p = OUT/"registration.json"
    base.immutable(p, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)], parent=study.artifact(summary),
        arms=model.ARMS, seeds=rel.eq.SEEDS, dir_steps=DIR_STEPS, risk_steps=RISK_STEPS, batch=BATCH, checkpoint_every=SAVE,
        inner="three other donor cities fit each direction model using frozen causal anchor radii; "
              "nested EqMotion and action reference supply the scored donor profiles, both exclude the donor being scored",
        confidence="paired lower gain regressions plus exact support and conditional risk; gain-only uses the same gain outputs",
        outer="fixed position forecasts unchanged; same original calibration prefix and held development scoring",
        ranking="fixed top 1, 5, 10 within each fold, identical intervention count, no policy fitting",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, threads=4, workers=0))
    return base.sha(p), data, src, roles, inner, outer, control, refs, parent, old, f


def batches(x):
    return tuple(torch.from_numpy(np.concatenate(z)) for z in zip(*x, strict=True))


def direction_data(data, truth, seed, cities):
    rows = defaultdict(list)
    for n, (y, v) in truth.items():
        d = data[n]
        a = d[f"anchor_{seed}"]
        if np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, _, _, _ = bank.candidate(d, a)
        for j in range(2):
            x, c, _, _, frame, scale = proj.inputs(d, p[:, j], np.arange(len(p)))
            z = (x, c, frame, scale, np.where(v[..., None], y, 0.), v)
            rows[d["city"]].append(tuple(a.astype(bool if a.dtype == bool else np.float32) for a in z))
    if set(rows) != set(cities):
        raise ValueError("Missing direction donors")
    return {k: batches(v) for k, v in sorted(rows.items())}


def fit(ident, name, data, is_direction):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+(433 if is_direction else 819))
    width = next(iter(data.values()))[0].shape[-1]
    net = (proj.head if is_direction else model.head)(width)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    total = DIR_STEPS if is_direction else RISK_STEPS
    p = LOCAL/f"{name}.pt"
    step0, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, total+1, SAVE):
            raise ValueError("Wrong resumed fit")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        step0, trace = ck["step"], ck["trace"]
    for step in range(step0+1, total+1):
        chosen = [(r, rng.integers(len(r[0]), size=BATCH//len(data))) for r in data.values()]
        b = [torch.cat([r[k][ix] for r, ix in chosen]) for k in range(len(next(iter(data.values()))))]
        if is_direction:
            x, c, frame, scale, y, v = b
            err = torch.linalg.vector_norm(c+proj.residual(net, x, frame, scale)-y, dim=-1)
            loss = ((err*v).sum(1)/v.sum(1).clamp_min(1)/scale).mean()
            terms = (loss,)
        else:
            x, gain, status = b
            loss, terms = model.loss(net(x), gain, status)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(t.detach()) for t in terms])
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(p)
            beat("direction" if is_direction else "risk", fold=name, step=step, total=total)
    return net.eval(), dict(checkpoint=study.artifact(p), identity=ident, trace=trace,
                            rows=sum(len(r[0]) for r in data.values()))


def confidence_examples(key, data, truth, role, inner, ref, seed):
    held, cal, donors = (ref["identity"][k] for k in ("held", "calibration", "donors"))
    raw = timing.old.donor_forecasts(inner, data, held, role, seed)
    if set(raw) != set(truth):
        raise ValueError("Wrong nested donor forecast roster")
    pools = direction_data(data, truth, seed, donors)
    groups, lineage = {}, {}
    for city in donors:
        training = {c: v for c, v in pools.items() if c != city}
        if len(training) != 3 or set(training) & {city, held, cal}:
            raise ValueError("Direction fit saw excluded city")
        ident = dict(registration=key, training=list(training), seed=seed, producer="causal_anchor_radii")
        direction, receipt = fit(ident, f"dir_{'_'.join(training)}_{seed}", training, is_direction=True)
        a_ref = ref["references"][city]
        action, cut = timing.rel.pair.inner_action(a_ref, ref["identity"], city)
        eq_ref = inner["models"][f"{role['inner'][city]}_{seed}"]
        if set(eq_ref["fit"]["identity"]["donors"]) != set(training):
            raise ValueError("Position and EqMotion training roster differs")
        lineage[city] = dict(direction=receipt, eqmotion=eq_ref["forecast"], action=a_ref["checkpoint"],
                             target_city=city, training=list(training))
        rows = []
        for n, (y, v) in truth.items():
            d = data[n]
            if d["city"] != city:
                continue
            a = timing.incumbent(d, seed, cut, raw[n], action, action_cut=cut)
            if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
                continue
            p, _, opts, _, _ = bank.candidate(d, a)
            all_rows = np.arange(len(p))
            projected = np.stack([rel.base_positions(direction, d, p[:, j], all_rows) for j in range(2)], axis=1)
            np.testing.assert_allclose(np.linalg.norm(projected-d["cv"][:, None], axis=-1),
                                       np.linalg.norm(p-d["cv"][:, None], axis=-1), rtol=1e-8, atol=1e-8)
            x, scale = model.inputs(d, projected, opts, cut[0])
            gain, status = model.targets(d, projected, opts, y, v, cut[0], scale)
            exact, _ = bank.model.eligibility(d["cv"], d["ols4"], projected, y, v, cut[0], opts)
            if not np.array_equal(exact, status.all(1)):
                raise ValueError("Exact support/risk/utility target did not replay")
            rows.append((x, gain, status))
        if not rows:
            raise ValueError("No source-excluded projections")
        groups[city] = batches(rows)
        beat("inner_ready", held=held, seed=seed, excluded=city, options=len(groups[city][0]))
    return groups, lineage


@torch.inference_mode()
def predict(net, d, path, ix, cut):
    x, _ = model.inputs(d, np.repeat(path[:, None], 2, axis=1), ix[None], cut[0])
    a, b = model.scores(net(torch.from_numpy(x)).numpy()[0])
    return dict(gain_only=float(a), gain_risk=float(b))


def freeze(key, data, src, roles, inner, outer, control, refs, parent, baseline):
    done = {}
    for name, ref in parent["folds"].items():
        f = study.read(study.checked(ref))
        old = study.read(study.checked(baseline["folds"][name]))
        if old["identity"]["parent"] != ref:
            raise ValueError("Wrong frozen baseline")
        held, cal, donors, seed, cut = (f["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut"))
        role = roles[held]
        if held in donors or cal in donors or held == cal or donors != role["donors"] or cal != role["calibration"]:
            raise ValueError("Overlapping roles")
        ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, parent=ref)
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            truth = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            examples, lineage = confidence_examples(key, data, truth, role, inner, refs["folds"][name], seed)
            net, fit_ref = fit(ident, f"risk_{name}", examples, is_direction=False)
            pp = study.read(study.checked(f["identity"]["parent"]))
            rows = {}
            for role in ("calibration", "held"):
                rows[role] = []
                for (n, d, _, ix, digest, _), q, z in zip(rel.inputs(data, pp, control["folds"][name], outer["folds"][name], role),
                                                        f["rows"][role], old["rows"][role], strict=True):
                    if not n == q["clip"] == z["clip"] or digest != q["hash"] or ix.tolist() != q["edited"]:
                        raise ValueError("Wrong fixed projected path")
                    path = d["cv"].copy()
                    path[ix] = q["paths"][0]
                    scores = {k: z["score"][k] for k in model.ARMS[:2]}
                    scores.update(predict(net, d, path, ix, cut))
                    rows[role].append(dict(clip=str(n), city=d["city"], score=scores))
            base.immutable(mark, dict(identity=ident, rows=rows, fit=fit_ref, lineage=lineage))
        z = study.read(mark)
        if z["identity"] != ident:
            raise ValueError("Wrong confidence freeze")
        study.checked(z["fit"]["checkpoint"])
        for v in z["lineage"].values():
            study.checked(v["direction"]["checkpoint"])
        done[name] = study.artifact(mark)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, calibration_labels_used=False,
                                          held_labels_used=False, reserved_outcomes_opened=False))


def score(key, parent, previous):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or any(f[k] for k in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require frozen confidence")
    rules, all_held, top = {}, [], defaultdict(list)
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        rows = list(prior.role_rows(z, parent["folds"][name], name, "calibration"))
        rules[name] = {k: timing.old.action.prior.calibration([dict(score=r["scores"][k], strict=r["result"]["eligible"],
                    add_ok=True) for r in rows]) for k in model.ARMS}
    cp = OUT/"calibration.json"
    base.immutable(cp, dict(freeze=study.artifact(p), rules=rules, held_receipts_used=False))
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        rows = list(prior.role_rows(z, parent["folds"][name], name, "held"))
        for k in model.ARMS:
            top[k].append(ranks(rows, k))
        for r in rows:
            r["release"] = {k: rules[name][k]["threshold"] is not None and r["scores"][k] >= rules[name][k]["threshold"] for k in model.ARMS}
        all_held.extend(rows)
    held = {k: prior.held_metrics(all_held, k) for k in model.ARMS}
    result = dict(views=len(all_held), scenarios=len({r["clip"] for r in all_held}), forced_qualifies=sum(r["result"]["eligible"] for r in all_held),
        held=held, calibration_pass={k: sum(r[k]["accepted"] for r in rules.values()) for k in model.ARMS},
        matched_counts={k: dict(**{f"top{m}": sum(r[f"top{m}"] for r in a) for m in (1, 5, 10)},
            fold_mean_ap=float(np.mean([r["ap"] for r in a if r["ap"] is not None]))) for k, a in top.items()},
        freeze=study.artifact(p), calibration=study.artifact(cp), original_2pct=True, result_source="fresh_run",
        calibration_and_held_outcome_source="cached_verified", independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    if (result["views"] != previous["held"]["views"] or result["forced_qualifies"] != previous["held"]["arms"]["frozen_position"]["eligible"]):
        raise ValueError("Fixed projection score changed")
    old = study.read(prior.OUT/"summary.json")
    for k in model.ARMS[:2]:
        if held[k] != old["held"][k]:
            raise ValueError("Frozen comparator did not replay")
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
        key, data, src, roles, inner, outer, control, refs, parent, previous, baseline = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, refs, parent, baseline)
        else:
            score(key, parent, previous)


if __name__ == "__main__":
    main()
