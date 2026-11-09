"""Matched AV2 direction supervision with prospectively frozen calibration and held paths."""

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
from scripts import run_m3w_av2_projected_reference as prior
from scripts.run_m3w_av2_assignment_constraints import trajectory, views as held_views
from src.world_model import m3w_av2_relative_projection as model

np, torch, study, base, prev, eq, av = prior.np, prior.torch, prior.study, prior.base, prior.prev, prior.eq, prior.av
timing, bank, proj = prior.timing, prior.bank, prior.model
OUT, LOCAL = prior.OUT.parent/"av2_relative_projection_v1", prior.LOCAL.parent/"av2_relative_projection_v1"
STEPS, SAVE, BATCH = 200, 50, 32
ARMS = ("soft", "frozen_position", "individual", "relative")


def beat(phase, **kw):
    r = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **kw)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(r)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(r, flush=True)


def register():
    k, data, src, roles, inner, outer, control, refs, _, _ = prior.register()
    path = prior.OUT/"summary.json"
    s = study.read(path)
    parent = study.read(study.checked(s["freeze"]))
    if parent["registration"] != k or not s["calibration_only"] or s["held_labels_opened"] or s["reserved_outcomes_opened"]:
        raise ValueError("Require sealed donor-only position model")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)], parent=study.artifact(path),
        arms=ARMS, steps=STEPS, save_every=SAVE, batch=BATCH, seeds=eq.SEEDS, max_angle=model.MAX_ANGLE,
        pair_weight=model.PAIR_WEIGHT, donors="same city-excluded forecast radii and donor labels; raw backbone frozen",
        loss="matched residual heads; only relative head adds known pair error and missing-step disagreement on every affected edge",
        budget="same two recipients and per-step radii, same initial projected position path, three bounded angle knots",
        calibration="original strict nonempty both-arm prefix; confidence from known history prefixes; no change means CV outside release",
        primary="relative vs individual and frozen position on paired held development trajectories; strict calibrated release separate",
        independent_confirmation=False, reserved_outcomes_opened=False, original_2pct=True, threads=4, workers=0))
    return base.sha(reg), data, src, roles, inner, outer, control, refs, parent, s


def frozen_model(parent):
    ck = torch.load(study.checked(parent["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != parent["identity"] or ck["step"] != prior.STEPS:
        raise ValueError("Wrong frozen position checkpoint")
    net = proj.head(parent["fit"]["width"])
    net.load_state_dict(ck["nets"][0])
    return net.eval()


@torch.inference_mode()
def base_positions(net, d, path, ix):
    x, c, _, delta, frame, scale = proj.inputs(d, path, ix)
    ts = [torch.from_numpy(a.astype(np.float32)) for a in (x, frame, scale)]
    r = proj.residual(net, *ts).numpy().astype(float)
    return c+proj.project(torch.from_numpy(r), torch.from_numpy(delta)).numpy()


def examples(net, data, truth, role, inner, ref, seed, cut):
    raw = timing.old.donor_forecasts(inner, data, ref["identity"]["held"], role, seed)
    if set(raw) != set(truth):
        raise ValueError("Wrong donor forecasts")
    actions = {k: timing.rel.pair.inner_action(v, ref["identity"], k) for k, v in ref["references"].items()}
    groups = defaultdict(list)
    for n, (y, v) in truth.items():
        d = data[n]
        action, ac = actions[d["city"]]
        a = timing.incumbent(d, seed, cut, raw[n], action, action_cut=ac)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, _, _ = bank.candidate(d, a)
        pp = np.stack([base_positions(net, d, p[:, j], np.arange(len(p))) for j in range(2)], axis=1)
        rows = []
        for ix in opts:
            path = bank.model.render(d["cv"], pp, ix)
            z = (*model.inputs(d, path, ix), *model.labels(d, path, ix, y, v))
            rows.append(tuple(np.asarray(a, dtype=bool if np.asarray(a).dtype == bool else np.float32) for a in z))
        groups[d["city"]].append(rows)
    if set(groups) != set(role["donors"]):
        raise ValueError("Require all four donor cities")
    return dict(sorted(groups.items()))


def batch(groups, rng):
    queries = [v[i] for v in groups.values() for i in rng.integers(len(v), size=BATCH//len(groups))]
    parts = list(zip(*(r[rng.integers(len(r))] for r in queries), strict=True))
    for k in range(6, 10):
        m = max(len(a) for a in parts[k])
        parts[k] = [np.pad(a, ((0, m-len(a)), *((0, 0),)*(a.ndim-1))) for a in parts[k]]
    return tuple(torch.from_numpy(np.stack(a)) for a in parts)


def fit(ident, name, groups):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+227)
    width = next(iter(groups.values()))[0][0][0].shape[-1]
    pnet = model.Refine(width)
    nets = (pnet, copy.deepcopy(pnet))
    opts = [torch.optim.AdamW(n.parameters(), lr=.001) for n in nets]
    p = LOCAL/f"{name}.pt"
    step0, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong resume checkpoint")
        for n, o, s, t in zip(nets, opts, ck["nets"], ck["optimizers"], strict=True):
            n.load_state_dict(s)
            o.load_state_dict(t)
        rng.bit_generator.state = ck["rng"]
        step0, trace = ck["step"], ck["loss"]
    for step in range(step0+1, STEPS+1):
        x, *args = batch(groups, rng)
        row = []
        for n, o, w in zip(nets, opts, (0., model.PAIR_WEIGHT), strict=True):
            angle = n(x)
            a, b = model.losses(angle, *args)
            loss = a+w*b+.02*angle.square().mean()
            o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n.parameters(), 5., error_if_nonfinite=True)
            o.step()
            row.append([float(a.detach()), float(b.detach())])
        if step % SAVE == 0:
            trace.append(row)
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, nets=[n.state_dict() for n in nets], optimizers=[o.state_dict() for o in opts],
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return tuple(n.eval() for n in nets), dict(checkpoint=study.artifact(p), width=width, loss=trace,
        donor_queries=sum(map(len, groups.values())))


def inputs(data, parent, cf, of, role):
    city, seed, cut = (parent["identity"][k] for k in ("calibration" if role == "calibration" else "held", "seed", "cut"))
    cal = study.read(study.checked(parent["identity"]["calibration_parent"]))
    if role == "calibration":
        for (n, d, p, opts, _, digest), q, r in zip(prior.prior.view_paths(data, cal["names"], cf, of, seed, cut),
                                                   cal["rows"], parent["rows"], strict=True):
            ix = opts[q["choices"][1]]
            if n != r["clip"] or digest != r["hash"] or ix.tolist() != r["edited"] or d["city"] != city:
                raise ValueError("Wrong calibration candidate")
            yield n, d, bank.model.render(d["cv"], p, ix), ix, digest, np.asarray(r["paths"][1])
    else:
        soft = study.read(study.checked(cal["identity"]["parent"]))
        for n, d, p, opts, q in held_views(data, cf, city, seed, soft):
            ix = opts[q["choice"][0]]
            yield n, d, bank.model.render(d["cv"], p, ix), ix, q["hash"], None


@torch.inference_mode()
def predict(nets, backbone, d, soft, ix, expected):
    c = d["cv"]
    raw = c.copy()
    raw[ix] = base_positions(backbone, d, soft, ix)
    if expected is not None:
        np.testing.assert_array_equal(raw[ix], expected)
    x, cv, delta, _ = model.inputs(d, raw, ix)
    paths = [raw]
    for n in nets:
        a = c.copy()
        a[ix] = cv+model.rotate(delta, n(torch.from_numpy(x)).numpy().astype(float), np)
        paths.append(a)
    for p in paths:
        np.testing.assert_allclose(np.linalg.norm(p-c, axis=-1), np.linalg.norm(soft-c, axis=-1), rtol=1e-8, atol=1e-8)
    return [p[ix].tolist() for p in paths], [model.confidence(d, raw, p) for p in paths[1:]]


def freeze(key, data, src, roles, inner, outer, control, refs, parent):
    done = {}
    for name, ref in parent["folds"].items():
        f = study.read(study.checked(ref))
        held, cal, donors, seed, cut = (f["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut"))
        role = roles[held]
        if held in donors or cal in donors or held == cal or donors != role["donors"] or cal != role["calibration"]:
            raise ValueError("Overlapping city roles")
        ident = dict(f["identity"], registration=key, parent=ref)
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            backbone = frozen_model(f)
            nets, receipt = fit(ident, name, examples(backbone, data, dy, role, inner, refs["folds"][name], seed, cut))
            rows = {}
            for role in ("calibration", "held"):
                rows[role] = []
                for n, d, soft, ix, digest, expected in inputs(data, f, control["folds"][name], outer["folds"][name], role):
                    p, conf = predict(nets, backbone, d, soft, ix, expected)
                    rows[role].append(dict(clip=str(n), hash=digest, edited=ix.tolist(), paths=p, confidence=conf))
            base.immutable(mark, dict(identity=ident, rows=rows, fit=receipt))
        z = study.read(mark)
        if z["identity"] != ident:
            raise ValueError("Wrong fold receipt")
        study.checked(z["fit"]["checkpoint"])
        done[name] = study.artifact(mark)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, calibration_labels_used=False,
        held_labels_used=False, reserved_outcomes_opened=False))


def fold_score(data, src, f, cf, of, role):
    parent = study.read(study.checked(f["identity"]["parent"]))
    city = f["identity"][role]
    if city in f["identity"]["donors"] or f["identity"]["held"] == f["identity"]["calibration"]:
        raise ValueError("Wrong outcome role")
    names = [r["clip"] for r in f["rows"][role]]
    if any(data[n]["city"] != city for n in names):
        raise ValueError("Wrong outcome roster")
    truth = prev.load_targets(src, data, names)
    rows = []
    for (n, d, soft, ix, digest, _), r in zip(inputs(data, parent, cf, of, role), f["rows"][role], strict=True):
        if n != r["clip"] or digest != r["hash"] or ix.tolist() != r["edited"]:
            raise ValueError("Wrong frozen path")
        y, v = truth[n]
        cv = d["cv"]
        paths = np.broadcast_to(cv, (3, *cv.shape)).copy()
        paths[:, ix] = r["paths"]
        np.testing.assert_allclose(np.linalg.norm(paths-cv, axis=-1), np.broadcast_to(np.linalg.norm(soft-cv, axis=-1), paths.shape[:-1]), atol=1e-8)
        row = dict(clip=str(n), city=city, seed=f["identity"]["seed"], complete=bool(v.all()), arms={})
        for k, p in zip(ARMS, (soft, *paths), strict=True):
            m = prev.metrics(p, d, y, v, f["identity"]["cut"])
            risk = m["risk"]
            row["arms"][k] = dict(eligible=eq.ceiling.eligible_additive(m), risk=risk["supported"],
                easy_over=(risk["known_easy_net_degradation"] or 0.) > .02,
                ade=m["ade"], fde=m["fde"], tail=m["tail"], hard=m["hard"], gain=m["gain"],
                vs_position=trajectory(p, paths[0], d, y, v), vs_individual=trajectory(p, paths[1], d, y, v),
                pair_vs_position=float(av.frontier.pair_reduction(p[None], paths[0], y, v)[0, 1]),
                pair_vs_individual=float(av.frontier.pair_reduction(p[None], paths[1], y, v)[0, 1]))
        for k, conf in zip(ARMS[2:], r["confidence"], strict=True):
            a = row["arms"][k]
            a.update(confidence=conf, strict=bool(row["arms"]["frozen_position"]["eligible"] and a["eligible"]
                     and a["vs_position"] is not None and a["vs_position"] >= -1e-8 and a["pair_vs_position"] > 1e-10))
        rows.append(row)
    return rows


def summarize(rows):
    return {k: dict(eligible=sum(r["arms"][k]["eligible"] for r in rows), risk=sum(r["arms"][k]["risk"] for r in rows),
        easy_over=sum(r["arms"][k]["easy_over"] for r in rows), strict=sum(r["arms"][k].get("strict", False) for r in rows),
        worst_p95=max(r["arms"][k]["tail"] for r in rows if r["arms"][k]["tail"] is not None),
        **{m: av.interval(rows, lambda r: r["arms"][k][m]) for m in ("ade", "fde", "hard", "gain", "vs_position",
            "vs_individual", "pair_vs_position", "pair_vs_individual")},
        eligible_vs_individual=av.interval(rows, lambda r: int(r["arms"][k]["eligible"])-int(r["arms"]["individual"]["eligible"])))
        for k in ARMS}


def score(key, data, src, outer, control, parent, replay):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or any(seal[k] for k in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require all forecasts frozen before scoring")
    scores, rules = {}, {}
    for role in ("calibration", "held"):
        rows = []
        for name, ref in seal["folds"].items():
            f = study.read(study.checked(ref))
            mark = LOCAL/f"{name}.{role}.score.json"
            if not mark.exists():
                if replay:
                    raise ValueError("Replay may not read outcomes")
                r = fold_score(data, src, f, control["folds"][name], outer["folds"][name], role)
                base.immutable(mark, dict(freeze=ref, role=role, rows=r))
            z = study.read(mark)
            if z["freeze"] != ref or z["role"] != role:
                raise ValueError("Wrong scoring receipt")
            if role == "calibration":
                rules[name] = {k: timing.old.action.prior.calibration([
                    dict(score=r["arms"][k]["confidence"], strict=r["arms"][k]["strict"], add_ok=True)
                    for r in z["rows"]]) for k in ARMS[2:]}
            else:
                for r in z["rows"]:
                    for k in ARMS[2:]:
                        t = rules[name][k]["threshold"]
                        r["arms"][k]["release"] = t is not None and r["arms"][k]["confidence"] >= t
            rows.extend(z["rows"])
        if role == "calibration":
            base.immutable(OUT/"calibration.json", dict(freeze=study.artifact(p), rules=rules, held_labels_used=False))
        scores[role] = dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows),
            arms=summarize(rows), by_city={c: {k: sum(r["arms"][k]["eligible"] for r in rows if r["city"] == c) for k in ARMS}
                for c in sorted({r["city"] for r in rows})})
        if role == "held":
            scores[role]["releases"] = {k: dict(calibration_pass=sum(f[k]["accepted"] for f in rules.values()),
                selected=sum(r["arms"][k]["release"] for r in rows),
                strict=sum(r["arms"][k]["release"] and r["arms"][k]["strict"] for r in rows)) for k in ARMS[2:]}
    cal = scores["calibration"]
    if cal["views"] != parent["views"] or cal["arms"]["frozen_position"]["eligible"] != parent["arms"]["raw_projection"]["eligible"]:
        raise ValueError("Frozen projected calibration reference changed")
    base.immutable(OUT/"summary.json", dict(**scores, freeze=study.artifact(p), calibration=study.artifact(OUT/"calibration.json"),
        result_source="fresh_run", independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False))
    print(study.json.dumps(scores, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, inner, outer, control, refs, parent, summary = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, refs, parent)
        else:
            score(key, data, src, outer, control, summary, replay=phase == "replay")


if __name__ == "__main__":
    main()
