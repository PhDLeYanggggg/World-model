"""Learn a causal trajectory reference without changing who is edited or by how much."""

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
from scripts import run_m3w_av2_causal_reference as prior
from scripts.run_m3w_av2_assignment_constraints import trajectory
from src.world_model import m3w_av2_projected_reference as model

np, torch, study, base, prev, eq, av = prior.np, prior.torch, prior.study, prior.base, prior.prev, prior.eq, prior.av
timing, old, bank = prior.timing, prior.old, prior.bank
OUT, LOCAL = prior.OUT.parent/"av2_projected_reference_v1", prior.LOCAL.parent/"av2_projected_reference_v1"
ARMS = model.ARMS
STEPS, SAVE, BATCH = 400, 50, 32


def beat(phase, **kw):
    z = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **kw)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(z)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(z, flush=True)


def register():
    key, data, src, roles, outer, control, _ = prior.register()
    path = prior.OUT/"summary.json"
    s = study.read(path)
    f = study.read(study.checked(s["freeze"]))
    if f["registration"] != key or not s["calibration_only"] or any(f[k] for k in ("labels_read", "held_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require sealed calibration reference")
    _, _, _, _, inner, _, _, refs, _, _ = old.register()
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(path), arms=ARMS, steps=STEPS, save_every=SAVE, batch=BATCH, seeds=eq.SEEDS,
        scope="donor training, calibration-only scoring, no held or reserved labels",
        training="same initialization and minibatches; whole donor queries, sampled legal actor assignments; "
                 "raw position objective vs raw + 0.5 projected ADE + 0.5 floor shortfall + 2.0 original risk excess",
        budget="frozen soft recipients and radial magnitude at every step; no added edits; full directional projection",
        primary="standalone qualification, conservative trajectory and easy harm vs frozen soft; CV and OLS4 unedited",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, threads=4, workers=0))
    return base.sha(reg), data, src, roles, inner, outer, control, refs, f, s


def examples(data, truth, role, inner, ref, seed, cut):
    raw = timing.old.donor_forecasts(inner, data, ref["identity"]["held"], role, seed)
    if set(raw) != set(truth):
        raise ValueError("Wrong nested donor forecasts")
    actions = {k: timing.rel.pair.inner_action(v, ref["identity"], k) for k, v in ref["references"].items()}
    groups = defaultdict(list)
    for n, (y, v) in truth.items():
        d = data[n]
        net, ac = actions[d["city"]]
        a = timing.incumbent(d, seed, cut, raw[n], net, action_cut=ac)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, _, _ = bank.candidate(d, a)
        all_rows = np.arange(len(p))
        inputs = [model.inputs(d, p[:, j], all_rows) for j in range(2)]
        x = tuple(np.stack((left[opts[:, 0]], right[opts[:, 1]]), axis=1)
                  for left, right in zip(*inputs, strict=True))
        labels = tuple(np.stack(z) for z in zip(*(model.target(d, ix, y, v, cut[0]) for ix in opts), strict=True))
        groups[d["city"]].append(tuple(z.astype(bool if z.dtype == bool else np.float32) for z in (*x, *labels)))
    if set(groups) != set(role["donors"]):
        raise ValueError("Need all four training cities")
    return dict(sorted(groups.items()))


def fit(ident, name, groups):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+30161)
    width = next(iter(groups.values()))[0][0].shape[-1]
    a = model.head(width)
    nets = (a, copy.deepcopy(a))
    opt = [torch.optim.AdamW(n.parameters(), lr=.001) for n in nets]
    p = LOCAL/f"{name}.pt"
    step0, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong fit identity")
        for n, o, w, r in zip(nets, opt, ck["nets"], ck["optimizers"], strict=True):
            n.load_state_dict(w)
            o.load_state_dict(r)
        rng.bit_generator.state = ck["rng"]
        step0, trace = ck["step"], ck["loss"]
    for step in range(step0+1, STEPS+1):
        q = [v[i] for v in groups.values() for i in rng.integers(len(v), size=BATCH//len(groups))]
        picks = [tuple(z[k] for z in row) for row, k in ((r, rng.integers(len(r[0]))) for r in q)]
        x, c, o, delta, frame, scale, y, v, easy, context = (torch.from_numpy(np.stack(z)) for z in zip(*picks, strict=True))
        trace_row = []
        for j, (n, optimizer) in enumerate(zip(nets, opt, strict=True)):
            r = model.residual(n, x, frame, scale)
            raw, proj, floors, risk = model.objectives(r, c, o, delta, scale, y, v, easy, context)
            loss = raw if j == 0 else raw+.5*proj+.5*floors+2.*risk
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            trace_row.append([float(a.detach()) for a in (raw, proj, floors, risk)])
        if step % SAVE == 0:
            trace.append(trace_row)
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, nets=[n.state_dict() for n in nets], optimizers=[o.state_dict() for o in opt],
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return tuple(n.eval() for n in nets), dict(checkpoint=study.artifact(p), width=width, loss=trace,
        donor_queries=sum(map(len, groups.values())), donor_options=sum(len(z[0]) for g in groups.values() for z in g))


@torch.inference_mode()
def predict(nets, d, soft, rows):
    x, c, o, delta, frame, scale = model.inputs(d, soft, rows)
    project = lambda z: c+model.project(torch.from_numpy(z), torch.from_numpy(delta)).numpy()
    ts = [torch.from_numpy(a.astype(np.float32)) for a in (x, frame, scale)]
    choices = [project(o-c), *(project(model.residual(n, *ts).numpy().astype(float)) for n in nets)]
    r = np.linalg.norm(delta, axis=-1)
    for p in choices:
        np.testing.assert_allclose(np.linalg.norm(p-c, axis=-1), r, atol=1e-8, rtol=1e-8)
        if not np.any(p != c, axis=(1, 2)).all():
            raise ValueError("Projection lost an edit")
    return np.stack(choices)


def freeze(key, data, src, roles, inner, outer, control, refs, parent):
    done = {}
    for name, ref in parent["folds"].items():
        f = study.read(study.checked(ref))
        held, city, donors, seed, cut = (f["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut"))
        role = roles[held]
        if held in donors or city in donors or city == held or donors != role["donors"] or city != role["calibration"]:
            raise ValueError("Overlapping source roles")
        ident = dict(f["identity"], registration=key, calibration_parent=ref)
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            nets, receipt = fit(ident, name, examples(data, dy, role, inner, refs["folds"][name], seed, cut))
            rows = []
            for (n, d, p, opts, _, digest), q in zip(prior.view_paths(data, f["names"], control["folds"][name], outer["folds"][name], seed, cut),
                                                    f["rows"], strict=True):
                if n != q["clip"] or digest != q["hash"] or d["city"] != city:
                    raise ValueError("Wrong frozen calibration bank")
                edited = opts[q["choices"][1]]
                soft = bank.model.render(d["cv"], p, edited)
                z = predict(nets, d, soft, edited)
                rows.append(dict(clip=str(n), hash=digest, edited=edited.tolist(), paths=z.tolist()))
            base.immutable(mark, dict(identity=ident, rows=rows, fit=receipt))
        z = study.read(mark)
        if z["identity"] != ident or len(z["rows"]) != len(f["rows"]):
            raise ValueError("Wrong frozen receipt")
        study.checked(z["fit"]["checkpoint"])
        done[name] = study.artifact(mark)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, calibration_labels_used=False,
        held_labels_used=False, reserved_outcomes_opened=False))


def fold_score(data, truth, f, cf, of):
    city, seed, cut = (f["identity"][k] for k in ("calibration", "seed", "cut"))
    parent = study.read(study.checked(f["identity"]["calibration_parent"]))
    rows = []
    for (n, d, p, opts, _, digest), q, z in zip(prior.view_paths(data, parent["names"], cf, of, seed, cut),
                                               parent["rows"], f["rows"], strict=True):
        ix = opts[q["choices"][1]]
        if n != q["clip"] or n != z["clip"] or digest != z["hash"] or ix.tolist() != z["edited"] or d["city"] != city:
            raise ValueError("Wrong frozen proposal")
        y, v = truth[n]
        cv, ols = d["cv"], d["ols4"]
        soft = bank.model.render(cv, p, ix)
        new = np.broadcast_to(cv, (3, *cv.shape)).copy()
        new[:, ix] = z["paths"]
        np.testing.assert_allclose(np.linalg.norm(new-cv, axis=-1), np.broadcast_to(np.linalg.norm(soft-cv, axis=-1), new.shape[:-1]), atol=1e-8)
        row = dict(clip=str(n), city=city, seed=seed, complete=bool(v.all()), arms={})
        for k, path in zip((*ARMS, "cv", "ols4"), (soft, *new, cv, ols), strict=True):
            m = prev.metrics(path, d, y, v, cut)
            risk = m["risk"]
            row["arms"][k] = dict(eligible=eq.ceiling.eligible_additive(m), risk=risk["supported"],
                easy_defined=risk["completion_easy_risk_upper"] is not None,
                easy_over=(risk["known_easy_net_degradation"] or 0.) > .02,
                ade=m["ade"], fde=m["fde"], tail=m["tail"], hard=m["hard"], gain=m["gain"],
                vs_soft=trajectory(path, soft, d, y, v),
                pair_vs_soft=float(av.frontier.pair_reduction(path[None], soft, y, v)[0, 1]))
        rows.append(row)
    return rows


def summarize(rows, parent):
    arms = {}
    for k in (*ARMS, "cv", "ols4"):
        a = [r["arms"][k] for r in rows]
        arms[k] = dict(**{n: sum(z[n] for z in a) for n in ("eligible", "risk", "easy_defined", "easy_over")},
            complete_eligible=sum(r["complete"] and r["arms"][k]["eligible"] for r in rows),
            worst_p95=max(z["tail"] for z in a if z["tail"] is not None),
            **{n: av.interval(rows, lambda r: r["arms"][k][n]) for n in ("ade", "fde", "hard", "gain", "vs_soft", "pair_vs_soft")},
            eligible_vs_soft=av.interval(rows, lambda r: int(r["arms"][k]["eligible"])-int(r["arms"]["soft"]["eligible"])),
            both_eligible_pair_win=sum(r["arms"]["soft"]["eligible"] and z["eligible"] and z["vs_soft"] is not None
                                      and z["vs_soft"] >= -1e-8 and z["pair_vs_soft"] > 1e-10 for r, z in zip(rows, a, strict=True)))
    if (len(rows) != parent["views"] or any(arms["soft"][n] != parent["arms"]["soft"][n] for n in ("eligible", "easy_over"))
            or arms["soft"]["ade"] != parent["arms"]["soft"]["ade"]):
        raise ValueError("Original calibration reference did not replay")
    return dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows), arms=arms,
        by_city={city: {k: sum(r["arms"][k]["eligible"] for r in rows if r["city"] == city) for k in ARMS}
                 for city in sorted({r["city"] for r in rows})})


def score(key, data, src, outer, control, parent, replay):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or any(seal[k] for k in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require frozen calibration forecasts")
    rows = []
    for name, ref in seal["folds"].items():
        f = study.read(study.checked(ref))
        mark = LOCAL/f"{name}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay may not open outcomes")
            ident = f["identity"]
            names = [q["clip"] for q in f["rows"]]
            if ident["calibration"] in [ident["held"], *ident["donors"]] or any(data[n]["city"] != ident["calibration"] for n in names):
                raise ValueError("Calibration-only outcome access")
            truth = prev.load_targets(src, data, names)
            base.immutable(mark, dict(freeze=ref, rows=fold_score(data, truth, f, control["folds"][name], outer["folds"][name])))
        z = study.read(mark)
        if z["freeze"] != ref:
            raise ValueError("Wrong score receipt")
        rows.extend(z["rows"])
    s = summarize(rows, parent)
    base.immutable(OUT/"summary.json", dict(**s, freeze=study.artifact(p), result_source="fresh_run",
        calibration_only=True, original_2pct=True, held_labels_opened=False, reserved_outcomes_opened=False,
        independent_confirmation=False, deployment_changed=False))
    print(study.json.dumps(s, indent=2))


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
