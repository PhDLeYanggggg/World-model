"""Fit matched reference steering heads on donor cities, then score held development."""

import argparse
from collections import Counter
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
from scripts import run_m3w_av2_incumbent_timing as prior
from src.world_model import m3w_av2_steered_reference as model


np, torch, av, eq, prev, study, base, old = prior.np, prior.torch, prior.av, prior.eq, prior.prev, prior.study, prior.base, prior.old
OUT = prior.OUT.parent/"av2_steered_reference_v1"
LOCAL = prior.LOCAL.parent/OUT.name
STEPS, SAVE, BATCH = 300, 50, 64
ARMS = ("anchor", *model.ARMS)


def beat(phase, **fields):
    d = dict(phase=phase, pid=os.getpid(), epoch=time.time(), **fields)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(d)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(d, flush=True)


def register():
    _, data, source, roles, inner, _, control, refs, _ = prior.register()
    p = OUT.parent/"av2_direction_menu_v1"/"summary.json"
    report = study.read(p)
    f = study.read(study.checked(report["freeze"]))
    if report["reserved_outcomes_opened"] or f["held_labels_used"] or set(f["folds"]) != set(refs["folds"]):
        raise ValueError("Require the exact development reference roster")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)], parent=study.artifact(p),
        arms=ARMS, steps=STEPS, checkpoint_every=SAVE, batch=BATCH, seeds=eq.SEEDS,
        geometry="rotate each of the two exact incumbent edits by a smooth bounded angle; all step magnitudes unchanged",
        max_angle_radians=model.MAX_ANGLE, knots=model.KNOTS,
        training="same two independent recipient encoders, initial zero steering, batch draws and optimizer; "
                 "compare valid-position ADE vs ADE plus both negative conservative gain and all/easy/signed risk excess",
        guarded_weights=dict(ade=1., floors=1., risk=2.),
        label_roles="train only on four donor cities, reconstruct donor incumbents out of their own city; held and calibration cities never fit",
        main="held standalone eligibility on unchanged CV/OLS4 and original risk checks, guarded versus ADE; no release rule fit",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False,
        torch_threads=4, num_workers=0))
    return base.sha(reg), data, source, roles, inner, control, refs, report


def examples(data, truth, role, inner, ref, seed, cut):
    raw = old.donor_forecasts(inner, data, ref["identity"]["held"], role, seed)
    if set(raw) != set(truth):
        raise ValueError("Donor forecasts and outcomes disagree")
    actions = {k: prior.rel.pair.inner_action(v, ref["identity"], k) for k, v in ref["references"].items()}
    rows = []
    for n, (y, v) in truth.items():
        d = data[n]
        action, ac = actions[d["city"]]
        a = prior.incumbent(d, seed, cut, raw[n], action, action_cut=ac)
        if a is None or (s := model.inputs(d, a)) is None:
            continue
        i, *args = s
        rows.append((*args, *model.label(d, i, y, v, cut[0])))
    if not rows:
        raise ValueError("No donor training examples")
    return [torch.as_tensor(np.asarray(x), dtype=torch.bool if i in (6, 7) else torch.float32)
            for i, x in enumerate(zip(*rows, strict=True))]


def fit(ident, name, data):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+39217)
    nets = torch.nn.ModuleDict({k: model.Steer(data[0].shape[-1]) for k in model.ARMS})
    nets["guarded"].load_state_dict(nets["ade"].state_dict())
    opt = torch.optim.AdamW(nets.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    step0, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Invalid steering checkpoint")
        nets.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        step0, trace = ck["step"], ck["loss"]
    for step in range(step0+1, STEPS+1):
        at = rng.integers(len(data[0]), size=BATCH)
        x, *args = (v[at] for v in data)
        loss = [model.objective(n(x), *args, guarded=k == "guarded") for k, n in nets.items()]
        opt.zero_grad(set_to_none=True)
        sum(loss).backward()
        for n in nets.values():
            torch.nn.utils.clip_grad_norm_(n.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(z.detach()) for z in loss])
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=nets.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return nets.eval(), dict(checkpoint=study.artifact(p), donor_queries=len(data[0]), loss=trace)


@torch.inference_mode()
def predict(d, anchor, nets):
    s = model.inputs(d, anchor)
    if s is None:
        return [anchor.copy() for _ in model.ARMS]
    idx, x, c, _, delta, _ = s
    out = []
    for net in nets.values():
        p = anchor.copy()
        p[idx] = c+model.rotate(delta, net(torch.from_numpy(x)).numpy().astype(float), np)
        original = np.linalg.norm(anchor-d["cv"], axis=-1)
        np.testing.assert_allclose(np.linalg.norm(p-d["cv"], axis=-1), original, atol=1e-8, rtol=1e-8)
        np.testing.assert_array_equal(np.any(p != d["cv"], axis=(1, 2)), np.any(anchor != d["cv"], axis=(1, 2)))
        out.append(p)
    return out


def freeze(key, data, source, roles, inner, control, refs):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        if held in donors or cal in donors or held == cal:
            raise ValueError("Invalid donor source role")
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            cf, ref = control["folds"][name], refs["folds"][name]
            cut = ref["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            for r in (cf, ref):
                if any(r["identity"][k] != ident[k] for k in ("held", "calibration", "donors", "seed", "cut")):
                    raise ValueError("Reference fold differs")
            mark = LOCAL/f"{name}.json"
            if not mark.exists():
                nets, fit_ref = fit(ident, name, examples(data, dy, role, inner, ref, seed, cut))
                with np.load(study.checked(cf["forecast"]), allow_pickle=False) as a:
                    rows = []
                    for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                        d = data[n]
                        if d["city"] != held:
                            raise ValueError("Wrong held source")
                        rows.append(np.stack(predict(d, a["additive"][lo:hi], nets), axis=1))
                    forecast = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=a["names"], offsets=a["offsets"],
                        registration=key, paths=np.concatenate(rows)))
                base.immutable(mark, dict(identity=ident, fit=fit_ref, forecast=forecast))
            r = study.read(mark)
            if r["identity"] != ident:
                raise ValueError("Invalid frozen fold")
            study.checked(r["forecast"])
            study.checked(r["fit"]["checkpoint"])
            done[name] = r
            beat("freeze", completed=len(done), total=len(refs["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False,
                                         calibration_labels_used=False, reserved_outcomes_opened=False))


def fold_score(key, data, truth, f, cf):
    held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
    rows = []
    with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
          np.load(study.checked(cf["forecast"]), allow_pickle=False) as a):
        for k in ("names", "offsets"):
            np.testing.assert_array_equal(z[k], a[k])
        if z["registration"].item() != key:
            raise ValueError("Wrong steering forecast")
        for n, lo, hi in zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True):
            d, (y, v), anchor = data[n], truth[n], a["additive"][lo:hi]
            if d["city"] != held:
                raise ValueError("Wrong held roster")
            if len(eq.radii(d, seed)) != 2:
                continue
            row = dict(clip=str(n), city=held, seed=seed, complete=bool(v.all()), arms={})
            for k, p in zip(ARMS, [anchor, *z["paths"][lo:hi].swapaxes(0, 1)], strict=True):
                np.testing.assert_allclose(np.linalg.norm(p-d["cv"], axis=-1), np.linalg.norm(anchor-d["cv"], axis=-1), atol=1e-8, rtol=1e-8)
                m = prev.metrics(p, d, y, v, cut)
                risk = m["risk"]
                lower = prior.cost.audit.direct_trajectory(p, anchor, d["cv"], y, v)
                pair = av.frontier.pair_reduction(p[None], anchor, y, v)[0, 1]
                row["arms"][k] = dict(eligible=eq.ceiling.eligible_additive(m), risk=risk["supported"],
                    easy_defined=risk["completion_easy_risk_upper"] is not None,
                    signed_easy_over=(risk["known_easy_net_degradation"] or 0.) > .02,
                    ade=m["ade"], fde=m["fde"], tail=m["tail"], gain=m["gain"],
                    lower_vs_anchor=lower[2] if lower is not None else None, pair_vs_anchor=float(pair))
            rows.append(row)
    return rows


def summarize(rows, report):
    complete = [r for r in rows if r["complete"]]
    out = {}
    for k in ARMS:
        v = [r["arms"][k] for r in rows]
        out[k] = dict(**{n: sum(r[n] for r in v) for n in ("eligible", "risk", "easy_defined", "signed_easy_over")},
            complete_eligible=sum(r["arms"][k]["eligible"] for r in complete), worst_p95=max(r["tail"] for r in v if r["tail"] is not None),
            **{n: av.interval(rows, lambda r: r["arms"][k][n]) for n in ("ade", "fde", "gain", "lower_vs_anchor", "pair_vs_anchor")})
    if (len(rows), len(complete), out["anchor"]["eligible"], out["anchor"]["complete_eligible"]) != (
            report["views"], report["complete_views"], report["counts"]["anchor_eligible"], report["complete_counts"]["anchor_eligible"]):
        raise ValueError("Previous standalone eligibility did not reproduce")
    paired = {k+"_minus_"+a: av.interval(rows, lambda r: int(r["arms"][k]["eligible"])-int(r["arms"][a]["eligible"]))
              for k, a in (("ade", "anchor"), ("guarded", "anchor"), ("guarded", "ade"))}
    transitions = {k: dict(Counter(f"{int(r['arms']['anchor']['eligible'])}_to_{int(r['arms'][k]['eligible'])}" for r in rows))
                   for k in model.ARMS}
    by_city = {s: {k: sum(r["arms"][k]["eligible"] for r in rows if r["city"] == s) for k in ARMS}
               for s in sorted({r["city"] for r in rows})}
    return dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=len(complete),
                arms=out, paired_eligibility=paired, transitions=transitions, by_city=by_city)


def score(key, data, source, control, report, replay):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["held_labels_used"] or f["reserved_outcomes_opened"]:
        raise ValueError("Require sealed forecast choices")
    truth, rows = None, []
    for name, r in f["folds"].items():
        mark = LOCAL/f"{name}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay cannot read labels")
            if truth is None:
                truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
            z = fold_score(key, data, truth, r, control["folds"][name])
            base.immutable(mark, dict(freeze=r, rows=z))
        z = study.read(mark)
        if z["freeze"] != r:
            raise ValueError("Wrong score receipt")
        rows.extend(z["rows"])
    result = summarize(rows, report)
    base.immutable(OUT/"summary.json", dict(**result, freeze=study.artifact(p), result_source="fresh_run",
        fitted_heads=2*len(f["folds"]), original_2pct=True, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False, release_tested=False))
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
        key, data, source, roles, inner, control, refs, report = register()
        if phase == "fit":
            freeze(key, data, source, roles, inner, control, refs)
        else:
            score(key, data, source, control, report, replay=phase == "replay")


if __name__ == "__main__":
    main()
