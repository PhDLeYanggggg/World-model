"""Isolate observed context and support factorization in AV2 scene release."""

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
from scripts import run_m3w_av2_paired_qualification as pair
from src.world_model import m3w_av2_observability as model

av, prev, eq, np, torch, study, base, old, prior, cost = (
    pair.av, pair.prev, pair.eq, pair.np, pair.torch, pair.study, pair.base, pair.old, pair.prior, pair.cost)
OUT, LOCAL = pair.OUT.parent/"av2_observability_v2", pair.LOCAL.parent/"av2_observability_v2"
MODES, PRIMARY = (*model.MODES, "fixed_direct", "fixed_query_weighted"), "context_factored"
STEPS, SAVE, BATCH = 320, 40, 64


def beat(phase, **fields):
    path = LOCAL/"heartbeat.pending"
    path.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    path.replace(LOCAL/"heartbeat.json")


def register():
    parent, data, src, roles, inner, outer, control, references, _, _ = pair.register()
    p = pair.OUT/"summary.json"
    report = study.read(p)
    seal = study.read(study.checked(report["freeze"]))
    if (report["registration"] != parent or seal["registration"] != parent or seal["held_labels_used"]
            or report["reserved_outcomes_opened"] or set(seal["folds"]) != set(references["folds"])):
        raise ValueError("Require frozen AV2 candidate scores and checked references")
    r = OUT/"registration.json"
    base.immutable(r, dict(parent=study.artifact(p), parent_freeze=report["freeze"],
        code=[study.artifact(Path(v)) for v in (__file__, model.__file__)],
        primary=PRIMARY, modes=list(MODES), steps=STEPS, checkpoint=SAVE, batch=BATCH, seeds=eq.SEEDS,
        observation=list(av.OBS), forecast=list(av.FUT), columns=list(av.INPUT_COLUMNS),
        context="observed-only neighbor histories including fragments outside the scored roster; no category, actor type, map or stored velocity",
        supervision="donor-only: edited easy support S, exact reference eligibility A, A conditional on S, per-actor future coverage and easy given any label",
        ablation="same initialization, training draws and heads; mask context for motion control; direct and factored readouts share each fit",
        fixed_decision="same paths and direct candidate score as prior run; only rerank scenes",
        calibration="original exact both-arm prefix, no threshold from held labels",
        workers=0, torch_threads=4, original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(r), data, src, roles, inner, outer, control, references, seal


def read_context(data, source):
    out, stats = {}, Counter()
    for n, d in data.items():
        f = study.checked(source["objects"][n]["local"])
        rows = av.pq.read_table(f, columns=list(av.INPUT_COLUMNS), use_threads=False,
            filters=[("timestep", "in", list(av.OBS)), ("observed", "=", True)]).to_pylist()
        motion, ctx, partial = model.past_context(rows, d["ids"], d["h"], av.OBS)
        out[n] = (motion, ctx)
        stats.update(scenarios=1, observed_rows=len(rows), fragment_tracks=partial, scenes_with_fragments=int(partial > 0))
    return out, dict(stats)


def input_for(s, d, context):
    _, x, _ = pair.features(s)
    a = s[1]
    edited = np.flatnonzero(av.motion.peak(a-d["cv"]) > 0.)
    if len(edited) != 2:
        raise ValueError("Availability head requires the unchanged two edits")
    actors = torch.from_numpy(np.column_stack(context))[edited]
    return x, actors, edited


def target_for(d, a, y, v, cut, edited):
    actors, support = model.labels(d["cv"], y, v, edited, cut[0])
    m = prev.metrics(a, d, y, v, cut)
    a_ok = eq.ceiling.eligible_additive(m)
    if support != (m["risk"]["completion_easy_risk_upper"] is not None) or (a_ok and not support):
        raise ValueError("Support factorization did not reproduce original qualification")
    return torch.tensor([float(a_ok), float(support)]), torch.from_numpy(actors)


def examples(inner, data, dy, context, role, ident, refs):
    held, seed, cut = (ident[k] for k in ("held", "seed", "cut"))
    raw = old.donor_forecasts(inner, data, held, role, seed)
    models = {c: pair.inner_action(z, refs["identity"], c) for c, z in refs["references"].items()}
    out = []
    for n, (y, v) in dy.items():
        d = dict(data[n], bank=eq.expanded(data[n], raw[n]))
        action, ac = models[d["city"]]
        if (s := cost.slate(d, seed, cut, action, action_cut=ac)) is None:
            continue
        x, actors, edited = input_for(s, d, context[n])
        event, aux = target_for(d, s[1], y, v, cut, edited)
        out.append((x, actors, event, aux))
    return tuple(torch.stack(v) for v in zip(*out, strict=True))


def fit(ident, name, examples):
    x, a, y, aux = examples
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+2861)
    nets = torch.nn.ModuleDict({k: model.Gate(x.shape[1]) for k in ("motion", "context")})
    nets["context"].load_state_dict(nets["motion"].state_dict())
    opt = torch.optim.AdamW(nets.parameters(), lr=.001)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong resume checkpoint")
        start, trace = ck["step"], ck["trace"]
        nets.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    for step in range(start+1, STEPS+1):
        ix = rng.integers(len(x), size=BATCH)
        losses = [model.gate_loss(*net(x[ix], ablate(a[ix], k)), y[ix], aux[ix]) for k, net in nets.items()]
        opt.zero_grad(set_to_none=True)
        sum(losses).backward()
        torch.nn.utils.clip_grad_norm_(nets.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(v.detach()) for v in losses])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=nets.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return nets.eval(), dict(checkpoint=study.artifact(path), trace=trace, queries=len(x),
        support=int(y[:, 1].sum()), eligible=int(y[:, 0].sum()))


def ablate(actors, kind):
    if kind == "context":
        return actors
    prefix, tail = actors.split((model.MOTION_WIDTH, model.CONTEXT_WIDTH), dim=-1)
    return torch.cat((prefix, torch.zeros_like(tail)), dim=-1)


@torch.inference_mode()
def predict(nets, s, d, ctx):
    x, actors, _ = input_for(s, d, ctx)
    q, aux = {}, {}
    for k, net in nets.items():
        logits, output = net(x[None], ablate(actors[None], k))
        for mode, value in zip(("direct", "factored"), model.scores(logits[0]), strict=True):
            q[f"{k}_{mode}"] = float(value)
        aux[k] = np.column_stack((output[0].sigmoid().numpy(), np.full(2, float(logits[0, 1].sigmoid()))))
    return q, aux


def candidate_model(fold):
    ck = torch.load(study.checked(fold["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != fold["identity"] or ck["step"] != pair.STEPS:
        raise ValueError("Wrong frozen candidate checkpoint")
    net = pair.model.Qualification(ck["net"]["candidate.0.weight"].shape[1]).eval()
    net.load_state_dict(ck["net"])
    return net


def calibrate(nets, candidate, action, forecaster, data, cy, ctx, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {m: [] for m in MODES}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        if (s := cost.slate(d, seed, cut, action)) is None:
            continue
        a, p, confidence, _, _ = pair.rank(candidate, s, d["cv"])
        q, _ = predict(nets, s, d, ctx[n])
        y, v = cy[n]
        _, label, _ = prev.label_terms(d, p["direct"][None], a, y, v, cut)
        ok = eq.ceiling.eligible_additive(prev.metrics(a, d, y, v, cut))
        scores = dict(fixed_direct=confidence["direct"], fixed_query_weighted=confidence["query_weighted"],
            **{k: value*confidence["direct"] for k, value in q.items()})
        for m, score in scores.items():
            rows[m].append(dict(score=score, strict=bool(label[0]), add_ok=ok))
    return {m: old.action.prior.calibration(rows[m]) for m in MODES}


def freeze(key, data, src, roles, inner, outer, control, refs, seal):
    ctx, inventory = read_context(data, src)
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            fold, source = seal["folds"][name], outer["folds"][name]
            cut = fold["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if fold["identity"] != dict(ident, registration=seal["registration"]):
                raise ValueError("Previous and new roles differ")
            path = LOCAL/f"{name}.json"
            if not path.exists():
                nets, fitted = fit(ident, name, examples(inner, data, dy, ctx, role, ident, refs["folds"][name]))
                action, cand = prior.load_action(control["folds"][name]), candidate_model(fold)
                table = calibrate(nets, cand, action, old.load_outer(source), data, cy, ctx, seed, cut)
                for m in ("direct", "query_weighted"):
                    if table[f"fixed_{m}"] != fold["calibration"]["modes"][m]:
                        raise ValueError("The unchanged calibration failed exact replay")
                with (np.load(study.checked(source["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(fold["forecast"]), allow_pickle=False) as fixed,
                      np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], a[k])
                        np.testing.assert_array_equal(raw[k], fixed[k])
                    qs, auxs = [], []
                    for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                        d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                        if (s := cost.slate(d, seed, cut, action)) is None:
                            qs.append(np.zeros(4))
                            auxs.append(np.zeros((2, 2, 3)))
                        else:
                            np.testing.assert_array_equal(s[1], a["additive"][lo:hi])
                            q, aux = predict(nets, s, d, ctx[n])
                            qs.append([q[m] for m in model.MODES])
                            auxs.append(list(aux.values()))
                    forecast = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=raw["names"],
                        offsets=raw["offsets"], registration=key, scores=np.asarray(qs), aux=np.asarray(auxs)))
                base.immutable(path, dict(identity=ident, fit=fitted, forecast=forecast, calibration=table))
            r = study.read(path)
            if r["identity"] != ident:
                raise ValueError("Wrong frozen receipt")
            study.checked(r["fit"]["checkpoint"])
            study.checked(r["forecast"])
            done[name] = r
            beat("freeze", fold=name, completed=len(done), total=len(seal["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, inventory=inventory, held_labels_used=False))


def score(key, data, src, control, parent):
    path = OUT/"freeze.json"
    seal = study.read(path)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(parent["folds"]):
        raise ValueError("Require frozen scores before outcome readout")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    series, rows, city = [], [], {}
    counts = {m: Counter() for m in MODES}
    for name, f in seal["folds"].items():
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        tally = city.setdefault(held, Counter())
        with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(parent["folds"][name]["forecast"]), allow_pickle=False) as p,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], p[k])
                np.testing.assert_array_equal(z[k], a[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Invalid held roster")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                cv, ref, pred = d["cv"], a["additive"][lo:hi], p["direct"][lo:hi]
                gains = dict.fromkeys(MODES, 0.)
                if p["score_direct"][i] > 0.:
                    edited = np.flatnonzero(av.motion.peak(ref-cv) > 0.)
                    event, aux = target_for(d, ref, y, v, cut, edited)
                    label = bool(event[0])
                    q = np.r_[z["scores"][i], 1., p["eligible"][i]]
                    series.append(dict(city=held, clip=str(n), seed=seed, event=event.tolist(),
                        aux_target=aux.tolist(), q=q.tolist(), aux=z["aux"][i].tolist()))
                    tally.update(two_slots=1, support=int(event[1]), eligible=int(label))
                    joint = None
                    for m, qv in zip(MODES, q, strict=True):
                        t = f["calibration"][m]["threshold"]
                        confidence = float(qv*p["score_direct"][i])
                        if m == "fixed_query_weighted":
                            confidence = p["score_query_weighted"][i]
                        if t is None or confidence < t:
                            continue
                        if joint is None:
                            _, strict, _ = prev.label_terms(d, pred[None], ref, y, v, cut)
                            joint = bool(strict[0])
                        m_pred = prev.metrics(pred, d, y, v, cut)
                        counts[m].update(held_released=1, held_both=int(joint and label),
                            risk_fail=int(not m_pred["risk"]["supported"]))
                        tally.update({f"{m}_released": 1, f"{m}_both": int(joint and label)})
                        gain = cost.audit.direct_trajectory(pred, cv, cv, y, v)
                        gains[m] = None if gain is None else gain[2]
                rows.append(dict(city=held, clip=str(n), seed=seed, gain=gains))
        beat("score", fold=name, views=len(rows), total=len(truth)*len(eq.SEEDS))
    cal = {m: {k: sum(f["calibration"][m][k] for f in seal["folds"].values())
               for k in ("tested", "feasible", "strict_with_add", "accepted")} for m in MODES}
    y = np.asarray([s["event"] for s in series])
    q = np.asarray([s["q"] for s in series])
    pred = np.asarray([s["aux"] for s in series])
    target = np.asarray([s["aux_target"] for s in series])
    q_metrics = {m: pair.diagnostics(np.column_stack((y[:, 0], q[:, i]))) for i, m in enumerate(MODES)
                 if m != "fixed_direct"}
    known = target[..., 1].astype(bool)
    aux_metrics = {k: dict(coverage_mae=float(np.abs(pred[:, i, :, 0]-target[..., 0]).mean()),
        easy_given_known=pair.diagnostics(np.column_stack((target[..., 2][known], pred[:, i, :, 1][known]))),
        scene_support=pair.diagnostics(np.column_stack((y[:, 1], pred[:, i, 0, 2]))))
        for i, k in enumerate(("motion", "context"))}
    intervals = {m: av.interval(series, lambda r: (r["q"][model.MODES.index(m)]-r["event"][0])**2
        -(r["q"][MODES.index("fixed_query_weighted")]-r["event"][0])**2) for m in model.MODES}
    out = dict(registration=key, freeze=study.artifact(path), result_source="fresh_run", primary=PRIMARY,
        fitted_models=2*len(seal["folds"]), held_scenarios=len(truth), inventory=seal["inventory"],
        calibration=cal, held_release={k: {f: v[f] for f in ("held_released", "held_both", "risk_fail")}
            for k, v in counts.items()},
        by_city={k: dict(v) for k, v in city.items()}, eligibility=q_metrics, auxiliaries=aux_metrics,
        paired_brier_delta_vs_fixed=intervals,
        released_trajectory_lower_gain_vs_cv={m: av.interval(rows, lambda r: r["gain"][m]) for m in MODES},
        original_2pct=True, candidate_paths_changed=False, deployment_changed=False,
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("inventory", "calibration", "held_release", "eligibility", "auxiliaries")}, indent=2))


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
        key, data, src, roles, inner, outer, control, refs, seal = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, refs, seal)
        else:
            score(key, data, src, control, seal)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
