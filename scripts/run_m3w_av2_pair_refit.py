"""Fit and assess exact-budget pair selection on dense AV2 development tracks."""

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
from scripts import run_m3w_av2_approach_replay as replay
from scripts import run_m3w_opendd_atomic_pair_admission as pair
from src.world_model import m3w_slate_relative_gain as relative

av, np, study, base, torch = replay.parent, replay.np, replay.study, replay.base, replay.torch
OUT = av.OUT.parent/"av2_pair_refit_v1"
LOCAL = av.LOCAL.parent/OUT.name
SEEDS, STEPS, SAVE, BATCH = av.slate.SEEDS, 320, 40, 128
ARMS = ("cv", "ols4", "additive", "frozen", "refit", "released")


def beat(phase, **fields):
    tmp = LOCAL/"heartbeat.pending"
    tmp.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    tmp.replace(LOCAL/"heartbeat.json")


def roles(cities, held):
    if len(set(cities)) != 6 or held not in cities:
        raise ValueError("Expected the six registered AV2 development cities")
    order = sorted(cities)
    cal = order[(order.index(held)+1) % len(order)]
    return cal, [c for c in order if c not in (held, cal)]


def register():
    key, old, source, _, _ = replay.register()
    r = replay.OUT/"replay_readout.json"
    prev = study.read(r)
    f = study.read(study.checked(prev["freeze"]))
    if (prev["registration"] != key or f["registration"] != key or f["outcomes_used"]
            or prev["reserved_outcomes_opened"] or not f["all_parent_predictions_reproduced"]
            or set(old["files"]) != set(f["banks"]) or set(f["banks"]) != set(source["objects"])):
        raise ValueError("Expected sealed AV2 publisher-train development forecasts")
    p = OUT/"registration.json"
    base.immutable(p, dict(parent=study.artifact(r), parent_freeze=prev["freeze"],
        code=[study.artifact(Path(x)) for x in (__file__, relative.__file__, pair.audit.__file__, replay.__file__)],
        observation=av.OBS, future=av.FUT, seeds=SEEDS, steps=STEPS, checkpoint_every=SAVE, batch=BATCH,
        split="six development city folds; next sorted city calibrates, other four alone fit and define easy/hard",
        reference="cached verified expanded additive forecast and original two peaks; frozen singleton predictions",
        target="full-grid pair lower gain, exact original risk and support, dual causal accuracy and additive noninferiority",
        score="predicted strict probability times positive predicted conservative pair gain; otherwise use frozen forecast",
        calibration="all nonempty score breakpoints; both released paths must pass original risk and dual reference checks, refit must improve over additive",
        source_roles="AV2 publisher train only; frozen OpenDD expanded forecast generators",
        cpu_threads=4, num_workers=0, original_2pct=True, independent_confirmation=False,
        reserved_outcomes_opened=False))
    return base.sha(p), old, source, f


def read_inputs(old, seal):
    out = {}
    for name in sorted(old["files"]):
        with (np.load(study.checked(old["files"][name]), allow_pickle=False) as original,
              np.load(study.checked(seal["banks"][name]), allow_pickle=False) as expanded):
            d = {k: original[k] for k in ("city", "h", "ids", "cv", "ols4", "turn",
                                         *(f"anchor_{s}" for s in SEEDS))}
            d["city"] = d["city"].item()
            d.update({k: expanded[k] for k in expanded.files})
            _, own, _, _ = replay.menu.model.candidates(d["h"], len(av.FUT))
            d["bank"] = np.concatenate((d["ols4"][:, None], d["turn"][:, None], own), axis=1)
            out[name] = d
    return out


def load_targets(source, data, names):
    return {n: av.targets(study.checked(source["objects"][n]["local"]), data[n]["ids"], data[n]["h"])[:2]
            for n in names}


def donor_cut(data, truth):
    errors = []
    for n, (y, v) in truth.items():
        full = v.all(1)
        e = np.linalg.norm(data[n]["cv"][full]-y[full], axis=-1).mean(1)
        errors.extend(e[e > 0])
    if not errors:
        raise ValueError("No donor-only error reference")
    return np.quantile(errors, [.25, .75])


def candidates(d, seed):
    c = d["cv"]
    a, j = (d[f"{k}_{seed}"] for k in ("additive", "relative_joint"))
    radius = av.motion.peak(d[f"anchor_{seed}"]-c)
    r = radius[radius > 0]
    if len(r) != 2:
        return a, j, None, None
    p, row, col, opts, grp = av.optional.options(c, d["bank"], r)
    opts = opts[grp == 2]
    paths = av.slate.assemble(c, p, row, col, opts)
    peaks = np.sort(av.motion.peak(paths-c), axis=1)[:, -2:]
    np.testing.assert_allclose(peaks, np.broadcast_to(np.sort(r), peaks.shape), rtol=1e-8, atol=1e-8)
    for control in (a, j):
        if not np.any(np.all(paths == control, axis=(1, 2, 3))):
            raise ValueError("Frozen comparator left its exact-budget candidate menu")
    x, _, b, unit, _ = relative.features(d["h"], c, d["ols4"], paths, a)
    return a, j, paths, (np.column_stack((x, np.log1p(b))).astype(np.float32), b, unit)


def label_terms(d, paths, a, y, valid, cut):
    c, o = d["cv"], d["ols4"]
    mass = float(np.where(valid, np.linalg.norm(c-y, axis=-1), 0.).sum())
    t = pair.audit.terms(paths, c, o, a, y, valid, cut[0], 1., max(mass, 1e-12), 1)
    e = (np.where(valid, np.linalg.norm(c-y, axis=-1), 0.).sum(1)/valid.sum(1).clip(1))
    cap = .02*e[(e > 0.) & (e <= cut[0])].sum()
    support = (t[:, -1] > 0.) & (mass > 0.)
    strict = (support & (t[:, 0] > 1e-10) & (t[:, 1:3] > 0.).all(1) & (t[:, 3] >= -1e-8)
              & (t[:, 4:6] <= 0.).all(1) & (t[:, 6] <= cap))
    return t[:, 0], strict, support


def examples(data, truth, seed, cut):
    out = {}
    for n, (y, v) in truth.items():
        d = data[n]
        a, _, paths, feat = candidates(d, seed)
        if feat is None:
            continue
        x, b, unit = feat
        gain, strict, support = label_terms(d, paths, a, y, v, cut)
        if not np.isfinite(x).all() or (np.abs(gain)/unit > b+1e-5).any():
            raise ValueError("Invalid features or gain target exceeds geometric bound")
        out[n] = tuple(torch.from_numpy(z.astype(np.float32)) for z in
                       (x, b, gain/unit, strict, support))
    return out


def network():
    nn = torch.nn
    return nn.Sequential(nn.Linear(relative.WIDTH+1, 48), nn.SiLU(),
                         nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))


def fit(identity, examples):
    held, seed = identity["held"], identity["seed"]
    p = LOCAL/f"{held}_{seed}.pt"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+5507)
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    start, losses = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != identity or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Cannot resume an incompatible checkpoint")
        start, losses = ck["step"], ck["loss"]
        rng.bit_generator.state = ck["rng"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
    queries = tuple(examples.values())
    if not queries:
        raise ValueError("No donor scenarios with two causal edit slots")
    for step in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(queries), 1./len(queries)))
        batches = [(z, rng.integers(len(z[0]), size=n)) for z, n in zip(queries, counts, strict=True) if n]
        x, b, g, strict, support = (torch.cat([z[k][i] for z, i in batches]) for k in range(5))
        score = net(x)
        loss_gain = torch.nn.functional.smooth_l1_loss(score[:, 0].tanh()*b, g, beta=.1)
        loss_strict = torch.nn.functional.binary_cross_entropy_with_logits(
            score[:, 1], strict, pos_weight=torch.tensor(5.))
        loss_support = torch.nn.functional.binary_cross_entropy_with_logits(score[:, 2], support)
        loss = loss_gain+loss_strict+.25*loss_support
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            losses.append([float(t.detach()) for t in (loss, loss_gain, loss_strict, loss_support)])
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=losses), tmp)
            tmp.replace(p)
            beat("fit", fold=f"{held}_{seed}", step=step, total=STEPS)
    return net.eval(), study.artifact(p), losses


@torch.inference_mode()
def predict(net, d, seed):
    a, j, paths, feat = candidates(d, seed)
    if feat is None:
        return j, -1.
    x, b, _ = feat
    output = net(torch.from_numpy(x))
    gain = output[:, 0].tanh().numpy()*b
    probability = output[:, 1].sigmoid().numpy()
    score = probability*np.maximum(gain, 0.)
    k = int(score.argmax())
    return (paths[k], float(score[k])) if score[k] > 0 else (j, -1.)


def calibrate(rows):
    """A nonempty prefix must pass both policy arms and improve on additive."""
    values = sorted({r["score"] for r in rows if r["score"] > 0.}, reverse=True)
    feasible = []
    for t in values:
        take = [r for r in rows if r["score"] >= t]
        if all(r["strict"] and r["add_ok"] for r in take):
            feasible.append((len(take), t))
    return (max(feasible)[1] if feasible else None), len(values), len(feasible)


def metrics(path, d, y, v, cut):
    c, o = d["cv"], d["ols4"]
    m = study.policy(path, c, c, y, v, ["scenario"]*len(c), easy_cut=cut[0], hard_cut=cut[1])
    risk = m["recordings"]["scenario"]
    gain = av.frozen.optional_gain(c, o, path, y, v, np.zeros(len(c), dtype=int))
    return dict(risk=risk, gain=gain, hard=m["subsets"]["hard"]["gain_percent"],
        fde=m["available_endpoint_FDE_diagnostic_pixel"], ade=m["known_ADE_pixel"],
        tail=m["p95_known_ADE_pixel"])


def fits(key, old, source, seal):
    data, done = read_inputs(old, seal), {}
    cities = sorted({d["city"] for d in data.values()})
    for held in cities:
        cal, train = roles(cities, held)
        names = lambda subset: [n for n, d in data.items() if d["city"] in subset]
        donor = load_targets(source, data, names(train))
        ycal = load_targets(source, data, names([cal]))
        cut = donor_cut(data, donor)
        for seed in SEEDS:
            k = f"{held}_{seed}"
            receipt = LOCAL/f"{k}.json"
            identity = dict(registration=key, held=held, calibration=cal, donors=train, seed=seed)
            if not receipt.exists():
                x = examples(data, donor, seed, cut)
                net, checkpoint, loss = fit(identity, x)
                calrows = []
                for n, (y, v) in ycal.items():
                    d = data[n]
                    path, s = predict(net, d, seed)
                    if s <= 0:
                        continue
                    a = d[f"additive_{seed}"]
                    _, strict, _ = label_terms(d, path[None], a, y, v, cut)
                    m = metrics(a, d, y, v, cut)
                    calrows.append(dict(score=s, strict=bool(strict[0]),
                        add_ok=bool(m["risk"]["supported"] and m["gain"] is not None and min(m["gain"]) > 0.)))
                threshold, tested, feasible = calibrate(calrows)
                forecasts = {}
                for n in names([held]):
                    d = data[n]
                    path, s = predict(net, d, seed)
                    forecasts[n] = av.turn.exp.previous.save_bank(LOCAL/f"{n}_{seed}.npz",
                        dict(refit=path, score=s, registration=key, seed=seed))
                base.immutable(receipt, dict(identity=identity, checkpoint=checkpoint, loss=loss,
                    donor_queries=len(x), donor_options=sum(len(z[0]) for z in x.values()),
                    donor_strict=int(sum(z[3].sum().item() for z in x.values())), cut=cut.tolist(),
                    cal_strict=sum(r["strict"] for r in calrows), threshold=threshold,
                    tested=tested, feasible=feasible, forecasts=forecasts))
            z = study.read(receipt)
            if z["identity"] != identity or set(z["forecasts"]) != set(names([held])):
                raise ValueError("Fold or held roster changed")
            study.checked(z["checkpoint"])
            for p in z["forecasts"].values():
                study.checked(p)
            done[k] = z
            beat("freeze", fold=k, done=len(done), total=len(cities)*len(SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, old, source, seal):
    p = OUT/"freeze.json"
    frozen = study.read(p)
    data = read_inputs(old, seal)
    cities = sorted({d["city"] for d in data.values()})
    if (frozen["registration"] != key or frozen["held_labels_used"]
            or set(frozen["folds"]) != {f"{c}_{s}" for c in cities for s in SEEDS}):
        raise ValueError("Require all development folds sealed before scoring")
    truth, rows, capacity = load_targets(source, data, data), [], Counter()
    for n, d in data.items():
        y, v = truth[n]
        for seed in SEEDS:
            z = frozen["folds"][f"{d['city']}_{seed}"]
            cal, train = roles(cities, d["city"])
            if z["identity"] != dict(registration=key, held=d["city"], calibration=cal, donors=train, seed=seed):
                raise ValueError("Invalid split receipt")
            with np.load(study.checked(z["forecasts"][n]), allow_pickle=False) as r:
                if r["registration"].item() != key or r["seed"].item() != seed:
                    raise ValueError("Wrong forecast checkpoint")
                learned, s = r["refit"], r["score"].item()
            a, j, paths, _ = candidates(d, seed)
            c, o, cut = d["cv"], d["ols4"], z["cut"]
            threshold = z["threshold"]
            release = learned if threshold is not None and s >= threshold else c
            values = dict(zip(ARMS, (c, o, a, j, learned, release), strict=True))
            qual = {}
            if paths is not None:
                _, options, _ = label_terms(d, paths, a, y, v, cut)
                _, q, _ = label_terms(d, np.stack((j, learned)), a, y, v, cut)
                qual = dict(oracle=int(options.any()), frozen=int(q[0]), refit=int(q[1]), two_slots=1)
                capacity.update(qual)
                if v.all():
                    capacity.update({f"complete_{k}": a for k, a in qual.items()})
            rp, jp = [av.frontier.pair_reduction(values[k][None], a, y, v)[0].tolist()
                      for k in ("refit", "frozen")]
            direct = av.frontier.pair_reduction(learned[None], j, y, v)[0].tolist()
            rows.append(dict(clip=n, city=d["city"], seed=seed, complete=bool(v.all()),
                metrics={k: metrics(path, d, y, v, cut) for k, path in values.items()},
                refit_pair=rp, frozen_pair=jp, direct_pair=direct, **qual))
        beat("score", done=len(rows)//len(SEEDS), total=len(data))
    totals = {}
    for a in ARMS:
        active = [r["metrics"][a]["risk"] for r in rows if r["metrics"][a]["risk"]["selected"]]
        totals[a] = dict(edits=sum(x["selected"] for x in active), edited=len(active),
            risk_pass=sum(x["supported"] for x in active),
            easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in active),
            easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in active))
        for k in ("ade", "fde", "gain", "hard", "tail"):
            totals[a][k] = av.interval(rows, lambda r: r["metrics"][a][k])
    if len({totals[a]["edits"] for a in ("additive", "frozen", "refit")}) != 1:
        raise ValueError("Forced comparison changed edit budget")
    contrasts = {k: {s: av.interval([r for r in rows if s == "all" or r["complete"]], lambda r: r[k])
                     for s in ("all", "complete")} for k in ("refit_pair", "frozen_pair", "direct_pair")}
    cal = {k: sum(z[k] for z in frozen["folds"].values()) for k in ("tested", "feasible", "cal_strict")}
    base.immutable(LOCAL/"rows.json", rows)
    result = dict(registration=key, freeze=study.artifact(p), source="fresh_run", fits=len(frozen["folds"]),
        seeds=SEEDS, cities=cities, scenarios=len(data), capacity=dict(capacity), metrics=totals,
        contrasts=contrasts, calibration=cal, original_2pct=True, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps({k: result[k] for k in ("capacity", "contrasts", "calibration")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, old, source, seal = register()
        (fits if phase == "fit" else score)(key, old, source, seal)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
