"""Train the pinned EqMotion K=1 control and test its exact-slot forecast menu."""

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
from scripts import run_m3w_av2_data_scale as scale
from scripts import audit_m3w_av2_paired_release_capacity as ceiling
from src.world_model import m3w_opendd_comoving_eqmotion as eq
from src.world_model import m3w_eqmotion_adapter as author

av, prev, np, torch, study, base = scale.av, scale.prev, scale.np, scale.torch, scale.study, scale.base
OUT, LOCAL = scale.OUT.parent/"av2_eqmotion_v1", scale.LOCAL.parent/"av2_eqmotion_v1"
STEPS, BATCH, SAVE = 800, 32, 50
SEEDS = prev.SEEDS
VARIANT = "centered_peers"
MATCHED = ("fixed_additive", "fixed_joint", "eq_additive", "eq_joint")
ARMS = ("cv", "ols4", "raw_eq", *MATCHED)


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), phase=phase, **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    previous, _, _, source, _, costs, relatives = scale.register("score")
    p = scale.OUT/"summary.json"
    r = study.read(p)
    f = study.read(study.checked(r["freeze"]))
    past = study.read(study.checked(r["inputs"]))
    if (r["registration"] != previous or f["registration"] != previous or f["held_labels_used"]
            or past["registration"] != previous or past["input_future_used"]
            or r["reserved_outcomes_opened"] or r["independent_confirmation"]):
        raise ValueError("Require the previously frozen AV2 development roles")
    _, upstream = author.verified_source()
    reg = OUT/"registration.json"
    base.immutable(reg, dict(parent=study.artifact(p), source_receipts=r["inputs"], previous_fit=r["freeze"],
        code=[study.artifact(Path(x)) for x in (__file__, eq.__file__, author.__file__, ceiling.__file__)],
        author=upstream, model=eq.ARCH, variant=VARIANT, seeds=SEEDS, steps=STEPS,
        checkpoint_every=SAVE, batch=BATCH, workers=0, torch_threads=4,
        scope="same observed AV2 publisher train, 8/12, one deterministic author head trained from scratch",
        roles="same four donor cities; calibration excluded; held scores on the same 896 added scenarios",
        optimizer="AdamW lr .001, masked robust normalized path loss, city-balanced actor batches",
        match="add EqMotion as one family to each original menu; only original exact peak slots; same frozen additive and joint decoders",
        capacity="strict dual motion and 2 percent risks, noninferior trajectory and positive conservative pair gain; compare both menus against original additive and relaxed two-arm ceiling",
        calibration="not run: forced public forecaster and candidate capacity only",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), previous, source, f, costs, relatives


def city_batches(data, truth, cities):
    pools = []
    for city in cities:
        items = []
        for n, (y, v) in truth.items():
            if data[n]["city"] != city:
                continue
            d = data[n]
            use = v.any(1)
            if use.any():
                x = eq.context(d["h"], [0, len(y)])
                items.append((*[t[use] for t in x], torch.from_numpy(y[use].astype(np.float32)),
                              torch.from_numpy(v[use])))
        if not items:
            raise ValueError("Empty donor target pool")
        pools.append(tuple(torch.cat(parts) for parts in zip(*items, strict=True)))
    return pools


def fit(ident, pools):
    seed, held = ident["seed"], ident["held"]
    p = LOCAL/f"{held}_{seed}.pt"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+3711)
    net = eq.Forecast(VARIANT)
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=.001)
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Cannot resume an incompatible EqMotion fit")
        start, trace = ck["step"], ck["loss"]
        rng.bit_generator.state = ck["rng"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
    for i in range(start+1, STEPS+1):
        sizes = rng.multinomial(BATCH, np.full(len(pools), 1./len(pools)))
        idx = [rng.integers(len(d[0]), size=n) for d, n in zip(pools, sizes, strict=True)]
        data = [torch.cat([d[k][ix] for d, ix in zip(pools, idx, strict=True)]) for k in range(6)]
        value = eq.loss(net, data[:4], data[4], data[5])
        opt.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if i % SAVE == 0:
            trace.append(float(value.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, net=net.state_dict(), opt=opt.state_dict(), step=i,
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=f"{held}_{seed}", step=i, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), loss=trace, donor_actors=sum(len(d[0]) for d in pools))


@torch.inference_mode()
def predict(net, data, names):
    hist = [data[n]["h"] for n in names]
    offsets = np.r_[0, np.cumsum([len(h) for h in hist])]
    x = eq.context(np.concatenate(hist), offsets)
    forecast = torch.cat([net(*(t[i:i+128] for t in x[:3])) for i in range(0, len(x[0]), 128)]).numpy().astype(float)
    if not np.isfinite(forecast).all():
        raise ValueError("EqMotion produced a nonfinite forecast")
    return forecast, offsets


def radii(d, seed):
    r = av.motion.peak(d[f"anchor_{seed}"]-d["cv"])
    return r[r > 0.]


def expanded(d, y):
    return np.concatenate((d["bank"], y[:, None]), axis=1)


def freeze(key, previous, source, parent, costs, relatives):
    data, done = scale.read_inputs(previous, source), {}
    cities = sorted({d["city"] for d in data.values()})
    frozen_models = {s: (av.frozen.load(s, costs, av.turn.refit.model.CostHead, av.turn.refit.STEPS),
                    av.frozen.load(s, relatives, av.slate.model.RelativeGain, av.slate.STEPS)) for s in SEEDS}
    for held in cities:
        cal, donors = prev.roles(cities, held)
        donor_y = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        pools = city_batches(data, donor_y, donors)
        names = [n for n, d in data.items() if d["city"] == held and d["added"]]
        for seed in SEEDS:
            k = f"{held}_{seed}"
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, arm=VARIANT)
            p = LOCAL/f"{k}.json"
            old = parent["folds"][k]
            if any(ident[x] != old["identity"][x] for x in ("held", "calibration", "donors", "seed")):
                raise ValueError("The public control must use the same donor cities")
            if not p.exists():
                net, item = fit(ident, pools)
                raw, offsets = predict(net, data, names)
                forecast = {a: [] for a in ("eq_additive", "eq_joint")}
                for n, lo, hi in zip(names, offsets[:-1], offsets[1:], strict=True):
                    d = data[n]
                    args = (d["h"], d["cv"], d["ols4"])
                    rr = radii(d, seed)
                    re = prev.replay.choose(*args, d["bank"], rr, *frozen_models[seed])
                    for a, v in re.items():
                        np.testing.assert_array_equal(v, d[f"{a}_{seed}"])
                    out = prev.replay.choose(*args, expanded(d, raw[lo:hi]), rr, *frozen_models[seed])
                    for a, v in out.items():
                        k2 = "eq_joint" if a == "relative_joint" else "eq_additive"
                        forecast[k2].append(v)
                bank = av.turn.exp.previous.save_bank(LOCAL/f"{k}.npz", dict(
                    registration=key, names=np.asarray(names), offsets=offsets, raw_eq=raw,
                    **{a: np.concatenate(v) for a, v in forecast.items()}))
                base.immutable(p, dict(identity=ident, fit=item, forecast=bank, donor_scenarios=len(donor_y),
                    held_scenarios=len(names), cut=old["identity"]["cut"]))
            r = study.read(p)
            if r["identity"] != ident or r["held_scenarios"] != len(names) or r["cut"] != old["identity"]["cut"]:
                raise ValueError("Incompatible held EqMotion forecast")
            study.checked(r["forecast"])
            study.checked(r["fit"]["checkpoint"])
            done[k] = r
            beat("freeze", fold=k, done=len(done), total=len(cities)*len(SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done,
        held_labels_used=False, calibration_labels_used=False))


def slates(d, rr, bank):
    paths, row, col, opts, group = av.optional.options(d["cv"], bank, rr)
    return av.slate.assemble(d["cv"], paths, row, col, opts[group == 2])


def opportunities(d, seed, pred, y, v, cut):
    rr = radii(d, seed)
    if len(rr) != 2:
        return {}
    refs = {}
    for name, bank in (("base", d["bank"]), ("expanded", expanded(d, pred["raw_eq"]))):
        paths = slates(d, rr, bank)
        _, q, _ = prev.label_terms(d, paths, pred["fixed_additive"], y, v, cut)
        _, pair = ceiling.alternative_capacity(d, paths, y, v, cut)
        refs[name] = (bool(q.any()), pair, len(paths))
    if refs["base"][0] > refs["expanded"][0] or refs["base"][1] > refs["expanded"][1]:
        raise ValueError("Adding one candidate family removed a legal witness")
    strict = {}
    for a, b in (("fixed_joint", "fixed_additive"), ("eq_joint", "eq_additive")):
        _, q, _ = prev.label_terms(d, pred[a][None], pred[b], y, v, cut)
        strict[f"{a}_strict"] = int(q[0])
        strict[f"{a}_both"] = int(q[0] and ceiling.eligible_additive(prev.metrics(pred[b], d, y, v, cut)))
    return dict(two_slots=1, base_options=refs["base"][2], expanded_options=refs["expanded"][2],
        base_strict=int(refs["base"][0]), expanded_strict=int(refs["expanded"][0]),
        base_paired_ceiling=int(refs["base"][1]), expanded_paired_ceiling=int(refs["expanded"][1]), **strict)


def score(key, previous, source, parent):
    data = scale.read_inputs(previous, source)
    p = OUT/"freeze.json"
    f = study.read(p)
    if (f["registration"] != key or f["held_labels_used"] or f["calibration_labels_used"]
            or set(f["folds"]) != set(parent["folds"])):
        raise ValueError("Freeze every forecast before outcome access")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    rows, cap = [], Counter()
    for name, z in f["folds"].items():
        ident = z["identity"]
        cut, seed, held = z["cut"], ident["seed"], ident["held"]
        with np.load(study.checked(z["forecast"]), allow_pickle=False) as raw:
            pred = dict(raw)
        expected = [n for n in truth if data[n]["city"] == held]
        if pred["registration"].item() != key or list(pred["names"]) != expected:
            raise ValueError("Held forecast roster does not match its target city")
        for n, lo, hi in zip(expected, pred["offsets"][:-1], pred["offsets"][1:], strict=True):
            d, (y, v) = data[n], truth[n]
            forecasts = dict(cv=d["cv"], ols4=d["ols4"], fixed_additive=d[f"additive_{seed}"],
                fixed_joint=d[f"relative_joint_{seed}"], **{a: pred[a][lo:hi] for a in ("raw_eq", "eq_additive", "eq_joint")})
            q = opportunities(d, seed, forecasts, y, v, cut)
            cap.update(q)
            if v.all():
                cap.update({f"complete_{k}": val for k, val in q.items()})
            effect = {f"{a}_vs_{b}": av.frontier.pair_reduction(forecasts[a][None], forecasts[b], y, v)[0].tolist()
                      for a, b in (("eq_joint", "eq_additive"), ("fixed_joint", "fixed_additive"),
                                   ("eq_joint", "fixed_joint"), ("eq_additive", "fixed_additive"), ("raw_eq", "cv"))}
            rows.append(dict(city=held, clip=n, seed=seed, complete=bool(v.all()),
                metrics={a: prev.metrics(forecasts[a], d, y, v, cut) for a in ARMS}, effects=effect))
        beat("score", fold=name, views=len(rows), total=len(truth)*len(SEEDS))
    metrics = {}
    for a in ARMS:
        risk = [r["metrics"][a]["risk"] for r in rows if r["metrics"][a]["risk"]["selected"]]
        metrics[a] = dict(edits=sum(x["selected"] for x in risk), edited=len(risk),
            risk_pass=sum(x["supported"] for x in risk), easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in risk),
            easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in risk),
            dual_risk_pass=sum(ceiling.eligible_additive(r["metrics"][a]) for r in rows))
        metrics[a].update({k: av.interval(rows, lambda r: r["metrics"][a][k]) for k in ("ade", "fde", "gain", "hard", "tail")})
    if len({metrics[a]["edits"] for a in MATCHED}) != 1:
        raise ValueError("The expanded producer changed the forced edit budget")
    effects = {k: {g: av.interval([r for r in rows if g == "all" or r["complete"]], lambda r: r["effects"][k])
                   for g in ("all", "complete")} for k in rows[0]["effects"]}
    base.immutable(LOCAL/"rows.json", rows)
    result = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run",
        fits=len(f["folds"]), updates=len(f["folds"])*STEPS, held_scenarios=len(truth),
        metrics=metrics, contrasts=effects, opportunity=dict(cap), original_2pct=True,
        calibration_evaluated=False, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps({k: result[k] for k in ("fits", "held_scenarios", "opportunity")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, previous, source, parent, costs, relatives = register()
        if phase == "fit":
            freeze(key, previous, source, parent, costs, relatives)
        else:
            score(key, previous, source, parent)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
