"""Learn matched AV2 decisions on genuinely out-of-city neural candidates."""

import argparse
from collections import Counter
import fcntl
from math import log
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_eqmotion_oof as oof
from scripts import run_m3w_av2_learned_reference as action
from scripts import run_m3w_av2_eqmotion_uniform as uniform
from src.world_model import m3w_av2_interaction_credit as credit

eq, av, prev, np, torch, study, base = oof.eq, oof.av, oof.prev, oof.np, oof.torch, oof.study, oof.base
OUT, LOCAL = oof.OUT.parent/"av2_oof_selection_v1", oof.LOCAL.parent/"av2_oof_selection_v1"
STEPS, SAVE, BATCH, POSITIVE = 320, 40, 128, 32
METHODS = ("risk_only", "interaction")
LEARNED = ("uniform", "additive", *METHODS)
ARMS = ("cv", "ols4", "raw_eq", "frozen_additive", "frozen_joint", "frozen_uniform",
        *LEARNED, "release_additive", "release_joint")
F = torch.nn.functional


def beat(phase, **fields):
    tmp = LOCAL/"heartbeat.pending"
    tmp.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    tmp.replace(LOCAL/"heartbeat.json")


def register():
    upstream, data, source, roles, jobs, held, _, _ = oof.register()
    p = oof.OUT/"freeze.json"
    inner = study.read(p)
    a = eq.OUT/"uniform_freeze.json"
    b = study.read(a)
    if (inner["registration"] != upstream or not inner["past_only_forecasts"]
            or inner["reserved_outcomes_opened"] or b["future_used"]
            or len(inner["models"]) != len(jobs)*len(eq.SEEDS) or set(b["forecasts"]) != set(held["folds"])):
        raise ValueError("Require complete nested and outer causal forecasts")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, credit.__file__, action.__file__)],
        nested=study.artifact(p), parent=study.artifact(eq.OUT/"freeze.json"), uniform=study.artifact(a),
        roles=roles, seeds=eq.SEEDS, steps=STEPS, checkpoint_every=SAVE, batch=BATCH,
        heads="shared independent action cost; shared slate qualification and conditional gain with or without nonadditive pair credit",
        credit="four-way paired error difference on edited recipients only; subtract twice the smaller edit distance on missing pair steps",
        training="city-excluded EqMotion producer for every donor trajectory; calibration and held use the frozen four-donor producer",
        score="probability of joint accuracy/risk qualification times conditional dual gain; add conditional interaction credit averaged over all roster pairs only in interaction arm",
        same_menu_and_peaks=True, calibration="same original 2 percent checks, dual gains and direct pair advantage over selected additive; choose greatest calibration coverage",
        num_workers=0, torch_threads=4, original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, source, roles, inner, held, b


def read_forecasts(ref, field):
    with np.load(study.checked(ref), allow_pickle=False) as z:
        return {str(n): z[field][a:b].copy() for n, a, b in zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)}


def donor_forecasts(frozen, data, held, role, seed):
    forecasts = {}
    for city, name in role["inner"].items():
        r = frozen["models"][f"{name}_{seed}"]
        tr = set(r["fit"]["identity"]["donors"])
        if tr != set(role["donors"])-{city} or tr & {city, held, role["calibration"]}:
            raise ValueError("A nested producer saw excluded cities")
        pp = read_forecasts(r["forecast"], "forecast")
        names = {n for n, d in data.items() if d["city"] == city}
        if not names <= pp.keys():
            raise ValueError("Missing out-of-city neural forecasts")
        forecasts.update({n: pp[n] for n in names})
    return forecasts


def load_outer(item):
    ck = torch.load(study.checked(item["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != item["identity"] or ck["step"] != eq.STEPS:
        raise ValueError("Wrong calibration EqMotion producer")
    model = eq.eq.Forecast(eq.VARIANT).eval()
    model.load_state_dict(ck["net"])
    return model


def options(d, m, cut):
    c, o = d["cv"], d["ols4"]
    p = av.slate.assemble(c, m["bank"], m["row"], m["col"], m["opts"])
    recipients = m["row"][m["opts"]]
    if (recipients[:, 0] == recipients[:, 1]).any():
        raise ValueError("Both edits target the same agent")
    x, g, b, unit = credit.describe(d["h"], c, o, p, recipients, cut[0])
    return p, recipients, x, g, b, unit


def pair_targets(d, m, y, v, cut):
    p, recipients, x, g, b, unit = options(d, m, cut)
    c, o = d["cv"], d["ols4"]
    e = np.where(v, np.linalg.norm(c-y, axis=-1), 0.)
    mass = float(e.sum())
    e = e.sum(1)/v.sum(1).clip(1)
    cap = .02*e[(e > 0.) & (e <= cut[0])].sum()
    t = prev.pair.audit.terms(p, c, o, c, y, v, cut[0], 1., max(mass, 1e-12), 1)
    support = (t[:, -1] > 0.) & (mass > 0.)
    good = support & (t[:, 1:3] > 0.).all(1) & (t[:, 4:6] <= 0.).all(1) & (t[:, 6] <= cap)
    gain = np.maximum(t[:, 1:3].min(1), 0.)*mass/(100.*v.size*unit)
    effect = credit.interaction_lower(c, p, recipients, y, v)/unit
    if (gain > g+1e-5).any() or (np.abs(effect) > b+1e-5).any():
        raise ValueError("Target exceeds an outcome-free geometry bound")
    return tuple(torch.from_numpy(z.astype(np.float32)) for z in (x, g, b, gain, effect, good, support))


def pair_loss(net, batch, pos):
    x, _, _, _, _, good, support = batch
    raw = net(x)
    gate = F.binary_cross_entropy_with_logits(raw[:, 2], good, pos_weight=torch.tensor(5.))
    aux = F.binary_cross_entropy_with_logits(raw[:, 3], support)
    x, b1, b2, g, c, _, _ = pos
    a, b = credit.values(net(x), b1, b2)
    return gate+.25*aux+F.mse_loss(a, g)+F.mse_loss(b, c)


def fit(ident, name, examples, is_pair):
    seed = ident["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+7283)
    net = credit.PairValue() if is_pair else action.Action()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        c = torch.load(path, weights_only=True, map_location="cpu")
        if c["identity"] != ident or c["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Checkpoint belongs to a different fit")
        start, trace = c["step"], c["loss"]
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["rng"]
    pos = [(d, torch.nonzero(d[5]).numpy().ravel()) for d in examples if d[5].any()] if is_pair else []
    if is_pair and not pos:
        raise ValueError("No donor examples for conditional slate training")

    def batch(ds, n):
        sizes = rng.multinomial(n, np.full(len(ds), 1./len(ds)))
        draw = [(d, ix[rng.integers(len(ix), size=k)]) for (d, ix), k in zip(ds, sizes, strict=True) if k]
        return [torch.cat([d[j][i] for d, i in draw]) for j in range(len(ds[0][0]))]

    rows = [(d, np.arange(len(d[0]))) for d in examples]
    for t in range(start+1, STEPS+1):
        x = batch(rows, BATCH)
        loss = pair_loss(net, x, batch(pos, POSITIVE)) if is_pair else action.action_loss(net, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % SAVE == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=t, loss=trace, net=net.state_dict(),
                            opt=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=t, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(path), loss=trace, donor_queries=len(examples),
        positive_donor_queries=len(pos), qualifying_options=sum(len(x) for _, x in pos))


def select_scores(raw, g, b, pairs):
    gain, pair = credit.values(raw, g, b)
    p = (raw[:, 2]-log(5.)).sigmoid()
    return torch.stack((p*gain, p*(gain+pair/pairs)), dim=-1)


@torch.inference_mode()
def choose(act, slate, d, seed, cut, singleton):
    m = action.menu(d, seed, cut)
    if m is None:
        return dict.fromkeys(LEARNED, singleton), dict.fromkeys(METHODS, -1.)
    p, _, x, g, b, _ = options(d, m, cut)
    value = act(torch.from_numpy(m["x"])).numpy()
    units = m["unit"][m["row"]]
    _, add, _ = action.rank(value, m["opts"], units)
    opts = m["opts"]
    family = m["col"] % d["bank"].shape[1]
    in_uniform = np.flatnonzero(family[opts[:, 0]] == family[opts[:, 1]])
    if not len(in_uniform):
        raise ValueError("Missing same-family OLS witness")
    _, u, _ = action.rank(value, opts[in_uniform], units)
    paths = dict(uniform=p[in_uniform[u[0]]], additive=p[add[0]])
    scores = select_scores(slate(torch.from_numpy(x)), torch.from_numpy(g), torch.from_numpy(b),
                           len(d["cv"])*(len(d["cv"])-1)//2).numpy()
    confidences = {}
    for k, m in enumerate(METHODS):
        i = int(scores[:, k].argmax())
        s = float(scores[i, k])
        paths[m] = p[i] if s > 0. else paths["additive"]
        confidences[m] = s if s > 0. else -1.
    for path in paths.values():
        rr = av.motion.peak(path-d["cv"])
        np.testing.assert_allclose(np.sort(rr[rr > 0.]), np.sort(eq.radii(d, seed)), atol=1e-8, rtol=1e-8)
    return paths, confidences


def train(key, data, source, roles, inner, held):
    done = {}
    for h, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        cal_names = [n for n, d in data.items() if d["city"] == cal and d["added"]]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, cal_names)
        for seed in eq.SEEDS:
            name = f"{h}_{seed}"
            old = held["folds"][name]
            cut = old["cut"]
            ident = dict(registration=key, held=h, calibration=cal, donors=donors, seed=seed, cut=cut)
            if any(old["identity"][k] != ident[k] for k in ("held", "calibration", "donors", "seed")):
                raise ValueError("Outer roles changed")
            p = LOCAL/f"{name}.json"
            if not p.exists():
                nested = donor_forecasts(inner, data, h, role, seed)
                if nested.keys() != dy.keys():
                    raise ValueError("Neural donor and target rosters differ")
                ae, pe = [], []
                for n, (y, v) in dy.items():
                    d = dict(data[n], bank=eq.expanded(data[n], nested[n]))
                    if (m := action.menu(d, seed, cut)) is not None:
                        ae.append(action.action_targets(d, m, y, v, cut))
                        pe.append(pair_targets(d, m, y, v, cut))
                a, fa = fit(dict(ident, model="action"), name+"_action", ae, False)
                j, fj = fit(dict(ident, model="slate"), name+"_slate", pe, True)
                raw, offsets = eq.predict(load_outer(old), data, cal_names)
                rr = {m: [] for m in METHODS}
                for n, lo, hi in zip(cal_names, offsets[:-1], offsets[1:], strict=True):
                    d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
                    paths, scores = choose(a, j, d, seed, cut, d["cv"])
                    y, v = cy[n]
                    ref = paths["additive"]
                    ok = eq.ceiling.eligible_additive(prev.metrics(ref, d, y, v, cut))
                    for m in METHODS:
                        if scores[m] > 0.:
                            _, q, _ = prev.label_terms(d, paths[m][None], ref, y, v, cut)
                            rr[m].append(dict(score=scores[m], strict=bool(q[0]), add_ok=ok))
                cal_metrics = {m: action.prior.calibration(rr[m]) for m in METHODS}
                chosen = max(METHODS, key=lambda m: cal_metrics[m]["accepted"])
                if not cal_metrics[chosen]["accepted"]:
                    chosen = None
                with np.load(study.checked(old["forecast"]), allow_pickle=False) as z:
                    raw = dict(z)
                paths, scores = {m: [] for m in LEARNED}, {m: [] for m in METHODS}
                for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                    d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                    pred, confidence = choose(a, j, d, seed, cut, raw["eq_additive"][lo:hi])
                    for m in LEARNED:
                        paths[m].append(pred[m])
                    for m in METHODS:
                        scores[m].append(confidence[m])
                f = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(
                    names=raw["names"], offsets=raw["offsets"], registration=key,
                    **{m: np.concatenate(v) for m, v in paths.items()},
                    **{f"score_{m}": np.asarray(v) for m, v in scores.items()}))
                base.immutable(p, dict(identity=ident, action=fa, slate=fj, forecast=f, calibration=cal_metrics,
                    chosen=chosen, nested_models={c: f"{g}_{seed}" for c, g in role["inner"].items()}))
            z = study.read(p)
            if z["identity"] != ident or z["nested_models"] != {c: f"{g}_{seed}" for c, g in role["inner"].items()}:
                raise ValueError("Incompatible frozen forecasts")
            for f in (z["forecast"], z["action"]["checkpoint"], z["slate"]["checkpoint"]):
                study.checked(f)
            done[name] = z
            beat("freeze", fold=name, done=len(done), total=len(held["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, data, source, held, uniform_control):
    p = OUT/"freeze.json"
    freeze = study.read(p)
    if freeze["registration"] != key or freeze["held_labels_used"] or set(freeze["folds"]) != set(held["folds"]):
        raise ValueError("Require all decisions and calibrations frozen")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    rows, counter = [], Counter()
    for name, z in freeze["folds"].items():
        city, seed, cut = (z["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(z["forecast"]), allow_pickle=False) as a,
              np.load(study.checked(held["folds"][name]["forecast"]), allow_pickle=False) as b,
              np.load(study.checked(uniform_control["forecasts"][name]), allow_pickle=False) as u):
            for field in ("names", "offsets"):
                np.testing.assert_array_equal(a[field], b[field])
                np.testing.assert_array_equal(a[field], u[field])
            if a["registration"].item() != key or any(data[n]["city"] != city for n in a["names"]):
                raise ValueError("Held roster changed")
            for i, (n, lo, hi) in enumerate(zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                c = d["cv"]
                pred = dict(cv=c, ols4=d["ols4"], raw_eq=b["raw_eq"][lo:hi], frozen_additive=b["eq_additive"][lo:hi],
                    frozen_joint=b["eq_joint"][lo:hi], frozen_uniform=u["path"][lo:hi],
                    **{m: a[m][lo:hi] for m in LEARNED})
                method = z["chosen"]
                use = method is not None and a[f"score_{method}"][i] >= z["calibration"][method]["threshold"]
                pred.update(release_additive=pred["additive"] if use else c, release_joint=pred[method] if use else c)
                m = {k: prev.metrics(x, d, y, v, cut) for k, x in pred.items()}
                ok = eq.ceiling.eligible_additive(m["additive"])
                if len(eq.radii(d, seed)) == 2:
                    legal = eq.slates(d, eq.radii(d, seed), eq.expanded(d, pred["raw_eq"]))
                    _, feasible, _ = prev.label_terms(d, legal, pred["additive"], y, v, cut)
                    counter.update(two_slots=1, additive_eligible=int(ok), any_strict=int(feasible.any()),
                        paired_ceiling=int(ok and feasible.any()),
                        credit_changes_selection=int(not np.array_equal(pred["risk_only"], pred["interaction"])))
                    for arm in METHODS:
                        _, q, _ = prev.label_terms(d, pred[arm][None], pred["additive"], y, v, cut)
                        if q[0] and not feasible.any():
                            raise ValueError("Selected action lies outside its matched oracle")
                        counter.update({f"{arm}_strict": int(q[0]), f"{arm}_both": int(ok and q[0])})
                if use:
                    _, q, _ = prev.label_terms(d, pred[method][None], pred["additive"], y, v, cut)
                    counter.update(release_views=1, release_both=int(ok and q[0]))
                pairs = {f"{k}_vs_{b}": av.frontier.pair_reduction(pred[k][None], pred[b], y, v)[0].tolist()
                         for k, b in (("interaction", "risk_only"), ("interaction", "additive"), ("interaction", "uniform"),
                                      ("interaction", "frozen_joint"), ("risk_only", "additive"), ("additive", "uniform"))}
                rows.append(dict(clip=str(n), city=city, seed=seed, complete=bool(v.all()), metrics=m, pair=pairs))
        beat("score", fold=name, views=len(rows), total=len(truth)*len(eq.SEEDS))
    out = {}
    for arm in ARMS:
        risk = [r["metrics"][arm]["risk"] for r in rows if r["metrics"][arm]["risk"]["selected"]]
        out[arm] = dict(edits=sum(x["selected"] for x in risk), edited=len(risk), risk_pass=sum(x["supported"] for x in risk),
            dual_risk_pass=sum(eq.ceiling.eligible_additive(r["metrics"][arm]) for r in rows),
            easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in risk),
            easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in risk))
        out[arm].update({k: av.interval(rows, lambda r: r["metrics"][arm][k]) for k in ("ade", "fde", "gain", "hard", "tail")})
    if len({out[m]["edits"] for m in ("frozen_joint", "frozen_uniform", *LEARNED)}) != 1:
        raise ValueError("Forced edit budget changed")
    if out["release_additive"]["edits"] != out["release_joint"]["edits"]:
        raise ValueError("Unequal optional paired edit count")
    cal = {m: {k: sum(z["calibration"][m][k] for z in freeze["folds"].values())
               for k in ("tested", "feasible", "local_strict", "strict_with_add", "accepted")} for m in METHODS}
    pair = {k: {g: av.interval([r for r in rows if g == "all" or r["complete"]], lambda r: r["pair"][k])
                for g in ("all", "complete")} for k in rows[0]["pair"]}
    result = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run", fits=2*len(freeze["folds"]),
        held_scenarios=len(truth), metrics=out, contrasts=pair, counts=dict(counter), calibration=cal,
        chosen=dict(Counter(z["chosen"] or "cv" for z in freeze["folds"].values())),
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False, original_2pct=True)
    base.immutable(LOCAL/"rows.json", rows)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps({k: result[k] for k in ("fits", "counts", "calibration", "chosen")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, source, roles, inner, held, control = register()
        if phase == "fit":
            train(key, data, source, roles, inner, held)
        else:
            score(key, data, source, held, control)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
