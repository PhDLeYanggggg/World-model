"""Train and validate whole-pair release before applying the held-location rule."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

from scipy.stats import rankdata

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_distinct_supported_slates as audit
from src.world_model import m3w_atomic_pair_admission as model, m3w_slate_imitation as geometry

run = audit.run
np, torch, study, base = run.np, run.torch, run.study, run.base
SITES, SEEDS = run.SITES, run.SEEDS
OUT, LOCAL = run.OUT.parent/"opendd_atomic_pair_admission_v1", run.LOCAL.parent/"opendd_atomic_pair_admission_v1"
STEPS, EVERY, BATCH, LIMIT = 600, 50, 128, 24
THRESHOLDS = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9)
ARMS = ("forced_add", "forced_joint", "cal_add", "cal_joint")
F = torch.nn.functional


def roles(held):
    cal = SITES[(SITES.index(held)+1) % len(SITES)]
    return cal, tuple(s for s in SITES if s not in (held, cal))


def beat(**fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    _, cohorts, _, _, _, _ = run.register()
    p = OUT/"registration.json"
    base.immutable(p, dict(cohorts=cohorts, code=[study.artifact(Path(x)) for x in (__file__, model.__file__,
        audit.__file__, geometry.__file__)], parent_capacity=study.artifact(audit.OUT/"summary.json"),
        roles={s: roles(s) for s in SITES}, seed=SEEDS, steps=STEPS, batch=BATCH, samples_per_query=LIMIT,
        producer="analytic OLS4, causal motion variant, tapered turn and decaying acceleration for all roles",
        slots="exact peaks of two past-hindcast-positive OLS4 trajectories; other queries fallback CV",
        training_target="full-slate local positive conservative gains vs CV/OLS4, three original risks and easy support",
        calibration="one disjoint location; retain most pairs with all edited recordings passing for both arms",
        thresholds=THRESHOLDS, no_threshold_when_no_valid_nonempty=True,
        comparison="one shared gate on the joint score; additive and joint use exactly the same slots",
        workers=0, threads=4, observed_steps=8, forecast_steps=12, original_2pct=True,
        development_only=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), cohorts


def causal(ref):
    h, q, cv, ols, bank = run.turn.menu.parent.source_inputs(ref)
    bank = run.with_acceleration(h, cv, run.turn.append_turn(h, bank))
    return dict(h=h, q=q, cv=cv, ols=ols, bank=bank)


def menus(data):
    h, cv, ols, bank = (data[k] for k in ("h", "cv", "ols", "bank"))
    _, _, _, hindcast = run.shared.costs.features(h, cv, ols, ols[:, None])
    radius = run.turn.turn.peak(ols-cv)
    for ix in run.slate.chunks(data["q"]):
        if ix.stop-ix.start < 3:
            continue
        p = np.flatnonzero((hindcast[ix] > 0.) & (radius[ix] > 1e-9))
        if len(p) < 2:
            continue
        chosen = p[np.argsort(-hindcast[ix][p], kind="stable")[:2]]
        c, o, hh = cv[ix], ols[ix], h[ix]
        paths, row, col, opts = run.slate.legal(c, bank[ix], radius[ix][chosen])
        x, scale, legal, _ = run.shared.costs.features(hh, c, o, paths)
        if not legal[row, col].all():
            raise ValueError("Illegal action at matched radii")
        yield ix, dict(c=c, ols=o, h=hh, paths=paths, row=row, col=col, opts=opts,
                       x=x[row, col], scale=scale[row], radius=radius[ix][chosen])


def encode(p, opts):
    r = p["row"][opts]
    h = p["h"][r]
    scale = p["scale"][opts].mean(1)
    displacement = p["paths"][p["row"], p["col"]]-p["c"][p["row"]]
    x, g, _ = geometry.pack(p["x"], p["h"][p["row"], -1], displacement, p["scale"], opts)
    observed = np.linalg.norm(h[:, 0]-h[:, 1], axis=-1)[:, [0, 3, 7]]
    motion = np.linalg.norm((h[:, 0, -1]-h[:, 0, -3])-(h[:, 1, -1]-h[:, 1, -3]), axis=-1)
    context = np.log1p(np.column_stack((observed, motion))/scale[:, None])
    return x, np.column_stack((g, context)).astype(np.float32)


def paths_for(p, opts):
    return run.slate.assemble(p["c"], p["paths"], p["row"], p["col"], opts)


def labels(p, paths, y, v, cut):
    c, o = p["c"], p["ols"]
    e, _ = audit.errors(c, y, v)
    full_easy = .02*e[(e > 0.) & (e <= cut)].sum()
    terms = audit.terms(paths, c, o, c, y, v, cut, 1., 100., 1)
    support = terms[:, -1] > 0.
    good = support & (terms[:, 1:3] > 0.).all(1) & (terms[:, 4:6] <= 0.).all(1) & (terms[:, 6] <= full_easy)
    return np.column_stack((good, support)).astype(np.float32)


def examples(x, y, v, cut):
    rng = np.random.default_rng(2187)
    result = []
    for ix, p in menus(x):
        take = np.sort(rng.choice(len(p["opts"]), min(LIMIT, len(p["opts"])), replace=False))
        opts = p["opts"][take]
        result.append((*encode(p, opts), labels(p, paths_for(p, opts), y[ix], v[ix], cut)))
    if not result:
        return None
    return tuple(torch.from_numpy(np.concatenate(a)) for a in zip(*result, strict=True))


def fit(key, held, cal, seed, donors, binding):
    name = f"{held}_{seed}"
    ident = dict(registration=key, held=held, calibration=cal, seed=seed, donors=binding,
                 examples={k: len(v[0]) for k, v in donors.items()})
    p = LOCAL/f"{name}.pt"
    torch.manual_seed(seed)
    net = model.PairAdmission()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+827)
    start, trace = 0, []
    if p.exists():
        c = torch.load(p, map_location="cpu", weights_only=True)
        if c["identity"] != ident or c["step"] not in range(EVERY, STEPS+1, EVERY):
            raise ValueError("Resume checkpoint identity mismatch")
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["optimizer"])
        start, trace, rng.bit_generator.state = c["step"], c["trace"], c["rng"]
    source = list(donors.values())
    for step in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(source), 1./len(source)))
        ids = [rng.integers(len(d[0]), size=n) for d, n in zip(source, counts, strict=True)]
        x, g, y = (torch.cat([a[k][i] for a, i in zip(source, ids, strict=True)]) for k in range(3))
        prediction = net(x, g)
        loss = (F.binary_cross_entropy_with_logits(prediction, y[:, None].expand_as(prediction), reduction="none")
                * prediction.new_tensor([1., .25])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % EVERY == 0:
            trace.append(float(loss.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(p)
            beat(phase="fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), identity=ident, trace=trace)


@torch.inference_mode()
def predict(net, data):
    cv, q = data["cv"], data["q"]
    add, joint = cv.copy(), cv.copy()
    scores = np.full(len(cv), -1.)
    unary = np.full(len(cv), -1.)
    for ix, p in menus(data):
        result = []
        for k in range(0, len(p["opts"]), 256):
            x, g = encode(p, p["opts"][k:k+256])
            result.append(net(torch.from_numpy(x), torch.from_numpy(g)).sigmoid().numpy()[:, :, 0])
        v = np.concatenate(result)
        a, j = v.argmax(0)
        add[ix], joint[ix] = paths_for(p, p["opts"][[a, j]])
        unary[ix], scores[ix] = v[a, 0], v[j, 1]
        for path in (add[ix], joint[ix]):
            rr = run.turn.turn.peak(path-cv[ix])
            np.testing.assert_allclose(np.sort(rr[rr > 0.]), np.sort(p["radius"]), rtol=1e-8, atol=1e-8)
    return dict(q=q, cv=cv, ols4=data["ols"], forced_add=add, forced_joint=joint,
                joint_probability=scores, additive_probability=unary)


def release(paths, threshold):
    mask = paths["joint_probability"] >= threshold if threshold is not None else np.zeros(len(paths["cv"]), bool)
    return {f"cal_{a}": np.where(mask[:, None, None], paths[f"forced_{a}"], paths["cv"]) for a in ("add", "joint")}


def assess(paths, data, cut):
    out = []
    for rec in np.unique(data["rec"]):
        at = data["rec"] == rec
        cv, ols, y, v = (data[k][at] for k in ("cv", "ols", "y", "v"))
        q = np.unique(data["q"][at], return_inverse=True)[1]
        w = 1./np.bincount(q)[q]
        mass = float(w@np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        arms, pairs = {}, {}
        for arm, path in paths.items():
            z = path[at]
            m = study.policy(z, cv, cv, y, v, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])
            g = [100.*float(w@audit.lower(k, z, y, v))/mass for k in (cv, ols)] if mass > 0. else None
            arms[arm] = dict(risk=m["recordings"]["r"], gain=g, ade=m["known_ADE_pixel"],
                fde=m["available_endpoint_FDE_diagnostic_pixel"], tail=m["p95_known_ADE_pixel"],
                hard=m["subsets"]["hard"]["gain_percent"])
        for variant in ("forced", "cal"):
            a, j = f"{variant}_add", f"{variant}_joint"
            if a in paths:
                pairs[variant] = run.turn.expanded.actual_contrast(cv, ols, paths[j][at], paths[a][at], y, v, q, cut)
        out.append(dict(record=str(rec), arms=arms, pairs=pairs))
    return out


def choose_threshold(paths, data, cut):
    table = []
    for t in THRESHOLDS:
        rows = assess(release(paths, t), data, cut)
        edited = [x for r in rows for x in r["arms"].values() if x["risk"]["selected"]]
        eligible = bool(edited and all(x["risk"]["supported"] and x["gain"] is not None
                                      and min(x["gain"]) > 0. for x in edited))
        table.append(dict(threshold=t, arm_edits=sum(x["risk"]["selected"] for x in edited),
                          edited_arm_views=len(edited), pass_arm_views=sum(x["risk"]["supported"] for x in edited),
                          eligible=eligible))
    feasible = [a for a in table if a["eligible"]]
    best = max(feasible, key=lambda a: (a["arm_edits"], a["threshold"])) if feasible else None
    return best["threshold"] if best else None, table


def freeze(key, cohorts):
    src = run.turn.exp.loader(cohorts)
    inputs = {s: causal(cohorts[s]) for s in SITES}
    fits, refs, calibration = {}, {}, {}
    for held in SITES:
        cal, train = roles(held)
        cut = np.quantile(np.concatenate([src[s]["e"] for s in train]), [.25, .75])
        donors, binding = {}, {}
        for donor in train:
            d = src[donor]
            local_cut = float(np.quantile(np.concatenate([src[s]["e"] for s in train if s != donor]), .25))
            ex = examples(inputs[donor], d["y"], d["v"], local_cut)
            if ex is not None:
                donors[donor] = ex
                binding[donor] = dict(source=cohorts[donor], cut=local_cut,
                                     strict=int(ex[-1][:, 0].sum().item()), supported=int(ex[-1][:, 1].sum().item()))
        for seed in SEEDS:
            name = f"{held}_{seed}"
            path = LOCAL/f"{name}_freeze.json"
            if not path.exists():
                net, fit_info = fit(key, held, cal, seed, donors, binding)
                prior = predict(net, inputs[cal])
                threshold, table = choose_threshold(prior, src[cal], cut)
                future = predict(net, inputs[held])
                future.update(release(future, threshold))
                forecast = run.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", future)
                base.immutable(path, dict(registration=key, fit=fit_info, forecast=forecast,
                    calibration=dict(site=cal, cut=cut.tolist(), train=train, threshold=threshold, table=table)))
            f = study.read(path)
            ident = f["fit"]["identity"]
            if (f["registration"] != key or ident["held"] != held or ident["calibration"] != cal
                    or not set(ident["donors"]) <= set(train)):
                raise ValueError("Calibration/held role crossed in stored forecast")
            study.checked(f["fit"]["checkpoint"])
            study.checked(f["forecast"])
            fits[name], refs[name], calibration[name] = f["fit"], f["forecast"], f["calibration"]
            beat(phase="freeze", completed=len(refs), total=len(SITES)*len(SEEDS), fold=name)
            print(name, calibration[name]["threshold"], flush=True)
    base.immutable(OUT/"freeze.json", dict(registration=key, fits=fits, forecasts=refs,
        calibration=calibration, held_targets_used=False))


def rank_audit(p, d, cut):
    outcomes = []
    for ix in run.slate.chunks(d["q"]):
        s = p["joint_probability"][ix][0]
        if s < 0.:
            continue
        x = dict(c=d["cv"][ix], ols=d["ols"][ix])
        truth = labels(x, p["forced_joint"][None, ix], d["y"][ix], d["v"][ix], cut)[0, 0]
        outcomes.append((s, truth))
    if not outcomes:
        return dict(queries=0, good=0, auc=None)
    prob, truth = np.asarray(outcomes).T
    hit = truth > 0.
    n = int(hit.sum())
    rank = rankdata(prob)
    return dict(queries=len(prob), good=n, auc=float((rank[hit].sum()-n*(n+1)/2)/(n*(len(hit)-n)))
                if 0 < n < len(hit) else None)


def summarize(rows):
    results = {}
    for arm in ("cv", "ols4", *ARMS):
        all_ = [r["arms"][arm] for r in rows]
        active = [a for a in all_ if a["risk"]["selected"]]
        results[arm] = dict(edits=sum(a["risk"]["selected"] for a in active), edited=len(active),
            risk_pass=sum(a["risk"]["supported"] for a in active),
            easy_undefined=sum(a["risk"]["completion_easy_risk_upper"] is None for a in active),
            easy_over_2=sum((a["risk"]["known_easy_net_degradation"] or 0.) > .02 for a in active),
            gain=run.turn.effects.site_means([(r["record"], r["arms"][arm]["gain"]) for r in rows if r["arms"][arm]["gain"] is not None]),
            **{k: float(np.mean(x)) if (x := [r[k] for r in all_ if r[k] is not None]) else None for k in ("ade","fde","tail","hard")})
    for kind in ("forced", "cal"):
        items = [dict(record=r["record"], effect=r["pairs"][kind]["pair_native"], qualified=r["pairs"][kind]["qualified"])
                 for r in rows if r["pairs"][kind] is not None]
        results[kind] = run.turn.effects.summary(items)
    return results


def score(key, cohorts):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["held_targets_used"] or len(f["forecasts"]) != len(SITES)*len(SEEDS):
        raise ValueError("Require every held forecast frozen before scoring")
    rows, sites, ranking = [], {}, {}
    data = run.turn.exp.loader(cohorts)
    for s in SITES:
        ds, local = data[s], []
        for seed in SEEDS:
            name = f"{s}_{seed}"
            with np.load(study.checked(f["forecasts"][name]), allow_pickle=False) as a:
                for k, x in (("q", ds["q"]), ("cv", ds["cv"]), ("ols4", ds["ols"])):
                    np.testing.assert_array_equal(a[k], x)
                cut = f["calibration"][name]["cut"]
                local.extend(assess({k: a[k] for k in ("cv", "ols4", *ARMS)}, ds, cut))
                ranking[name] = rank_audit(a, ds, cut[0])
            beat(phase="score", fold=name)
        sites[s] = summarize(local)
        rows.extend({**r, "record": f"{s}:{r['record']}"} for r in local)
    total = summarize(rows)
    for arm in ("cv", "ols4", *ARMS):
        total[arm]["location_ci_gain_pp"] = run.turn.effects.interval([s[arm]["gain"] for s in sites.values()])
    for k in ("forced", "cal"):
        total[k]["location_ci_pair_native"] = run.turn.effects.interval([s[k]["site_mean_native"] for s in sites.values()])
        if total[f"{k}_add"]["edits"] != total[f"{k}_joint"]["edits"]:
            raise ValueError("Matched edit count differs")
    chosen = {k: a["threshold"] for k, a in f["calibration"].items()}
    base.immutable(OUT/"summary.json", dict(registration=key, freeze=study.artifact(p), totals=total, sites=sites,
        calibrated_thresholds=chosen, ranking=ranking, fits=len(f["fits"]), views=len(rows),
        calibration_locations={k: a["site"] for k, a in f["calibration"].items()},
        training_support={k: a["identity"]["donors"] for k, a in f["fits"].items()},
        independent_confirmation=False, reserved_outcomes_opened=False, original_2pct=True, result_source="fresh_run"))
    print(study.json.dumps(dict(totals=total, calibrated=Counter(chosen.values()).most_common()), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts = register()
        (freeze if phase == "freeze" else score)(key, cohorts)
        beat(phase=phase+"_complete")


if __name__ == "__main__":
    main()
