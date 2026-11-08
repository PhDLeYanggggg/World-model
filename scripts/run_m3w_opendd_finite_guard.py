"""Warm-start a matched continuation with recording-normalized finite harm."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_cv_tangent as parent
from src.world_model import m3w_cv_finite_guard as loss

run, torch, np = parent.run, parent.torch, parent.np
OUT = parent.OUT.parent / "opendd_finite_guard_v1"
PRIVATE = parent.PRIVATE.parent / OUT.name
ARMS = ("continued", "finite_guard", "curvature_guard")
STEPS = 400
PAIRS = (("continued", "tangent"), ("finite_guard", "continued"), ("curvature_guard", "finite_guard"),
         ("finite_guard", "half"), ("curvature_guard", "half"),
         ("finite_guard", "previous_shape"), ("curvature_guard", "previous_shape"))


def register():
    key, donors, previous = parent.register()
    p = parent.OUT / "analysis.json"
    old = run.read(p)
    if old["registration"] != key or old["confirmation_access"]:
        raise ValueError("Require the verified CV-tangent source-held experiment")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), loss=run.artifact(Path(loss.__file__)),
        tangent=run.artifact(p), parent=key, arms=list(ARMS), steps=STEPS, threads=4, workers=0,
        lr=.0005, seed_order=parent.job.SEEDS, checkpoint_every=100,
        matched_half_radius_primary=True, quarter_radius_diagnostic=True, original_2pct=True,
        exact_recipients=True, confirmation_access=False, calibration_access=False)))
    return run.sha(f), key, donors, previous, {Path(a["path"]).stem: a for a in old["models"]}


def fit_data(sources, candidates, held):
    sites = [s for s in sorted(sources) if s != held]
    cuts = parent.old.base.cuts_for([sources[s] for s in sites])
    features, groups, offset = [], [], 0
    for s in sites:
        d, p = sources[s], candidates[s]
        cv = ((d["cv"]@d["rot"])/d["scale"][:, None, None]).astype("float32")
        y = ((d["y"]@d["rot"])/d["scale"][:, None, None]).astype("float32")
        radius = (parent.RADIUS*p["ratio"]).to(torch.float32)
        easy = (d["cv_ade"] > 0.) & (d["cv_ade"] <= cuts[0])
        legal = p["r"] > 0.
        norm = np.ones(len(easy))
        for rec in set(d["recordings"]):
            at = (d["recordings"] == rec) & easy & legal
            if at.any():
                norm[at] = np.clip(d["scale"][at]/d["cv_ade"][at].mean(), .25, 8.)
        features.append((p["x"], p["unit"], radius, torch.from_numpy(cv), torch.from_numpy(y),
            torch.from_numpy(d["valid"]), torch.from_numpy(easy), torch.from_numpy(norm.astype("float32"))))
        groups.append(p["pool"]+offset)
        offset += len(cv)
    return list(map(torch.cat, zip(*features, strict=True))), groups, cuts


def load_warm(warm, old_key, site, seed, width):
    state = torch.load(run.checked(warm), map_location="cpu", weights_only=True)
    ident = state["identity"]
    if (ident["key"], ident["held"], ident["seed"], ident["arm"], state["step"]) != (
            old_key, site, seed, "tangent", parent.STEPS) or site in ident["fit"]:
        raise ValueError("The tangent checkpoint must exclude the scored location")
    net = parent.model.CVTangent(width)
    net.load_state_dict(state["model"])
    return net.eval(), state["optimizer"]


def fit(key, old_key, warm, site, seed, arm, data, groups, beat, replay):
    x = data[0]
    net, state = load_warm(warm, old_key, site, seed, x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.0005)
    opt.load_state_dict(state)
    for g in opt.param_groups:
        g["lr"] = .0005
    rng = torch.Generator().manual_seed(seed+17831)
    f = PRIVATE / f"{site}_{seed}_{arm}.pt"
    identity = dict(key=key, site=site, seed=seed, arm=arm, warm=warm)
    step, trace = 0, []
    if f.exists():
        state = torch.load(f, map_location="cpu", weights_only=True)
        if state["identity"] != identity or not 0 < state["step"] <= STEPS:
            raise ValueError("Continuation receipt changed")
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        step, trace = state["step"], state["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot continue training")
    net.train()
    for step in range(step+1, STEPS+1):
        i = parent.anchor.study.direction.timing.draw_batch(groups, 128, rng)
        x, unit, radius, cv, y, valid, easy, norm = (a[i] for a in data)
        pred, logit = net(x, unit, radius)
        incumbent = unit*radius[:, None, None]
        value = loss.objective(pred, logit, incumbent, cv, y, valid, easy, norm, arm)
        opt.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0:
            trace.append(float(value.detach()))
            tmp = f.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, model=net.state_dict(), optimizer=opt.state_dict(),
                rng=rng.get_state(), trace=trace), tmp)
            tmp.replace(f)
            beat(site=site, seed=seed, arm=arm, step=step, loss=trace[-1])
    return net.eval(), run.artifact(f)


def score(d, p, support, cuts, fraction):
    cv = d["cv"]
    paths = {k: cv+fraction*(v-cv) for k, v in p.items() if k not in ("guard", "matched_half", "matched_previous")}
    policies, records = {}, {}
    for k, a in paths.items():
        if k != "cv":
            np.testing.assert_array_equal((a != cv).any((1, 2)), support)
        v = parent.policy(a, cv, cv, d["y"], d["valid"], d["recordings"], easy_cut=cuts[0], hard_cut=cuts[1])
        rr = v["recordings"]
        records[k] = {rec: {"pass": r["supported"],
            "undefined": r["completion_easy_risk_upper"] is None,
            "whole_easy_absent": r["known_easy_net_degradation"] is None,
            "all_violation": r["completion_all_risk_upper"] is not None and r["completion_all_risk_upper"] > .02,
            "easy_violation": r["completion_easy_risk_upper"] is not None and r["completion_easy_risk_upper"] > .02,
            "net_easy_violation": r["known_easy_net_degradation"] is not None and r["known_easy_net_degradation"] > .02}
            for rec, r in rr.items()}
        policies[k] = dict(vs_cv=parent.old.base.paired_summary(d, a, cv, support),
            worst_easy_pct=max((100*r["known_easy_net_degradation"] for r in rr.values()
                                if r["known_easy_net_degradation"] is not None), default=None),
            hard_gain=v["subsets"]["hard"]["gain_percent"],
            p95_ade=v["p95_known_ADE_pixel"])
    direct = {f"{a}_vs_{b}": parent.old.base.paired_summary(d, paths[a], paths[b], support) for a, b in PAIRS}
    return dict(policies=policies, records=records, direct=direct,
                pairs=parent.old.pair_errors(d, paths)["mean_relative_vector_error"])


def summary(views, key):
    table = [r[key] for rr in views.values() for r in rr]
    draws = np.random.default_rng(817).integers(len(views), size=(3000, len(views)))
    def ci(fn):
        x = np.array([np.mean([fn(r[key]) for r in v]) for v in views.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    policies, transitions = {}, {}
    for k in table[0]["policies"]:
        block = [r for v in table for r in v["records"][k].values()]
        policies[k] = {f: sum(r[f] for r in block) for f in block[0]}
        policies[k]["views"] = len(block)
        policies[k]["vs_cv"] = ci(lambda r: r["policies"][k]["vs_cv"]["lower_pp"])
        policies[k]["worst_easy_pct"] = max(v["policies"][k]["worst_easy_pct"] for v in table)
        flags = [(v["records"]["tangent"][rec]["pass"], v["records"][k][rec]["pass"])
                 for v in table for rec in v["records"]["tangent"]]
        transitions[k] = dict(gained=sum(not a and b for a, b in flags), lost=sum(a and not b for a, b in flags))
    return dict(policies=policies, transitions_vs_parent=transitions,
        direct={k: ci(lambda r: r["direct"][k]["lower_pp"]) for k in table[0]["direct"]},
        pair_relative_vs_continued={k: ci(lambda r: r["pairs"][k]-r["pairs"]["continued"])
                                   for k in ARMS[1:]})


def execute(key, old_key, donors, previous, warm, beat, replay):
    old = parent.old
    outer, inner = old.load_producers(torch, parent.producer, donors)
    sources = {s: old.site_data(s, torch, parent.producer, parent.shape) for s in old.SITES}
    rows, fits = {}, []
    for pos, held in enumerate(old.SITES):
        rows[held] = []
        d = sources[held]
        for seed in parent.job.SEEDS:
            prior = warm[f"{held}_{seed}_tangent"]
            c = {s: old.candidate(sources[s], inner[tuple(sorted((s, held))), seed], torch, parent.producer)
                 for s in old.SITES if s != held}
            data, groups, cuts = fit_data(sources, c, held)
            cells = {k: fit(key, old_key, prior, held, seed, k, data, groups, beat, replay) for k in ARMS}
            fits.extend(v[1] for v in cells.values())
            candidate = old.candidate(d, outer[seed][pos], torch, parent.producer)
            net, _ = load_warm(prior, old_key, held, seed, candidate["x"].shape[1])
            ready = {"tangent": (net, cuts, prior), **{k: (v[0], cuts, v[1]) for k, v in cells.items()}}
            ready["finite"] = ready["continued"]  # Common scoring helper expects both parent slots.
            p, radius = parent.paths(d, candidate, ready, held, seed, previous)
            p.pop("finite")
            for k in ARMS:
                np.testing.assert_allclose(np.linalg.norm(p[k]-d["cv"], axis=-1).max(1),
                                           radius, rtol=1e-7, atol=1e-14)
            rows[held].append(dict(seed=seed, edits=int((radius > 0.).sum()),
                half=score(d, p, radius > 0., cuts, 1.), quarter=score(d, p, radius > 0., cuts, .5)))
    if len(fits) != 63:
        raise ValueError("Require three complete matched continuation arms")
    reports = {k: summary(rows, k) for k in ("half", "quarter")}
    compact = {s: [dict(seed=v["seed"], edits=v["edits"], **{
        k: {f: x for f, x in v[k].items() if f != "records"} for k in reports}) for v in rr]
        for s, rr in rows.items()}
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, fits=fits, summary=reports,
        edits=sum(r["edits"] for v in rows.values() for r in v), by_location=compact,
        confirmation_access=False, deployment=False, original_2pct_unchanged=True)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    args = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, old_key, donors, previous, warm = register()

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 2))
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            if v["step"] == STEPS:
                print(v, flush=True)

        execute(key, old_key, donors, previous, warm, beat, args.replay)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
