"""Fit and score causal optional bridge policies on held OpenDD locations."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use the native arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_optional_reference_bridge as capacity
from scripts import run_m3w_opendd_map_path_value as old
from src.world_model import m3w_reference_bridge_policy as model

run, np, torch, source, transfer = old.run, old.np, old.torch, old.source, old.transfer
OUT = old.OUT.parent / "reference_bridge_policy_v1"
PRIVATE = old.PRIVATE.parent / OUT.name
STEPS, BATCH = 700, 128
ARMS = ("cv", "ols4", "tangent", "joint", "independent", "matched_independent", "matched_motion", "bridge_only")


def register():
    key, frozen, _, _, donors = old.register()
    p = capacity.OUT / "readout.json"
    a = run.read(p)
    if not a["labels_hindsight_only"] or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require bound capacity findings, not a deployed policy")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        capacity=run.artifact(p), previous=key, only_opendd=True, steps=[8, 12], updates=STEPS, batch=BATCH,
        budget="up to two recipients and at least one CV fallback in each query",
        objective="two conservative gains plus predicted all/easy positive harm and reference mass",
        donor_bank="pair-excluded producer, refiner excludes held location", score_uses_original_cv_2pct=True,
        seeds=old.prior.SEEDS, threads=4, workers=0, no_held_tuning=True, confirmation_access=False)))
    return run.sha(f), frozen, donors


def prepare(d, prediction):
    cv, ols, anchor, _, delta = prediction
    paths, on = transfer.matched_paths(cv, anchor, delta, .5)
    base = np.stack([paths[k] for k in transfer.ARMS], axis=1)
    bank = capacity.menus(cv, ols, base, on)["union"]
    x, scale, legal, motion = model.features(d["h"], cv, ols, bank)
    return dict(cv=cv, ols=ols, bank=bank, tangent=paths["tangent"], x=x, scale=scale, legal=legal, motion=motion)


def fit(key, site, seed, donor_sites, data, beat, replay):
    file = PRIVATE / f"{site}_{seed}.pt"
    identity = dict(registration=key, held=site, seed=seed, donors=donor_sites)
    torch.manual_seed(seed)
    net = model.BridgeValue()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+8209)
    step, trace = 0, []
    if file.exists():
        s = torch.load(file, weights_only=True, map_location="cpu")
        if s["identity"] != identity or s["step"] not in range(100, STEPS+1, 100):
            raise ValueError("Bridge checkpoint changed training role")
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        rng.bit_generator.state = s["rng"]
        step, trace = s["step"], s["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot fit")
    for t in range(step+1, STEPS+1):
        j = rng.integers(len(data), size=BATCH)
        pick = [rng.choice(len(x[0]), size=int((j == k).sum())) for k, x in enumerate(data)]
        x, y, m = (torch.cat([data[k][i][ix] for k, ix in enumerate(pick)]) for i in range(3))
        loss = model.loss(net(x), y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = file.with_suffix(".pending")
            torch.save(dict(identity=identity, net=net.state_dict(), opt=opt.state_dict(), step=t, trace=trace,
                            rng=rng.bit_generator.state), tmp)
            tmp.replace(file)
            beat(phase="fit", site=site, seed=seed, step=t)
    return net.eval(), dict(identity=identity, checkpoint=run.artifact(file), trace=trace), STEPS-step


@torch.inference_mode()
def score(d, p, net, cuts):
    cv, ols, bank = (p[k] for k in ("cv", "ols", "bank"))
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    estimates = net(torch.from_numpy(p["x"])).numpy()*p["scale"][:, None, None]
    c = model.query_choices(estimates, p["legal"], q, p["motion"])
    arms = dict(cv=cv, ols4=ols, tangent=p["tangent"])
    for k, choice in c.items():
        at = np.flatnonzero(choice >= 0)
        path = cv.copy()
        path[at] = bank[at, choice[at]]
        np.testing.assert_array_equal((path != cv).any((1, 2)), choice >= 0)
        arms[k] = path
    count = np.bincount(q[c["joint"] >= 0], minlength=q.max()+1)
    for name in ("matched_independent", "matched_motion"):
        np.testing.assert_array_equal(np.bincount(q[c[name] >= 0], minlength=len(count)), count)
    if not (count <= np.minimum(2, np.bincount(q)-1)).all():
        raise ValueError("Unexpected query intervention budget")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()

    def utility(a, b):
        return float(100*transfer.paired(arms[b], arms[a], y, valid)[1].sum()/mass)

    scores = {}
    for k, path in arms.items():
        v = capacity.policy(path, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])
        rows = v["recordings"].values()
        paired_both = sum(r["supported"] and all(transfer.paired(ref[rec == name], path[rec == name],
            y[rec == name], valid[rec == name])[1].sum() > 0. for ref in (cv, ols))
            for name, r in v["recordings"].items())
        scores[k] = dict(vs_cv=utility(k, "cv"), vs_ols4=utility(k, "ols4"),
            risk_passes=sum(r["supported"] for r in rows), undefined=sum(r["completion_easy_risk_upper"] is None for r in rows),
            dual_positive_passes=paired_both, views=len(rows), edits=int((path != cv).any((1, 2)).sum()),
            available_ade=v["known_ADE_pixel"], available_fde=v["available_endpoint_FDE_diagnostic_pixel"],
            unknown_edits=sum(r["unknown_selected"] for r in rows))
    return dict(arms=scores, direct={f"joint_vs_{k}": utility("joint", k) for k in ("matched_independent", "matched_motion", "bridge_only", "tangent")},
        joint_nonempty_queries=int((count > 0).sum()), joint_two_edit_queries=int((count == 2).sum()),
        action_counts={k: np.bincount(v[v >= 0], minlength=bank.shape[1]).tolist() for k, v in c.items()})


def summarize(groups):
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in rows]) for rows in groups.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    a = [r for rows in groups.values() for r in rows]
    scores = {k: dict(vs_cv=ci(lambda r: r["arms"][k]["vs_cv"]), vs_ols4=ci(lambda r: r["arms"][k]["vs_ols4"]),
        **{f: sum(r["arms"][k][f] for r in a) for f in ("risk_passes", "undefined", "dual_positive_passes", "views", "edits", "unknown_edits")}) for k in ARMS}
    return dict(arms=scores, direct={k: ci(lambda r: r["direct"][k]) for k in a[0]["direct"]},
        joint_nonempty_queries=sum(r["joint_nonempty_queries"] for r in a),
        joint_two_edit_queries=sum(r["joint_two_edit_queries"] for r in a),
        action_counts={k: np.sum([v["action_counts"][k] for v in a], axis=0).tolist() for k in a[0]["action_counts"]})


def execute(key, frozen, donor_hashes, beat, replay, cache):
    nets = transfer.frozen_models(*frozen[:5])
    _, inner = transfer.parent.old.load_producers(torch, transfer.parent.producer, donor_hashes)
    d = {k: old.load_site(k) for k in source.SITES}
    heads, sites, updates = {}, {}, 0
    for j, held in enumerate(source.SITES):
        sites[held] = []
        train = [k for k in source.SITES if k != held]
        cut = np.quantile(np.concatenate([d[k]["cut_samples"] for k in train]), [.25, .75])
        for seed in old.prior.SEEDS:
            file = cache / f"{held}_{seed}.json"
            if not file.exists():
                outer, refiners = nets[seed][j]
                arrays = []
                for k in train:
                    a = prepare(d[k], transfer.forecast(d[k]["h"], [(inner[tuple(sorted((held, k))), seed], refiners)]))
                    y = model.targets(a["cv"], a["ols"], a["bank"], d[k]["y"], d[k]["valid"], cut[0], a["scale"])
                    on = a["legal"].any(1)
                    arrays.append(tuple(torch.from_numpy(v[on]) for v in (a["x"], y, a["legal"])))
                head, receipt, n = fit(key, held, seed, train, arrays, beat, replay)
                updates += n
                p = prepare(d[held], transfer.forecast(d[held]["h"], [(outer, refiners)]))
                r = score(d[held], p, head, cut)
                run.once(file, run.stable(dict(registration=key, held=held, seed=seed, head=receipt, **r)))
            a = run.read(file)
            if (a["registration"], a["held"], a["seed"]) != (key, held, seed):
                raise ValueError("Unexpected scoring checkpoint lineage")
            run.checked(a["head"]["checkpoint"])
            heads[f"{held}/{seed}"] = a["head"]
            sites[held].append({k: v for k, v in a.items() if k != "head"})
            beat(phase="held", site=held, seed=seed)
    if len(heads) != 21:
        raise ValueError("Missing one of the three-seed held folds")
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(f), summary=summarize(sites), by_site=sites,
        original_2pct=True, no_held_tuning=True, held_fully_excluded=True, confirmation_access=False, deployment=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] != "fit":
                print(row, flush=True)

        state = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as t:
                updates = execute(*state, beat, True, Path(t))
        else:
            updates = execute(*state, beat, False, PRIVATE)
        beat(phase="complete", updates=updates, replay=args.replay)


if __name__ == "__main__":
    main()
