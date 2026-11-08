"""Replay frozen OpenDD shape refiners on the previously exposed TUMDOT roster."""

import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_finite_guard as fit
from src.evaluation.m3w_opendd_location import paired
from src.world_model import m3w_ols_replacement as motion

parent, run, np, torch = fit.parent, fit.run, fit.np, fit.torch
source = parent.old.frozen
OUT = fit.OUT.parent / "tumdot_matched_finite_transfer_v1"
PRIVATE = fit.PRIVATE.parent / OUT.name
ARMS = ("incumbent", "hindcast", "previous_shape", "tangent", "finite_guard", "curvature_guard")
PAIRS = (("tangent", "incumbent"), ("tangent", "hindcast"), ("tangent", "previous_shape"),
         ("finite_guard", "tangent"), ("finite_guard", "hindcast"), ("curvature_guard", "finite_guard"))
FRACTIONS = {"half": .5, "quarter": .25}


def register():
    key, tangent_key, _, prior_shape, warm = fit.register()
    source_key = source.register()
    prior_read = source.PUBLIC / "analysis.json"
    refit_read = fit.OUT / "analysis.json"
    a, b = run.read(prior_read), run.read(refit_read)
    if (a["registration"] != source_key or a["formal_confirmation"] or a["dronecrowd_confirmation_access"]
            or b["registration"] != key or b["confirmation_access"]):
        raise ValueError("Require exposed transfer diagnostics and the original training lineage")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), cohort=run.artifact(prior_read),
        original_files=run.artifact(source.PUBLIC / "registration.json"), finite=run.artifact(refit_read),
        original_sampling=[8, 12, 12], no_refit_or_selection=True, fractions=FRACTIONS,
        radius_match="per-row peak of original fixed ensemble times fraction",
        ensemble="one refiner per matching OpenDD producer, then equal-model world average",
        seeds=parent.job.SEEDS, original_2pct=True, confirmation_access=False, deployment=False)))
    models = {Path(p["path"]).stem: p for p in b["fits"]}
    return run.sha(f), key, tangent_key, prior_shape, warm, models, a


def frozen_models(key, tangent_key, prior, warm, recent):
    seeds = parent.job.SEEDS
    producer = source.load_networks(torch, parent.producer)
    result = {}
    for seed in seeds:
        result[seed] = []
        for i, site in enumerate(source.SITES):
            old = prior[f"{site}/{seed}/easy_guard"]
            f = parent.old.PRIVATE / f"{site}_{seed}_easy_guard.pt"
            if run.sha(f) != old["sha256"]:
                raise ValueError("Earlier shape checkpoint hash changed")
            prev = torch.load(f, weights_only=True, map_location="cpu")
            if prev["identity"] != old["identity"]:
                raise ValueError("Earlier shape checkpoint roles changed")
            width = prev["model"]["0.weight"].shape[1]
            p = parent.producer.network(width)
            p.load_state_dict(prev["model"])
            t, _ = fit.load_warm(warm[f"{site}_{seed}_tangent"], tangent_key, site, seed, width)
            models = dict(previous_shape=p.eval(), tangent=t)
            for arm in ARMS[-2:]:
                f = recent[f"{site}_{seed}_{arm}"]
                z = torch.load(run.checked(f), map_location="cpu", weights_only=True)
                if z["identity"] != dict(key=key, site=site, seed=seed, arm=arm,
                                           warm=warm[f"{site}_{seed}_tangent"]) or z["step"] != fit.STEPS:
                    raise ValueError("Guard checkpoint did not come from the registered OpenDD continuation")
                net = parent.model.CVTangent(width)
                net.load_state_dict(z["model"])
                models[arm] = net.eval()
            result[seed].append((producer[seed][i], models))
    return result


def matched_paths(cv, anchor, deltas, fraction):
    """The reference budget, not the candidate, defines the edit roster."""
    radius = fraction*np.linalg.norm(anchor-cv, axis=-1).max(1)
    on = radius > 0.
    values = {}
    for k, v in deltas.items():
        result = cv+parent.shape.at_radius(v, radius)
        np.testing.assert_array_equal((result != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(result-cv, axis=-1).max(1), radius, atol=1e-13, rtol=1e-7)
        values[k] = result
    return values, on


@torch.inference_mode()
def forecast(h, models):
    cv, ols, hindcast, eligible = source.motion_bank(h, motion)
    x, rot, scale = parent.producer.design(h, np.full((len(h), 2, 2, 8, 2), np.nan))
    x = torch.from_numpy(x)
    d = dict(cv=cv, ols=ols, old=hindcast, legal=eligible, x=x, peer=x, rot=rot, scale=scale)
    direct = []
    variants = {k: [] for k in ARMS[2:]}
    for net, heads in models:
        local = parent.producer.decode(net(x)).numpy().astype(float)
        direct.append(local)
        p = parent.old.candidate(d, net, torch, parent.producer)
        on = p["r"] > 0.
        for k, head in heads.items():
            value = np.zeros_like(cv)
            if k == "previous_shape":
                value[on] = parent.shape.shape(head(p["x"][on]), p["unit"][on]).numpy()
            else:
                value[on] = head(p["x"][on], p["unit"][on], torch.ones(int(on.sum())))[0].numpy()
            world = value @ rot.transpose(0, 2, 1)
            variants[k].append(parent.shape.at_radius(world, p["r"]))
    blend, historical_hindcast = source.blend(cv,
        (np.mean(direct, axis=0)*scale[:, None, None]) @ rot.transpose(0, 2, 1), hindcast, eligible, .05)
    deltas = {"incumbent": blend-cv, "hindcast": historical_hindcast-cv,
              **{k: np.mean(v, axis=0) for k, v in variants.items()}}
    return cv, ols, blend, historical_hindcast, deltas


def score(rec, y, valid, cv, ols, paths, support, cuts):
    base = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    values = {}
    for k, p in paths.items():
        s = source.score(p, cv, ols, y, valid, rec, cuts)
        values[k] = dict(s, edits=int(support.sum()))
    direct = {f"{a}_vs_{b}": float(100*paired(paths[b], paths[a], y, valid)[1].sum()/base)
              for a, b in PAIRS}
    return dict(policies=values, direct=direct)


def summarize(records, part):
    flat = [v[part] for r in records for v in r["seeds"]]
    names = flat[0]["policies"]
    policies = {}
    for k in names:
        rows = [r["policies"][k] for r in flat]
        policies[k] = dict(lower_vs_cv=float(np.mean([r["lower_pp"]["cv"] for r in rows])),
            lower_vs_ols=float(np.mean([r["lower_pp"]["ols4"] for r in rows])),
            passes=sum(r["risk"]["supported"] for r in rows),
            undefined_easy=sum(r["risk"]["completion_easy_risk_upper"] is None for r in rows),
            absent_whole_easy=sum(r["risk"]["known_easy_net_degradation"] is None for r in rows),
            max_easy_pct=max((100*r["risk"]["known_easy_net_degradation"] for r in rows
                              if r["risk"]["known_easy_net_degradation"] is not None), default=None))
    direct = {k: float(np.mean([r["direct"][k] for r in flat])) for k in flat[0]["direct"]}
    return dict(policies=policies, direct=direct, views=len(flat))


def execute(reg, key, tangent_key, prior, warm, fitted, historical, beat):
    models = frozen_models(key, tangent_key, prior, warm, fitted)
    cuts = source.donor_cuts(motion)
    np.testing.assert_array_equal(cuts, historical["open_dd_positive_complete_cv_cuts"])
    data = []
    for rec, (name, _) in source.ROSTER.items():
        f = PRIVATE / f"{rec}.json"
        if not f.exists():
            tracks, _, _ = source.read_tracks(source.SOURCE / name)
            h, y, starts = source.windows(tracks)
            valid = np.isfinite(y).all(2)
            old = historical["recorded_files"][rec]
            if (len(h), len(starts), int(valid.all(1).sum()), int((~valid.any(1)).sum())) != (
                    old["targets"], old["queries"], old["complete"], old["unknown"]):
                raise ValueError("Previously exposed roster changed")
            values = []
            for i, seed in enumerate(parent.job.SEEDS):
                cv, ols, base, original_hindcast, deltas = forecast(h, models[seed])
                old_cell = old["views"][i]
                replay = source.score(base, cv, ols, y, valid, rec, cuts, original_hindcast)
                if old_cell["seed"] != seed or old_cell["arms"]["blend05"] != replay:
                    raise ValueError("Original transfer did not reproduce exactly")
                value = dict(seed=seed)
                for part, fraction in FRACTIONS.items():
                    p, support = matched_paths(cv, base, deltas, fraction)
                    value[part] = score(rec, y, valid, cv, ols, p, support, cuts)
                values.append(value)
            run.once(f, run.stable(dict(registration=reg, recording=rec, rows=len(h), queries=len(starts),
                                        complete=int(valid.all(1).sum()), wholly_missing=int((~valid.any(1)).sum()),
                                        seeds=values)))
        row = run.read(f)
        if row["registration"] != reg or row["recording"] != rec:
            raise ValueError("Transfer checkpoint mismatch")
        data.append(row)
        beat(recording=rec, complete=len(data))
    if len(data) != 9 or sum(r["rows"] for r in data) != 1456:
        raise ValueError("Incomplete original TUMDOT roster")
    summary = {p: summarize(data, p) for p in FRACTIONS}
    by_day = {day: {p: summarize([r for r in data if r["recording"].endswith(day)], p) for p in FRACTIONS}
              for day in ("d1", "d2", "d3")}
    run.once(OUT / "analysis.json", run.stable(dict(registration=reg, summary=summary, by_day=by_day,
        recordings=data, campaigns=1, no_new_training=True, no_target_tuning=True,
        historical_replay_exact=True, source_is_previously_exposed=True, confirmation_access=False,
        deployment=False)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            v.update(pid=os.getpid(), seconds=round(time.monotonic()-start, 2))
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            print(v, flush=True)

        execute(*register(), beat)
        print(f"Complete: {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
