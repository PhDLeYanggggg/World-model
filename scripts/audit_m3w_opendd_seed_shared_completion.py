"""Audit frozen OpenDD context results using one unknown location across seeds."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_context_exchange as ctx
from src.evaluation.m3w_seed_shared_completion import seed_shared_completion


PUBLIC = ctx.PUBLIC.parent/"opendd_seed_shared_completion_v1"
PRIVATE = ctx.PRIVATE.parent/PUBLIC.name
PAIRS = (("traffic", "half"), ("traffic", "pedestrian"), ("traffic", "shuffled_traffic"))


def register():
    parent, donors, inputs = ctx.register()
    frozen = json.loads((ctx.PUBLIC/"head_freeze.json").read_text())
    old = json.loads((ctx.PUBLIC/"analysis.json").read_text())
    replay = json.loads((ctx.PUBLIC/"replay_execution.json").read_text())
    digest = ctx.prior.sha(ctx.PUBLIC/"analysis.json")
    if (old["registration"] != parent or frozen["registration"] != parent or
            replay["analysis_sha256"] != digest or not replay["fresh_exact_replay"] or
            replay["new_heads_this_execution"] != 0 or old["formal_confirmation"]):
        raise ValueError("Require the replay-verified exposed parent")
    spec = PUBLIC/"registration.json"
    ctx.prior.write_json(spec, dict(
        parent=parent, parent_analysis_sha256=digest,
        heads_sha256=ctx.prior.sha(ctx.PUBLIC/"head_freeze.json"),
        code_sha256=ctx.prior.sha(Path(__file__)),
        math_sha256=ctx.prior.sha(Path(seed_shared_completion.__code__.co_filename)),
        pairs=PAIRS, sites=ctx.SITES, seeds=ctx.SEEDS, resampling="site_bootstrap_after_seed_mean",
        bootstrap_seed=817, bootstrap_draws=3000, normalized_by="fixed_observed_CV_mass",
        old_risk_unchanged=True, no_fits=True, independent_confirmation=False,
    ), immutable=True)
    return ctx.prior.sha(spec), parent, donors, inputs, frozen, old


def build_paths(site, pos, d, parent, frozen, outer, torch, model, beat):
    paths = {a: [] for a in ("half", "traffic", "pedestrian", "shuffled_traffic")}
    for seed in ctx.SEEDS:
        c = ctx.base.shape_run.candidate(d, outer[seed][pos], torch, model)
        features = ctx.features(d, c["inc"])
        paths["half"].append(d["cv"]+.5*(c["inc"]-d["cv"]))
        for arm in ("traffic", "pedestrian", "shuffled_traffic"):
            head, _, fits = ctx.fit(site, seed, arm, parent, None, frozen["heads"][f"{site}/{seed}/{arm}"])
            if fits:
                raise ValueError("Frozen analysis must not fit")
            x, q, info, norm = ctx.direct.actions(d, c, features[arm])
            estimates = head.predict(x)
            estimates[:, :4] *= norm[:, None]
            v, _ = ctx.direct.apply(d["cv"], c["inc"], d["offsets"], q, info, estimates)
            paths[arm].append(v["direct_guard"])
        beat(stage="seed_read", site=site, seed=seed)
    return {k: np.asarray(v, dtype=np.float64) for k, v in paths.items()}


def score_site(d, paths, old_site):
    # No policy may change the target roster or the fixed normalization.
    mass = float(np.nansum(np.linalg.norm(d["cv"]-d["y"], axis=-1)))
    if mass <= 0:
        raise ValueError("Observed CV mass is not positive")
    result = {}
    for arm, ref in PAIRS:
        key = f"{arm}_vs_{ref}"
        parent_key = "traffic_vs_shuffled" if ref == "shuffled_traffic" else key
        actual = [ctx.paired(d, p, q) for p, q in zip(paths[arm], paths[ref], strict=True)]
        np.testing.assert_allclose(actual, [r["paired"][parent_key] for r in old_site], atol=2e-9, rtol=2e-7)
        values = seed_shared_completion(paths[ref], paths[arm], d["y"], d["valid"], d["cv"])
        values = {k: v if k.endswith("_steps") else 100*v/mass for k, v in values.items()}
        np.testing.assert_allclose(values["diagonal_lower"], np.mean(actual), atol=2e-9, rtol=2e-7)
        result[key] = values
    return result


def summarize(results):
    draw = np.random.default_rng(817).integers(len(ctx.SITES), size=(3000, len(ctx.SITES)))
    out = {}
    for arm, ref in PAIRS:
        name = f"{arm}_vs_{ref}"
        rows = [results[s][name] for s in ctx.SITES]
        fields = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows])
            fields[k] = int(v.sum()) if k.endswith("_steps") else dict(
                point=float(v.mean()), ci95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                positive_locations=int((v > 0).sum()))
        fields["witness_sign_reversal"] = fields["template_low"]["point"] < 0 < fields["template_high"]["point"]
        out[name] = fields
    return out


def run(binding, parent, donors, inputs, frozen, old):
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    outer, _ = ctx.base.shape_run.load_producers(torch, model, donors)
    results = {}

    def beat(**value):
        ctx.prior.write_json(PRIVATE/"heartbeat.json", dict(pid=os.getpid(), **value))
        print(value, flush=True)

    for pos, site in enumerate(ctx.SITES):
        d = ctx.site_data(site, torch, model, shape, inputs[site])
        paths = build_paths(site, pos, d, parent, frozen, outer, torch, model, beat)
        results[site] = score_site(d, paths, old["by_location"][site])
    out = PUBLIC/"analysis.json"
    ctx.prior.write_json(out, dict(
        registration_sha256=binding, by_site=results, contrasts=summarize(results),
        original_per_seed_lower_scores_reproduced=True, fit_count=0,
        role="opendd_exposed_development_sensitivity", original_risk_unchanged=True,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False,
    ), immutable=True)
    beat(stage="complete", analysis_sha256=ctx.prior.sha(out))


def main():
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise SystemExit("Use native arm64 .venv-pytorch")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "score"))
    args = cli.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = register()
        if args.phase == "score":
            run(*spec)
        else:
            print(spec[0])


if __name__ == "__main__":
    main()
