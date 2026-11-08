"""Test whether observed-window replay predicts the matched future path ranking."""

import argparse
import fcntl
from functools import partial
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use the native arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_tumdot_matched_finite_transfer as prior
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_prefix_path_replay as model

run, np, torch, source = prior.run, prior.np, prior.torch, prior.source
OUT = prior.OUT.parent / "prefix_path_replay_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
ARMS, SEEDS = prior.ARMS, prior.parent.job.SEEDS
CONTROL = ("cv", "ols4", "hindcast", "tangent")
METHODS = ("linear_prefix", "padding_consensus")
REFERENCE = ARMS.index("hindcast")


def register():
    parent, *frozen = prior.register()
    a = prior.OUT / "analysis.json"
    before = run.read(a)
    if before["registration"] != parent or before["confirmation_access"] or not before["source_is_previously_exposed"]:
        raise ValueError("Require the exposed matched transfer and verified source checkpoints")
    p = OUT / "registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        original=run.artifact(a), fit_sources="location-excluded OpenDD producers; no new fits",
        evaluation="exposed OpenDD and TUMDOT only", prefixes=model.PREFIXES, padding=model.PADDING,
        obs_pred=[8, 12], fractions=prior.FRACTIONS, fallback="matched hindcast", matched_recipients=True,
        primary="padding_consensus_vs_hindcast_at_half_radius", no_future_inputs=True,
        original_2pct=True, no_fit_or_calibration=True, confirmation_access=False, seeds=SEEDS)))
    return run.sha(p), frozen, before


def infer(h, predict):
    cv, ols, inc, _, deltas = predict(h)
    replays = {m: model.replay(h, predict, prior.matched_paths, ARMS, m) for m in model.PADDING}
    choice = model.choose(replays, REFERENCE)
    outputs = {}
    for part, fraction in prior.FRACTIONS.items():
        paths, on = prior.matched_paths(cv, inc, deltas, fraction)
        bank = np.stack([paths[a] for a in ARMS], axis=1)
        values = dict(cv=cv, ols4=ols, hindcast=paths["hindcast"], tangent=paths["tangent"])
        for k, c in choice.items():
            pred = bank[np.arange(len(bank)), c]
            np.testing.assert_array_equal((pred != cv).any((1, 2)), on)
            np.testing.assert_allclose(np.linalg.norm(pred-cv, axis=-1).max(1),
                                       fraction*np.linalg.norm(inc-cv, axis=-1).max(1), atol=1e-13, rtol=1e-7)
            values[k] = pred
        outputs[part] = (values, bank, on)
    return outputs, replays, choice


def score(values, bank, on, replays, choice, y, valid, rec, cuts):
    cv = values["cv"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    if mass <= 0.:
        raise ValueError("Missing CV utility mass")
    complete = valid.all(1) & on
    actual = -np.linalg.norm(bank[complete]-y[complete, None], axis=-1).mean(2)
    ranks = {m: model.rank_counts(g[complete], actual, n[complete] > 0) for m, (g, n) in replays.items()}
    b = dict(rank=ranks, edits=int(on.sum()), complete_edited=int(complete.sum()),
             observed_prefix_support={k: np.bincount(n[on], minlength=5).tolist() for k, (_, n) in replays.items()},
             path_choices={k: np.bincount(v[on], minlength=len(ARMS)).tolist() for k, v in choice.items()})
    arms = {}
    for k, p in values.items():
        a = policy(p, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])
        risk = a["recordings"].values()
        easy = [r["completion_easy_risk_upper"] for r in risk if r["completion_easy_risk_upper"] is not None]
        arms[k] = dict(lower={name: float(100*prior.paired(values[name], p, y, valid)[1].sum()/mass)
                               for name in CONTROL}, pass_views=sum(r["supported"] for r in risk), views=len(risk),
                       undefined_easy=sum(r["completion_easy_risk_upper"] is None for r in risk),
                       worst_easy_risk_pct=100*max(easy) if easy else None,
                       ade=a["known_ADE_pixel"], fde=a["available_endpoint_FDE_diagnostic_pixel"])
    return dict(arms=arms, **b)


def compact(groups, part, bootstrap):
    cells = [c[part] for group in groups for c in group]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups))) if bootstrap else None

    def paired_summary(fn):
        a = np.array([np.mean([fn(c[part]) for c in group]) for group in groups])
        return dict(point=float(a.mean()), interval95=np.quantile(a[draws].mean(1), [.025, .975]).tolist()
                    if draws is not None else None)

    arms = {}
    for k in cells[0]["arms"]:
        values = [v["arms"][k] for v in cells]
        e = [v["worst_easy_risk_pct"] for v in values if v["worst_easy_risk_pct"] is not None]
        arms[k] = dict(lower={b: paired_summary(lambda c: c["arms"][k]["lower"][b]) for b in CONTROL},
            pass_views=sum(v["pass_views"] for v in values), views=sum(v["views"] for v in values),
            undefined_easy=sum(v["undefined_easy"] for v in values), worst_easy_risk_pct=max(e, default=None),
            ade=float(np.mean([v["ade"] for v in values])), fde=float(np.mean([v["fde"] for v in values])))
    rank = {}
    for mode in model.PADDING:
        v = [c["rank"][mode] for c in cells]
        rank[mode] = {k: sum(x[k] for x in v) for k in v[0]}
        # Rank correlation is descriptive on overlapping windows; report the group statistic separately.
        if all(c[part]["rank"][mode]["comparable"] > 0 for group in groups for c in group):
            rank[mode]["group_mean"] = paired_summary(lambda c: c["rank"][mode]["concordant"]/c["rank"][mode]["comparable"])
    return dict(arms=arms, rank=rank, edits=sum(c["edits"] for c in cells),
                observed_prefix_support={k: np.sum([c["observed_prefix_support"][k] for c in cells], axis=0).tolist()
                                         for k in model.PADDING},
                path_choices={k: np.sum([c["path_choices"][k] for c in cells], axis=0).tolist() for k in METHODS})


def execute(reg, frozen, previous, beat):
    nets = prior.frozen_models(*frozen[:5])
    open_dd = {}
    for site in source.SITES:
        p = source.prior.PRIVATE / f"{site}_cohort.npz"
        with np.load(p, allow_pickle=False) as z:
            h, y, rec = z["history"], z["future"], np.repeat(z["recording"], np.diff(z["offsets"]))
        cv, _, _, _ = prior.motion.rollout(h, 12)
        valid = np.isfinite(y).all(2)
        ade = np.linalg.norm(cv[valid.all(1)]-y[valid.all(1)], axis=-1).mean(1)
        open_dd[site] = (h, y, valid, rec, ade[ade > 0.])
    domains = {"opendd": {}, "tumdot": {}}
    for domain, items in (("opendd", source.SITES), ("tumdot", tuple(source.ROSTER))):
        for j, name in enumerate(items):
            p = PRIVATE / f"{domain}_{name}.json"
            if not p.exists():
                if domain == "opendd":
                    h, y, valid, rec, _ = open_dd[name]
                    cuts = np.quantile(np.concatenate([d[-1] for s, d in open_dd.items() if s != name]), [.25, .75])
                else:
                    file, _ = source.ROSTER[name]
                    tracks, _, _ = source.read_tracks(source.SOURCE / file)
                    h, y, _ = source.windows(tracks)
                    valid, rec = np.isfinite(y).all(2), np.full(len(y), name)
                    cuts = np.asarray(frozen[-1]["open_dd_positive_complete_cv_cuts"])
                rows = []
                for i, seed in enumerate(SEEDS):
                    member = [nets[seed][j]] if domain == "opendd" else nets[seed]
                    predictor = partial(prior.forecast, models=member)
                    preds, replays, choice = infer(h, predictor)
                    row = dict(seed=seed)
                    for part, (v, bank, on) in preds.items():
                        row[part] = score(v, bank, on, replays, choice, y, valid, rec, cuts)
                        if domain == "tumdot":
                            old = previous["recordings"][j]["seeds"][i][part]["policies"]
                            for k in ("hindcast", "tangent"):
                                if (row[part]["arms"][k]["lower"]["cv"] != old[k]["lower_pp"]["cv"]
                                        or row[part]["arms"][k]["pass_views"] != int(old[k]["risk"]["supported"])):
                                    raise ValueError("Unchanged target controls did not replay")
                    rows.append(row)
                run.once(p, run.stable(dict(registration=reg, domain=domain, name=name, rows=rows)))
            v = run.read(p)
            if (v["registration"], v["domain"], v["name"]) != (reg, domain, name):
                raise ValueError("Interrupted readout cohort mismatch")
            domains[domain][name] = v["rows"]
            beat(domain=domain, name=name, finished=len(domains[domain]))
    report = {d: {k: compact(list(v.values()), k, d == "opendd") for k in prior.FRACTIONS} for d, v in domains.items()}
    run.once(OUT / "analysis.json", run.stable(dict(registration=reg, summary=report, groups=domains,
        opendd_sites=7, tumdot_campaigns=1, no_training=True, no_future_inputs=True, target_tuning=False,
        original_2pct=True, confirmation_access=False, deployment=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true", help="Regenerate all groups in a fresh scratch directory")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**state):
            v = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **state)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", v)
            print(v, flush=True)

        args_reg = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as d:
                replay_root(args_reg, beat, Path(d))
        else:
            execute(*args_reg, beat)
        beat(phase="complete", replay=args.replay)


def replay_root(args, beat, scratch):
    global PRIVATE
    cache, PRIVATE = PRIVATE, scratch
    try:
        execute(*args, beat)
    finally:
        PRIVATE = cache


if __name__ == "__main__":
    main()
