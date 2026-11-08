"""Fit path-choice heads on OpenDD and transport them without fitting on TUMDOT."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use the native macOS arm64 .venv-pytorch interpreter")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_tumdot_matched_finite_transfer as transfer
from src.world_model import m3w_matched_path_choice as model

run, np, torch, source = transfer.run, transfer.np, transfer.torch, transfer.source
OUT = transfer.OUT.parent / "matched_path_choice_v1"
PRIVATE = transfer.PRIVATE.parent / OUT.name
SITES, SEEDS, ARMS = source.SITES, transfer.parent.job.SEEDS, transfer.ARMS
STEPS, BATCH, ANCHOR = 500, 128, ARMS.index("hindcast")
VARIANTS = ("full", "action_only")


def register():
    parent, *frozen = transfer.register()
    prior = transfer.OUT / "analysis.json"
    previous = run.read(prior)
    if previous["registration"] != parent or not previous["source_is_previously_exposed"] or previous["confirmation_access"]:
        raise ValueError("Require the fixed exposed diagnostic and source-held producer bindings")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        previous=run.artifact(prior), donor_sites=SITES, training="OpenDD only; each row from its site-excluded producer",
        exposed_transport="TUMDOT only after freezing all heads; never an independent test", seeds=SEEDS,
        matched_arms=ARMS, checkpoint_every=100, steps=STEPS, batch=BATCH, cpu_threads=4, num_workers=0,
        fitting_arms=VARIANTS, horizon=[8, 12], objective="normalized lower utility + easy harm + pairwise costs",
        primary=dict(radius="half", arm="full_agreement", fixed_fallback="hindcast"),
        original_2pct=True, target_tuning=False, confirmation_access=False, deployment=False)))
    return run.sha(f), frozen, previous


def prepare(h, cv, paths, support):
    bank = np.stack([paths[name] for name in ARMS], axis=1)
    x, radius = model.features(h[support], cv[support], bank[support])
    return bank, x, radius


def fit_data(seed, nets, cuts):
    all_rows, fixed = [], []
    for i, site in enumerate(SITES):
        p = source.prior.PRIVATE / f"{site}_cohort.npz"
        with np.load(p, allow_pickle=False) as z:
            h, y = z["history"], z["future"]
        valid = np.isfinite(y).all(2)
        cv, _, base, _, deltas = transfer.forecast(h, [nets[seed][i]])
        paths, support = transfer.matched_paths(cv, base, deltas, .5)
        bank, x, radius = prepare(h, cv, paths, support)
        y_head = model.targets(cv[support], bank[support], radius, y[support], valid[support], cuts[0])
        ref_mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
        fixed.append(100*12*(y_head[..., 0]*radius).sum(0)/ref_mass)
        all_rows.append((torch.from_numpy(x), torch.from_numpy(y_head)))
    fixed = np.mean(fixed, axis=0)
    return all_rows, int(fixed.argmax()), {a: float(v) for a, v in zip(ARMS, fixed, strict=True)}


def action_only(x):
    # Keep rollout shape and relative radius; remove history and CV endpoint.
    return torch.cat((torch.zeros_like(x[..., :16]), x[..., 16:]), dim=-1)


def fit(reg, seed, variant, data, beat, replay):
    identity = dict(registration=reg, seed=seed, arm=variant, sources=SITES)
    ck = PRIVATE / f"{variant}_{seed}.pt"
    torch.manual_seed(seed)
    head = model.ActionValue()
    opt = torch.optim.AdamW(head.parameters(), lr=.001)
    rng = np.random.default_rng(seed+8209)
    step, trace = 0, []
    if ck.exists():
        d = torch.load(ck, weights_only=True, map_location="cpu")
        if d["identity"] != identity or d["step"] not in range(100, STEPS+1, 100):
            raise ValueError("Unexpected model or interrupted checkpoint state")
        head.load_state_dict(d["head"])
        opt.load_state_dict(d["optimizer"])
        rng.bit_generator.state = d["rng"]
        step, trace = d["step"], d["trace"]
    if replay and step != STEPS:
        raise ValueError("Replay never trains")
    for i in range(step, STEPS):
        j = rng.integers(len(data), size=BATCH)
        selections = [(a, rng.choice(len(x), size=int((j == a).sum()))) for a, (x, _) in enumerate(data)]
        x, y = (torch.cat([data[a][k][indices] for a, indices in selections]) for k in (0, 1))
        if variant == "action_only":
            x = action_only(x)
        loss = model.loss(head(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if (i+1) % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = ck.with_suffix(".pending")
            torch.save(dict(identity=identity, head=head.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, step=i+1, trace=trace), tmp)
            tmp.replace(ck)
            beat(phase="fit", seed=seed, variant=variant, step=i+1, loss=trace[-1])
    return head.eval(), dict(checkpoint=run.artifact(ck), identity=identity, trace=trace, updates_this_run=STEPS-step)


@torch.inference_mode()
def score(rec, cv, ols, y, valid, paths, support, h, heads, source_best, cuts):
    bank, x, radius = prepare(h, cv, paths, support)
    at = np.flatnonzero(support)
    values = dict(paths, donor_best_fixed=paths[ARMS[source_best]])
    counts = {}
    for variant, head in heads.items():
        inputs = torch.from_numpy(x)
        estimate = head(inputs if variant == "full" else action_only(inputs)).numpy()
        for rule, choice in model.select(estimate, ANCHOR).items():
            name = f"{variant}_{rule}"
            p = cv.copy()
            p[at] = bank[at, choice]
            np.testing.assert_array_equal((p != cv).any((1, 2)), support)
            np.testing.assert_allclose(np.linalg.norm(p[at]-cv[at], axis=-1).max(1), radius[:, 0], atol=1e-13, rtol=1e-7)
            counts[name] = np.bincount(choice, minlength=len(ARMS)).tolist()
            values[name] = p
    out = {n: source.score(p, cv, ols, y, valid, rec, cuts, paths["hindcast"]) for n, p in values.items()}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = {f"{a}_vs_{b}": float(100*transfer.paired(values[b], values[a], y, valid)[1].sum()/mass)
             for a, b in (("full_agreement", "donor_best_fixed"), ("full_agreement", "action_only_agreement"),
                          ("full_easy_penalty", "full_utility"), ("full_utility", "action_only_utility"))}
    return dict(arms=out, counts=counts, pairs=pairs, edits=int(support.sum()))


def summarize(rows, key):
    cells = [v[key] for row in rows for v in row["seeds"]]
    result = source.mean_summary(cells)
    for name, v in result.items():
        if name in cells[0]["counts"]:
            v["path_choices"] = np.sum([c["counts"][name] for c in cells], axis=0).tolist()
    return dict(arms=result, direct={n: float(np.mean([r["pairs"][n] for r in cells])) for n in cells[0]["pairs"]},
                views=len(cells))


def execute(reg, frozen, previous, beat, replay):
    nets = transfer.frozen_models(*frozen[:5])
    cuts = np.asarray(frozen[-1]["open_dd_positive_complete_cv_cuts"])
    fit_meta, heads, fixed, updates = {}, {}, {}, 0
    for seed in SEEDS:
        data, fixed[seed], fixed_costs = fit_data(seed, nets, cuts)
        heads[seed] = {}
        fit_meta[str(seed)] = dict(rows=[len(x) for x, _ in data], fixed=ARMS[fixed[seed]],
                                  fixed_costs=fixed_costs, heads={})
        for variant in VARIANTS:
            heads[seed][variant], v = fit(reg, seed, variant, data, beat, replay)
            updates += v.pop("updates_this_run")
            fit_meta[str(seed)]["heads"][variant] = v
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=reg, fit=fit_meta, target_labels_used=False)))
    rows = []
    for rec, (filename, _) in source.ROSTER.items():
        p = PRIVATE / f"{rec}.json"
        if not p.exists():
            tracks, _, _ = source.read_tracks(source.SOURCE / filename)
            h, y, _ = source.windows(tracks)
            valid = np.isfinite(y).all(2)
            old = next(r for r in previous["recordings"] if r["recording"] == rec)
            cells = []
            for seed in SEEDS:
                cv, ols, base, _, deltas = transfer.forecast(h, nets[seed])
                entry = dict(seed=seed)
                for part, fraction in transfer.FRACTIONS.items():
                    paths, support = transfer.matched_paths(cv, base, deltas, fraction)
                    entry[part] = score(rec, cv, ols, y, valid, paths, support, h, heads[seed], fixed[seed], cuts)
                    hist = next(r for r in old["seeds"] if r["seed"] == seed)[part]["policies"]
                    for name in ARMS:
                        expected = {k: v for k, v in hist[name].items() if k != "edits"}
                        got = {k: v for k, v in entry[part]["arms"][name].items() if k != "lower_pp"}
                        got["lower_pp"] = {k: entry[part]["arms"][name]["lower_pp"][k] for k in expected["lower_pp"]}
                        if expected != got:
                            raise ValueError("Frozen target control does not reproduce exactly")
                cells.append(entry)
            run.once(p, run.stable(dict(registration=reg, heads=run.sha(f), recording=rec, seeds=cells)))
        row = run.read(p)
        if row["registration"] != reg or row["heads"] != run.sha(f) or row["recording"] != rec:
            raise ValueError("Target checkpoint roles changed")
        rows.append(row)
        beat(phase="exposed_transfer", recording=rec, complete=len(rows))
    report = {p: summarize(rows, p) for p in transfer.FRACTIONS}
    days = {day: summarize([r for r in rows if r["recording"].endswith(day)], "half") for day in ("d1", "d2", "d3")}
    run.once(OUT / "analysis.json", run.stable(dict(registration=reg, heads=run.sha(f), summary=report, by_day=days,
        recordings=len(rows), original_controls_exact=True, target_tuning=False, exposed_campaigns=1,
        original_2pct=True, confirmation_access=False, deployment=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
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
            if state.get("step", STEPS) == STEPS:
                print(v, flush=True)

        updates = execute(*register(), beat, args.replay)
        beat(phase="complete", updates=updates, replay=args.replay)


if __name__ == "__main__":
    main()
