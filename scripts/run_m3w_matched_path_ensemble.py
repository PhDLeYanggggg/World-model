"""Isolate source candidate averaging in the frozen matched path-choice study."""

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
from scripts import run_m3w_matched_path_choice as base

run, np, torch, source = base.run, base.np, base.torch, base.source
transfer, model = base.transfer, base.model
OUT = base.OUT.parent / "matched_path_ensemble_v1"
PRIVATE = base.PRIVATE.parent / OUT.name


def register():
    parent, frozen, previous = base.register()
    paths = [base.OUT / n for n in ("head_freeze.json", "analysis.json")]
    heads, old = [run.read(p) for p in paths]
    if (any(v["registration"] != parent for v in (heads, old)) or old["heads"] != run.sha(paths[0])
            or heads["target_labels_used"] or old["target_tuning"] or old["confirmation_access"]):
        raise ValueError("Require the registered single-producer path-choice controls")
    shape = transfer.parent.old
    donor_reg, donors = shape.register()
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), single=[run.artifact(p) for p in paths],
        base=parent, donor_lineage=donor_reg, source_members="one site-excluded + six pair-excluded producers",
        refiner="same site-excluded refiner per arm across all source members",
        source_anchor="identical single-producer recipients and per-row peak", target_bank="unchanged seven-member bank",
        protocol=[8, 12], steps=base.STEPS, seeds=base.SEEDS, threads=4, workers=0,
        original_2pct=True, exposed_target_only=True, target_tuning=False, confirmation_access=False)))
    return run.sha(f), frozen, previous, donors, heads, old


def source_members(site, seed, single, inner):
    producer, refiners = single
    return [(producer, refiners)] + [(inner[tuple(sorted((site, peer))), seed], refiners)
                                    for peer in base.SITES if peer != site]


def make_donor(seed, nets, inner, cuts):
    train, stats = [], []
    for i, site in enumerate(base.SITES):
        p = source.prior.PRIVATE / f"{site}_cohort.npz"
        with np.load(p, allow_pickle=False) as z:
            h, y = z["history"], z["future"]
        valid = np.isfinite(y).all(2)
        single = nets[seed][i]
        cv, _, anchor, _, a = transfer.forecast(h, [single])
        cv2, _, _, _, b = transfer.forecast(h, source_members(site, seed, single, inner))
        np.testing.assert_array_equal(cv, cv2)
        pa, support = transfer.matched_paths(cv, anchor, a, .5)
        pb, matched = transfer.matched_paths(cv, anchor, b, .5)
        np.testing.assert_array_equal(support, matched)
        aa, xa, ra = base.prepare(h, cv, pa, support)
        bb, xb, rb = base.prepare(h, cv, pb, support)
        np.testing.assert_allclose(ra, rb, atol=1e-13, rtol=1e-7)
        ya, yb = [model.targets(cv[support], p[support], r, y[support], valid[support], cuts[0])
                  for p, r in ((aa, ra), (bb, rb))]
        ordered = np.sort(ya[..., 0], axis=1)
        strict = (ordered[:, -1]-ordered[:, -2]) > 1e-6
        path_shift = np.linalg.norm(aa[support]-bb[support], axis=-1).mean(2)/ra
        stats.append(dict(site=site, rows=len(xb), mean_normalized_path_shift=float(path_shift.mean()),
            mean_absolute_gain_target_shift=float(np.abs(ya[..., 0]-yb[..., 0]).mean()),
            strict_original_oracle_rows=int(strict.sum()),
            changed_best_on_strict_rows=int((strict & (ya[..., 0].argmax(1) != yb[..., 0].argmax(1))).sum())))
        train.append((torch.from_numpy(xb), torch.from_numpy(yb)))
    return train, stats


def fit(reg, seed, variant, data, beat, replay):
    ck = PRIVATE / f"{variant}_{seed}.pt"
    ident = dict(registration=reg, seed=seed, variant=variant, sources=base.SITES)
    torch.manual_seed(seed)
    head = model.ActionValue()
    opt = torch.optim.AdamW(head.parameters(), lr=.001)
    rng = np.random.default_rng(seed+8209)
    step, trace = 0, []
    if ck.exists():
        state = torch.load(ck, weights_only=True, map_location="cpu")
        if state["identity"] != ident or state["step"] not in range(100, base.STEPS+1, 100):
            raise ValueError("Ensemble source checkpoint roles changed")
        step, trace = state["step"], state["trace"]
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
    if replay and step != base.STEPS:
        raise ValueError("Replay cannot fit")
    for t in range(step+1, base.STEPS+1):
        which = rng.integers(len(data), size=base.BATCH)
        pick = [rng.choice(len(x), size=int((which == i).sum())) for i, (x, _) in enumerate(data)]
        x, y = (torch.cat([data[i][k][at] for i, at in enumerate(pick)]) for k in (0, 1))
        loss = model.loss(head(x if variant == "full" else base.action_only(x)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = ck.with_suffix(".pending")
            torch.save(dict(identity=ident, step=t, head=head.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(ck)
            beat(phase="fit", seed=seed, variant=variant, step=t, loss=trace[-1])
    return head.eval(), dict(identity=ident, checkpoint=run.artifact(ck), trace=trace), base.STEPS-step


def load_single(meta, seed, variant):
    m = meta["fit"][str(seed)]["heads"][variant]
    z = torch.load(run.checked(m["checkpoint"]), weights_only=True, map_location="cpu")
    if run.stable(z["identity"]) != m["identity"] or z["step"] != base.STEPS:
        raise ValueError("Previous fitted head roles changed")
    head = model.ActionValue()
    head.load_state_dict(z["head"])
    return head.eval()


@torch.inference_mode()
def compare(rec, h, cv, ols, y, valid, paths, support, pair, cuts, hist):
    bank, x, radius = base.prepare(h, cv, paths, support)
    inp = torch.from_numpy(x)
    at = np.flatnonzero(support)
    arms, choices, rollout, diffs = {}, {}, {}, {}
    for mode, heads in pair.items():
        for variant, head in heads.items():
            prediction = head(inp if variant == "full" else base.action_only(inp)).numpy()
            for rule, choice in model.select(prediction, base.ANCHOR).items():
                k = f"{variant}_{rule}"
                a = cv.copy()
                a[at] = bank[at, choice]
                np.testing.assert_array_equal((a != cv).any((1, 2)), support)
                np.testing.assert_allclose(np.linalg.norm(a[at]-cv[at], axis=-1).max(1),
                                           radius[:, 0], atol=1e-13, rtol=1e-7)
                name = f"{mode}_{k}"
                arms[name] = source.score(a, cv, ols, y, valid, rec, cuts, paths["hindcast"])
                if mode == "single" and arms[name] != hist["arms"][k]:
                    raise ValueError("Single-producer control did not replay exactly")
                rollout[name], choices[name] = a, choice
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for k in (f"{v}_{r}" for v in base.VARIANTS for r in ("utility", "easy_penalty", "agreement")):
        a, b = (f"{m}_{k}" for m in ("single", "ensemble"))
        observed, lower = transfer.paired(rollout[a], rollout[b], y, valid)
        diffs[k] = dict(observed_pp=float(100*observed.sum()/mass), lower_pp=float(100*lower.sum()/mass),
                        changed=int((choices[a] != choices[b]).sum()), matched_rows=len(at))
    return dict(arms=arms, pairs=diffs)


def summarize(rows, part):
    cells = [v[part] for row in rows for v in row["seeds"]]
    stats = source.mean_summary(cells)
    for name, v in stats.items():
        risk = [row["arms"][name]["risk"] for row in cells]
        easy = [a["completion_easy_risk_upper"] for a in risk if a["completion_easy_risk_upper"] is not None]
        v["worst_easy_risk_pct"] = 100*max(easy) if easy else None
    direct = {k: dict(observed_pp=float(np.mean([v["pairs"][k]["observed_pp"] for v in cells])),
                      lower_pp=float(np.mean([v["pairs"][k]["lower_pp"] for v in cells])),
                      changed=sum(v["pairs"][k]["changed"] for v in cells),
                      matched_rows=sum(v["pairs"][k]["matched_rows"] for v in cells)) for k in cells[0]["pairs"]}
    return dict(arms=stats, ensemble_vs_single=direct, views=len(cells))


def execute(reg, frozen, previous, donors, heads, old, beat, replay):
    nets = transfer.frozen_models(*frozen[:5])
    parent = transfer.parent
    _, inner = parent.old.load_producers(torch, parent.producer, donors)
    cuts = np.asarray(frozen[-1]["open_dd_positive_complete_cv_cuts"])
    fit_meta, pairs, updates = {}, {}, 0
    for seed in base.SEEDS:
        data, stats = make_donor(seed, nets, inner, cuts)
        if [r["rows"] for r in stats] != heads["fit"][str(seed)]["rows"]:
            raise ValueError("Source fitting cohort differs from the single-producer control")
        pairs[seed] = dict(single={}, ensemble={})
        fit_meta[str(seed)] = dict(stats=stats, heads={})
        for variant in base.VARIANTS:
            pairs[seed]["single"][variant] = load_single(heads, seed, variant)
            pairs[seed]["ensemble"][variant], m, n = fit(reg, seed, variant, data, beat, replay)
            fit_meta[str(seed)]["heads"][variant] = m
            updates += n
    freeze = OUT / "head_freeze.json"
    run.once(freeze, run.stable(dict(registration=reg, fit=fit_meta, target_labels_used=False)))
    rows = []
    for rec, (file, _) in source.ROSTER.items():
        p = PRIVATE / f"{rec}.json"
        if not p.exists():
            prior = run.read(base.PRIVATE / f"{rec}.json")
            if prior["registration"] != old["registration"] or prior["heads"] != old["heads"] or prior["recording"] != rec:
                raise ValueError("Unexpected prior recording identity")
            tracks, _, _ = source.read_tracks(source.SOURCE / file)
            h, y, _ = source.windows(tracks)
            valid = np.isfinite(y).all(2)
            cells = []
            for i, seed in enumerate(base.SEEDS):
                cv, ols, anchor, _, deltas = transfer.forecast(h, nets[seed])
                row = dict(seed=seed)
                for part, f in transfer.FRACTIONS.items():
                    paths, support = transfer.matched_paths(cv, anchor, deltas, f)
                    row[part] = compare(rec, h, cv, ols, y, valid, paths, support, pairs[seed], cuts, prior["seeds"][i][part])
                cells.append(row)
            run.once(p, run.stable(dict(registration=reg, heads=run.sha(freeze), recording=rec, seeds=cells)))
        v = run.read(p)
        if v["registration"] != reg or v["heads"] != run.sha(freeze) or v["recording"] != rec:
            raise ValueError("Unexpected transfer cache identity")
        rows.append(v)
        beat(phase="exposed_transfer", recording=rec, complete=len(rows))
    summary = {part: summarize(rows, part) for part in transfer.FRACTIONS}
    for part, m in summary.items():
        for name, v in m["arms"].items():
            if name.startswith("single_") and any(v[k] != u for k, u in old["summary"][part]["arms"][name[7:]].items()
                                                  if k != "path_choices"):
                raise ValueError("Previous aggregate control changed")
    by_day = {d: summarize([r for r in rows if r["recording"].endswith(d)], "half") for d in ("d1", "d2", "d3")}
    run.once(OUT / "analysis.json", run.stable(dict(registration=reg, heads=run.sha(freeze), summary=summary,
        by_day=by_day, recordings=len(rows), single_control_exact=True, original_2pct=True, confirmation_access=False,
        exposed_campaigns=1, target_tuning=False, deployment=False)))
    return updates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    args = cli.parse_args()
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
            if state.get("step", base.STEPS) == base.STEPS:
                print(v, flush=True)

        updates = execute(*register(), beat, args.replay)
        beat(phase="complete", updates=updates, replay=args.replay)


if __name__ == "__main__":
    main()
