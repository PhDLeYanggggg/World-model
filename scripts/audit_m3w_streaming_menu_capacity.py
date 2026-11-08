"""Separate the nested menu and streaming credit bottlenecks on development data."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recording_count as study
from src.evaluation import m3w_streaming_menu_capacity as solver

run, np, torch, parent = study.run, study.np, study.torch, study.parent
OUT = study.OUT.parent / "streaming_menu_capacity_v1"
PRIVATE = study.PRIVATE.parent / OUT.name
FULL = ("full_total", "full_stream")
NESTED = ("nested_total", "nested_stream")
ARMS = (*FULL, *NESTED)
POS = "verified_positive_both"
CONTRASTS = (("full_total", "nested_total"), ("full_stream", "nested_stream"),
             ("full_total", "full_stream"), ("nested_total", "nested_stream"))


def register():
    key, a, _, _, ph = study.register()
    files = [study.OUT / f"{k}.json" for k in ("analysis", "head_freeze", "matched_diagnostic")]
    p, h, m = [run.read(f) for f in files]
    if (any(v["registration"] != key for v in (p, h, m)) or p["heads"] != run.sha(files[1])
            or m["analysis"]["sha256"] != run.sha(files[0]) or p["confirmation_access"] or not p["original_2pct"]
            or study.RATE != .2):
        raise ValueError("Require bound recording-count experiment")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), solver=run.artifact(Path(solver.__file__)),
        parent=[run.artifact(f) for f in files], arms=ARMS, native_steps=[8, 12], original_2pct=True,
        confirmation_access=False, labels_hindsight_only=True, no_training=True,
        joint_query_cap="min(2, agents - 1)", total_budget="floor(eligible queries / 5)",
        token_budget=dict(initial=0, earn=.2, bank=2., cost=1.),
        development_locations=parent.source.SITES, seeds=study.prior.SEEDS, timeout_seconds=2.)))
    return run.sha(reg), a, ph


def build_menu(d, bank, rank):
    _, _, paths, _, _, _ = bank
    menu = study.menus(d, bank, rank)
    row, arm = menu["row"], menu["arm"]
    restricted = np.broadcast_to(bank[0][:, None], paths.shape).copy()
    at = row >= 0
    restricted[row[at], arm[at]] = paths[row[at], arm[at]]
    return row, restricted, menu["legal"][:, 1]


def execute(key, a, ph, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    total, paired, rows = {k: Counter() for k in ARMS}, {}, {}
    for s, d in sites.items():
        source = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(source, allow_pickle=False) as z:
            records, starts = z["recording"], z["start"]
        if (np.diff(starts)[records[1:] == records[:-1]] <= 0).any():
            raise ValueError("Recording order is not chronological")
        cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
        bank = parent.prior.analytic(d)
        cv, ols, b, _, _, legal = bank
        q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
        cap = np.minimum(2, np.bincount(q)-1)
        eligible = (cap > 0) & (np.bincount(q[legal.any(1)], minlength=len(cap)) > 0)
        nested = {}
        for cell in a["by_site"][s]:
            seed = cell["seed"]
            rank = study.load(ph["heads"][f"{s}/{seed}/additive_rank"], lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            row, restricted, el = build_menu(d, bank, rank)
            np.testing.assert_array_equal(eligible, el)
            nested[seed] = row, restricted
        site = {k: Counter() for k in ARMS}
        pair = {f"{left}_to_{right}": Counter() for left, right in CONTRASTS}
        sizes = Counter()
        for i, name in enumerate(sorted(set(rec))):
            file = PRIVATE / f"{s}_{i:04d}.json"
            if not file.exists():
                ix = rec == name
                qi = records == name
                _, qlocal = np.unique(q[ix], return_inverse=True)
                if len(np.unique(qlocal)) != qi.sum() or not np.array_equal(np.unique(q[ix]), np.flatnonzero(qi)):
                    raise ValueError("Query-to-recording alignment changed")
                base = cv[ix], ols[ix]
                tail = y[ix], valid[ix], qlocal, name, cuts, eligible[qi], cap[qi]
                full = {k: solver.assess(*base, b[ix], *tail, streaming=k.endswith("stream")) for k in FULL}
                entries = []
                index = np.flatnonzero(ix)
                for seed, (row, restricted) in nested.items():
                    r = row[qi].copy()
                    at = r >= 0
                    if not np.isin(r[at], index).all():
                        raise ValueError("Nominated action moved between recordings")
                    r[at] = np.searchsorted(index, r[at])
                    v = {k: solver.assess(*base, restricted[ix], *tail, nested=r, streaming=k.endswith("stream")) for k in NESTED}
                    entries.append(dict(seed=seed, arms={**full, **v}))
                run.once(file, run.stable(dict(registration=key, site=s, ordinal=i, entries=entries)))
            receipt = run.read(file)
            if (receipt["registration"], receipt["site"], receipt["ordinal"]) != (key, s, i):
                raise ValueError("Capacity cache identity changed")
            for v in receipt["entries"]:
                statuses = {k: v["arms"][k]["status"] for k in ARMS}
                for k, status in statuses.items():
                    site[k][status] += 1
                for left, right in CONTRASTS:
                    lo, hi = statuses[left], statuses[right]
                    if lo not in (POS, "unresolved") and hi == POS:
                        raise ValueError(f"Restricted menu has a witness outside the unrestricted set: {left}, {right}")
                    pair[f"{left}_to_{right}"][f"{lo} -> {hi}"] += 1
                sizes["views"] += 1
            beat(phase="solving", site=s, recordings=i+1)
        rows[s] = dict(arms=site, contrasts=pair, views=sizes["views"])
        for k in ARMS:
            total[k].update(site[k])
        for k, values in pair.items():
            paired.setdefault(k, Counter()).update(values)
        beat(phase="site_complete", site=s, witnesses={k: site[k][POS] for k in ARMS})
    if any(sum(t.values()) != 606 for t in total.values()):
        raise ValueError("Require complete recording-seed roster")
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=total, paired=paired, by_site=rows,
        distinct_recordings=202, full_bank_shared_across_three_seeds=True, original_2pct=True,
        confirmation_access=False, labels_hindsight_only=True, no_training=True)))


def main():
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
            if v["phase"] != "solving":
                print(row, flush=True)

        execute(*register(), beat)
        beat(phase="complete", updates=0)


if __name__ == "__main__":
    main()
