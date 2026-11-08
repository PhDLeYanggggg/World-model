"""Compare first-query, fixed-delay and donor-selected one-edit stopping."""

import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_first_edit_target as prior

run, parent, paired, two = prior.run, prior.parent, prior.paired, prior.two
OUT = prior.OUT.parent / "one_edit_stopping_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS = 400
THRESHOLDS = (0., .05, .10, .20, .35, .50, .70, .85)
DELAYS = (0, 1, 2, 3, 7)
GRID = dict((f"threshold_{i}", (0, t)) for i, t in enumerate(THRESHOLDS))
GRID.update((f"delay_{k}", (k, 0.)) for k in DELAYS)
ARMS = ("immediate", "fixed_delay", "learned_stop")


def register():
    k, a, _, _, _ = prior.register()
    p, f = prior.OUT / "analysis.json", prior.OUT / "head_freeze.json"
    old, h = [run.read(x) for x in (p, f)]
    if (any(v["registration"] != k for v in (old, h)) or old["heads"] != run.sha(f)
            or old["confirmation_access"] or not old["original_2pct"]):
        raise ValueError("Require original first-edit roster and risk evidence")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(prior.model.__file__)),
        parents=[run.artifact(p), run.artifact(f)], native_steps=[8, 12], seeds=two.study.prior.SEEDS,
        steps=STEPS, batch=128, threads=4, workers=0,
        train="all eligible windows; sample source, recording, eligible query and candidate uniformly in sequence",
        donors="for each held site and donor, predictions use a head fitted excluding both",
        selection="pool three seeds; macro-average donor joint-pass rate, tie by CV gain then smaller grid index",
        clock="at most one edit, no fallback edit at the last query, no final-length input",
        grid=GRID, primary="learned stop vs donor-selected fixed delay, full roster and retrospectively matched utility",
        confirmation_access=False, original_2pct=True)))
    return run.sha(reg), a, old


def records(d, packet):
    legal = packet[0][-1]
    q = d["query"]
    size = np.bincount(q)
    for name in np.unique(d["rec"]):
        at = np.flatnonzero(d["rec"] == name)
        row, arm = np.nonzero(legal[at] & (size[q[at]] >= 2)[:, None])
        if len(row):
            query = q[at[row]]
            if (np.diff(query) < 0).any():
                raise ValueError("Expected chronological candidate rows")
            _, counts = np.unique(query, return_counts=True)
            yield at, row, arm, np.r_[0, counts.cumsum()]


def stopping(scores, skip, threshold):
    """Yield the first accepted window, without seeing the future or length."""
    for index, score in enumerate(scores):
        if index >= skip and score >= threshold:
            return index
    return None


def batcher(sites):
    def sample(rng):
        src = rng.integers(len(sites), size=128)
        rec = [sites[k][rng.integers(len(sites[k]))] for k in src]
        window = [rng.integers(len(bounds)-1) for _, _, bounds in rec]
        row = [rng.integers(r[2][w], r[2][w+1]) for r, w in zip(rec, window, strict=True)]
        return tuple(torch.stack([v[j][i] for v, i in zip(rec, row, strict=True)]) for j in range(2))
    return sample


@torch.inference_mode()
def score_windows(packet, entries, net):
    x = packet[2]
    all_features = torch.from_numpy(np.concatenate([x[at[row], arm] for at, row, arm, _ in entries]))
    out = []
    p = prior.model.scores(net(all_features).numpy())["direct"]
    cursor = 0
    for at, row, arm, bounds in entries:
        best = [i + p[cursor+i:cursor+j].argmax() for i, j in zip(bounds[:-1], bounds[1:], strict=True)]
        out.append((at[row[best]], arm[best], p[cursor+np.asarray(best)]))
        cursor += len(row)
    if cursor != len(p):
        raise ValueError("Candidate/window misalignment")
    return out


def plans(windows, rules):
    out = {k: {} for k in rules}
    for r, (rows, arms, score) in enumerate(windows):
        for k, (skip, threshold) in rules.items():
            j = stopping(score, skip, threshold)
            if j is not None:
                out[k][r] = int(rows[j]), int(arms[j]), j
    return out


def donor_select(rows, prefix):
    choices = [k for k in rows[0] if k.startswith(prefix)]
    def key(k):
        return (np.mean([r[k]["dual_positive"]/r[k]["views"] for r in rows]),
                np.mean([r[k]["vs_cv"] for r in rows]))
    # max() keeps the first grid entry for an exact tie.
    best = max(choices, key=key)
    return best, {k: list(map(float, key(k))) for k in choices}


def forecasts(d, packet, selected, cuts):
    cv, ols = packet[0][:2]
    paths = packet[1]
    predictions, metrics = {}, {}
    for k, edits in selected.items():
        p = cv.copy()
        for row, arm, _ in edits.values():
            p[row] = paths[row, arm]
        predictions[k] = p
        metrics[k] = parent.score(d, cv, ols, p, cuts)
        if metrics[k]["edits"] != len(edits):
            raise ValueError("Expected one nonzero edit per stopped recording")
    return predictions, metrics


def paired_pp(d, cv, p, q, at):
    y, valid = d["y"][at], d["valid"][at]
    mass = np.linalg.norm(cv[at]-y, axis=-1)[valid].sum()
    return float(100.*paired(q[at], p[at], y, valid)[1].sum()/mass) if mass > 0. else None


def evaluate(d, packet, entries, windows, rules, cuts, old):
    selected = plans(windows, rules)
    pred, stats = forecasts(d, packet, selected, cuts)
    cv = packet[0][0]
    all_rows = np.arange(len(cv))
    effects, common = {}, Counter()
    for other in ("immediate", "fixed_delay"):
        edits = selected["learned_stop"].keys() & selected[other].keys()
        ix = [entries[r][0] for r in sorted(edits)]
        common[other] = len(edits)
        effects[other] = dict(full=paired_pp(d, cv, pred["learned_stop"], pred[other], all_rows),
            matched=paired_pp(d, cv, pred["learned_stop"], pred[other], np.concatenate(ix)) if ix else None)
    ceiling = Counter()
    position = {k: Counter(str(j) for _, _, j in v.values()) for k, v in selected.items()}
    short_ids = {i for i, (_, _, _, b) in enumerate(entries) if len(b) <= 5}
    for at, row, arm, b in entries:
        hit = prior.checked_oracle(d, packet, at, row, arm, cuts)[:, 2]
        per_query = np.array([hit[i:j].any() for i, j in zip(b[:-1], b[1:], strict=True)])
        ceiling.update(first=int(per_query[0]), any=int(per_query.any()), later_only=int(not per_query[0] and per_query.any()))
    reference = old["support"]
    if (ceiling["first"], ceiling["any"], len(short_ids)) != (
            reference["qualifies_first_query"], reference["qualifies_any_query"], reference.get("short", 0)):
        raise ValueError("Timing oracle changed relative to the first-edit readout")
    short = {}
    if short_ids:
        at = np.concatenate([entries[i][0] for i in sorted(short_ids)])
        sd = {k: d[k][at] for k in ("rec", "y", "valid")}
        short = {k: parent.score(sd, cv[at], packet[0][1][at], p[at], cuts) for k, p in pred.items()}
    return dict(metrics=stats, short=short, effects=effects, common=common, positions=position, ceiling=ceiling,
                short_edits={k: len(v.keys() & short_ids) for k, v in selected.items()})


def execute(key, a, old, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    packets = {s: two.prepare(d) for s, d in sites.items()}
    entries = {s: list(records(d, packets[s])) for s, d in sites.items()}
    all_heads, result, settings, updates = {}, {}, {}, 0
    n_fits, n_reused = 0, 0

    @lru_cache(maxsize=96)
    def cut(use):
        return np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])

    @lru_cache(maxsize=48)
    def train_rows(site, use):
        packet = packets[site]
        cut_easy = cut(tuple(s for s in use if s != site))[0]
        return [(torch.from_numpy(packet[2][at[row], arm]),
                 torch.from_numpy(prior.target(sites[site], packet, at, row, arm, cut_easy)), bounds)
                for at, row, arm, bounds in entries[site]]

    @lru_cache(maxsize=9)
    def fitted(excluded, seed):
        nonlocal updates, n_fits, n_reused
        use = tuple(s for s in sites if s not in excluded)
        name = "_".join(excluded)+f"_{seed}"
        ident = dict(registration=key, excluded=list(excluded), fit=list(use), seed=seed)
        batch = batcher([train_rows(s, use) for s in use])
        head, meta, u = two.fit.fit(ident, name, seed, lambda: prior.model.network(packets[use[0]][2].shape[-1]),
                                    STEPS, batch, prior.model.loss, replay, beat)
        if name not in all_heads:
            n_fits += int(u > 0)
            n_reused += int(u == 0)
        all_heads[name] = meta
        updates += u
        return head

    for held, d in sites.items():
        donors = tuple(s for s in sites if s != held)
        seeds = [c["seed"] for c in a["by_site"][held]]
        donor_scores = []
        for s in donors:
            use = tuple(u for u in donors if u != s)
            for seed in seeds:
                net = fitted(tuple(sorted((held, s))), seed)
                p = plans(score_windows(packets[s], entries[s], net), GRID)
                _, m = forecasts(sites[s], packets[s], p, cut(use))
                donor_scores.append(m)
        stop, stop_grid = donor_select(donor_scores, "threshold")
        fixed, fixed_grid = donor_select(donor_scores, "delay")
        rules = dict(immediate=(0, 0.), fixed_delay=GRID[fixed], learned_stop=GRID[stop])
        settings[held] = dict(chosen=dict(fixed_delay=fixed, learned_stop=stop), rules=rules,
                              donor_cells=len(donor_scores), fixed_grid=fixed_grid, stop_grid=stop_grid)
        result[held] = []
        for seed in seeds:
            net = fitted((held,), seed)
            ref = next(x for x in old["by_site"][held] if x["seed"] == seed)
            cell = evaluate(d, packets[held], entries[held], score_windows(packets[held], entries[held], net),
                            rules, cut(donors), ref)
            result[held].append(dict(seed=seed, **cell))
            beat(phase="held", site=held, seed=seed, rules=rules, metrics=cell["metrics"])
    flat = [x for v in result.values() for x in v]

    def ci(ref, scope):
        per = [[x["effects"][ref][scope] for x in v if x["effects"][ref][scope] is not None] for v in result.values()]
        z = np.array([np.mean(v) for v in per if v])
        if not len(z):
            return None
        draw = np.random.default_rng(817).integers(len(z), size=(3000, len(z)))
        return dict(point=float(z.mean()), interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist(),
                    worst_location=float(z.min()), supported_locations=len(z))

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    total = {s: {k: {f: sum(x[s][k][f] for x in flat if k in x[s]) for f in fields} for k in ARMS}
             for s in ("metrics", "short")}
    total.update(effects={k: {s: ci(k, s) for s in ("full", "matched")} for k in ("immediate", "fixed_delay")},
                 common=sum((Counter(x["common"]) for x in flat), Counter()),
                 positions={k: sum((Counter(x["positions"][k]) for x in flat), Counter()) for k in ARMS},
                 ceiling=sum((Counter(x["ceiling"]) for x in flat), Counter()))
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=all_heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(f), summary=total,
        selected_by_donors=settings, by_site=result, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates, n_fits, n_reused


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    two.fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            if v["phase"] != "fit":
                print(z, flush=True)

        updates, fits, reused = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=updates, fitted=fits, reused=reused)


if __name__ == "__main__":
    main()
