"""Separate causal cross-query scheduling from second-edit coordination."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_two_edit as prior

run, parent, paired, study = prior.run, prior.parent, prior.paired, prior.study
OUT = prior.OUT.parent / "cross_query_edit_v2"
PRIVATE = prior.PRIVATE.parent / OUT.name
ARMS = ("same", "split_same_start", "early_independent", "early_joint")
WEIGHTS = np.array([2., 4., 1.])


def register():
    key, a, ph, _, source = prior.register()
    p, f, r = [prior.OUT / s for s in ("analysis.json", "head_freeze.json", "readout_repair_v2.json")]
    old, heads, repair = [run.read(x) for x in (p, f, r)]
    if (any(v["registration"] != key for v in (old, heads, repair)) or old["heads"] != run.sha(f)
            or old["confirmation_access"] or not old["original_2pct"]):
        raise ValueError("Require the registered two-edit checkpoints and readout")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(x) for x in (p, f, r)],
        native_steps=[8, 12], seeds=study.prior.SEEDS, arms=ARMS, no_training=True, original_2pct=True,
        primary="early joint minus early independent, identical first edit and first two legal query times",
        timing="original simultaneous pair vs split with same first path, matched two-edit views only for direct timing effect",
        full_roster_risk=True, no_label_or_roster_lookahead=True, bootstrap_draws=3000, bootstrap_seed=817,
        model="unchanged held-location two-edit candidate value checkpoints",
        oracle="freeze the first edit; exact second edit at registered next query or any later eligible query; short roster only",
        joint="sum predicted moments then apply the same risk penalties; both second choices use the first window scale",
        previous_mixed_scale_readout=run.artifact(OUT.parent / "cross_query_edit_v1" / "analysis.json"),
        confirmation_access=False)))
    return run.sha(reg), a, ph, old, heads["heads"], key, source


def local_score(value, scale, unit):
    v = value[:, :5]*(scale[:, None]/unit)
    return v[:, :2].min(1)-v[:, 2:].clip(min=0.)@WEIGHTS+.025*expit(value[:, 5])


def pick_joint(v, scale, prev, prev_scale, unit):
    z = (v[:, :5]*scale[:, None]+prev[:5]*prev_scale)/unit
    p, p0 = expit(v[:, 5]), expit(prev[5])
    return int((z[:, :2].min(1)-z[:, 2:].clip(min=0.)@WEIGHTS+.025*(p+p0-p*p0)).argmax())


def schedule(query_packets):
    """Consume the current window once; do not condition on a second window existing."""
    picks = {k: [] for k in ARMS}
    first = None
    for query, row, arm, v, scale, unit, capacity in query_packets:
        if len(row) == 0:
            continue
        score = local_score(v, scale, unit)
        best = int(score.argmax())
        option = lambda j: (query, int(row[j]), int(arm[j]))
        if first is None:
            first = (v[best], scale[best], unit)
            picks["early_independent"].append(option(best))
            picks["early_joint"].append(option(best))
        elif len(picks["early_joint"]) == 1:
            independent = int(local_score(v, scale, first[2]).argmax())
            picks["early_independent"].append(option(independent))
            picks["early_joint"].append(option(pick_joint(v, scale, *first)))
        if not picks["same"] and capacity >= 2 and len(np.unique(row)) >= 2:
            i, j = prior.model.pairs(row, np.zeros(len(row), dtype=int), [2])
            pair = int((score[i]+score[j]).argmax())
            left, right = i[pair], j[pair]
            if score[right] > score[left]:
                left, right = right, left
            picks["same"].extend((option(left), option(right)))
            picks["split_same_start"].append(option(left))
        elif len(picks["split_same_start"]) == 1:
            picks["split_same_start"].append(option(best))
        if len(picks["early_joint"]) == len(picks["split_same_start"]) == 2:
            break
    return picks


def second_ceiling(cv, ols, y, valid, first, candidate, path, cuts):
    if not len(candidate):
        return False
    a, p = first
    row = np.r_[a, candidate]
    paths = np.concatenate((p[None], path))
    g, r, e, offset = prior.coefficients(cv, ols, y, valid, row, paths, cuts[0])
    ref = prior.moments(cv, cv, y, valid, cuts[0])["ref_easy"].sum()
    risk = (r[0]+r[1:] <= np.array([0., 0., .02*ref])).all(1)
    good = risk & (e[0] | e[1:]) & ((g[0]+g[1:]+[0., offset]) > 0.).all(1)
    if not good.any():
        return False
    k = 1+np.flatnonzero(good)[0]
    out = cv.copy()
    out[row[[0, k]]] = paths[[0, k]]
    score = parent.score(dict(y=y, valid=valid, rec=np.full(len(cv), "local")), cv, ols, out, cuts)
    if score["edits"] != 2 or score["dual_positive"] != 1:
        raise ValueError("Second-edit witness did not reproduce")
    return True


@torch.inference_mode()
def evaluate(d, packet, menu, rec, net, cuts):
    bank, paths, x = packet
    cv, ols, _, _, scale, legal = bank
    q, y, valid = d["query"], d["y"], d["valid"]
    names, inv = np.unique(rec, return_inverse=True)
    active = menu["legal"][:, 1]
    short = np.bincount(inv, weights=active) < 5
    short_rows = short[inv[q]]
    forecasts = {k: cv.copy() for k in ARMS}
    count = {k: Counter() for k in ARMS}
    common, diagnostic = [], Counter()
    for name, small in zip(names, short, strict=True):
        at = np.flatnonzero(d["rec"] == name)
        iq, invq = np.unique(q[at], return_inverse=True)
        sizes = np.bincount(invq)

        def stream():
            for k, query in enumerate(iq):
                if not active[query]:
                    continue
                rows = at[invq == k]
                i, arm = np.nonzero(legal[rows])
                row = rows[i]
                v = net(torch.from_numpy(x[row, arm])).numpy()
                yield query, row, arm, v, scale[row], np.median(scale[rows]), min(2, sizes[k]-1)

        p = schedule(stream())
        for k, picks in p.items():
            for _, r, arm in picks:
                forecasts[k][r] = paths[r, arm]
            count[k][str(len(picks))] += 1
            if small:
                count[k][f"short_{len(picks)}"] += 1
        if p["split_same_start"] and p["split_same_start"][0] != p["same"][0]:
            raise ValueError("Timing contrast changed the first edit")
        if p["early_independent"][:1] != p["early_joint"][:1] or [x[0] for x in p["early_independent"]] != [x[0] for x in p["early_joint"]]:
            raise ValueError("Coordination contrast changed the clock or first edit")
        if len(p["split_same_start"]) == 2:
            common.extend(at)
            diagnostic["timing_common"] += 1
            diagnostic["timing_common_short"] += int(small)
        if len(p["early_joint"]) == 2:
            diagnostic["second_action_changes"] += int(p["early_joint"][1] != p["early_independent"][1])
        if small and p["early_joint"]:
            t, r, arm = p["early_joint"][0]
            i, j = np.nonzero(legal[at] & (active[q[at]] & (q[at] > t))[:, None])
            first = np.searchsorted(at, r), paths[r, arm]
            args = cv[at], ols[at], y[at], valid[at], first
            any_later = second_ceiling(*args, i, paths[at[i], j], cuts)
            diagnostic["first_fixed_any_later_witness"] += int(any_later)
            if len(p["early_joint"]) == 2:
                on = q[at[i]] == p["early_joint"][1][0]
                earliest = second_ceiling(*args, i[on], paths[at[i[on]], j[on]], cuts)
                diagnostic["first_fixed_next_query_witness"] += int(earliest)
                if earliest and not any_later:
                    raise ValueError("Earlier subset cannot exceed all later candidates")
    metrics, short_metrics = {}, {}
    for k, p in forecasts.items():
        metrics[k] = parent.score(d, cv, ols, p, cuts)
        if metrics[k]["edits"] != count[k]["1"]+2*count[k]["2"]:
            raise ValueError("Counted edits do not match emitted paths")
        short_metrics[k] = (parent.score({u: d[u][short_rows] for u in ("rec", "y", "valid")},
            cv[short_rows], ols[short_rows], p[short_rows], cuts) if short.any() else None)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    primary = float(100.*paired(forecasts["early_independent"], forecasts["early_joint"], y, valid)[1].sum()/mass)
    timing = None
    if common:
        ix = np.array(common, dtype=int)
        local_mass = np.linalg.norm(cv[ix]-y[ix], axis=-1)[valid[ix]].sum()
        if local_mass > 0.:
            timing = float(100.*paired(forecasts["same"][ix], forecasts["split_same_start"][ix], y[ix], valid[ix])[1].sum()/local_mass)
    return dict(metrics=metrics, short=short_metrics, count=count, diagnostic=diagnostic, joint_pp=primary, timing_pp=timing)


def execute(key, a, ph, old, heads, original_key, source, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    out = {}
    for s, d in sites.items():
        f = PRIVATE / f"{s}.json"
        fresh = not f.exists()
        if fresh:
            packet = prior.prepare(d)
            cuts = np.quantile(np.concatenate([v["cut_samples"] for k, v in sites.items() if k != s]), [.25, .75])
            with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
                rec = z["recording"]
            cells = []
            for row in a["by_site"][s]:
                seed = row["seed"]
                info = heads[f"{s}_{seed}"]
                if info["identity"] != dict(registration=original_key, held=s, fit=[u for u in sites if u != s], seed=seed):
                    raise ValueError("Held-location value lineage changed")
                net = prior.fit.load(info, lambda: prior.model.network(packet[2].shape[2]), prior.STEPS)
                rank = study.load(ph["heads"][f"{s}/{seed}/additive_rank"],
                                  lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
                with torch.inference_mode():
                    menu = study.menus(d, packet[0], rank)
                    cell = evaluate(d, packet, menu, rec, net, cuts)
                ref = next(r for r in old["by_site"][s] if r["seed"] == seed)["full"]["independent"]
                for k, v in ref.items():
                    np.testing.assert_allclose(cell["metrics"]["same"][k], v, rtol=0., atol=1e-10)
                cells.append(dict(seed=seed, **cell))
            run.once(f, run.stable(dict(registration=key, cells=cells)))
        z = run.read(f)
        if z["registration"] != key:
            raise ValueError("Cached cross-query site registration changed")
        out[s] = z["cells"]
        beat(site=s, fresh=fresh)

    flat = [c for v in out.values() for c in v]

    def interval(field):
        values = [[r[field] for r in v if r[field] is not None] for v in out.values()]
        x = np.array([np.mean(v) for v in values if v])
        if not len(x):
            return None
        draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_location=float(x.min()), supported_locations=len(x))

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    total = {k: {a: {f: sum(c[k][a][f] for c in flat if c[k][a] is not None) for f in fields} for a in ARMS}
             for k in ("metrics", "short")}
    total.update(count={k: sum((Counter(c["count"][k]) for c in flat), Counter()) for k in ARMS},
                 diagnostic=sum((Counter(c["diagnostic"]) for c in flat), Counter()),
                 joint_pp=interval("joint_pp"), timing_pp=interval("timing_pp"))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=total, by_site=out,
        original_2pct=True, confirmation_access=False, no_training=True, deployment=False, submission_ready=False)))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            print(z, flush=True)

        execute(*register(), beat)
        beat(complete=True)


if __name__ == "__main__":
    main()
