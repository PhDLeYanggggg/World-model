"""Audit the frozen gain/support disagreement and an observed-prefix risk control."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_direct_count_value as study
from src.evaluation import m3w_support_value_attribution as method
from src.evaluation.m3w_selection_forecast_readout import policy

parent, run, torch = study.parent, study.run, study.torch
OUT = study.OUT.parent / "support_value_attribution_v1"
CAUSAL = ("motion_gain_only", "motion", "past_proxy", "prefix_risk")
ORACLE = ("gain_oracle", "risk_oracle", "support_oracle", "all_oracle_local")


def register():
    key, a, ph, _ = study.register()
    files = [study.OUT / f"{k}.json" for k in ("analysis", "head_freeze")]
    p, h = [run.read(f) for f in files]
    if (any(v["registration"] != key for v in (p, h)) or p["heads"] != run.sha(files[1])
            or p["confirmation_access"] or not p["original_2pct"]):
        raise ValueError("Require the frozen action-value head")
    out = OUT / "registration.json"
    run.once(out, run.stable(dict(code=run.artifact(Path(__file__)), method=run.artifact(Path(method.__file__)),
        parents=[run.artifact(p) for p in files], held_seeds=study.study.prior.SEEDS, native_steps=[8, 12],
        no_new_training=True, prefix_cut="other locations, 25th percentile of positive observed CV errors per prefix",
        primary="fixed-count oracle substitution and exact risk ceiling; causal prefix-risk control under unchanged tokens",
        matching="original motion-gain-only recording total; whole-history retrospective allocation, not deployable",
        original_2pct=True, confirmation_access=False)))
    return run.sha(out), a, ph, p, h


def install(cv, bank, menu, counts):
    row, arm = menu["row"], menu["arm"]
    use = np.arange(2) < counts[:, None]
    a = np.full(len(cv), -1, dtype=int)
    a[row[use]] = arm[use]
    return parent.selected(cv, bank, a)


def fixed_witness(d, bank, menu, counts, cuts):
    cv, ols, b = bank[:3]
    y, valid, rec, qr = (d[k] for k in ("y", "valid", "rec", "query_rec"))
    prior = study.study.prior.model
    p, offset = prior.label_parts(cv, ols, b, y, valid, cuts[0])
    raw = [prior.moments(cv, a, y, valid, cuts[0]) for a in b.swapaxes(0, 1)]
    p = np.concatenate((p, np.column_stack([z["easy_signed"] for z in raw])[..., None]), axis=-1)
    agg = study.model.accumulate(p, menu["row"], menu["arm"])
    stats = Counter()
    for r in np.unique(qr):
        on, ix = qr == r, rec == r
        mass = np.linalg.norm(cv[ix]-y[ix], axis=-1)[valid[ix]].sum()
        g = agg[on]
        plan, status = method.fixed_count_witness(g[..., :2], g[..., [2, 3, 5]], g[..., 4] > 0.,
            menu["legal"][on], offset[ix].sum(), .02*raw[0]["ref_easy"][ix].sum(), mass, int(counts[on].sum()))
        if plan is not None:
            both = np.zeros(len(qr), dtype=int)
            both[on] = plan
            pred = install(cv, b, menu, both)[ix]
            check = parent.score({k: d[k][ix] for k in ("y", "valid", "rec")}, cv[ix], ols[ix], pred, cuts)
            if check["dual_positive"] != 1:
                raise ValueError("Fixed-total witness fails original risk or gain")
        stats[status] += 1
    return dict(stats)


def contingency(y, legal, counts):
    on = legal.copy()
    on[:, 0] = False
    pick = np.where(counts > 0)[0]

    def collect(z):
        gain, excess, easy = z[:, 6] > 0., z[:, 7] > 0., z[:, 9] > 0.
        return dict(actions=len(z), gain=int(gain.sum()), all_excess=int(excess.sum()), easy=int(easy.sum()),
                    gain_and_easy=int((gain & easy).sum()), gain_easy_no_all_excess=int((gain & easy & ~excess).sum()))

    return dict(available=collect(y[on]), selected=collect(y[pick, counts[pick]]))


def undefined(d, cv, pred, cuts):
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    stats = policy(pred, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]
    out = Counter()
    for v in stats.values():
        if v["supported"]:
            out["risk_pass"] += 1
        elif v["selected"] == 0:
            out["zero_edits"] += 1
        elif v["known_easy_net_degradation"] is None:
            out["no_recording_easy_reference"] += 1
        elif v["completion_easy_risk_upper"] is None:
            out["no_selected_easy_reference"] += 1
        else:
            out["measurable_risk_failure"] += 1
    return dict(out)


@torch.inference_mode()
def evaluate(d, bank, menu, net, cuts, prefix, prefix_cuts):
    cv, ols, b = bank[:3]
    rec, starts, legal = d["query_rec"], d["starts"], menu["legal"]
    x, scale, proxy = study.features(d, bank, menu)
    pred = net(torch.from_numpy(x["motion"])).numpy()
    y = study.labels(d, bank, menu, cuts[0], scale)
    scores = method.oracle_scores(pred, y)
    scores.update(motion_gain_only=study.model.score(pred, risk=False), past_proxy=proxy,
                  prefix_risk=method.prefix_scores(prefix, prefix_cuts, menu["row"], menu["arm"], scale))
    online = {k: study.study.model.streaming_count(scores[k], legal, rec, starts) for k in CAUSAL}
    base = online["motion_gain_only"]
    matched = {k: np.zeros(len(base), dtype=int) for k in scores}
    for r in np.unique(rec):
        ix = rec == r
        n = int(base[ix].sum())
        if n:
            for k, s in scores.items():
                plan = study.exact_score_plan(s[ix], legal[ix], n)
                if not method.respects_tokens(plan, legal[ix, 1]):
                    raise ValueError("Matched attribution violates token replay")
                matched[k][ix] = plan

    def score(counts):
        paths = {k: install(cv, b, menu, v) for k, v in counts.items()}
        metrics = {k: parent.score(d, cv, ols, p, cuts) for k, p in paths.items()}
        return paths, metrics

    op, om = score(online)
    mp, mm = score(matched)
    mass = np.linalg.norm(cv-d["y"], axis=-1)[d["valid"]].sum()

    def gain(paths, a, ref):
        return float(100.*parent.transfer.paired(paths[ref], paths[a], d["y"], d["valid"])[1].sum()/mass)

    return dict(online=om, matched=mm, causes=undefined(d, cv, op["motion_gain_only"], cuts),
        contingency=contingency(y, legal, base), ceiling=fixed_witness(d, bank, menu, base, cuts),
        direct=dict(prefix_risk_vs_gain=gain(op, "prefix_risk", "motion_gain_only"),
                    prefix_risk_vs_proxy=gain(op, "prefix_risk", "past_proxy"),
                    **{f"matched_{k}_vs_motion": gain(mp, k, "motion") for k in (*ORACLE, "prefix_risk", "past_proxy")}))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, ph, p, h = register()
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        path = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(path, allow_pickle=False) as z:
            d.update(query_rec=z["recording"], starts=z["start"])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    prefix = {s: method.prefix_errors(d["h"]) for s, d in sites.items()}
    result = {}
    for s, d in sites.items():
        donors = [k for k in sites if k != s]
        cuts = np.quantile(np.concatenate([sites[k]["cut_samples"] for k in donors]), [.25, .75])
        ec = [np.concatenate([prefix[k][i][0] for k in donors]) for i in range(3)]
        pc = [float(np.quantile(v[v > 0.], .25)) for v in ec]
        result[s] = []
        for cell in a["by_site"][s]:
            seed = cell["seed"]
            rank = study.study.load(ph["heads"][f"{s}/{seed}/additive_rank"], lambda: study.study.prior.model.SwapValue(False), study.study.prior.STEPS)
            menu = study.study.menus(d, banks[s], rank)
            width = study.features(d, banks[s], menu)[0]["motion"].shape[1]
            net = study.study.load(h["heads"][f"{s}_{seed}_motion"], lambda: study.model.Value(width), study.STEPS)
            row = evaluate(d, banks[s], menu, net, cuts, prefix[s], pc)
            old = next(v for v in p["by_site"][s] if v["seed"] == seed)
            for k in CAUSAL[:-1]:
                if run.stable(row["online"][k]) != run.stable(old["online"][k]):
                    raise ValueError("Frozen causal control did not reproduce")
            result[s].append(dict(seed=seed, prefix_cuts=pc, **row))
        print(s, flush=True)
    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    def collect(family, keys):
        return {k: dict(vs_cv=ci(lambda r: r[family][k]["vs_cv"]),
            **{f: sum(r[family][k][f] for r in flat) for f in ("edits", "dual_positive", "risk_pass", "views", "undefined")}) for k in keys}

    def merged(get):
        c = Counter()
        for r in flat:
            c.update(get(r))
        return c

    summary = dict(online=collect("online", CAUSAL), matched=collect("matched", (*CAUSAL, *ORACLE)),
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]}, ceiling=merged(lambda r: r["ceiling"]),
        causes=merged(lambda r: r["causes"]), contingency={k: merged(lambda r: r["contingency"][k]) for k in ("available", "selected")})
    if any(v["edits"] != summary["online"]["motion_gain_only"]["edits"] for v in summary["matched"].values()):
        raise ValueError("The attribution did not preserve exact totals")
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, by_site=result,
        labels_used_only_for_oracles_and_evaluation=True, matched_not_online=True, no_training=True, original_2pct=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
