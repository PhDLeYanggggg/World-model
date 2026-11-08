"""Calibrate sign, magnitude and gain using strictly pair-excluded predictions."""

import argparse
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import expit

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_factorized_count_risk as prior
from src.world_model import m3w_donor_risk_readout as model

base, parent, run, torch = prior.study, prior.parent, prior.run, prior.torch
study = base.study
OUT = prior.OUT.parent / "donor_risk_readout_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
ARMS = ("gain_only", "prefix", "factorized", "calibrated_soft", "calibrated_sign", "calibrated_gain", "calibrated_both")
STEPS = base.STEPS


def register():
    key, a, ph, p, h = prior.register()
    paths = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze", "risk_readout")]
    f, n, r = [run.read(x) for x in paths]
    if (any(x["registration"] != key for x in (f, n, r)) or f["heads"] != run.sha(paths[1])
            or r["analysis"]["sha256"] != run.sha(paths[0]) or f["confirmation_access"]):
        raise ValueError("Require frozen factorized and sign diagnostics")
    frozen_nominations = base.prior.study.OUT / "head_freeze.json"
    upstream = base.prior.register()[3]
    path = OUT / "registration.json"
    run.once(path, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in (*paths, frozen_nominations)], seeds=study.prior.SEEDS, native_steps=[8, 12], updates=STEPS,
        exclusions="outer and calibration donor excluded from factorized fit and cutoff; additional fit site excluded from nomination",
        primary="soft risk substitution vs frozen factorized, preserving original motion gain",
        sign_threshold=.5, no_threshold_tuning=True, exact_recording_count_matched_diagnostic=True,
        original_2pct=True, confirmation_access=False)))
    return run.sha(path), a, ph, p, h, f, n, upstream


def probes(past, row, arm, scale):
    blocks = []
    for ec, ea in past:
        z = np.stack((ec[:, None]-ea, np.broadcast_to(ec[:, None], ea.shape)), axis=-1)
        blocks.append(base.model.accumulate(z, row, arm))
    return np.concatenate(blocks, axis=-1)/scale[:, None, None]


def pred_quality(raw, signed, fit, truth, legal, readout):
    use = legal.copy()
    use[:, 0] = False
    output = {}
    for k, ix, ev in (("all", 2, 7), ("easy", 3, 8)):
        z = truth[..., ix][use] > 0.
        j = ix-2
        probs = {"donor_rate": np.full(len(z), readout["risk"][j]["prior"]),
                 "signed_event": expit(signed[..., ev][use]), "calibrated": fit["prob"][..., j][use]}
        signs = {key: v >= .5 for key, v in probs.items()}
        signs["factorized_zero_sign"] = raw[..., ix][use] > 0.
        output[k] = dict(n=len(z), positive=int(z.sum()),
            brier_sum={key: float(np.square(v-z).sum()) for key, v in probs.items()},
            confusion={key: [int((v & z).sum()), int((v & ~z).sum())] for key, v in signs.items()})
    output["gain_mae"] = {k: float(np.abs(y[..., 0][use]-truth[..., 0][use]).mean())
                          for k, y in (("frozen_motion", signed), ("calibrated", fit["gain"]))}
    return output


def evaluate(d, bank, m, signed, value, fit, proxy, y, readout, cuts):
    cv, ols, b = bank[:3]
    rec, starts, legal = d["query_rec"], d["starts"], m["legal"]
    r = signed.copy()
    r[..., [2, 3, 4, 5, 9]] = value[..., [2, 3, 4, 5, 9]]
    g = r.copy()
    g[..., :2] = fit["gain"]
    score = base.model.score
    soft = fit["prob"]*fit["size"]
    s = dict(gain_only=score(signed, risk=False), prefix=proxy, factorized=score(r),
             calibrated_soft=model.score(signed[..., :2], soft, fit["support"]),
             calibrated_sign=model.score(signed[..., :2], soft*(fit["prob"] >= .5), fit["support"]),
             calibrated_gain=score(g), calibrated_both=model.score(fit["gain"], soft, fit["support"]))
    online = {k: study.model.streaming_count(v, legal, rec, starts) for k, v in s.items()}
    matched = {k: np.zeros(len(rec), dtype=int) for k in ARMS}
    for name in np.unique(rec):
        on = rec == name
        n = int(online["gain_only"][on].sum())
        for k in ARMS:
            if n:
                matched[k][on] = base.exact_score_plan(s[k][on], legal[on], n)
                if not prior.attribution.method.respects_tokens(matched[k][on], legal[on, 1]):
                    raise ValueError("Matched allocation cannot replay online credit")
    mass = np.linalg.norm(cv-d["y"], axis=-1)[d["valid"]].sum()

    def compute(c):
        p = {k: prior.attribution.install(cv, b, m, v) for k, v in c.items()}
        return dict(scores={k: parent.score(d, cv, ols, v, cuts) for k, v in p.items()},
            direct={f"{k}_vs_{ref}": float(100.*parent.transfer.paired(p[ref], p[k], d["y"], d["valid"])[1].sum()/mass)
                    for k, ref in (("calibrated_soft", "factorized"), ("calibrated_sign", "factorized"),
                                   ("calibrated_gain", "factorized"), ("calibrated_gain", "prefix"),
                                   ("calibrated_both", "factorized"))})

    return dict(online=compute(online), matched=compute(matched), quality=pred_quality(value, signed, fit, y, legal, readout))


def execute(key, a, ph, p, h, f, n, upstream, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        path = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(path, allow_pickle=False) as z:
            d.update(query_rec=z["recording"], starts=z["start"])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    past = {s: prior.attribution.method.prefix_errors(d["h"]) for s, d in sites.items()}
    cut = lambda fit: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in fit]), [.25, .75])
    rank_builder = lambda: study.prior.model.SwapValue(False)
    steps = 0
    rank_heads, pair_heads, readouts, rows = {}, {}, {}, {}

    @lru_cache(maxsize=24)
    def rank_data(site, threshold):
        return study.prior.prepare(sites[site], banks[site], threshold)

    @lru_cache(maxsize=None)
    def triple_rank(omit, seed):
        nonlocal steps
        fit = [s for s in sites if s not in omit]
        name = "rank_"+"_".join(omit)+f"_{seed}"
        data = []

        def batch(rng):
            if not data:
                c = cut(fit)[0]
                data.extend(rank_data(s, float(c)) for s in fit)
            return study.prior.batch(data, rng)

        identity = dict(registration=key, omitted=omit, fit=fit, seed=seed)
        out, rank_heads[name], u = study.fit(identity, name, seed, rank_builder, STEPS, batch,
            lambda net, x, y, pair, m: study.prior.model.objective(net(x, pair, m), y, True), replay, beat)
        steps += u
        return out

    @lru_cache(maxsize=None)
    def pair_model(omit, seed):
        nonlocal steps
        fit = [s for s in sites if s not in omit]
        name = "moments_"+"_".join(omit)+f"_{seed}"
        batch = None

        def make(rng):
            nonlocal batch
            if batch is None:
                data = []
                for s in fit:
                    ex = tuple(sorted((*omit, s)))
                    rk = triple_rank(ex, seed)
                    d, b = sites[s], banks[s]
                    m = study.menus(d, b, rk)
                    x, scale, _ = base.features(d, b, m)
                    y = prior.model.targets(*b[:3], d["y"], d["valid"], cut([k for k in sites if k not in ex])[0], m["row"], m["arm"], scale)
                    on = m["legal"][:, 1]
                    data.append((dict(x=x["motion"][on], legal=m["legal"][on]), y[on]))
                batch = base.batches(data)
            return batch(rng)

        # Load or verify every prerequisite even during zero-update replay.
        for s in fit:
            triple_rank(tuple(sorted((*omit, s))), seed)
        ident = dict(registration=key, omitted=omit, fit=fit, seed=seed)
        out, pair_heads[name], u = study.fit(ident, name, seed, lambda: prior.model.Factorized(width), STEPS, make,
            lambda net, x, legal, y: prior.model.objective(net(x), y, legal), replay, beat)
        steps += u
        return out

    # Width is fixed by the upstream causal schema; no evaluation row is used to fit it.
    width = 3*study.prior.model.WIDTH+3+2+base.model.PEER_WIDTH
    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            cal = []
            for s in donors:
                omit = tuple(sorted((held, s)))
                net = pair_model(omit, seed)
                nm = "rank_"+"_".join(omit)+f"_{seed}"
                if tuple(upstream["rank_heads"][nm]["identity"]["omitted"]) != omit:
                    raise ValueError("Missing frozen pair-excluded ranker")
                rk = study.load(upstream["rank_heads"][nm], rank_builder, study.RANK_STEPS)
                b, dv = banks[s], sites[s]
                m = study.menus(dv, b, rk)
                xx, scale, _ = base.features(dv, b, m)
                with torch.inference_mode():
                    raw = net(torch.from_numpy(xx["motion"]))
                    sv = prior.model.moments_for_score(raw)
                rr, gg = model.design(raw.numpy(), sv, probes(past[s], m["row"], m["arm"], scale))
                y = base.labels(dv, b, m, cut([k for k in sites if k not in omit])[0], scale)
                on = m["legal"].copy()
                on[:, 0] = False
                cal.append((rr[on], gg[on], y[on]))
            readout = model.fit(cal)
            readouts[f"{held}_{seed}"] = dict(donors=donors, fitted=readout)
            b = banks[held]
            rk = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], rank_builder, study.prior.STEPS)
            m = study.menus(d, b, rk)
            xx, scale, proxy = base.features(d, b, m)
            signed_net = study.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(width), STEPS)
            full = study.load(n["heads"][f"{held}_{seed}"], lambda: prior.model.Factorized(width), STEPS)
            with torch.inference_mode():
                t = torch.from_numpy(xx["motion"])
                raw, sv = full(t), signed_net(t).numpy()
                value = prior.model.moments_for_score(raw)
            rr, gg = model.design(raw.numpy(), value, probes(past[held], m["row"], m["arm"], scale))
            fit = model.apply(readout, rr, gg)
            cuts = cut(donors)
            yy = base.labels(d, b, m, cuts[0], scale)
            z = evaluate(d, b, m, sv, value, fit, proxy, yy, readout, cuts)
            old = next(v for v in f["by_site"][held] if v["seed"] == seed)
            for family in ("online", "matched"):
                if run.stable(z[family]["scores"]["factorized"]) != run.stable(old[family]["scores"]["factorized_risk"]):
                    raise ValueError("Frozen factorized control changed")
            ref = next(v for v in p["by_site"][held] if v["seed"] == seed)
            if run.stable(z["online"]["scores"]["prefix"]) != run.stable(ref["online"]["past_proxy"]):
                raise ValueError("Frozen prefix control changed")
            rows[held].append(dict(seed=seed, **z))
            beat(phase="held", held=held, seed=seed)
    return finish(key, rows, rank_heads, pair_heads, readouts, steps)


def finish(key, rows, rank_heads, pair_heads, readouts, steps):
    if len(rank_heads) != 105 or len(pair_heads) != 63 or len(readouts) != 21:
        raise ValueError("Incomplete nested calibration roster")
    draw = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))
    flat = [v for vv in rows.values() for v in vv]

    def ci(fn):
        x = np.array([np.mean([fn(v) for v in vv]) for vv in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    out = {r: dict(arms={k: {f: sum(v[r]["scores"][k][f] for v in flat)
            for f in ("edits", "dual_positive", "risk_pass", "undefined", "views")} for k in ARMS},
            direct={k: ci(lambda v: v[r]["direct"][k]) for k in flat[0][r]["direct"]}) for r in ("online", "matched")}
    out["gain_mae"] = {k: ci(lambda v: v["quality"]["gain_mae"][k]) for k in flat[0]["quality"]["gain_mae"]}
    for name in ("all", "easy"):
        cells = [v["quality"][name] for v in flat]
        count, positive = (sum(v[k] for v in cells) for k in ("n", "positive"))
        out[name] = dict(n=count, positive=positive,
            brier={k: sum(v["brier_sum"][k] for v in cells)/count for k in cells[0]["brier_sum"]},
            sign={k: dict(recall=sum(v["confusion"][k][0] for v in cells)/positive,
                          false_positive=sum(v["confusion"][k][1] for v in cells)/(count-positive)) for k in cells[0]["confusion"]})
    path = OUT / "head_freeze.json"
    run.once(path, run.stable(dict(registration=key, triple_ranks=rank_heads, pair_moments=pair_heads, calibration=readouts)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(path), summary=out, by_site=rows,
        original_2pct=True, confirmation_access=False, matching_label_blind_not_online=True)))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] != "fit":
                print(row, flush=True)

        n = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=n, replay=args.replay)


if __name__ == "__main__":
    main()
