"""Cross-fit a release-time gain deferral decision on OpenDD development."""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_timing_replay as timing
from scripts import run_m3w_opendd_protected_gain as parent
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_opendd_gain_wait as model

record, run, v, two, hard, prior = parent.record, parent.run, parent.v, parent.two, parent.hard, parent.prior
clock, shape, study = parent.rank.clock, parent.rank.shape, parent.study
OUT = parent.OUT.parent/"opendd_gain_wait_v1"
PRIVATE = parent.PRIVATE.parent/OUT.name
STEPS = 500
HEADS = ("context", "anchor_only")
POLICIES = ("always_wait", "utility", "guard", "anchor_guard")
ARMS = ("original", *POLICIES, *(k+"_matched" for k in POLICIES), "binary_oracle")


def register():
    key, *inputs = parent.register()
    a = timing.legacy.read(timing.PROTECTED_GAIN)
    timing.read_protected_clock()
    if a["registration"] != key:
        raise ValueError("Require the frozen protected gain experiment")
    parents = [parent.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "clock_registration", "clock_capacity")]
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in parents],
        steps=STEPS, seeds=two.study.prior.SEEDS, threads=4, workers=0, native_steps=[8, 12],
        release="original gain trigger; support clock/recipient/path stays original including gain trigger bookkeeping",
        offer="first later query with observed support and a non-support neural path of at least original gain radius",
        action="frozen gain ranking within that query; rescale down only to original gain radius",
        expiry_queries=model.DEADLINE, expiry="keep only the original support edit",
        training="outer and donor excluded upstream; wait targets use the realized causal first offer, not future maximum",
        guard="positive predicted conservative wait-minus-now utility; predicted deterioration and expiry logits both negative",
        ablation="retrain without observed support context and query-prefix summaries",
        primary="guard vs immediate on the full cohort; realized-budget matched immediate is a retrospective control",
        oracle="two-policy label-aware ceiling, preserving original risk pass, joint pass and defined easy reference",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), *inputs, a


def release_context(seq, t, first, support, q, feat, sizes):
    """Only queries through the original gain trigger contribute features."""
    known = support is not None and q[support] <= q[first]
    context = np.zeros(feat[1].shape[1]+2, np.float32)
    if known:
        context[:-2] = feat[1][support]
        context[-2:] = 1., np.log1p(q[first]-q[support])
    prefix = seq[max(0, t-model.DEADLINE):t+1]
    counts = [len(rows) for rows in prefix]
    prior_sizes = [float(sizes[rows].max()) if len(rows) else 0. for rows in prefix]
    anchor = np.r_[feat[0][first], np.log1p(sizes[first])]
    x = np.r_[anchor, context, np.log1p([t, len(seq[t]), np.mean(counts), np.max(prior_sizes)])]
    return x.astype(np.float32).clip(-16., 16.), len(anchor), known


@torch.inference_mode()
def prepare(d, cv, path, release, first_head):
    (big, small), feat, blocks, gain, easy = clock.prepare(d, cv, path, release, first_head)
    sizes = shape.peak(big-cv)
    original, support_only, waited = (cv.copy() for _ in range(3))
    q = d["query"]
    rows, x, width = [], [], None
    for r, seq in blocks.items():
        first, support, _ = clock.base.model.choose((ix, gain[ix], easy[ix]) for ix in seq)
        if support is not None:
            original[support] = support_only[support] = waited[support] = small[support]
        if first is None:
            continue
        original[first] = big[first]
        t = next(i for i, ix in enumerate(seq) if len(ix) and q[ix[0]] == q[first])
        f, width, known = release_context(seq, t, first, support, q, feat, sizes)

        def packets():
            seen = known
            for ix in islice_queries(seq, t):
                if support is not None and (ix == support).any():
                    seen = True
                    ix = ix[ix != support]
                yield seen, ix, sizes[ix], gain[ix]

        offer = model.first_offer(packets(), sizes[first])
        if offer is not None:
            elapsed, j = offer
            if j == support or q[j] <= q[first]:
                raise ValueError("Deferral must preserve support and move gain strictly later")
            waited[j] = cv[j]+(big[j]-cv[j])*(sizes[first]/sizes[j])
            np.testing.assert_allclose(shape.peak((waited[j]-cv[j])[None])[0], sizes[first], atol=1e-12, rtol=1e-10)
        else:
            elapsed, j = 0, None
        rows.append((r, first, j, elapsed, known))
        x.append(f)
    if not rows:
        raise ValueError("No original gain releases")
    return rows, np.stack(x), width, original, support_only, waited


def islice_queries(seq, release):
    for i in range(release+1, len(seq)):
        yield seq[i]


def labels(d, cv, ols, cut, pack):
    roster, _, _, original, _, waited = pack
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    risk = [policy(p, cv, cv, y, valid, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
            for p in (original, waited)]
    gain = [np.stack([v.paired(b, p, y, valid)[1].sum(1) for b in (cv, ols)])
            for p in (original, waited)]
    delta = v.paired(original, waited, y, valid)[1].sum(1)
    unit = prior.feature(d, cv, original)[1]
    truth, target, info = [], [], Counter()
    for r, first, j, _, known in roster:
        ix = rec == r
        a, b = (m[r] for m in risk)
        a_joint, b_joint = (z["supported"] and bool((g[:, ix].sum(1) > 0.).all())
                            for z, g in zip((a, b), gain, strict=True))
        lost = a["completion_easy_risk_upper"] is not None and b["completion_easy_risk_upper"] is None
        risk_loss = a["supported"] and not b["supported"]
        deterioration = lost or risk_loss or (a_joint and not b_joint)
        amount = float(delta[ix].sum())
        expired = j is None
        target.append([np.clip(amount/(12.*max(float(unit[first]), 1e-8)), -8., 8.),
                       float(deterioration), float(expired)])
        truth.append(dict(new_joint=bool(b_joint and not a_joint), lost_reference=bool(lost),
                          harm=amount < 0., amount=amount))
        info.update(releases=1, expired=int(expired), support_seen=int(known),
                    expired_support_seen=int(expired and known), favorable=int(amount > 0 and not deterioration and not expired),
                    lost_reference=int(lost), new_joint=int(b_joint and not a_joint))
    return np.asarray(target, np.float32), truth, info


def view(x, width, mode):
    if mode == "anchor_only":
        x = x.clone()
        x[:, width:] = 0.
    return x


@torch.inference_mode()
def evaluate(d, bank, pack, nets, cuts, ref, ceiling):
    cv, ols = bank[:2]
    roster, x, width, original, support_only, waited = pack
    y, valid = d["y"], d["valid"]
    target, truth, counts = labels(d, cv, ols, cuts, pack)
    x = torch.from_numpy(x)
    z, a = (nets[k](view(x, width, k)).numpy() for k in HEADS)
    binary_oracle = ((target[:, 1:] == 0.).all(1)
                     & np.array([p["new_joint"] or p["amount"] > 0. for p in truth]))
    decisions = dict(always_wait=np.ones(len(roster), dtype=bool), utility=model.choose_wait(z, False),
                     guard=model.choose_wait(z, True), anchor_guard=model.choose_wait(a, True),
                     binary_oracle=binary_oracle)
    pred = dict(original=original)
    fixed = (support_only != cv).any((1, 2))
    for k, take in decisions.items():
        pred[k] = original.copy()
        if k in POLICIES:
            pred[k+"_matched"] = original.copy()
        for use, (_, first, j, elapsed, known), label in zip(take, roster, truth, strict=True):
            if not use:
                continue
            pred[k][first] = cv[first]
            if j is not None:
                pred[k][j] = waited[j]
            elif k in POLICIES:
                pred[k+"_matched"][first] = cv[first]
            counts.update({k+"_waited": 1, k+"_retimed": int(j is not None), k+"_expired": int(j is None),
                           k+"_support_seen": int(known), k+"_lost_reference": int(label["lost_reference"]),
                           k+"_new_joint": int(label["new_joint"]), k+"_harmed": int(label["harm"]),
                           k+"_elapsed_sum": elapsed})
    for p in pred.values():
        np.testing.assert_array_equal(p[fixed], original[fixed])
    stats = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    if any(stats["original"][f] != ref["metrics"]["original"][f] for f in v.prior.FIELDS):
        raise ValueError("Frozen original no longer replays")
    for k in POLICIES:
        if stats[k]["edits"] != stats[k+"_matched"]["edits"]:
            raise ValueError("Mismatch in realized recording budgets")
        if stats[k]["edits"] != stats["original"]["edits"]-counts[k+"_expired"]:
            raise ValueError("A non-expiring action changed the edit budget")
    if not (stats["original"]["dual_positive"] <= stats["binary_oracle"]["dual_positive"]
            <= ceiling["delay_reference_preserved"]) or counts["binary_oracle_lost_reference"]:
        raise ValueError("Two-policy oracle left its verified bounds")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]
    pairs += [(k, "original") for k in (*POLICIES, "binary_oracle")]
    pairs += [(k, k+"_matched") for k in POLICIES]+[("guard", "anchor_guard"), ("guard", "utility")]
    effects = {f"{k}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[k], y, valid, mass)
               for k, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=counts)


def execute(key, outer, old, source, gains, _, producers, releases, nested, ref, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    cap = timing.read_protected_clock()
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0

    def prepare_site(s, path, r, f):
        d, cv = sites[s], banks[s][0]
        release = two.fit.load(r, lambda: hard.model.network(hard_width), hard.STEPS)
        n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
        first = two.fit.load(f, lambda: prior.model.network(n), prior.STEPS)
        return prepare(d, cv, path, release, first)

    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit, donor = [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                r = releases["release_"+"_".join(pair)+f"_{seed}"]
                f = nested[f"first_{held}_{s}_{seed}"]
                if (set(r["identity"]["fit"]) & set(pair) or set(f["identity"]["excluded"]) != set(pair)
                        or set(f["identity"]["fit"]) & set(pair)):
                    raise ValueError("Nomination models expose held or donor labels")
                path = record.read_prediction(refs, pair, s, seed)
                p = prepare_site(s, path, r, f)
                y, _, info = labels(sites[s], *banks[s][:2], cuts([t for t in use if t != s]), p)
                fit.append((torch.from_numpy(p[1]), torch.from_numpy(y), p[2]))
                donor.update(info)
            name = f"{held}_{seed}"
            n = fit[0][0].shape[1]
            nets = {}
            for mode in HEADS:
                ident = dict(registration=key, held=held, fit=use, seed=seed, arm=mode)
                tag = f"{name}_{mode}"
                batch = two.batcher([(view(x, w, mode), y) for x, y, w in fit])
                nets[mode], freeze[tag], u = two.fit.fit(ident, tag, seed, lambda: model.network(n), STEPS,
                                                          batch, model.loss, replay, beat)
                updates += u
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                p = prepare_site(held, z["comoving_ego"], outer[name], gains[name])
            z = evaluate(d, banks[held], p, nets, cuts(use), lookup(ref), lookup(cap)["counts"])
            result[held].append(dict(seed=seed, donor=donor, **z))
            beat(phase="site", site=held, seed=seed, updates=updates, releases=donor["releases"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
                   effects={k: ci(k) for k in flat[0]["effects"]},
                   diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if p.get("phase") != "fit":
                print(z, flush=True)

        two.fit.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
