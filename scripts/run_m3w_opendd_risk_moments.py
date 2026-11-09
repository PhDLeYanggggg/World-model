"""Test direct harm supervision and a causal risk ledger on the fixed turn menu."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_slate_relative_gain as slate
from src.evaluation import m3w_recording_risk_attribution as attribution
from src.world_model import m3w_causal_risk_moments as model

np, torch, study, base, turn, shared = slate.np, slate.torch, slate.study, slate.base, slate.turn, slate.shared
SITES, SEEDS = slate.SITES, slate.SEEDS
OUT = slate.OUT.parent / "opendd_risk_moments_v1"
LOCAL = slate.LOCAL.parent / OUT.name
STEPS, BATCH, EVERY = 600, 128, 50
ARMS = ("additive", "pair_parent", "soft", "local", "prefix", "prefix_two")


def register():
    key, cohorts, parent, refs, cal = slate.register()
    files = [slate.OUT / f"{k}.json" for k in ("registration", "freeze", "readout", "risk_attribution")]
    prior, result = map(study.read, files[1:3])
    if (prior["registration"] != key or result["freeze"] != study.artifact(files[1])
            or not prior["held_inference_causal_only"] or result["reserved_outcomes_opened"]):
        raise ValueError("Require frozen direct pair forecasts and donors")
    p = OUT / "registration.json"
    base.immutable(p, dict(parents=[study.artifact(x) for x in files],
        code=[study.artifact(Path(x)) for x in (__file__, model.__file__, shared.costs.__file__,
             attribution.__file__, slate.__file__)], sites=SITES, seeds=SEEDS, steps=STEPS, batch=BATCH,
        checkpoint_every=EVERY, modes=ARMS, original_2pct=True, observed_steps=8, forecast_steps=12,
        train="squared error for observed harm, easy harm, signed easy, CV/easy masses and unknown debit",
        scale="donor-only per-output RMS with 0.01 floor, site-balanced sampling",
        fixed="same menu, per-query edit counts and radii; prefix_two retains the original singleton",
        prefix="per-recording chronological predicted masses only; reset and no future queries",
        utilities="frozen two-reference gain and slate relative gain; not re-fit",
        development_only=True, independent_confirmation=False, thresholds_tuned=False))
    return base.sha(p), cohorts, parent, prior, refs, cal


def examples(d, bank, cut):
    cv, ols, h = (d[k] for k in ("cv", "ols", "h"))
    paths = cv[:, None]+np.concatenate([a*(bank-cv[:, None]) for a in turn.refit.model.AMPLITUDES], axis=1)
    x, unit, live, _ = shared.costs.features(h, cv, ols, paths)
    radius = np.linalg.norm(paths-cv[:, None], axis=-1).max(-1) / unit[:, None]
    target = model.target(cv, paths, d["y"], d["v"], cut)/unit[:, None, None]
    if not np.isfinite(target).all() or not live.any():
        raise ValueError("Invalid risk training examples")
    return tuple(torch.from_numpy(a[live].astype(np.float32)) for a in (x, radius, target))


def fit(key, site, seed, donors, refs):
    scale = sum(d[-1].square().mean(0) for d in donors.values()).div(len(donors)).sqrt().clamp_min(.01)
    identity = dict(registration=key, held=site, seed=seed, source=refs, scale=scale.tolist(),
                    donor_rows={s: len(d[-1]) for s, d in donors.items()})
    path = LOCAL / f"{site}_{seed}.pt"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1808)
    net = model.MomentHead()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    start, trace = 0, []
    if path.exists():
        st = torch.load(path, weights_only=True, map_location="cpu")
        if st["identity"] != identity or st["step"] not in range(EVERY, STEPS+1, EVERY):
            raise ValueError("Checkpoint identity differs")
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["optimizer"])
        start, trace, rng.bit_generator.state = st["step"], st["trace"], st["rng"]
    data = list(donors.values())
    for step in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(data), 1./len(data)))
        picks = [rng.integers(len(d[0]), size=n) for d, n in zip(data, counts, strict=True)]
        x, r, y = (torch.cat([d[k][i] for d, i in zip(data, picks, strict=True)]) for k in range(3))
        loss = ((net(x, r)-y)/scale).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % EVERY == 0:
            trace.append(float(loss.detach()))
            pending = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            trace=trace, rng=rng.bit_generator.state), pending)
            pending.replace(path)
            beat(phase="fit", fold=f"{site}_{seed}", step=step, total=STEPS)
    return net.eval(), dict(identity=identity, checkpoint=study.artifact(path), loss=trace)


def beat(**fields):
    pending = LOCAL / "heartbeat.pending"
    pending.write_text(study.json.dumps(dict(pid=os.getpid(), **fields)) + "\n")
    pending.replace(LOCAL / "heartbeat.json")


@torch.inference_mode()
def moments(net, x, r, unit):
    return net(torch.from_numpy(x), torch.from_numpy((r/unit).astype(np.float32))).numpy().astype(float)*unit[:, None]


def gather(z, p, d, identity):
    for key, value in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"]),
                       ("additive", z["turn_additive"])):
        np.testing.assert_array_equal(p[key], value, err_msg=identity)


def time_order(ref, q):
    with np.load(study.checked(ref), allow_pickle=False) as f:
        rec, start, offsets = (f[k] for k in ("recording", "start", "offsets"))
    np.testing.assert_array_equal(q, np.repeat(np.arange(len(rec)), np.diff(offsets)))
    for r in np.unique(rec):
        if (np.diff(start[rec == r]) <= 0).any():
            raise ValueError("Recording queries must already be in chronological order")
    return rec


@torch.inference_mode()
def infer(net, cost, gain_head, h, z, prior, rec):
    q, cv, ols, bank, native = (z[k] for k in ("q", "cv", "ols4", "bank", "native"))
    add, pair = prior["additive"], prior["full_grid_joint"]
    outputs = dict(additive=add, pair_parent=pair, **{k: cv.copy() for k in ARMS[2:]})
    counts, ledgers, known_easy = Counter(), {}, {}
    prefix = ("prefix", "prefix_two")
    x0, unit, _, _ = shared.costs.features(h, cv, ols, cv[:, None])
    row_easy = moments(net, x0[:, 0], np.zeros(len(h)), unit)[:, 4]
    peaks = turn.turn.peak(native-cv)
    for ix in slate.chunks(q):
        r = rec[int(q[ix][0])]
        if r not in ledgers:
            ledgers[r] = {k: np.zeros(6) for k in prefix}
        risk, easy = ledgers[r], known_easy.get(r, 0.)
        full_easy = float(row_easy[ix].sum())
        radii = peaks[ix][peaks[ix] > 0.]
        if len(radii):
            hist, c, o, b = (v[ix] for v in (h, cv, ols, bank))
            paths, slots = shared.slot_menu(c, b, radii)
            row = np.concatenate([s[0] for s in slots])
            arm = np.concatenate([s[1]+k*b.shape[1] for k, s in enumerate(slots)])
            opts = shared.legal_assignments(slots[0][0], slots[1][0] if len(radii) == 2 else None)[:, :len(radii)]
            feat, scale, legal, _ = shared.costs.features(hist, c, o, paths)
            if not len(opts) or not legal[row, arm].all():
                raise ValueError("No legal action at the frozen edit radii")
            x = feat[row, arm]
            action = paths[row, arm]
            a = moments(net, x, turn.turn.peak(action-c[row]), scale[row])
            np.testing.assert_allclose(a[:, 4], row_easy[ix][row], atol=5e-7, rtol=1e-5)
            predicted = a[opts].sum(1)
            util = (cost(torch.from_numpy(x)).numpy()[:, :2]*scale[row, None])[opts].sum(1).min(1)
            if len(radii) == 2:
                for start in range(0, len(opts), 128):
                    sel = opts[start:start+128]
                    paths2 = slate.assemble(c, paths, row, arm, sel)
                    util[start:start+len(sel)] += slate.WEIGHT*slate.gain(gain_head, hist, c, o, paths2, add[ix])[:, 2]
            for mode in ARMS[2:]:
                if mode == "prefix_two" and len(radii) == 1:
                    local = add[ix]
                    fx, fs, live, _ = shared.costs.features(hist, c, o, local[:, None])
                    rr = turn.turn.peak(local-c)
                    live = live[:, 0]
                    term = moments(net, fx[live, 0], rr[live], fs[live]).sum(0)
                else:
                    flavor = "prefix" if mode in prefix else mode
                    j = model.choose(util, predicted, full_easy, risk.get(mode, 0.), easy, flavor)
                    local = slate.assemble(c, paths, row, arm, opts[j:j+1])[0]
                    term = predicted[j]
                if mode in prefix:
                    risk[mode] += term
                outputs[mode][ix] = local
                counts[mode] += int(not np.allclose(local, pair[ix], atol=1e-9, rtol=1e-9))
            counts[f"queries_{len(radii)}_slots"] += 1
        known_easy[r] = easy+full_easy
    for name, p in outputs.items():
        a, b = turn.turn.peak(p-cv), peaks
        np.testing.assert_allclose(a[np.lexsort((a, q))], b[np.lexsort((b, q))], atol=1e-9, rtol=1e-9)
        if not np.array_equal(np.bincount(q, weights=a > 0.), np.bincount(q, weights=b > 0.)):
            raise ValueError(f"{name} changed the frozen edit count")
    return dict(q=q, cv=cv, ols4=ols, **outputs), dict(counts)


def state(ref, site, step):
    s = torch.load(study.checked(ref["checkpoint"]), weights_only=True, map_location="cpu")
    if (s["identity"] != ref["identity"] or s["identity"]["held"] != site or s["step"] != step
            or site in s["identity"].get("donors", s["identity"].get("donor_inputs", []))):
        raise ValueError("Incorrect producer lineage")
    return s["net"]


def freeze(key, cohorts, parent, prior, refs, cal):
    src = turn.exp.loader(cohorts)
    fits, banks, counts = {}, {}, {}
    for site in SITES:
        h, q, _, _, _ = turn.menu.parent.source_inputs(cohorts[site])
        rec = time_order(cohorts[site], q)
        for seed in SEEDS:
            name = f"{site}_{seed}"
            donors, binding = {}, {}
            for donor in SITES:
                if donor == site:
                    continue
                pair = tuple(sorted((site, donor)))
                d = src[donor]
                nn = turn.refit.pair_forecast(d["h"], pair, seed, refs, cal)
                bank = turn.append_turn(d["h"], np.concatenate((d["bank"], nn[:, None]), axis=1))
                cut = float(np.quantile(np.concatenate([src[s]["e"] for s in SITES if s not in pair]), .25))
                donors[donor] = examples(d, bank, cut)
                binding[donor] = dict(source=cohorts[donor], forecast=refs[f'{"+".join(pair)}_{seed}'], cut=cut)
            net, fits[name] = fit(key, site, seed, donors, binding)
            gain_head = slate.model.RelativeGain().eval()
            gain_head.load_state_dict(state(prior["fits"][name], site, slate.STEPS))
            cost = turn.refit.model.CostHead().eval()
            cost.load_state_dict(state(parent["fits"][name], site, turn.refit.STEPS))
            with (np.load(study.checked(parent["banks"][name]), allow_pickle=False) as z,
                  np.load(study.checked(prior["forecasts"][name]), allow_pickle=False) as p):
                gather(z, p, src[site], name)
                pred, counts[name] = infer(net, cost, gain_head, h, z, p, rec)
            banks[name] = turn.exp.previous.save_bank(LOCAL / f"{name}.npz", pred)
            beat(phase="freeze", fold=name, complete=len(fits), total=21)
            print(name, counts[name], flush=True)
    base.immutable(OUT/"freeze.json", dict(registration=key, fits=fits, banks=banks, counts=counts,
        no_future_in_policy=True, scoring_not_run=True))


def fold_score(d, pred, cut):
    rows = []
    for r in np.unique(d["rec"]):
        at = d["rec"] == r
        cv, ols, y, v = (d[k][at] for k in ("cv", "ols", "y", "v"))
        q = np.unique(d["q"][at], return_inverse=True)[1]
        paths = {k: pred[k][at] for k in ARMS}
        metrics, effect = {}, {}
        for k, p in paths.items():
            m = study.policy(p, cv, cv, y, v, ["r"]*len(y), easy_cut=cut[0], hard_cut=cut[1])
            s = m["recordings"]["r"]
            metrics[k] = dict(risk=s, ade=m["known_ADE_pixel"], fde=m["available_endpoint_FDE_diagnostic_pixel"],
                tail=m["p95_known_ADE_pixel"], hard_gain=m["subsets"]["hard"]["gain_percent"])
            for ref in ("additive", "pair_parent"):
                if k == ref or k == "additive":
                    continue
                effect[f"{k}_vs_{ref}"] = turn.expanded.actual_contrast(cv, ols, p, paths[ref], y, v, q, cut)
        rows.append(dict(record=str(r), arms=metrics, comparisons=effect))
    return rows


def summarize(rows):
    arms, contrasts = {}, {}
    for k in ARMS:
        v = [r["arms"][k] for r in rows]
        risk = [x["risk"] for x in v if x["risk"]["selected"]]
        report = attribution.summarize(risk) if risk else dict(
            views=0, switches=0, unknown_switches=0, checks={}, disjoint_recording_patterns={})
        for m in ("ade", "fde", "tail", "hard_gain"):
            x = [s[m] for s in v if s[m] is not None]
            report[m] = float(np.mean(x)) if x else None
        arms[k] = report
    for k in rows[0]["comparisons"]:
        estimates = [(r["record"], r["comparisons"][k]) for r in rows if r["comparisons"][k] is not None]
        contrasts[k] = turn.effects.summary([dict(record=r, effect=x["pair_native"], qualified=x["qualified"])
                                             for r, x in estimates])
        contrasts[k]["individual_pp"] = turn.effects.site_means([(r, x["accuracy_pp"]) for r, x in estimates])
    return dict(arms=arms, contrasts=contrasts)


def score(key, cohorts, prior):
    path = OUT/"freeze.json"
    f = study.read(path)
    expected = {f"{s}_{seed}" for s in SITES for seed in SEEDS}
    if (f["registration"] != key or not f["no_future_in_policy"] or not f["scoring_not_run"]
            or set(f["banks"]) != expected or set(f["fits"]) != expected):
        raise ValueError("Require all held forecasts frozen before readout")
    src = turn.exp.loader(cohorts)
    sites, folds, merged = {}, {}, []
    for s in SITES:
        d = src[s]
        cut = np.quantile(np.concatenate([src[t]["e"] for t in SITES if t != s]), [.25, .75])
        rows = []
        for seed in SEEDS:
            name = f"{s}_{seed}"
            fit = f["fits"][name]["identity"]
            if fit["held"] != s or set(fit["source"]) != set(SITES)-{s}:
                raise ValueError("Held source entered harm training")
            cache = LOCAL/f"{name}_score.json"
            bind = dict(registration=key, forecast=f["banks"][name], prior=prior["forecasts"][name])
            if not cache.exists():
                with (np.load(study.checked(bind["forecast"]), allow_pickle=False) as p,
                      np.load(study.checked(bind["prior"]), allow_pickle=False) as z):
                    for k, x in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"]),
                                 ("additive", z["additive"]), ("pair_parent", z["full_grid_joint"])):
                        np.testing.assert_array_equal(p[k], x)
                    # Materialize columns once, rather than repeatedly decompress per recording.
                    values = {k: p[k] for k in ARMS}
                base.immutable(cache, dict(binding=bind, rows=fold_score(d, values, cut)))
            saved = study.read(cache)
            if saved["binding"] != bind:
                raise ValueError("Score resume inputs changed")
            rows.extend(saved["rows"])
            folds[name] = study.artifact(cache)
            beat(phase="score", fold=name, complete=len(folds), total=len(expected))
            print(name, len(folds), flush=True)
        sites[s] = summarize(rows)
        merged.extend({**r, "record": f"{s}:{r['record']}"} for r in rows)
    total = summarize(merged)
    for name, r in total["contrasts"].items():
        for k in ("site_mean_native", "individual_pp"):
            r.pop(k)
            r[f"{k}_location_bootstrap"] = turn.effects.interval([s["contrasts"][name][k] for s in sites.values()])
    for v in total["arms"].values():
        if (v["views"], v["switches"]) != (439, 3690):
            raise ValueError("Changed evaluation population or intervention budget")
    parent_risk = study.read(slate.OUT/"risk_attribution.json")["risk"]
    for a, b in (("additive", "additive"), ("pair_parent", "full_grid_joint")):
        if total["arms"][a]["checks"] != parent_risk[b]["checks"]:
            raise ValueError("Frozen risk controls did not replay")
    base.immutable(OUT/"readout.json", dict(registration=key, freeze=study.artifact(path), checkpoints=folds,
        totals=total, sites=sites, counts=dict(sum((Counter(v) for v in f["counts"].values()), Counter())),
        new_fits=len(expected), result_source="fresh_run", original_2pct=True,
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False))
    print(study.json.dumps(total, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, prior, refs, cal = register()
        if phase == "freeze":
            freeze(key, cohorts, parent, prior, refs, cal)
        else:
            score(key, cohorts, prior)


if __name__ == "__main__":
    main()
