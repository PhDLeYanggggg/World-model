"""Location-excluded acceleration refit and matched support-gate experiment."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_optional_edits as prev
from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.world_model import m3w_approach_candidate_paths as motion
from src.world_model import m3w_supported_acceleration_policy as model

np, torch, study, base, turn, slate, shared = prev.np, prev.torch, prev.study, prev.base, prev.turn, prev.slate, prev.shared
risk = prev.risk
SITES, SEEDS = prev.SITES, prev.SEEDS
OUT, LOCAL = prev.OUT.parent/"opendd_supported_acceleration_v1", prev.LOCAL.parent/"opendd_supported_acceleration_v1"
STEPS, BATCH, EVERY = 600, 128, 50
RULES = dict(moments=0., support25=.25, support50=.5)
GROUPS = ("forced", *RULES)
ARMS = tuple(f"{k}_{a}" for k in GROUPS for a in ("add", "joint"))
F = torch.nn.functional


def beat(**info):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), **info))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    _, cohorts, parent, prior, _ = prev.register()
    _, checked, _, _, _, _, _, refs, cal = turn.refit.register()
    if checked != cohorts:
        raise ValueError("Donor and evaluation cohorts disagree")
    p = OUT/"registration.json"
    base.immutable(p, dict(
        cohorts=cohorts, code=[study.artifact(Path(x)) for x in (__file__, model.__file__, motion.__file__)],
        parent_turn=study.artifact(turn.OUT/"freeze.json"), parent_relative=study.artifact(slate.OUT/"freeze.json"),
        parent_optional=study.artifact(prev.OUT/"readout.json"),
        menu="same parent trajectories plus only the decaying acceleration candidate",
        donor_rule="exclude held site; source self predictors and easy cuts exclude donor and held",
        fit="two signed lower gains (Huber), six normalized risk moments (MSE), edited easy support (BCE)",
        gates="chronological intersection of both arms' predicted constraints, evidence max across selected actors",
        support_thresholds=RULES, threshold_selection=False, calibrated_risk_guarantee=False,
        match="forced exactly parent slots; optional common subset per query, same counts and peaks within pair",
        non_release="return exact CV; no empty-coverage risk pass",
        steps=STEPS, batch=BATCH, checkpoint_every=EVERY, seeds=SEEDS, sites=SITES, workers=0, threads=4,
        observed=8, horizon=12, original_2pct=True, reserved_outcomes_opened=False, independent_confirmation=False))
    return base.sha(p), cohorts, parent, prior, refs, cal


def with_acceleration(h, cv, bank):
    d = np.diff(h, axis=1)
    a = (d[:, -2:].mean(1)-d[:, -4:-2].mean(1))/2.
    phase = np.cumsum(np.cumsum(.72 ** np.arange(12)))
    return np.concatenate((bank, (cv+a[:, None]*phase[None, :, None])[:, None]), axis=1)


def examples(d, bank, cut):
    h, cv, ols, y, v = (d[k] for k in ("h", "cv", "ols", "y", "v"))
    paths = cv[:, None]+np.concatenate([a*(bank-cv[:, None]) for a in turn.refit.model.AMPLITUDES], axis=1)
    x, scale, live, _ = shared.costs.features(h, cv, ols, paths)
    _, _, gain, _ = candidate_terms(cv, ols, paths, y, v)
    m = risk.model.target(cv, paths, y, v, cut)
    event = m[..., 4] > 0.
    rr = turn.turn.peak(paths-cv[:, None])/scale[:, None]
    return tuple(torch.from_numpy(a[live].astype(np.float32)) for a in (
        x, rr, gain/(12.*scale[:, None, None]), m/scale[:, None, None], event))


def fit(key, held, seed, donors, binding):
    name = f"{held}_{seed}"
    path = LOCAL/f"{name}.pt"
    norm = (sum(d[3].square().mean(0) for d in donors.values())/len(donors)).sqrt().clamp_min(.01)
    ident = dict(registration=key, held=held, seed=seed, donors=binding, moments_norm=norm.tolist(),
                 donor_examples={s: len(x[0]) for s, x in donors.items()})
    torch.manual_seed(seed)
    net = model.SupportedCost()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(3871+seed)
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["identity"] != ident or ck["step"] not in range(EVERY, STEPS+1, EVERY):
            raise ValueError("Training checkpoint identity differs")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        step, trace, rng.bit_generator.state = ck["step"], ck["trace"], ck["rng"]
    data = list(donors.values())
    for step in range(step+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(data), 1./len(data)))
        indices = [rng.integers(len(d[0]), size=n) for d, n in zip(data, counts, strict=True)]
        x, r, g, m, s = (torch.cat([d[k][i] for d, i in zip(data, indices, strict=True)]) for k in range(5))
        gain, moments, support = net(x, r)
        terms = (F.smooth_l1_loss(gain, g, beta=.05), F.mse_loss(moments/norm, m/norm),
                 F.binary_cross_entropy_with_logits(support, s))
        loss = terms[0]+terms[1]+.25*terms[2]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % EVERY == 0:
            trace.append([float(t.detach()) for t in terms])
            p = path.with_suffix(".pending")
            torch.save(dict(identity=ident, net=net.state_dict(), optimizer=opt.state_dict(),
                            step=step, trace=trace, rng=rng.bit_generator.state), p)
            p.replace(path)
            beat(phase="fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(identity=ident, checkpoint=study.artifact(path), trace=trace)


@torch.inference_mode()
def infer(net, relative, h, q, cv, ols, bank, original, rec):
    out = {k: cv.copy() for k in ARMS}
    peaks = turn.turn.peak(original-cv)
    ledgers, support, easy, decisions = {}, {}, {}, Counter()
    x, scale, _, _ = shared.costs.features(h, cv, ols, cv[:, None])
    _, m, _ = net(torch.from_numpy(x[:, 0]), torch.zeros(len(h)))
    easy_mass = m[:, 4].numpy()*scale
    for ix in slate.chunks(q):
        r = rec[q[ix][0]]
        easy[r] = easy.get(r, 0.)+float(easy_mass[ix].sum())
        radii = peaks[ix][peaks[ix] > 0.]
        if not len(radii):
            continue
        c, o, hist = cv[ix], ols[ix], h[ix]
        paths, row, col, opts, group = prev.options(c, bank[ix], radii)
        feat, unit, legal, _ = shared.costs.features(hist, c, o, paths)
        if not legal[row, col].all():
            raise ValueError("An existing slot lost its causal action")
        rr = turn.turn.peak(paths[row, col]-c[row])/unit[row]
        g, m, s = net(torch.from_numpy(feat[row, col]), torch.from_numpy(rr.astype(np.float32)))
        g = g.numpy()*unit[row, None]
        m = model.packed(m.numpy()*unit[row, None], opts).sum(1)
        s = model.packed(s.sigmoid().numpy()[:, None], opts).max(1)[:, 0]
        gs = model.packed(g, opts)
        gains, add = gs.sum(1), gs.min(2).sum(1)
        joint = gains.min(1)
        forced = np.flatnonzero(group == (2 if len(radii) == 2 else 0))
        best_add = forced[int(add[forced].argmax())]
        ref = prev.paint(c, paths, row, col, opts[best_add])
        # Singletons do not get a learned pair bonus.
        double = np.flatnonzero(group == 2)
        for start in range(0, len(double), 128):
            at = double[start:start+128]
            trial = slate.assemble(c, paths, row, col, opts[at])
            joint[at] += slate.WEIGHT*slate.gain(relative, hist, c, o, trial, ref)[:, 2]
        for rule in GROUPS:
            if rule == "forced":
                live = forced
            else:
                live = np.flatnonzero(np.logical_and.reduce([
                    model.feasible(gains, m, s, easy[r], ledgers.get((r, rule, arm), 0.),
                                   support.get((r, rule, arm), 0.), RULES[rule])
                    for arm in ("add", "joint")]))
            if not len(live):
                decisions[f"{rule}_abstain"] += 1
                continue
            pick = live[int(joint[live].argmax())]
            same = live[group[live] == group[pick]]
            a = same[int(add[same].argmax())]
            j = a if group[pick] != 2 else pick
            for arm, i in (("add", a), ("joint", j)):
                out[f"{rule}_{arm}"][ix] = prev.paint(c, paths, row, col, opts[i])
                k = (r, rule, arm)
                if rule != "forced":
                    ledgers[k] = ledgers.get(k, 0.)+m[i]
                    support[k] = max(support.get(k, 0.), s[i])
            decisions[f"{rule}_released"] += 1
            decisions[f"{rule}_changed"] += int(j != a)
    for rule in GROUPS:
        left, right = (turn.turn.peak(out[f"{rule}_{a}"]-cv) for a in ("add", "joint"))
        np.testing.assert_allclose(left[np.lexsort((left, q))], right[np.lexsort((right, q))], atol=1e-8, rtol=1e-8)
        if rule == "forced":
            np.testing.assert_allclose(left[np.lexsort((left, q))], peaks[np.lexsort((peaks, q))], atol=1e-8, rtol=1e-8)
    return dict(q=q, cv=cv, ols4=ols, **out), dict(decisions)


def freeze(key, cohorts, parent, prior, refs, cal):
    src = turn.exp.loader(cohorts)
    fits, forecasts, counts = {}, {}, {}
    for held in SITES:
        h, q, cv, ols, _ = turn.menu.parent.source_inputs(cohorts[held])
        rec = risk.time_order(cohorts[held], q)
        for seed in SEEDS:
            name = f"{held}_{seed}"
            fold = LOCAL/f"{name}_freeze.json"
            if not fold.exists():
                donors, binding = {}, {}
                for donor in SITES:
                    if donor == held:
                        continue
                    d = src[donor]
                    pair = tuple(sorted((held, donor)))
                    nn = turn.refit.pair_forecast(d["h"], pair, seed, refs, cal)
                    bank = turn.append_turn(d["h"], np.concatenate((d["bank"], nn[:, None]), axis=1))
                    cut = float(np.quantile(np.concatenate([src[s]["e"] for s in SITES if s not in pair]), .25))
                    donors[donor] = examples(d, with_acceleration(d["h"], d["cv"], bank), cut)
                    binding[donor] = dict(rows=cohorts[donor], cut=cut, model=refs[f'{"+".join(pair)}_{seed}'])
                net, receipt = fit(key, held, seed, donors, binding)
                del donors
                relative = slate.model.RelativeGain().eval()
                relative.load_state_dict(risk.state(prior["fits"][name], held, slate.STEPS))
                with np.load(study.checked(parent["banks"][name]), allow_pickle=False) as z:
                    for k, x in (("q", q), ("cv", cv), ("ols4", ols)):
                        np.testing.assert_array_equal(x, z[k])
                    bank = with_acceleration(h, cv, z["bank"])
                    values, c = infer(net, relative, h, q, cv, ols, bank, z["native"], rec)
                forecast = turn.exp.previous.save_bank(LOCAL/f"{name}.npz", values)
                base.immutable(fold, dict(registration=key, fit=receipt, forecast=forecast, counts=c))
            saved = study.read(fold)
            if saved["registration"] != key or set(saved["fit"]["identity"]["donors"]) != set(SITES)-{held}:
                raise ValueError("Held site entered fit or stale fold")
            study.checked(saved["fit"]["checkpoint"])
            study.checked(saved["forecast"])
            fits[name], forecasts[name], counts[name] = saved["fit"], saved["forecast"], saved["counts"]
            beat(phase="freeze", fold=name, completed=len(fits), total=len(SITES)*len(SEEDS))
            print(name, counts[name], flush=True)
    base.immutable(OUT/"freeze.json", dict(registration=key, fits=fits, forecasts=forecasts,
        counts=counts, held_future_used_in_policy=False, complete_before_scoring=True))


def read_fold(d, paths, cut):
    out = []
    for r in np.unique(d["rec"]):
        at = d["rec"] == r
        cv, ols, y, v = (d[k][at] for k in ("cv", "ols", "y", "v"))
        q = np.unique(d["q"][at], return_inverse=True)[1]
        w = 1./np.bincount(q)[q]
        den = float(w @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        arms, comparisons = {}, {}
        for arm in ARMS:
            p = paths[arm][at]
            m = study.policy(p, cv, cv, y, v, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])
            gain = [100.*float(w@turn.exact.headroom.lower(b, p, y, v))/den for b in (cv, ols)] if den > 0. else None
            arms[arm] = dict(risk=m["recordings"]["r"], gain_pp=gain, ade=m["known_ADE_pixel"],
                fde=m["available_endpoint_FDE_diagnostic_pixel"], tail=m["p95_known_ADE_pixel"],
                hard=m["subsets"]["hard"]["gain_percent"])
        for rule in GROUPS:
            a, j = (paths[f"{rule}_{arm}"][at] for arm in ("add", "joint"))
            comparisons[rule] = turn.expanded.actual_contrast(cv, ols, j, a, y, v, q, cut)
        out.append(dict(record=str(r), arms=arms, comparisons=comparisons))
    return out


def summarize(rows):
    summary = {}
    for arm in ARMS:
        x = [r["arms"][arm] for r in rows]
        edits = [a["risk"] for a in x if a["risk"]["selected"]]
        summary[arm] = dict(views=len(x), edited_views=len(edits), edits=sum(r["selected"] for r in edits),
            risk_pass=sum(r["supported"] for r in edits), undefined_easy=sum(r["completion_easy_risk_upper"] is None for r in edits),
            easy_over_2pct=sum((r["known_easy_net_degradation"] or 0.) > .02 for r in edits),
            dual_accuracy_risk_pass=sum(a["risk"]["supported"] and a["gain_pp"] is not None
                                       and min(a["gain_pp"]) > 1e-8 for a in x),
            gain_pp=turn.effects.site_means([(r["record"], r["arms"][arm]["gain_pp"]) for r in rows
                                            if r["arms"][arm]["gain_pp"] is not None]),
            **{k: float(np.mean(z)) if (z := [r[k] for r in x if r[k] is not None]) else None
               for k in ("ade", "fde", "tail", "hard")})
    for rule in GROUPS:
        pairs = [(r["record"], r["comparisons"][rule]) for r in rows if r["comparisons"][rule] is not None]
        summary[rule] = turn.effects.summary([dict(record=r, effect=v["pair_native"], qualified=v["qualified"])
                                               for r, v in pairs])
    return summary


def score(key, cohorts):
    path = OUT/"freeze.json"
    frozen = study.read(path)
    if (frozen["registration"] != key or frozen["held_future_used_in_policy"] or not frozen["complete_before_scoring"]
            or set(frozen["forecasts"]) != {f"{s}_{k}" for s in SITES for k in SEEDS}):
        raise ValueError("Require completed location-excluded forecasts")
    src, sites, rows = turn.exp.loader(cohorts), {}, []
    for site in SITES:
        d = src[site]
        cut = np.quantile(np.concatenate([v["e"] for s, v in src.items() if s != site]), [.25, .75])
        parts = []
        for seed in SEEDS:
            name = f"{site}_{seed}"
            with np.load(study.checked(frozen["forecasts"][name]), allow_pickle=False) as z:
                for k, x in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"])):
                    np.testing.assert_array_equal(z[k], x)
                parts.extend(read_fold(d, {k: z[k] for k in ARMS}, cut))
            beat(phase="score", fold=name, rows=len(parts))
        sites[site] = summarize(parts)
        rows.extend({**r, "record": f"{site}:{r['record']}"} for r in parts)
    total = summarize(rows)
    for arm in ARMS:
        total[arm]["gain_bootstrap_pp"] = turn.effects.interval([s[arm]["gain_pp"] for s in sites.values()])
    for rule in GROUPS:
        total[rule]["pair_bootstrap_native"] = turn.effects.interval([s[rule]["site_mean_native"] for s in sites.values()])
        if total[f"{rule}_add"]["edits"] != total[f"{rule}_joint"]["edits"]:
            raise ValueError("Different realized budget")
    base.immutable(OUT/"readout.json", dict(registration=key, freeze=study.artifact(path), result_source="fresh_run",
        fits=len(frozen["fits"]), seed_count=len(SEEDS), recording_views=len(rows),
        unique_recordings=len({r["record"] for r in rows}), sites=sites, totals=total,
        counts=dict(sum((Counter(v) for v in frozen["counts"].values()), Counter())),
        original_2pct=True, reserved_outcomes_opened=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    print(study.json.dumps(total, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for path in (OUT, LOCAL):
        path.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, prior, refs, cal = register()
        if phase == "freeze":
            freeze(key, cohorts, parent, prior, refs, cal)
        else:
            score(key, cohorts)
        beat(phase=phase+"_complete")


if __name__ == "__main__":
    main()
