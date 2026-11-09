"""Train direct conservative slate gain on the frozen OpenDD turn menu."""

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
from scripts import run_m3w_opendd_turn_intervention as turn
from src.world_model import m3w_slate_relative_gain as model

base, study, np, torch = turn.base, turn.study, turn.np, turn.torch
shared = turn.refit.model.shared
SITES, SEEDS = turn.SITES, turn.SEEDS
OUT = turn.OUT.parent / "opendd_slate_relative_gain_v1"
LOCAL = turn.LOCAL.parent / OUT.name
STEPS, BATCH, CHECKPOINT, OPTIONS = 400, 128, 50, 16
WEIGHT = .25
ARMS = ("additive", "risk_joint", "ols_proxy_joint", "observed_joint", "full_grid_joint")


def register():
    key, cohorts, _, _, _, _, _, refs, cal = turn.register()
    files = [turn.OUT / f"{n}.json" for n in ("registration", "freeze", "readout")]
    frozen, readout = map(study.read, files[1:])
    expected = {f"{site}_{seed}" for site in SITES for seed in SEEDS}
    if (frozen["registration"] != key or frozen["held_labels_used_in_model"]
            or readout["freeze"] != study.artifact(files[1]) or readout["reserved_outcomes_opened"]
            or set(frozen["banks"]) != expected or set(frozen["fits"]) != expected):
        raise ValueError("Require all frozen and verified turn-menu controls")
    p = OUT / "registration.json"
    base.immutable(p, dict(parents=[study.artifact(f) for f in files],
        code=[study.artifact(Path(f)) for f in (__file__, model.__file__, shared.__file__,
             turn.expanded.frontier.__file__)],
        sites=SITES, seeds=SEEDS, steps=STEPS, batch=BATCH, checkpoint_every=CHECKPOINT,
        sampled_slates_per_reference=OPTIONS, pair_weight=WEIGHT, original_2pct=True,
        donors="outer- and donor-excluded candidate producer; no donor policy with outer exposure",
        donor_slots="two distinct recipients drawn from legal nonzero path peaks; no future conditioning",
        references="OLS-nearest slate and a uniformly sampled legal slate at the same two peaks",
        loss="equal smooth L1 for antisymmetric known grid gain and symmetric missing-grid debit",
        comparisons=ARMS, observed_steps=8, forecast_steps=12,
        development_only=True, independent_confirmation=False, no_threshold_search=True))
    return base.sha(p), cohorts, frozen, refs, cal


def beat(**fields):
    p = LOCAL / "heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), **fields)) + "\n")
    p.replace(LOCAL / "heartbeat.json")


def legal(cv, bank, radii):
    paths, slots = shared.slot_menu(cv, bank, radii)
    row = np.concatenate([s[0] for s in slots])
    arm = np.concatenate([s[1] + k*bank.shape[1] for k, s in enumerate(slots)])
    opts = shared.legal_assignments(slots[0][0], slots[1][0])
    if not len(opts):
        raise ValueError("Two-edit menu has no legal choice")
    return paths, row, arm, opts


def assemble(cv, paths, row, arm, opts):
    result = np.broadcast_to(cv, (len(opts), *cv.shape)).copy()
    result[np.arange(len(opts))[:, None], row[opts]] = paths[row[opts], arm[opts]]
    return result


def chunks(q):
    cuts = np.r_[0, np.flatnonzero(np.diff(q))+1, len(q)]
    return (slice(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:], strict=True))


def examples(d, bank, seed):
    rng = np.random.default_rng(seed)
    data = []
    counts = Counter()
    for ix in chunks(d["q"]):
        h, cv, ols, b = (x[ix] for x in (d["h"], d["cv"], d["ols"], bank))
        if len(h) < 3:
            continue
        peak = np.linalg.norm(b-cv[:, None], axis=-1).max(-1)
        rr, cc = np.nonzero(peak > 1e-9)
        if len(np.unique(rr)) < 2:
            continue
        a = int(rng.integers(len(rr)))
        second = rng.choice(np.flatnonzero(rr != rr[a]))
        radii = peak[rr[[a, second]], cc[[a, second]]]
        paths, row, arm, opts = legal(cv, b, radii)
        draw = rng.choice(len(opts), size=min(OPTIONS, len(opts)), replace=False)
        p = assemble(cv, paths, row, arm, opts[draw])
        error = np.linalg.norm(paths[row, arm] - ols[row], axis=-1).mean(1)
        ols_anchor = int(error[opts].sum(1).argmin())
        reference_ids = {ols_anchor, int(rng.integers(len(opts)))}
        y, v = d["y"][ix], d["v"][ix]
        counts["queries"] += 1
        counts["queries_with_observed_pairs"] += int((v.sum(0) >= 2).any())
        for anchor in sorted(reference_ids):
            ref = assemble(cv, paths, row, arm, opts[anchor:anchor+1])[0]
            x, xr, bound, unit, _ = model.features(h, cv, ols, p, ref)
            labels = model.labels(p, ref, y, v, unit)
            live = bound > 1e-9
            if live.any():
                data.append((x[live], xr[live], bound[live], labels[live]))
                counts["examples"] += int(live.sum())
    return (tuple(torch.from_numpy(np.concatenate(x)) for x in zip(*data, strict=True))
            if data else None), dict(counts)


def fit(key, site, seed, donors, input_refs):
    identity = dict(registration=key, held=site, seed=seed, donor_inputs=input_refs,
                    donor_rows={s: len(x[-1]) for s, x in donors.items()})
    path = LOCAL / f"{site}_{seed}.pt"
    torch.manual_seed(seed)
    net = model.RelativeGain()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+6011)
    start, trace = 0, []
    if path.exists():
        state = torch.load(path, weights_only=True, map_location="cpu")
        if state["identity"] != identity or state["step"] not in range(CHECKPOINT, STEPS+1, CHECKPOINT):
            raise ValueError("Mismatched resume checkpoint")
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["optimizer"])
        start, trace, rng.bit_generator.state = state["step"], state["trace"], state["rng"]
    xs = list(donors.values())
    for step in range(start+1, STEPS+1):
        n = rng.multinomial(BATCH, np.full(len(xs), 1./len(xs)))
        ii = [rng.integers(len(x[-1]), size=k) for x, k in zip(xs, n, strict=True)]
        x, xr, b, y = (torch.cat([z[k][i] for z, i in zip(xs, ii, strict=True)]) for k in range(4))
        loss = torch.nn.functional.smooth_l1_loss(net(x, xr, b), y, beta=.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % CHECKPOINT == 0:
            trace.append(float(loss.detach()))
            pending = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), pending)
            pending.replace(path)
            beat(phase="fit", fold=f"{site}_{seed}", step=step, total=STEPS)
    return net.eval(), dict(identity=identity, checkpoint=study.artifact(path), loss=trace)


@torch.inference_mode()
def gain(net, h, cv, ols, p, ref):
    x, xr, b, scale, proxy = model.features(h, cv, ols, p, ref)
    val = net(*(torch.from_numpy(k) for k in (x, xr, b))).numpy().astype(float) * scale
    return np.column_stack((proxy, val[:, 0], val[:, 0]-val[:, 1]))


@torch.inference_mode()
def infer(net, cost, h, z):
    q, cv, ols, bank, native, control, additive, risk = (z[k] for k in
        ("q", "cv", "ols4", "bank", "native", "control", "turn_additive", "turn_joint"))
    output = dict(additive=additive, risk_joint=risk, **{k: risk.copy() for k in ARMS[2:]})
    counts = Counter()
    probe = []
    peak = turn.turn.peak(native-cv)
    for ix in chunks(q):
        r = peak[ix][peak[ix] > 0.]
        if len(r) != 2:
            continue
        g, c, o, b = (x[ix] for x in (h, cv, ols, bank))
        paths, row, arm, opts = legal(c, b, r)
        x, scale, valid, _ = shared.costs.features(g, c, o, paths)
        if not valid[row, arm].all():
            raise ValueError("Frozen edit became invalid")
        value = cost(torch.from_numpy(x[row, arm])).numpy().astype(float) * scale[row, None]
        a, j = turn.refit.model.ranks(value, opts)
        for score, ref in ((a, additive), (j, risk)):
            ix_best = int(score.argmax())
            np.testing.assert_array_equal(assemble(c, paths, row, arm, opts[ix_best:ix_best+1])[0], ref[ix])
        rank = []
        for start in range(0, len(opts), 128):
            take = opts[start:start+128]
            p = assemble(c, paths, row, arm, take)
            rank.append(gain(net, g, c, o, p, additive[ix]))
        rank = np.concatenate(rank)
        idx = (j[:, None] + WEIGHT*rank).argmax(0)
        for k, sel in zip(ARMS[2:], idx, strict=True):
            output[k][ix] = assemble(c, paths, row, arm, opts[sel:sel+1])[0]
            counts[k] += int(not np.allclose(output[k][ix], risk[ix], atol=1e-9, rtol=1e-9))
        # Freeze score probes on equally spaced legal options before evaluation labels are opened.
        sample = np.linspace(0, len(opts)-1, min(8, len(opts)), dtype=int)
        probe.extend((int(q[ix][0]), int(option), *rank[option]) for option in sample)
        counts["two_edit_queries"] += 1
    turn.pin_singletons(cv, q, control, native, output)
    return dict(q=q, cv=cv, ols4=ols, **output, probe=np.asarray(probe).reshape(-1, 5)), dict(counts)


def freeze(key, cohorts, parent, refs, cal):
    src = turn.exp.loader(cohorts)
    fits, forecasts, counts = {}, {}, {}
    for site in SITES:
        for seed in SEEDS:
            name = f"{site}_{seed}"
            donors, receipts, support = {}, {}, {}
            for donor in SITES:
                if donor == site:
                    continue
                pair = tuple(sorted((donor, site)))
                d = src[donor]
                nn = turn.refit.pair_forecast(d["h"], pair, seed, refs, cal)
                bank = turn.append_turn(d["h"], np.concatenate((d["bank"], nn[:, None]), axis=1))
                train, support[donor] = examples(d, bank, seed + 1000*SITES.index(donor))
                if train is not None:
                    donors[donor] = train
                    receipts[donor] = dict(source=cohorts[donor], forecast=refs[f'{"+".join(pair)}_{seed}'])
            if not donors:
                raise ValueError("No donor examples")
            net, fits[name] = fit(key, site, seed, donors, receipts)
            earlier = parent["fits"][name]
            state = torch.load(study.checked(earlier["checkpoint"]), weights_only=True, map_location="cpu")
            if (state["identity"] != earlier["identity"] or state["identity"]["held"] != site
                    or site in state["identity"]["donors"] or state["step"] != turn.refit.STEPS):
                raise ValueError("Frozen unary costs have different training lineage")
            cost = turn.refit.model.CostHead().eval()
            cost.load_state_dict(state["net"])
            # Inference has no access to labels or future masks.
            h, q, _, _, _ = turn.menu.parent.source_inputs(cohorts[site])
            with np.load(study.checked(parent["banks"][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(q, z["q"])
                pred, switches = infer(net, cost, h, z)
            counts[name] = dict(donor_examples=support, changed_queries=switches)
            forecasts[name] = turn.exp.previous.save_bank(LOCAL / f"{name}.npz", pred)
            beat(phase="freeze", fold=name, completed=len(fits), total=21)
            print(name, switches, flush=True)
    base.immutable(OUT / "freeze.json", dict(registration=key, fits=fits, forecasts=forecasts,
        counts=counts, held_inference_causal_only=True, fold_scoring_not_run=True))


def probe_metrics(d, z, p):
    terms = Counter()
    bank, native = z["bank"], z["native"]
    peak = turn.turn.peak(native-d["cv"])
    saved = p["probe"]
    q = d["q"]
    by_q = {int(q[ix][0]): ix for ix in chunks(q)}
    for query in np.unique(saved[:, 0]):
        ix = by_q[int(query)]
        at = saved[:, 0] == query
        select = saved[at, 1].astype(int)
        r = peak[ix][peak[ix] > 0.]
        cv, b = d["cv"][ix], bank[ix]
        paths, row, arm, opts = legal(cv, b, r)
        forecasts = assemble(cv, paths, row, arm, opts[select])
        target = model.labels(forecasts, p["additive"][ix], d["y"][ix], d["v"][ix], 1.).astype(float)
        full = target[:, 0]-target[:, 1]
        guess = saved[at, 2:]
        terms["options"] += len(full)
        terms["null_squared"] += float((full**2).sum())
        for k, val in zip(("proxy", "observed", "full_grid"), guess.T, strict=True):
            terms[f"{k}_squared"] += float(((full-val)**2).sum())
        terms["observed_target_squared"] += float(((target[:, 0]-guess[:, 1])**2).sum())
        terms["debit_squared"] += float(((target[:, 1] - guess[:, 1]+guess[:, 2])**2).sum())
    return dict(terms)


def fold_score(d, pred, cut):
    rows = []
    for record in np.unique(d["rec"]):
        at = d["rec"] == record
        q = np.unique(d["q"][at], return_inverse=True)[1]
        cv, ols, y, v = (d[k][at] for k in ("cv", "ols", "y", "v"))
        outputs = {k: pred[k][at] for k in ARMS}
        row = dict(record=str(record), comparisons={}, arms={})
        for arm, p in outputs.items():
            s = study.policy(p, cv, cv, y, v, ["record"]*len(y), easy_cut=cut[0], hard_cut=cut[1])
            r = s["recordings"]["record"]
            row["arms"][arm] = dict(risk=r["supported"], easy=r["known_easy_net_degradation"],
                edits=r["selected"], ade=s["known_ADE_pixel"], fde=s["available_endpoint_FDE_diagnostic_pixel"])
            for ref in ("additive", "risk_joint"):
                if arm == ref or (arm == "additive" and ref == "risk_joint"):
                    continue
                vj = turn.expanded.actual_contrast(cv, ols, p, outputs[ref], y, v, q, cut)
                row["comparisons"][f"{arm}_vs_{ref}"] = vj
        rows.append(row)
    return rows


def summarize(rows):
    result = dict(arms={}, comparisons={})
    for arm in ARMS:
        sample = [r["arms"][arm] for r in rows]
        easy = [s["easy"] for s in sample if s["easy"] is not None]
        result["arms"][arm] = dict(edits=sum(s["edits"] for s in sample), risk_pass=sum(s["risk"] for s in sample),
            easy_estimable=len(easy), easy_above_2pct=sum(x > .02 for x in easy),
            easy_mean=float(np.mean(easy)) if easy else None)
        for metric in ("ade", "fde"):
            vals = [r[metric] for r in sample if r[metric] is not None]
            result["arms"][arm][metric] = float(np.mean(vals)) if vals else None
    for k in rows[0]["comparisons"]:
        scored = [(r["record"], r["comparisons"][k]) for r in rows if r["comparisons"][k] is not None]
        items = [dict(record=rec, effect=x["pair_native"], qualified=x["qualified"]) for rec, x in scored]
        result["comparisons"][k] = turn.effects.summary(items)
        result["comparisons"][k]["site_mean_accuracy_pp"] = turn.effects.site_means(
            [(rec, x["accuracy_pp"]) for rec, x in scored])
    return result


def score(key, cohorts, parent):
    path = OUT / "freeze.json"
    f = study.read(path)
    expected = {f"{s}_{seed}" for s in SITES for seed in SEEDS}
    if (f["registration"] != key or not f["held_inference_causal_only"] or not f["fold_scoring_not_run"]
            or set(f["forecasts"]) != expected or set(f["fits"]) != expected):
        raise ValueError("All predictions must be frozen before scoring")
    src = turn.exp.loader(cohorts)
    by_site, receipts, all_rows, probes = {}, {}, [], Counter()
    for s in SITES:
        d = src[s]
        cut = np.quantile(np.concatenate([src[t]["e"] for t in SITES if t != s]), [.25, .75])
        rows, site_probe = [], Counter()
        for seed in SEEDS:
            name = f"{s}_{seed}"
            ident = f["fits"][name]["identity"]
            if ident["held"] != s or set(ident["donor_rows"]) != set(ident["donor_inputs"]) or s in ident["donor_rows"]:
                raise ValueError("Invalid donor fit identity")
            binding = dict(registration=key, pred=f["forecasts"][name], parent=parent["banks"][name])
            cache = LOCAL / f"{name}_score.json"
            if not cache.exists():
                with (np.load(study.checked(binding["pred"]), allow_pickle=False) as p,
                      np.load(study.checked(binding["parent"]), allow_pickle=False) as z):
                    for k, x in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"]),
                                 ("additive", z["turn_additive"]), ("risk_joint", z["turn_joint"])):
                        np.testing.assert_array_equal(p[k], x)
                    base.immutable(cache, dict(binding=binding, rows=fold_score(d, p, cut),
                                               probe=probe_metrics(d, z, p)))
            saved = study.read(cache)
            if saved["binding"] != binding:
                raise ValueError("Held scoring checkpoint mismatch")
            rows.extend(saved["rows"])
            site_probe.update(saved["probe"])
            receipts[name] = study.artifact(cache)
            beat(phase="score", fold=name, completed=len(receipts), total=len(expected))
            print(name, len(receipts), flush=True)
        by_site[s] = dict(**summarize(rows), probe=dict(site_probe))
        probes.update(site_probe)
        all_rows.extend({**r, "record": f"{s}:{r['record']}"} for r in rows)
    total = summarize(all_rows)
    for k, v in total["comparisons"].items():
        for metric, out in (("site_mean_native", "location_bootstrap_native"),
                            ("site_mean_accuracy_pp", "location_bootstrap_accuracy_pp")):
            v.pop(metric)
            v[out] = turn.effects.interval([s["comparisons"][k][metric] for s in by_site.values()])
    if {r["edits"] for r in total["arms"].values()} != {3690}:
        raise ValueError("The registered realized edit counts did not replay")
    base.immutable(OUT / "readout.json", dict(registration=key, freeze=study.artifact(path),
        checkpoints=receipts, sites=by_site, totals=total, prediction_probe=dict(probes),
        result_source="fresh_run", frozen_inputs="cached_verified", new_fits=len(expected),
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    print(study.json.dumps(dict(totals=total, prediction_probe=dict(probes)), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for path in (OUT, LOCAL):
        path.mkdir(parents=True, exist_ok=True)
    with (LOCAL / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, refs, cal = register()
        if args.phase == "freeze":
            freeze(key, cohorts, parent, refs, cal)
        else:
            score(key, cohorts, parent)


if __name__ == "__main__":
    main()
