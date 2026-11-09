"""Separate the capacity of a past-only turn forecast from learned joint utility."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Use the native arm64 Python environment")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_opendd_expanded_interaction_frontier as expanded
from scripts import report_m3w_opendd_interaction_effects as effects
from scripts import train_m3w_opendd_expanded_costs as refit
from src.world_model import m3w_radius_matched_turn_menu as turn

base, study, np, torch = refit.base, refit.study, refit.np, refit.torch
menu, exp, exact = refit.menu, refit.exp, expanded.original.original
OUT = expanded.OUT.parent / "opendd_turn_intervention_v1"
LOCAL = expanded.LOCAL.parent / OUT.name
SITES, SEEDS = refit.SITES, refit.SEEDS
ARMS = ("previous_additive", "previous_joint", "turn_additive", "turn_joint")


def register():
    parent_key, cohorts, _, old_menu, _, weights, producer, refs, calibration = refit.register()
    parent, upper = refit.OUT / "freeze.json", expanded.OUT / "readout.json"
    old_fit, old_upper = map(study.read, (parent, upper))
    if (old_fit["registration"] != parent_key or not old_fit["held_scoring_not_run"]
            or old_upper["reserved_outcomes_opened"]):
        raise ValueError("Expected verified older development controls")
    p = OUT / "registration.json"
    base.immutable(p, dict(parents=[study.artifact(x) for x in (parent, upper, refit.OUT / "registration.json")],
        code=[study.artifact(Path(x)) for x in (__file__, turn.__file__, turn.velocity.__code__.co_filename,
             refit.__file__, expanded.__file__, effects.__file__, exact.__file__)],
        sites=SITES, seeds=SEEDS, updates_per_fold=refit.STEPS, original_2pct=True,
        singleton="original matched additive, identical in all four arms",
        new_forecast="registered tapered turn from the observed eight positions; no new thresholds",
        training="same cost head, amplitudes, sampler, optimizer, steps and pair-excluded producer",
        goals_used=False, central_velocity_used=False, observed_steps=8, forecast_steps=12,
        development_only=True, independent_confirmation=False, bootstrap_draws=3000))
    return base.sha(p), cohorts, old_menu, old_fit, old_upper, weights, producer, refs, calibration


def append_turn(h, bank):
    if h.shape[1:] != (8, 2) or bank.shape[0] != len(h) or bank.shape[2:] != (12, 2):
        raise ValueError("Expected eight observed and twelve forecast positions")
    if not np.isfinite(h).all() or not np.isfinite(bank).all():
        raise ValueError("Only finite causal inputs belong to the forecast menu")
    return np.concatenate((bank, turn.turn_path(h, 12)[:, None]), axis=1)


def pin_singletons(cv, q, original_additive, original_joint, predictions):
    """Use the same original singletons, checking the entire slot multiset per query."""
    peak = turn.peak(original_joint - cv)
    count = np.bincount(q, weights=peak > 0., minlength=int(q.max()) + 1)
    if not np.isin(count, (0, 1, 2)).all():
        raise ValueError("Expected at most two slots in any query")
    at = count[q] == 1
    for path in predictions.values():
        path[at] = original_additive[at]
        radius = turn.peak(path - cv)
        if not np.array_equal(np.bincount(q, weights=radius > 0., minlength=len(count)), count):
            raise ValueError("New menu changed the number of edits")
        idx = np.lexsort((peak, q))
        got = np.lexsort((radius, q))
        np.testing.assert_allclose(radius[got], peak[idx], atol=1e-9, rtol=1e-9)
    return count[q] == 2


def train(key, cohorts, old_menu, old_fit, weights, producer, refs, calibration):
    src = exp.loader(cohorts)
    fits, banks, selection = {}, {}, {}
    for site in SITES:
        h, q, cv, ols, motion = menu.parent.source_inputs(cohorts[site])
        for seed in SEEDS:
            name = f"{site}_{seed}"
            donors = {}
            for donor in SITES:
                if donor == site:
                    continue
                pair = tuple(sorted((donor, site)))
                d = src[donor]
                nn = refit.pair_forecast(d["h"], pair, seed, refs, calibration)
                bank = append_turn(d["h"], np.concatenate((d["bank"], nn[:, None]), axis=1))
                easy = float(np.quantile(np.concatenate([src[s]["e"] for s in SITES if s not in pair]), .25))
                donors[donor] = refit.model.training_examples(d["h"], d["cv"], d["ols"], bank, d["y"], d["v"], easy)
                if not len(donors[donor][0]):
                    raise ValueError("Missing donor actions")
            net, fits[name] = refit.fit(key, site, seed, donors)
            nn = menu.self_forecast(h, site, seed, weights[name], producer)
            bank = append_turn(h, np.concatenate((motion, nn[:, None]), axis=1))
            ident = old_fit["fits"][name]["identity"]
            if ident["held"] != site or set(ident["donors"]) != set(SITES) - {site}:
                raise ValueError("Parent cost fit is not location-excluded")
            with (np.load(study.checked(old_menu["banks"][name]), allow_pickle=False) as a,
                  np.load(study.checked(old_fit["banks"][name]), allow_pickle=False) as b):
                for z in (a, b):
                    for k, expect in (("q", q), ("cv", cv), ("ols4", ols)):
                        np.testing.assert_array_equal(z[k], expect)
                native, control = a["base_joint"], a["base_additive"]
                new, _ = refit.predict(net, h, q, cv, ols, bank, native)
                choices = dict(zip(ARMS, (b["trained_additive"], b["trained_joint"],
                                         *[new[k] for k in refit.ARMS[2:]]), strict=True))
            double = pin_singletons(cv, q, control, native, choices)
            selection[name] = dict(two_edit_queries=len(np.unique(q[double])))
            for arm in ARMS[2:]:
                selected = menu.scaled_to_path(cv, bank[:, -1], choices[arm])
                earlier = np.any([menu.scaled_to_path(cv, p, choices[arm]) for p in bank[:, :-1].swapaxes(0, 1)], axis=0)
                selection[name][arm] = int(np.count_nonzero(selected & ~earlier & double))
            values = dict(q=q, cv=cv, ols4=ols, bank=bank, native=native, control=control, **choices)
            banks[name] = exp.previous.save_bank(LOCAL / f"{name}.npz", values)
            refit.beat(phase="freeze", last=name, completed=len(banks), total=len(SITES)*len(SEEDS))
            print(name, selection[name], flush=True)
    base.immutable(OUT / "freeze.json", dict(registration=key, banks=banks, fits=fits,
        selection=selection, held_labels_used_in_model=False, forecast_scored=False))


def contrasts(paths, y, v, q, cut):
    cv, ols = paths["cv"], paths["ols4"]
    # The old and new arms use the same helper, row weights and easy denominator.
    result = {}
    for name, joint, control in (
            ("previous_joint_effect", "previous_joint", "previous_additive"),
            ("turn_joint_effect", "turn_joint", "turn_additive"),
            ("turn_additive_effect", "turn_additive", "previous_additive"),
            ("turn_joint_menu_effect", "turn_joint", "previous_joint")):
        result[name] = expanded.actual_contrast(cv, ols, paths[joint], paths[control], y, v, q, cut)
    arms, _ = menu.recording(dict(cv=cv, ols4=ols, **dict(zip(menu.ARMS, (paths[k] for k in ARMS), strict=True))),
                             y, v, q, cut)
    return result, dict(zip(ARMS, arms.values(), strict=True))


def fold_score(d, z, prior, cut):
    for k, value in (("q", d["q"]), ("cv", d["cv"]), ("ols4", d["ols"])):
        np.testing.assert_array_equal(z[k], value)
    keys = ("cv", "ols4", "bank", "native", "control", *ARMS)
    arrays = {k: z[k] for k in keys}
    labels = np.unique(d["rec"])
    if set(prior) != set(map(str, labels)):
        raise ValueError("Old and new recording populations differ")
    for r in labels:
        at = d["rec"] == r
        q = np.unique(d["q"][at], return_inverse=True)[1]
        p = {k: value[at] for k, value in arrays.items()}
        y, v = d["y"][at], d["v"][at]
        args = (p["cv"], p["ols4"], p["bank"], p["control"], p["native"], y, v, q, cut)
        cert = exact.assess(*args, double_only=True)
        result = expanded.original.assess(*args, cert)
        previous = prior[str(r)]
        known = []
        if result["edits"] != previous["edits"]:
            raise ValueError("New capacity test changed the exact edit budget")
        if result["edits"]:
            if np.any(np.asarray(result["relaxed_pair_native"]) < np.asarray(previous["relaxed_pair_native"]) - 1e-8):
                raise ValueError("New menu lost an old relaxed action")
        for objective in expanded.original.OBJECTIVES:
            before, after = expanded.support(previous, objective), expanded.support(result, objective)
            if before is not None and (after is None or after < before - 1e-8):
                raise ValueError("New exact solve lost a verified old feasible witness")
            known.append(before)
        comparison, arm = contrasts(p, y, v, q, cut)
        yield dict(record=str(r), base_optima=known, actual=comparison["turn_joint_effect"],
                   contrasts=comparison, arms=arm, **result)


def summarize(rows):
    groups = {}
    for k in rows[0]["contrasts"]:
        pairs = [dict(record=r["record"], effect=r["contrasts"][k]["pair_native"],
                      qualified=r["contrasts"][k]["qualified"]) for r in rows if r["contrasts"][k] is not None]
        groups[k] = effects.summary(pairs)
        groups[k]["site_mean_accuracy_pp"] = effects.site_means(
            [(r["record"], r["contrasts"][k]["accuracy_pp"]) for r in rows if r["contrasts"][k] is not None])
    arm = {}
    for k in ARMS:
        samples = [r["arms"][k] for r in rows]
        arm[k] = {m: sum(r[m] for r in samples) for m in ("edits", "risk", "dual")}
        for m in ("ade", "fde", "easy_degradation"):
            vals = [r[m] for r in samples if r[m] is not None]
            arm[k][f"recording_seed_mean_{m}"] = float(np.mean(vals)) if vals else None
            if m == "easy_degradation":
                arm[k]["easy_estimable"] = len(vals)
                arm[k]["easy_above_2pct"] = sum(x > .02 for x in vals)
    return dict(capacity=expanded.summarize(rows), arms=arm, contrasts=groups)


def score(key, cohorts, old_upper):
    p = OUT / "freeze.json"
    frozen = study.read(p)
    expected = {f"{site}_{seed}" for site in SITES for seed in SEEDS}
    if (frozen["registration"] != key or frozen["forecast_scored"] or frozen["held_labels_used_in_model"]
            or set(frozen["banks"]) != expected or set(frozen["fits"]) != expected):
        raise ValueError("All predictions must be frozen before scoring")
    src = exp.loader(cohorts)
    receipts, rows, sites = {}, [], {}
    for site in SITES:
        d = src[site]
        cut = np.quantile(np.concatenate([src[s]["e"] for s in SITES if s != site]), [.25, .75])
        site_rows = []
        for seed in SEEDS:
            name = f"{site}_{seed}"
            fit = frozen["fits"][name]["identity"]
            if fit["held"] != site or set(fit["donors"]) != set(SITES) - {site}:
                raise ValueError("Held site entered fitting")
            cache = LOCAL / f"{name}_score.json"
            binding = dict(registration=key, bank=frozen["banks"][name], previous=old_upper["checkpoints"][name])
            if not cache.exists():
                before = study.read(study.checked(binding["previous"]))
                prior = {r["record"]: r for r in before["rows"]}
                with np.load(study.checked(binding["bank"]), allow_pickle=False) as z:
                    base.immutable(cache, dict(binding=binding, rows=list(fold_score(d, z, prior, cut))))
            saved = study.read(cache)
            if saved["binding"] != binding:
                raise ValueError("Score checkpoint belongs to different inputs")
            site_rows.extend(saved["rows"])
            receipts[name] = study.artifact(cache)
            refit.beat(phase="score", last=name, completed=len(receipts), total=len(expected))
            print(name, len(receipts), flush=True)
        sites[site] = summarize(site_rows)
        rows.extend({**r, "record": f"{site}:{r['record']}"} for r in site_rows)
    total = summarize(rows)
    if total["capacity"]["edits"] != old_upper["totals"]["edits"]:
        raise ValueError("Budget did not replay")
    for k, r in total["contrasts"].items():
        r.pop("site_mean_native")
        r.pop("site_mean_accuracy_pp")
        r["location_bootstrap_native"] = effects.interval([s["contrasts"][k]["site_mean_native"] for s in sites.values()])
        r["location_bootstrap_accuracy_pp"] = effects.interval([s["contrasts"][k]["site_mean_accuracy_pp"] for s in sites.values()])
    result = dict(registration=key, result_source="fresh_run", inputs="cached_verified", new_fits=len(expected),
        freeze=study.artifact(p), checkpoints=receipts, sites=sites, totals=total,
        original_2pct=True, reserved_outcomes_opened=False, independent_confirmation=False, deployment_changed=False)
    base.immutable(OUT / "readout.json", result)
    print(study.json.dumps(total, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    # Reuse the frozen optimizer implementation with this process's own checkpoint root.
    refit.LOCAL = LOCAL
    with (LOCAL / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, old_menu, old_fit, old_upper, weights, producer, refs, calibration = register()
        if args.phase == "freeze":
            train(key, cohorts, old_menu, old_fit, weights, producer, refs, calibration)
        else:
            score(key, cohorts, old_upper)


if __name__ == "__main__":
    main()
