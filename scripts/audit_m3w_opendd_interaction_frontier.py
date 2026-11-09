"""Measure the best pair-relative improvement on the frozen development menu."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_exact_slot_headroom as original
from src.evaluation import m3w_interaction_frontier as frontier

base, study, exp, np = original.base, original.study, original.exp, original.np
SITES, SEEDS, prior = original.SITES, original.SEEDS, original.prior
OUT = original.OUT.parent / "opendd_interaction_frontier_v1"
LOCAL = original.LOCAL.parent / OUT.name
OBJECTIVES = ("coobserved", "full_grid_lower")
SECONDS = 3.
CERTIFICATES = ("no_easy_support", "nonpositive_reference_bound", "infeasible")


def register():
    original_key, cohorts, freeze, prior_result = original.register()
    p = original.OUT / "readout.json"
    accuracy = study.read(p)
    if accuracy["registration"] != original_key or not accuracy["hindsight_only"]:
        raise ValueError("Require the original accuracy-only proof")
    reg = OUT / "registration.json"
    base.immutable(reg, dict(accuracy_parent=study.artifact(p), shared_menu_freeze=study.artifact(
        prior.OUT / "freeze.json"), code=[study.artifact(Path(x)) for x in (__file__, frontier.__file__)],
        sites=SITES, seeds=SEEDS, objectives=OBJECTIVES, seconds_per_solver=SECONDS,
        exact_slots=True, frozen_singletons=True, original_2pct=True,
        additional_hindsight_constraint="paired accuracy lower gain versus additive >= 0 (1e-8 pp roundoff)",
        coobserved="unchanged observed pair metric; all query weights held fixed",
        full_grid_lower="all requested multi-agent pair steps, missing steps debit relative path distance",
        development_hindsight_only=True, reserved_outcomes_opened=False))
    return base.sha(reg), cohorts, freeze, prior_result, accuracy


def paired_utility(path, cv, ols, control, y, v, w, mass):
    return 100. / mass * np.array([w @ original.headroom.lower(a, path, y, v) for a in (cv, ols, control)])


def assess(cv, ols, bank, control, joint, y, v, q, cut, before):
    d, reason = original.headroom.make_design(cv, ols, bank, control, joint, y, v, q, cut[0], double_only=True)
    if d is None:
        if before["state"] != reason:
            raise ValueError("Previous no-edit or reference state did not replay")
        return dict(state=reason, edits=0)
    den = frontier.denominators(q, v)
    menu = frontier.query_options(d, cv, control, y, v, q, den)
    relaxed = frontier.relaxed_pair_bound(menu)
    current = frontier.aggregate_pair(joint, control, y, v, q, den)
    if np.any(current > relaxed + 1e-8):
        raise ValueError("Learned choice exceeds relaxed interaction ceiling")
    row = dict(state="eligible", edits=d["slots"], pair_queries=den.tolist(),
               learned_pair_native=current.tolist(), relaxed_pair_native=relaxed.tolist(),
               accuracy_parent_state=before["state"], solve={})
    if before["state"] in CERTIFICATES:
        row["state"] = "reused_accuracy_certificate"
        return row
    if (before.get("solver", {}).get("upper_pp") is not None
            and before["solver"]["upper_pp"] < -frontier.UTILITY_TOL_PP - 1e-7):
        row["state"] = "accuracy_would_decrease"
        return row
    weights = 1. / np.bincount(q)[q]
    for k, name in enumerate(OBJECTIVES):
        if den[k] == 0:
            row["solve"][name] = dict(status="undefined")
            continue
        take, log = frontier.solve(menu, d, k, SECONDS)
        if take is not None:
            actions = np.concatenate([menu["actions"][i] for i in take])
            path = cv.copy()
            path[d["row"][actions]] = d["path"][actions]
            for j in np.unique(q):
                at = q == j
                radii = np.linalg.norm(joint[at] - cv[at], axis=-1).max(-1)
                selected = np.linalg.norm(path[at] - cv[at], axis=-1).max(-1)
                np.testing.assert_allclose(np.sort(selected), np.sort(radii), atol=1e-9, rtol=1e-9)
                if np.count_nonzero(radii) < 2:
                    np.testing.assert_allclose(path[at], control[at], atol=1e-9, rtol=1e-9)
            terms = paired_utility(path, cv, ols, control, y, v, weights, d["reference"])
            effect = frontier.aggregate_pair(path, control, y, v, q, den)
            np.testing.assert_allclose(effect, menu["pair"][take].sum(0), atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(terms, d["offset"] + menu["gain"][take].sum(0), atol=1e-7, rtol=1e-7)
            verdict = study.policy(path, cv, cv, y, v, ["record"] * len(y),
                                   easy_cut=cut[0], hard_cut=cut[1])["recordings"]["record"]
            ok = (verdict["supported"] and verdict["selected"] == d["slots"]
                  and min(terms[:2]) > 0. and terms[2] >= -frontier.UTILITY_TOL_PP)
            log.update(verified=bool(ok), effect_native=effect.tolist(), accuracy_pp=terms.tolist())
            if not ok:
                log["status"] = "numerical_boundary"
        row["solve"][name] = log
    return row


def summary(rows):
    result = dict(views=len(rows), records=len({r["record"] for r in rows}),
                  edit_views=sum(bool(r["edits"]) for r in rows), edits=sum(r["edits"] for r in rows),
                  states=dict(Counter(r["state"] for r in rows)))
    for k, name in enumerate(OBJECTIVES):
        scored = [r for r in rows if r.get("pair_queries", [0, 0])[k]]
        solved = [r["solve"][name] for r in scored if name in r["solve"]]
        valid = [(r["record"], r["solve"][name]["effect_native"][k]) for r in scored
                 if r["solve"].get(name, {}).get("verified")]
        positives = [v for _, v in valid if v > frontier.INTERACTION_TOL]
        result[name] = dict(estimable_edit_views=len(scored),
            relaxed_positive=sum(r["relaxed_pair_native"][k] > frontier.INTERACTION_TOL for r in scored),
            learned_positive=sum(r["learned_pair_native"][k] > frontier.INTERACTION_TOL for r in scored),
            verified_feasible=len(valid), verified_positive=len(positives),
            verified_positive_recordings=len({r for r, v in valid if v > frontier.INTERACTION_TOL}),
            positive_witness_range_native=[min(positives), max(positives)] if positives else None,
            solve_states=dict(Counter(r["status"] for r in solved)))
    return result


def run(key, cohorts, freeze, shared_result, accuracy):
    source = exp.loader(cohorts)
    groups, merged, receipts = {}, [], {}
    for site in SITES:
        d = source[site]
        cut = np.quantile(np.concatenate([source[s]["e"] for s in SITES if s != site]), [.25, .75])
        site_rows = []
        for seed in SEEDS:
            name = f"{site}_{seed}"
            cache = LOCAL / f"{name}.json"
            binding = dict(registration=key, site=site, seed=seed, bank=freeze["forecasts"][name],
                           prior_certificate=accuracy["checkpoints"][name])
            if not cache.exists():
                old = study.read(study.checked(binding["prior_certificate"]))
                records = {r["record"]: r["arms"]["two_edits_only"] for r in old["rows"]}
                with np.load(study.checked(binding["bank"]), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["q"], d["q"])
                    np.testing.assert_array_equal(z["cv"], d["cv"])
                    np.testing.assert_array_equal(z["ols4"], d["ols"])
                    control, joint = z["matched_independent"], z["joint"]
                rows = []
                for r in np.unique(d["rec"]):
                    at = d["rec"] == r
                    q = np.unique(d["q"][at], return_inverse=True)[1]
                    row = assess(d["cv"][at], d["ols"][at], d["bank"][at], control[at], joint[at],
                                 d["y"][at], d["v"][at], q, cut, records[str(r)])
                    rows.append(dict(record=str(r), **row))
                base.immutable(cache, dict(binding=binding, rows=rows))
            saved = study.read(cache)
            if saved["binding"] != binding:
                raise ValueError("Resume inputs changed")
            site_rows.extend(saved["rows"])
            receipts[name] = study.artifact(cache)
            pending = LOCAL / "heartbeat.pending"
            pending.write_text(study.json.dumps(dict(pid=os.getpid(), completed=len(receipts), total=21,
                                                      last=name)) + "\n")
            pending.replace(LOCAL / "heartbeat.json")
            print(name, summary(saved["rows"]), flush=True)
        groups[site] = summary(site_rows)
        if groups[site]["edits"] != shared_result["groups"][site]["arms"]["joint"]["edits"]:
            raise ValueError("Shared edit budget did not replay")
        merged.extend({**r, "record": f"{site}:{r['record']}"} for r in site_rows)
    totals = summary(merged)
    base.immutable(OUT / "readout.json", dict(registration=key, result_source="fresh_run",
        totals=totals, sites=groups, checkpoints=receipts, original_2pct=True,
        hindsight_only=True, reserved_outcomes_opened=False, independent_confirmation=False))
    print(study.json.dumps(totals, indent=2))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prior.torch.set_num_threads(4)
    prior.torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(*register())


if __name__ == "__main__":
    main()
