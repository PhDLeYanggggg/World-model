"""Test whether the frozen held-source forecast expands the interaction frontier."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_opendd_interaction_frontier as original
from scripts import audit_m3w_opendd_self_menu_capacity as capacity

base, study, np = original.base, original.study, original.np
frontier, menu = original.frontier, capacity.menu
OUT = original.OUT.parent / "opendd_expanded_interaction_frontier_v1"
LOCAL = original.LOCAL.parent / OUT.name


def register():
    old_key, cohorts, _, _, _ = original.register()
    new_key, new_cohorts, weights, producer, frozen, _ = capacity.register()
    old_path = original.OUT / "readout.json"
    new_path = capacity.OUT / "readout.json"
    old, new = map(study.read, (old_path, new_path))
    if (old["registration"] != old_key or new["registration"] != new_key or cohorts != new_cohorts
            or old["reserved_outcomes_opened"] or new["reserved_outcomes_opened"]):
        raise ValueError("Expanded and base interaction inputs do not align")
    p = OUT / "registration.json"
    base.immutable(p, dict(parents=[study.artifact(x) for x in (old_path, new_path, menu.OUT / "freeze.json")],
        implementation=study.artifact(Path(__file__)), original_2pct=True,
        scope="only add the frozen source-held self-history path",
        reference="original additive policy in both interaction ceilings; singleton assignments frozen",
        secondary="actual expanded joint versus actual expanded additive, no changed inference",
        objectives=original.OBJECTIVES, seeds=original.SEEDS, sites=original.SITES,
        development_only=True, no_new_training=True))
    return base.sha(p), cohorts, weights, producer, frozen, old, new


def actual_contrast(cv, ols, joint, additive, y, valid, q, cut):
    selected = np.linalg.norm(joint - cv, axis=-1).max(-1) > 0
    if not selected.any():
        return None
    den = frontier.denominators(q, valid)
    if not den.all():
        return None
    effect = frontier.aggregate_pair(joint, additive, y, valid, q, den)
    weights = 1. / np.bincount(q)[q]
    mass = float(weights @ np.where(valid, np.linalg.norm(cv - y, axis=-1), 0.).sum(-1))
    if mass <= 0.:
        return None
    accuracy = original.paired_utility(joint, cv, ols, additive, y, valid, weights, mass)
    risk = study.policy(joint, cv, cv, y, valid, ["record"] * len(y),
                        easy_cut=cut[0], hard_cut=cut[1])["recordings"]["record"]["supported"]
    qualified = bool(risk and min(accuracy[:2]) > 0. and accuracy[2] >= -frontier.UTILITY_TOL_PP)
    return dict(pair_native=effect.tolist(), accuracy_pp=accuracy.tolist(), qualified=qualified)


def support(row, name):
    z = row.get("solve", {}).get(name, {})
    return z.get("effect_native", [0., 0.])[original.OBJECTIVES.index(name)] if z.get("verified") else None


def run(key, cohorts, weights, producer, frozen, old, new):
    source = original.exp.loader(cohorts)
    combined, sites, receipts = [], {}, {}
    for site in original.SITES:
        src = source[site]
        cut = np.quantile(np.concatenate([d["e"] for s, d in source.items() if s != site]), [.25, .75])
        site_rows = []
        for seed in original.SEEDS:
            name = f"{site}_{seed}"
            binding = dict(registration=key, fold=name, bank=frozen["banks"][name],
                           base=old["checkpoints"][name], accuracy=new["checkpoints"][name])
            p = LOCAL / f"{name}.json"
            if not p.exists():
                base_rows = {r["record"]: r for r in study.read(study.checked(binding["base"]))["rows"]}
                cert = study.read(study.checked(binding["accuracy"]))["rows"]
                with np.load(study.checked(binding["bank"]), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["q"], src["q"])
                    np.testing.assert_array_equal(z["cv"], src["cv"])
                    np.testing.assert_array_equal(z["ols4"], src["ols"])
                    a, j, ea, ej = (z[k] for k in ("base_additive", "base_joint", "expanded_additive", "expanded_joint"))
                candidate = menu.self_forecast(src["h"], site, seed, weights[name], producer)
                bank = np.concatenate((src["bank"], candidate[:, None]), axis=1)
                rows = []
                for r in np.unique(src["rec"]):
                    at = src["rec"] == r
                    q = np.unique(src["q"][at], return_inverse=True)[1]
                    args = (src["cv"][at], src["ols"][at])
                    labels = (src["y"][at], src["v"][at], q, cut)
                    got = original.assess(*args, bank[at], a[at], j[at], *labels, cert[str(r)])
                    previous = base_rows[str(r)]
                    if (got["edits"] != previous["edits"] or (got["edits"] and np.any(
                            np.asarray(got["relaxed_pair_native"]) < np.asarray(previous["relaxed_pair_native"]) - 1e-8))):
                        raise ValueError("Expanded menu altered budgets or excluded a base choice")
                    known = []
                    for objective in original.OBJECTIVES:
                        left, right = support(previous, objective), support(got, objective)
                        if left is not None and (right is None or right < left - 1e-8):
                            raise ValueError("Expanded frontier lost a verified base witness")
                        known.append(left)
                    actual = actual_contrast(*args, ej[at], ea[at], *labels)
                    rows.append(dict(record=str(r), base_optima=known, actual=actual, **got))
                base.immutable(p, dict(binding=binding, rows=rows))
            saved = study.read(p)
            if saved["binding"] != binding:
                raise ValueError("Expanded interaction checkpoint inputs changed")
            site_rows.extend(saved["rows"])
            receipts[name] = study.artifact(p)
            tmp = LOCAL / "heartbeat.pending"
            tmp.write_text(study.json.dumps(dict(pid=os.getpid(), completed=len(receipts), total=21, last=name)) + "\n")
            tmp.replace(LOCAL / "heartbeat.json")
            print(name, len(receipts), flush=True)
        sites[site] = summarize(site_rows)
        combined.extend({**r, "record": f"{site}:{r['record']}"} for r in site_rows)
    totals = summarize(combined)
    base.immutable(OUT / "readout.json", dict(registration=key, result_source="fresh_run",
        sites=sites, totals=totals, checkpoints=receipts, original_2pct=True,
        no_new_training=True, hindsight_only=True, reserved_outcomes_opened=False))
    print(study.json.dumps(totals, indent=2))


def summarize(rows):
    result = original.summary(rows)
    actual = [r["actual"] for r in rows if r["actual"] is not None]
    for k, name in enumerate(original.OBJECTIVES):
        result[name].update(
            new_wins_vs_base=sum((support(r, name) or 0.) > frontier.INTERACTION_TOL
                                 and (r["base_optima"][k] or 0.) <= frontier.INTERACTION_TOL for r in rows),
            actual_qualified_positive=sum(v["qualified"] and v["pair_native"][k] > frontier.INTERACTION_TOL
                                          for v in actual),
            actual_positive=sum(v["pair_native"][k] > frontier.INTERACTION_TOL for v in actual),
            actual_estimable=len(actual),
        )
    result["actual_status"] = dict(Counter("qualified" if r["qualified"] else "not_qualified" for r in actual))
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    menu.torch.set_num_threads(4)
    menu.torch.set_num_interop_threads(1)
    for directory in (OUT, LOCAL):
        directory.mkdir(parents=True, exist_ok=True)
    with (LOCAL / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(*register())
