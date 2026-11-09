"""Measure how much new causal paths expand the AV2 exact-slot frontier."""

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
from scripts import run_m3w_av2_native_trajectory as parent
from src.world_model import m3w_approach_candidate_paths as model

base, study, np = parent.base, parent.study, parent.np
OUT = parent.OUT.parent/"av2_approach_menu_v1"
LOCAL = parent.LOCAL.parent/OUT.name
ARMS = dict(base=(0, 1), own=(0, 1, 2, 3), neighbor=(0, 1, 4, 5), combined=tuple(range(6)))
NAMES = ("ols4", "tapered_turn", *model.OWN, *model.NEIGHBOR)
EPS = 1e-10


def beat(phase, done, total):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), phase=phase, done=done, total=total, epoch=time()))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    old_readout = parent.OUT/"readout.json"
    old = study.read(old_readout)
    freeze = study.read(study.checked(old["freeze"]))
    source = study.read(parent.SOURCE/"manifest.json")
    if (freeze["registration"] != old["registration"] or freeze["future_rows_used"]
            or not freeze["complete_before_scoring"] or old["reserved_outcomes_opened"]
            or old["independent_confirmation"] or set(freeze["files"]) != set(source["objects"])):
        raise ValueError("Require the complete development-only AV2 parent")
    p = OUT/"registration.json"
    base.immutable(p, dict(
        source=study.artifact(parent.SOURCE/"manifest.json"), parent_result=study.artifact(old_readout),
        parent_freeze=old["freeze"], code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        producer="closed-form last-step motion, own acceleration/smoothing, and pair closest approach",
        arms=ARMS, paths=NAMES, comparison="identical parent radii, both motion floors and additive reference",
        primary_support="all parent fully labelled two-slot scenarios, with each seed reported",
        risk_checks="same all harm / easy harm / signed easy <=.02; missing denominators fail",
        score="exact positive pair gain subject to accuracy and risk; maximum over distinct recipients",
        no_training=True, no_tuning=True, exposed_development=True, reserved_outcomes_opened=False))
    return base.sha(p), old, freeze, source


def freeze(key, old_freeze):
    receipts = {}
    for name, item in sorted(old_freeze["files"].items()):
        with np.load(study.checked(item), allow_pickle=False) as z:
            h = z["h"]
            if len(h) < 3 or not any(np.count_nonzero(parent.motion.peak(z[f"anchor_{s}"]-z["cv"])) == 2
                                     for s in parent.slate.SEEDS):
                continue
            cv, own, neighbor, partners = model.candidates(h, len(parent.FUT))
            np.testing.assert_array_equal(cv, z["cv"])
            bank = np.concatenate((z["ols4"][:, None], z["turn"][:, None], own, neighbor), axis=1)
            p = LOCAL/f"{name}.npz"
            receipts[name] = parent.turn.exp.previous.save_bank(p, dict(
                history=z["h"], bank=bank, partners=partners, registration=key))
        beat("freeze", len(receipts), len(old_freeze["files"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, banks=receipts, future_used=False))


def constraints(batch, cv, ols, additive, y, cut):
    """Evaluate original ratios on a complete single-scenario grid."""
    distance = np.linalg.norm(batch-y, axis=-1)
    ade = distance.mean(2)
    cv_ade, ols_ade, additive_ade = (np.linalg.norm(x-y, axis=-1).mean(1) for x in (cv, ols, additive))
    total = distance.sum((1, 2))
    accuracy = ((total < min(cv_ade.sum(), ols_ade.sum())*len(parent.FUT)-EPS)
                & (total <= additive_ade.sum()*len(parent.FUT)+EPS))
    changed = np.any(batch != cv, axis=(2, 3))
    easy = (cv_ade > 0.) & (cv_ade <= cut)
    delta = ade - cv_ade
    harm = np.maximum(delta, 0.)
    den = changed @ cv_ade
    eden = changed @ np.where(easy, cv_ade, 0.)
    full = cv_ade[easy].sum()
    all_risk = (den > 0.) & ((harm*changed).sum(1) <= .02*den)
    easy_risk = (eden > 0.) & ((harm*changed)[:, easy].sum(1) <= .02*eden)
    easy_net = (full > 0.) & (delta[:, easy].sum(1) <= .02*full)
    return accuracy, all_risk, easy_risk, easy_net


def evaluate(cv, ols, additive, y, bank, radii, cut):
    valid = np.ones(y.shape[:2], dtype=bool)
    paths, row, col, opts, group = parent.optional.options(cv, bank, radii)
    opts = opts[group == 2]
    by_arm = {}
    pool = np.zeros((len(opts), 5), float)
    for start in range(0, len(opts), 128):
        end = min(start+128, len(opts))
        p = parent.slate.assemble(cv, paths, row, col, opts[start:end])
        actual = np.sort(parent.motion.peak(p-cv), axis=1)[:, -2:]
        np.testing.assert_allclose(actual, np.broadcast_to(np.sort(radii), actual.shape), atol=1e-8, rtol=1e-8)
        pool[start:end, 0] = parent.frontier.pair_reduction(p, additive, y, valid)[:, 1]
        pool[start:end, 1:] = np.column_stack(constraints(p, cv, ols, additive, y, cut[0]))
    names = col[opts] % bank.shape[1]
    pair = pool[:, 0]
    for name, menu in ARMS.items():
        in_menu = np.isin(names, menu).all(1)
        improving = in_menu & (pair > EPS)
        accurate = improving & (pool[:, 1] > 0.)
        safe = accurate & (pool[:, 2:] > 0.).all(1)
        best = None
        if safe.any():
            at = int(np.argmax(np.where(safe, pair, -np.inf)))
            best = float(pair[at])
            path = parent.slate.assemble(cv, paths, row, col, opts[at:at+1])[0]
            real = study.policy(path, cv, cv, y, valid, ["r"]*len(cv), easy_cut=cut[0], hard_cut=cut[1])
            if not real["recordings"]["r"]["supported"]:
                raise ValueError("Vectorized safety check did not replay the witness")
        by_arm[name] = dict(assignments=int(in_menu.sum()), pair_positive=bool(improving.any()),
                            accuracy_pair_positive=bool(accurate.any()), best=best,
                            witness=[NAMES[i] for i in names[at]] if best is not None else [])
    if any(by_arm[k]["assignments"] == 0 or (by_arm[k]["best"] or 0.)+EPS < (by_arm["base"]["best"] or 0.)
           for k in ARMS):
        raise ValueError("An expanded menu lost base assignments")
    return by_arm


def score(key, old, parent_freeze, source):
    seal_path = OUT/"freeze.json"
    seal = study.read(seal_path)
    if seal["registration"] != key or seal["future_used"]:
        raise ValueError("Require frozen causal proposals")
    rows, data = [], {}
    for name, item in sorted(parent_freeze["files"].items()):
        with np.load(study.checked(item), allow_pickle=False) as z:
            if len(z["h"]) < 2:
                continue
            y, valid, _ = parent.targets(study.checked(source["objects"][name]["local"]), z["ids"], z["h"])
            complete = valid.all(1)
            error = np.linalg.norm(z["cv"][complete]-y[complete], axis=-1).mean(1)
            data[name] = (z["city"].item(), y, valid, error[error > 0.])
    for name, entry in sorted(seal["banks"].items()):
        city, y, valid, _ = data[name]
        if not valid.all():
            continue
        cut = np.quantile(np.concatenate([e for c, _, _, e in data.values() if c != city]), [.25, .75])
        with (np.load(study.checked(entry), allow_pickle=False) as p,
              np.load(study.checked(parent_freeze["files"][name]), allow_pickle=False) as z):
            np.testing.assert_array_equal(z["h"], p["history"])
            if p["registration"].item() != key:
                raise ValueError("Wrong candidate bank version")
            cv, ols = z["cv"], z["ols4"]
            for s in parent.slate.SEEDS:
                radii = parent.motion.peak(z[f"anchor_{s}"]-cv)
                if np.count_nonzero(radii) != 2:
                    continue
                arms = evaluate(cv, ols, z[f"additive_{s}"], y, p["bank"], radii[radii > 0.], cut)
                rows.append(dict(city=city, clip=name, seed=s, arms=arms,
                                 neighbors=int(np.count_nonzero(p["partners"] >= 0))))
        beat("score", len(rows), old["exact_complete_two_slot_ceiling"]["views"])
    result = {}
    n = len(rows)
    for a in ARMS:
        arm = [r["arms"][a] for r in rows]
        wins = [r for r in rows if r["arms"][a]["best"] is not None]
        new = [r for r in wins if r["arms"]["base"]["best"] is None]
        def increment(r):
            return float(r["arms"][a]["best"] is not None)-float(r["arms"]["base"]["best"] is not None)
        result[a] = dict(views=n, mean_assignments=float(np.mean([r["assignments"] for r in arm])),
                         pair_positive=sum(r["pair_positive"] for r in arm),
                         accuracy_pair_positive=sum(r["accuracy_pair_positive"] for r in arm),
                         strict_positive=len(wins), unique_strict_scenarios=len({r["clip"] for r in wins}),
                         added_unique_scenarios=len({r["clip"] for r in new}),
                         witness_path_counts=dict(Counter(k for r in wins for k in r["arms"][a]["witness"])),
                         added_fraction=parent.interval(rows, increment))
    ref = old["exact_complete_two_slot_ceiling"]
    if (n != ref["views"] or result["base"]["pair_positive"] != ref["relaxed_positive"]
            or result["base"]["strict_positive"] != ref["strict_positive"]
            or result["base"]["unique_strict_scenarios"] != ref["unique_strict_scenarios"]):
        raise ValueError("Base-menu outcomes did not replay")
    base.immutable(OUT/"readout.json", dict(registration=key, freeze=study.artifact(seal_path),
        source="fresh_run", base_replayed=True, neighbors_supported_views=sum(r["neighbors"] > 0 for r in rows),
        cities=dict(Counter(r["city"] for r in rows)), results=result, rows=rows,
        deployment_changed=False, reserved_outcomes_opened=False, stage5c_executed=False, smc_enabled=False))
    print(study.json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    for path in (OUT, LOCAL):
        path.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, old, seal, source = register()
        if phase == "freeze":
            freeze(key, seal)
        else:
            score(key, old, seal, source)
        beat(phase+"_complete", 1, 1)


if __name__ == "__main__":
    main()
