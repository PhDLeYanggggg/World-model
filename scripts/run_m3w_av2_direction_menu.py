"""Test direction-only forecast changes at the incumbent's two exact edit budgets."""

import argparse
from collections import Counter
import fcntl
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_av2_incumbent_timing as prior
from src.world_model import m3w_av2_direction_menu as model


np, torch, av, eq, prev, study, base, old = prior.np, prior.torch, prior.av, prior.eq, prior.prev, prior.study, prior.base, prior.old
OUT = prior.OUT.parent/"av2_direction_menu_v1"
LOCAL = prior.LOCAL.parent/OUT.name
MODES = model.MODES


def beat(phase, done, total):
    print(dict(phase=phase, pid=prior.refit.tm.os.getpid(), completed=done, total=total), flush=True)


def register():
    k, data, source, roles, _, outer, control, refs, _ = prior.register()
    p = prior.OUT/"summary.json"
    r = study.read(p)
    freeze = study.read(study.checked(r["freeze"]))
    if freeze["registration"] != k or freeze["held_labels_used"] or r["reserved_outcomes_opened"]:
        raise ValueError("Require verified incumbent timing development study")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)], parent=study.artifact(p),
        angles=[-model.ANGLE_DEG, model.ANGLE_DEG], menus=MODES, primary="equal-sized direction vs timing and incremental union coverage",
        budget="same frozen two recipients and peaks; rotation preserves magnitude of CV-relative offset at every step",
        selector="same observed-prefix replay score; calibration on original separate city; no new weight fitting",
        references="both CV and OLS4 and same independent path, original risk and missing-label checks",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, source, roles, outer, control, refs, r


def prepare(d, anchor, cut):
    if int(np.any(anchor != d["cv"], axis=(1, 2)).sum()) != 2:
        return None
    bank, edited, menus = model.menus(d["cv"], anchor)
    h = prior.local.history.features(d["h"], d["cv"], bank, edited, cut[0])
    score = prior.local.replay_score(h)
    choices = [int(m[score[m].argmax()]) for m in menus.values()]
    sha = hashlib.sha256(np.ascontiguousarray(bank).tobytes()).hexdigest()
    return bank, menus, choices, score[choices], sha


def labels(d, m, a, y, v, cut):
    bank = m[0]
    t = prior.targets(d, (bank, 1.), a, y, v, cut)
    own = (t[:, 2] > 0.) & (t[:, 5] > 0.) & (t[:, 6] >= 0.)
    for j in np.flatnonzero(own):
        if not eq.ceiling.eligible_additive(prev.metrics(bank[j], d, y, v, cut)):
            raise ValueError("Standalone candidate witness failed exact risk replay")
    if bool(own[0]) != bool(t[0, 1]):
        raise ValueError("Anchor eligibility changed")
    return t[:, 0] > 0., own


def calibrate(action, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows, stats = {k: [] for k in MODES}, Counter()
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = data[n]
        a = prior.incumbent(d, seed, cut, raw[lo:hi], action)
        if a is None or (m := prepare(d, a, cut)) is None:
            continue
        both, own = labels(d, m, a, *cy[n], cut)
        for k, j, s in zip(MODES, m[2], m[3], strict=True):
            stats[k+"_top"] += int(both[j])
            rows[k].append(dict(score=float(s), strict=bool(both[j]), add_ok=bool(own[0])))
    return dict(counts=dict(stats), rules={k: old.action.prior.calibration(rows[k]) for k in MODES})


def freeze(key, data, source, roles, outer, control, refs):
    done = {}
    for held, role in roles.items():
        cal = role["calibration"]
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            cf, of, ref = control["folds"][name], outer["folds"][name], refs["folds"][name]
            cut = ref["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, seed=seed, cut=cut)
            if any(cf["identity"][k] != ident[k] for k in ("held", "calibration", "seed", "cut")):
                raise ValueError("Incorrect control roles")
            mark = LOCAL/f"{name}.json"
            if not mark.exists():
                action = prior.rel.pair.prior.load_action(cf)
                table = calibrate(action, old.load_outer(of), data, cy, seed, cut)
                with np.load(study.checked(cf["forecast"]), allow_pickle=False) as a:
                    chosen, scores, digest = [], [], []
                    for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                        d = data[n]
                        if d["city"] != held:
                            raise ValueError("Wrong held city")
                        m = prepare(d, a["additive"][lo:hi], cut)
                        chosen.append(m[2] if m else [0]*len(MODES))
                        scores.append(m[3] if m else [-1.]*len(MODES))
                        digest.append(m[4] if m else "")
                    forecast = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=a["names"], offsets=a["offsets"],
                        choice=np.asarray(chosen), score=np.asarray(scores), digest=np.asarray(digest), registration=key))
                base.immutable(mark, dict(identity=ident, calibration=table, forecast=forecast))
            r = study.read(mark)
            if r["identity"] != ident:
                raise ValueError("Changed development fold")
            study.checked(r["forecast"])
            done[name] = r
            beat("freeze", len(done), len(refs["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False,
                                         reserved_outcomes_opened=False))


def fold_score(key, data, truth, name, f, cf):
    held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
    rows = []
    with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
          np.load(study.checked(cf["forecast"]), allow_pickle=False) as a):
        for k in ("names", "offsets"):
            np.testing.assert_array_equal(z[k], a[k])
        if z["registration"].item() != key:
            raise ValueError("Wrong frozen forecast")
        for i, (n, lo, hi) in enumerate(zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True)):
            d, (y, v), anchor = data[n], truth[n], a["additive"][lo:hi]
            if d["city"] != held:
                raise ValueError("Wrong held roster")
            if len(eq.radii(d, seed)) != 2:
                continue
            m = prepare(d, anchor, cut)
            if m is None or m[4] != z["digest"][i]:
                raise ValueError("Two-slot candidate menu differs from freeze")
            np.testing.assert_array_equal(z["choice"][i], m[2])
            both, own = labels(d, m, anchor, y, v, cut)
            row = dict(clip=str(n), city=held, seed=seed, complete=bool(v.all()), anchor_eligible=int(own[0]),
                       complete_recipient_labels=bool(v[np.any(anchor != d["cv"], axis=(1, 2))].all()))
            for j, (k, mask) in enumerate(m[1].items()):
                c = int(z["choice"][i, j])
                th = f["calibration"]["rules"][k]["threshold"]
                active = c > 0 and th is not None and z["score"][i, j] >= th
                row.update({k+"_options": len(mask)-1, k+"_standalone": int(own[mask].any()),
                    k+"_available": int(both[mask].any()), k+"_top": int(both[c]),
                    k+"_release": int(active), k+"_pass": int(active and both[c])})
            if row["union_available"] < max(row[k+"_available"] for k in MODES[:2]):
                raise ValueError("Union lost a matched witness")
            rows.append(row)
    return rows


def summarize(rows, seal, parent, key):
    counts = lambda x: {k: sum(r[k] for r in x) for k in ("anchor_eligible", *(a+"_"+t for a in MODES
        for t in ("options", "standalone", "available", "top", "release", "pass")))}
    all_ = counts(rows)
    previous = parent["counts"]
    if (len(rows), all_["anchor_eligible"], all_["timing_options"], all_["timing_available"], all_["timing_top"],
            all_["timing_release"], all_["timing_pass"]) != tuple(previous[k] for k in
            ("two_slots", "anchor_eligible", "options", "local_available", "replay_top", "replay_release", "replay_pass")):
        raise ValueError("Prior timing control did not reproduce exactly")
    complete = [r for r in rows if r["complete"]]
    return dict(registration=key, views=len(rows), scenarios=len({r["clip"] for r in rows}),
        complete_views=len(complete), complete_recipient_views=sum(r["complete_recipient_labels"] for r in rows), counts=all_,
        complete_counts=counts(complete), by_city={g: counts([r for r in rows if r["city"] == g]) for g in sorted({r["city"] for r in rows})},
        paired={a+"_minus_timing_"+m: av.interval(rows, lambda r: r[a+"_"+m]-r["timing_"+m])
                for a in MODES[1:] for m in ("available", "standalone", "top")},
        calibration_accepted={k: sum(f["calibration"]["rules"][k]["accepted"] for f in seal["folds"].values()) for k in MODES},
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False,
        result_source="fresh_run", no_new_training=True)


def score(key, data, source, control, parent, replay):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or seal["reserved_outcomes_opened"]:
        raise ValueError("Require frozen direction forecasts")
    truth = None
    rows = []
    for name, f in seal["folds"].items():
        mark = LOCAL/f"{name}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay cannot read outcomes")
            if truth is None:
                truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
            r = fold_score(key, data, truth, name, f, control["folds"][name])
            base.immutable(mark, dict(freeze=f, rows=r))
        r = study.read(mark)
        if r["freeze"] != f:
            raise ValueError("Incorrect scored receipt")
        rows.extend(r["rows"])
    out = summarize(rows, seal, parent, key)
    base.immutable(OUT/"summary.json", dict(**out, freeze=study.artifact(p)))
    print(study.json.dumps(out, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, source, roles, outer, control, refs, parent = register()
        if phase == "freeze":
            freeze(key, data, source, roles, outer, control, refs)
        else:
            score(key, data, source, control, parent, replay=phase == "replay")


if __name__ == "__main__":
    main()
