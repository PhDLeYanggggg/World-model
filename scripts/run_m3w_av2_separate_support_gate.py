"""Test a noncompensating support gate on the unchanged AV2 forecasts."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_observed_harm as old
from scripts.audit_m3w_av2_projection_risk_matched import weights

np, torch, study, base, av = old.np, old.torch, old.study, old.base, old.av
OUT, LOCAL = old.OUT.parent/"av2_separate_support_gate_v1", old.LOCAL.parent/"av2_separate_support_gate_v1"
GATED = {"gated_support": ("actor_support", .5), "gated_harm": ("conditional_harm", .5),
         "high_support_harm": ("conditional_harm", .75)}
ARMS = (*old.ARMS, *GATED)
PRIMARY = (*old.ARMS, "gated_support", "gated_harm")
FIELDS = ("qualified", "risk_fail", "unsupported", "observed_harm", "floor_fail")


def register():
    key, _, _, _, _, _, _, _, parent, sf, _, _ = old.register()
    p = old.OUT/"summary.json"
    s = study.read(p)
    hf = study.read(study.checked(s["freeze"]))
    if (hf["registration"] != key or hf["calibration_labels_used"] or hf["held_labels_used"]
            or s["independent_confirmation"] or s["reserved_outcomes_opened"]):
        raise ValueError("Require frozen development harm scores")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=study.artifact(Path(__file__)), parent=study.artifact(p), arms=ARMS,
        support_gates=GATED, primary="fixed top five per fold at common retained count, then original calibration release",
        sensitivity="0.75 support gate separately reported; never selected over the primary 0.50 gate",
        scope="all forecasts, fitted scores and two-edit budget unchanged; no fitting or raw future reads",
        threshold="unchanged largest nonempty all-qualified confidence prefix among admitted calibration rows",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), parent, sf, hf, s


def gate(scores, probability):
    if not 0. <= probability <= 1.:
        raise ValueError("Support probability must be in [0,1]")
    return dict(scores, **{k: scores[a] if probability >= cut else None for k, (a, cut) in GATED.items()})


def freeze(key, parent, sf, hf):
    done = {}
    for name, ref in hf["folds"].items():
        z = study.read(study.checked(ref))
        s = study.read(study.checked(sf["folds"][name]))
        if z["identity"]["source"] != parent["folds"][name] or s["identity"]["source"] != parent["folds"][name]:
            raise ValueError("Wrong fixed support roster")
        ident = dict(registration=key, source=parent["folds"][name], harm=ref, support=sf["folds"][name])
        rows = {}
        for role in ("calibration", "held"):
            rows[role] = []
            for r, a in zip(z["rows"][role], s["rows"][role], strict=True):
                if r["clip"] != a["clip"] or r["city"] != a["city"] or r["score"]["actor_support"] != a["score"]["actor_support"]:
                    raise ValueError("Frozen support scores changed")
                prob = a["probability"]["new"]
                rows[role].append(dict(clip=r["clip"], city=r["city"], score=gate(r["score"], prob)))
        p = LOCAL/f"{name}.json"
        base.immutable(p, dict(identity=ident, rows=rows))
        done[name] = study.artifact(p)
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, outcomes_read=False, reserved_outcomes_opened=False))


def calibrate(rows, arm):
    return old.timing.old.action.prior.calibration([
        dict(score=r["scores"][arm], strict=r["result"]["eligible"], add_ok=True)
        for r in rows if r["scores"][arm] is not None])


def matched_rows(rows, arms, city):
    n = min(5, *(sum(r["scores"][a] is not None for r in rows) for a in arms))
    if not n:
        return None
    values = np.asarray([(r["result"]["eligible"], not r["result"]["risk"], not r["support"], r["harm"],
                          r["result"]["gain"] is None or min(r["result"]["gain"]) <= 0.) for r in rows])
    counts = {}
    for a in arms:
        scores = np.asarray([r["scores"][a] if r["scores"][a] is not None else -np.inf for r in rows])
        w = weights(scores, n)
        if w[~np.isfinite(scores)].any():
            raise ValueError("A rejected view entered the matched comparison")
        counts[a] = dict(zip(FIELDS, map(float, w@values), strict=True))
    return dict(city=city, clip="equal_count", n=n, arms=counts)


def summarize(cells, arms):
    if not cells:
        return dict(n=0, folds=0, counts=None, paired_pp=None)
    counts = {a: {k: sum(r["arms"][a][k] for r in cells) for k in FIELDS} for a in arms}
    contrasts = [(a, b) for a, b in (("gated_harm", "conditional_harm"), ("gated_harm", "gated_support"),
                                    ("gated_harm", "actor_support"), ("high_support_harm", "conditional_harm"))
                 if a in arms and b in arms]
    return dict(n=sum(r["n"] for r in cells), folds=len(cells), counts=counts,
        paired_pp={f"{a}_minus_{b}": {k: av.interval(cells, lambda r: 100.*(r["arms"][a][k]-r["arms"][b][k])/r["n"])
                   for k in FIELDS} for a, b in contrasts})


def score(key, parent, previous):
    p = OUT/"freeze.json"
    f = study.read(p)
    if (f["registration"] != key or f["outcomes_read"] or f["reserved_outcomes_opened"]
            or set(f["folds"]) != set(parent["folds"])):
        raise ValueError("Require frozen scores before calibration")
    frozen = {n: study.read(study.checked(r)) for n, r in f["folds"].items()}
    rules, availability = {}, {}
    for n, z in frozen.items():
        rows = list(old.support.prior.role_rows(z, parent["folds"][n], n, "calibration"))
        rules[n] = {a: calibrate(rows, a) for a in ARMS}
        availability[n] = {a: sum(r["scores"][a] is not None for r in rows) for a in ARMS}
    cp = OUT/"calibration.json"
    base.immutable(cp, dict(freeze=study.artifact(p), rules=rules, admitted=availability, held_receipts_used=False))
    all_rows, cells, sens, admitted = [], [], [], Counter()
    for n, z in frozen.items():
        pf = parent["folds"][n]
        ann = study.read(old.support.LOCAL/f"{n}.support.json")
        if ann["parent"] != pf or z["identity"]["source"] != pf:
            raise ValueError("Wrong held support annotation")
        rows = list(old.support.prior.role_rows(z, pf, n, "held"))
        for r, b in zip(rows, ann["cells"], strict=True):
            if r["clip"] != b["clip"]:
                raise ValueError("Wrong held annotation row")
            r.update(support=b["support"], harm=b["cause"] == "observed_positive_harm", cause=b["cause"])
            r["release"] = {}
            for a in ARMS:
                sc, t = r["scores"][a], rules[n][a]["threshold"]
                admitted[a] += sc is not None
                r["release"][a] = sc is not None and t is not None and sc >= t
        city = study.read(study.checked(pf))["identity"]["held"]
        x = matched_rows(rows, PRIMARY, city)
        if x:
            cells.append(x)
        x = matched_rows(rows, ("conditional_harm", "high_support_harm"), city)
        if x:
            sens.append(x)
        all_rows.extend(rows)
    h = {a: old.support.prior.held_metrics(all_rows, a) for a in ARMS}
    if len(all_rows) != previous["views"] or any(h[a] != previous["held"][a] for a in old.ARMS):
        raise ValueError("Frozen comparators did not replay")
    result = dict(views=len(all_rows), scenarios=len({r["clip"] for r in all_rows}), matched=summarize(cells, PRIMARY),
        sensitivity=summarize(sens, ("conditional_harm", "high_support_harm")), held=h, held_admitted=admitted,
        calibration_admitted={a: sum(x[a] for x in availability.values()) for a in ARMS},
        calibration_pass={a: sum(v[a]["accepted"] for v in rules.values()) for a in ARMS},
        release_causes={a: Counter(r["cause"] for r in all_rows if r["release"][a]) for a in ARMS},
        freeze=study.artifact(p), calibration=study.artifact(cp), result_source="fresh_run", outcome_source="cached_verified",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, parent, sf, hf, previous = register()
        if phase == "freeze":
            freeze(key, parent, sf, hf)
        else:
            score(key, parent, previous)


if __name__ == "__main__":
    main()
