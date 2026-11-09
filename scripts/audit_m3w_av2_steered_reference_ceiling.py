"""Closed-form hindsight eligibility at fixed recipients and per-step radii."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_av2_steered_reference as stage


np, model, study, base, prev, eq = stage.np, stage.model, stage.study, stage.base, stage.prev, stage.eq
ARMS = ("bounded", "free")


def best_direction(cv, ols, anchor, y, valid, max_angle):
    """On a known step aim at truth; on a missing step minimize distance to OLS4."""
    delta = anchor-cv
    toward = np.where(valid[..., None], y, ols)-cv
    cross = delta[..., 0]*toward[..., 1]-delta[..., 1]*toward[..., 0]
    angle = np.arctan2(cross, (delta*toward).sum(-1)).clip(-max_angle, max_angle)
    p = cv+model.rotate(delta, angle, np)
    np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1), np.linalg.norm(delta, axis=-1), rtol=1e-8, atol=1e-8)
    return p


def main():
    key, data, source, _, _, control, _, _ = stage.register()
    p = stage.OUT/"summary.json"
    result = study.read(p)
    seal = study.read(study.checked(result["freeze"]))
    if seal["registration"] != key or seal["held_labels_used"] or result["reserved_outcomes_opened"]:
        raise ValueError("Require scored development-only folds")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    rows = []
    for name, f in seal["folds"].items():
        r = study.read(stage.LOCAL/f"{name}.score.json")
        if r["freeze"] != f:
            raise ValueError("Wrong held eligibility scores")
        prior = {z["clip"]: z for z in r["rows"]}
        seed, cut = (f["identity"][k] for k in ("seed", "cut"))
        with np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a:
            for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                d = data[n]
                if len(eq.radii(d, seed)) != 2:
                    continue
                y, v = truth[n]
                anchor = a["additive"][lo:hi]
                z = prior[str(n)]
                out = dict(clip=str(n), city=z["city"], complete=z["complete"], supported=z["arms"]["anchor"]["easy_defined"])
                for k, limit in zip(ARMS, (model.MAX_ANGLE, np.pi), strict=True):
                    path = best_direction(d["cv"], d["ols4"], anchor, y, v, limit)
                    m = prev.metrics(path, d, y, v, cut)
                    out[k] = eq.ceiling.eligible_additive(m)
                if out["bounded"] > out["free"] or max(m["eligible"] for m in z["arms"].values()) > out["bounded"]:
                    raise ValueError("Observed forecast exceeds its directional feasibility ceiling")
                rows.append(out)
    if len(rows) != result["views"]:
        raise ValueError("Missing fixed-budget views")
    out = dict(code=study.artifact(Path(__file__)), score=study.artifact(p), views=len(rows),
        ceilings={k: sum(r[k] for r in rows) for k in ARMS},
        complete_ceilings={k: sum(r[k] for r in rows if r["complete"]) for k in ARMS},
        fixed_easy_supported=sum(r["supported"] for r in rows),
        by_city={g: {k: sum(r[k] for r in rows if r["city"] == g) for k in ARMS} for g in sorted({r["city"] for r in rows})},
        exact_for_relaxed_geometry=True, hindsight_only=True, ignores_temporal_smoothness=True,
        original_2pct=True, reserved_outcomes_opened=False, result_source="fresh_run")
    base.immutable(stage.OUT/"direction_ceiling.json", out)
    print(study.json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
