"""Attribute misses using only the frozen forecast and score summaries."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recording_risk_pool as stage


np, run, study = stage.np, stage.run, stage.study


def main():
    key, _, _, roster, _, _ = stage.register()
    p = stage.PUBLIC/"analysis.json"
    result = run.read(p)
    seal = run.read(run.checked(result["freeze"]))
    if seal["registration"] != key or seal["confirmation_access"] or result["confirmation_access"]:
        raise ValueError("Invalid scored calibration summaries")
    groups = {}
    for f in seal["recordings"]:
        rec = f["recording"]
        r = run.read(stage.PRIVATE/f"{rec}.score.json")
        if r["freeze"] != f:
            raise ValueError("Invalid score receipt")
        for p0, z in zip(f["predicted_recording_masses"], r["observed"], strict=True):
            risk, gain = study.prediction(p0)
            actual = np.asarray([[1. if u is None else u for u in row["risk"]] for row in z])
            flags = np.asarray([[row["risk_pass"], row["dual_pass"]] for row in z], dtype=bool)
            groups.setdefault(roster[rec], []).append((risk, gain, actual-risk, flags))
    residuals = {g: [x[2] for x in rows] for g, rows in groups.items()}
    all_flags = np.stack([x[3] for v in groups.values() for x in v])
    baseline = all_flags[:, :1]
    rescued = (all_flags & ~baseline).sum(0)
    lost = (~all_flags & baseline).sum(0)
    n = int(all_flags[:, :, 1].any(1).sum())
    if n != result["hindsight_available_joint"]:
        raise ValueError("Menu opportunity differs from the primary scoring")
    methods = {m: dict(genuine_opportunities=n, rejects_genuine_by_risk=0,
                      rejects_genuine_by_gain=0, selected_other_arm=0, selected_genuine=0) for m in study.RULES}
    for held, rows in groups.items():
        offsets = study.group_offsets(residuals, held)
        for risk, gain, _, flags in rows:
            true = flags[:, 1]
            if not true.any():
                continue
            for name, offset in zip(study.RULES, offsets, strict=True):
                eligible = true & (risk+offset <= study.TOLERANCE).all(1)
                idx = study.select(risk, gain, offset)
                if not eligible.any():
                    reason = "rejects_genuine_by_risk"
                elif not (eligible & (gain > 0.)).any():
                    reason = "rejects_genuine_by_gain"
                elif idx is None or not true[idx]:
                    reason = "selected_other_arm"
                else:
                    reason = "selected_genuine"
                methods[name][reason] += 1
    for name, m in methods.items():
        if m["selected_genuine"] != result["held_source_selection"][name]["passed_both_and_risk"]:
            raise ValueError("Calibration attribution differs from scoring")
    run.once(stage.PUBLIC/"attribution.json", run.stable(dict(code=run.artifact(Path(__file__)), source=run.artifact(p),
        matched_arm_changes={k: dict(risk_rescued=int(rescued[i, 0]), risk_lost=int(lost[i, 0]),
                                    dual_rescued=int(rescued[i, 1]), dual_lost=int(lost[i, 1]))
                             for i, k in enumerate(study.ARMS)}, opportunities=methods, confirmation_access=False)))
    print(methods, flush=True)


if __name__ == "__main__":
    main()
