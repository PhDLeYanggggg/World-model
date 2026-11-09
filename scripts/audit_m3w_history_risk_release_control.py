"""Replay the same released recording views with the independent forecast."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_history_risk_relative_envelope as guard


stage, run, np = guard.stage, guard.run, guard.np


def main():
    key, _, _, roster, seal, _ = stage.register()
    (d, h, y, gain, groups, risk, dual), _ = stage.read_frozen(key, roster, seal)
    p = stage.PUBLIC/"relative_envelope.json"
    source = run.read(p)
    if source["confirmation_access"]:
        raise ValueError("Use calibration replay only")
    rows = []
    for f in seal["recordings"]:
        r = run.read(stage.parent.PRIVATE/f"{f['recording']}.score.json")
        if r["freeze"] != f:
            raise ValueError("Wrong score receipt")
        rows.extend(r["observed"])
    report = {}
    for name, x in (("history", h[..., None]), ("combined", np.stack((d, h), axis=-1))):
        _, bound = guard.relative_bounds(x, y, groups)
        m = stage.selections(bound, gain, risk, dual, groups)
        if m != source["metrics"][name]:
            raise ValueError("Released views changed")
        chosen = [stage.pool.select(r, g, np.zeros_like(r)) for r, g in zip(bound, gain, strict=True)]
        held = {}
        for g in np.unique(groups):
            pairs = [(i, j) for i, j in enumerate(chosen) if groups[i] == g and j is not None]
            if not pairs:
                continue
            masses = [rows[i][j]["masses"]["independent"] for i, j in pairs]
            if any(rows[i][j]["edits"] != rows[i][0]["edits"] for i, j in pairs):
                raise ValueError("Release counts differ from independent")
            held[str(g)] = dict(released=len(pairs), risk_pass=sum(bool(risk[i, j]) for i, j in pairs),
                independent_risk_pass=sum(bool(risk[i, 0]) for i, _ in pairs),
                dual_pass=sum(bool(dual[i, j]) for i, j in pairs), independent_dual_pass=sum(bool(dual[i, 0]) for i, _ in pairs),
                rescued_joint=sum(bool(dual[i, j] and not dual[i, 0]) for i, j in pairs),
                lost_joint=sum(bool(dual[i, 0] and not dual[i, j]) for i, j in pairs),
                arms={a: sum(j == k for _, j in pairs) for k, a in enumerate(stage.pool.ARMS)},
                lower_gain_vs_independent_pp=stage.parent.prior.merge_comparisons(masses)["completion_lower_pp"])
        report[name] = held
    run.once(stage.PUBLIC/"released_control.json", run.stable(dict(code=run.artifact(Path(__file__)),
        source=run.artifact(p), by_source=report, interpretation="Same selected recording-seed views and same per-query counts; conditional development contrast",
        original_2pct=True, confirmation_access=False)))
    print(report, flush=True)


if __name__ == "__main__":
    main()
