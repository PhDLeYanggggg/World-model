"""Check whether the history gate needs a risk-priced menu at all."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_history_risk_relative_envelope as guard


stage, run, np = guard.stage, guard.run, guard.np


def main():
    key, _, _, roster, seal, _ = stage.register()
    (_, h, y, gain, groups, risk, dual), _ = stage.read_frozen(key, roster, seal)
    _, bound = guard.relative_bounds(h[..., None], y, groups)
    ref = stage.PUBLIC/"relative_envelope.json"
    prior = run.read(ref)
    if prior["confirmation_access"] or stage.selections(bound, gain, risk, dual, groups) != prior["metrics"]["history"]:
        raise ValueError("Need the exact completed history release")
    result = stage.selections(bound[:, :1], gain[:, :1], risk[:, :1], dual[:, :1], groups)
    run.once(stage.PUBLIC/"independent_gate.json", run.stable(dict(code=run.artifact(Path(__file__)), source=run.artifact(ref),
        result=result, interpretation="Remove risk-priced arms, leaving the same cross-fitted guard and frozen independent forecasts",
        development_followup=True, original_2pct=True, confirmation_access=False)))
    print(result, flush=True)


if __name__ == "__main__":
    main()
