"""Compare absolute and log-scale residual guards at fixed 2% risk limits."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recording_history_risk as stage


np, run, study = stage.np, stage.run, stage.study


def relative_bounds(x, y, groups):
    log = lambda risk: np.log1p(100.*risk)
    result, point = np.zeros_like(y), np.zeros_like(y)
    for g in np.unique(groups):
        outer, held = np.flatnonzero(groups != g), np.flatnonzero(groups == g)
        point[held] = study.fit_predict(x, y, groups, outer, held)
        r = []
        for j in np.unique(groups[outer]):
            train, test = outer[groups[outer] != j], outer[groups[outer] == j]
            estimate = study.fit_predict(x, y, groups, train, test)
            r.append((log(y[test])-log(estimate)).max(0))
        result[held] = np.expm1(log(point[held])+np.maximum(np.max(r, axis=0), 0.))/100.
    return point, result


def main():
    key, _, _, roster, seal, _ = stage.register()
    p = stage.PUBLIC/"analysis.json"
    prior = run.read(p)
    if prior["registration"] != key or prior["confirmation_access"]:
        raise ValueError("Require the completed development study")
    (d, h, y, gain, groups, risk_pass, dual), sources = stage.read_frozen(key, roster, seal)
    if sources != prior["score_sources"]:
        raise ValueError("Score inputs changed")
    reg = stage.PUBLIC/"relative_envelope_registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), source=run.artifact(p),
        reason="absolute residual guard abstained everywhere; test scale-dependent residuals at the same thresholds",
        specification="maximum nested held-source residual on the existing log1p(100*risk) training scale, clipped below at zero",
        development_followup=True, confirmation_access=False, original_2pct=True)))
    out = {}
    for name, x in zip(study.SIGNALS, (d[..., None], h[..., None], np.stack((d, h), axis=-1)), strict=True):
        point, bound = relative_bounds(x, y, groups)
        q = stage.selections(point, gain, risk_pass, dual, groups)
        if q != prior["metrics"][name]["point"] or (bound+1e-12 < point).any():
            raise ValueError("Point prediction or upper-bound direction changed")
        out[name] = stage.selections(bound, gain, risk_pass, dual, groups)
    run.once(stage.PUBLIC/"relative_envelope.json", run.stable(dict(registration=run.artifact(reg), metrics=out,
        original_2pct=True, confirmation_access=False, deployment_changed=False)))
    print(out, flush=True)


if __name__ == "__main__":
    main()
