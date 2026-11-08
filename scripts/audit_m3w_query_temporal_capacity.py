"""An exact hindsight risk ceiling over nonnegative temporal envelopes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_query_temporal_envelope as study

job, run, np = study.job, study.run, study.np


def minimize_observed_error(displacement, target, valid, fraction=.25):
    """Project on the ray per step, then enforce one step at the peak radius."""
    norms = np.linalg.norm(displacement, axis=-1)
    radius = fraction*norms.max(1, keepdims=True)
    unit = np.divide(displacement, norms[..., None], out=np.zeros_like(displacement),
                     where=norms[..., None] > 0)
    target = np.where(valid[..., None], target, 0.)
    projection = (unit*target).sum(-1).clip(0., radius)
    base = projection[..., None]*unit
    forced = radius[..., None]*unit
    extra = np.linalg.norm(forced-target, axis=-1)-np.linalg.norm(base-target, axis=-1)
    extra = np.where(norms > 0., np.where(valid, extra, 0.), np.inf)
    step = extra.argmin(1)
    rows = np.arange(len(step))
    base[rows, step] = forced[rows, step]
    return base


def main():
    job.torch.set_num_threads(4)
    job.torch.set_num_interop_threads(1)
    binding, identity, cut, freeze = study.register()
    analysis_path = study.OUT / "analysis.json"
    analysis = run.read(analysis_path)
    if analysis["registration"] != binding or analysis["confirmation_access"]:
        raise ValueError("Require completed held-source readout")
    x, _, blocks = job.prior.prior.prepare(identity, lambda **_: None)
    pairs = {"oracle_vs_cv": (job.CV, "oracle", job.CV), "oracle_vs_flat": ("flat", "oracle", job.CV)}
    rows = []
    for fold in (r for r in freeze["calibration"] if r["seed"] == job.SEEDS[0]):
        sets, _ = study.inputs(x[0], blocks, fold, freeze["registration"])
        held = fold["held"]
        for b, i, _, d, _, _ in sets:
            if b["source"] != held:
                continue
            valid = b["valid"][i]
            known = valid.any(1)
            cv = b["cv"]
            flat = cv.copy()
            flat[i] += .25*d
            oracle = flat.copy()
            good = i[known]
            oracle[good] = cv[good]+minimize_observed_error(d[known], b["target"][good]-cv[good], valid[known])
            np.testing.assert_allclose(np.linalg.norm(oracle-cv, axis=2).max(1),
                                       np.linalg.norm(flat-cv, axis=2).max(1), rtol=1e-9, atol=1e-9)
            row = job.prior.prior.prior.base.score_recording({job.CV: cv, "flat": flat, "oracle": oracle},
                pairs, b["target"], b["valid"], b["inverse"], b["rec"],
                easy_cut=cut["easy"], hard_cut=cut["hard"])
            m, f = (row["metrics"][k]["risk"] for k in ("oracle", "flat"))
            for k in ("completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation"):
                if (m[k] is None) != (f[k] is None) or m[k] is not None and m[k] > f[k]+1e-10:
                    raise ValueError("Risk-minimizing envelope did not dominate the flat control")
            rows.append(dict(row, source=held, seed=job.SEEDS[0]))
    if len(rows) != 24:
        raise ValueError("Expected 24 recordings; fixed choices are seed invariant")
    summary = job.grouped_readout(rows, {b["rec"]: b["source"] for b in blocks}, [job.SEEDS[0]])
    run.once(study.OUT / "capacity.json", run.stable(dict(analysis=run.artifact(analysis_path),
        code=run.artifact(Path(__file__)), summary=summary, records=[
            dict(recording=r["recording"], source=r["source"], risk=r["metrics"]["oracle"]["risk"]) for r in rows],
        fixed_recipients=True, fixed_radius=True, allows_zero_per_step=True,
        label_aware=True, prediction_exported=False, confirmation_access=False,
        does_not_bound_other_recipients_or_directions=True)))


if __name__ == "__main__":
    main()
