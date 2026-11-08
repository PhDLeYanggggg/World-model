"""Use frozen scalar masses to decompose the observed risk reversal."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_shared_correction_audit as job
from src.evaluation.m3w_risk_mixture import gap


def main():
    run = job.run
    p = job.OUT/"analysis.json"
    a = run.read(p)
    if a["confirmation_opened"] or a["risk_rule_changed"] or a["deployment"]:
        raise ValueError("Use the opened diagnostic only")
    result = {}
    for arm in (job.RAW, job.SHARED, job.MAG):
        s, t = (a[d]["strata"][arm] for d in ("fitting", "opened_calibration"))
        result[arm] = dict(all=gap(s, t), easy=gap(s, t, "easy_harm", "easy_reference"))
    run.once(job.OUT/"risk_mixture.json", run.stable(dict(analysis=run.artifact(p),
        result=result, scope="descriptive_selected_reference_mass_decomposition_not_camera_causality")))


if __name__ == "__main__":
    main()
