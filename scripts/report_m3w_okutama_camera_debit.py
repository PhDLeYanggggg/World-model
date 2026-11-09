"""Separate observed gain from missing-step debit on frozen Okutama forecasts."""

from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_okutama_past_camera as run

np, study, base, turn = run.np, run.study, run.base, run.turn


def report():
    key, manifest, plan = run.register()
    path = run.OUT/"forecast_freeze.json"
    readout = run.OUT/"readout.json"
    freeze, result = map(study.read, (path, readout))
    if (freeze["registration"] != key or result["forecast_freeze"] != study.artifact(path)
            or result["future_images_accessed"] or result["reserved_outcomes_opened"]):
        raise ValueError("Require verified past-only study")
    rows = {arm: defaultdict(list) for arm in run.REFERENCE}
    endpoints = {arm: defaultdict(list) for arm in run.REFERENCE}
    for clip, r in plan.items():
        count = len(manifest["frames"][clip])
        labels = run.source.parse((run.source.SOURCE/f"{clip}.txt").read_bytes(), manifest["frames"][clip])
        points = run.source.trajectories(labels, count, "nongenerated")
        with np.load(study.checked(freeze["predictions"][clip]), allow_pickle=False) as z:
            y = points[r["start"]+np.arange(8, 20)*run.STRIDE][:, z["ids"]].transpose(1, 0, 2)-z["h"][:, -1:]
            v = np.isfinite(y).all(-1)
            y = np.where(v[..., None], y, 0.)
            for arm, ref in run.REFERENCE.items():
                before = np.linalg.norm(z[ref]-y, axis=-1)
                after = np.linalg.norm(z[arm]-y, axis=-1)
                denom = before[v].sum()
                if denom > 0.:
                    known = float(100.*(before-after)[v].sum()/denom)
                    debit = float(100.*np.linalg.norm(z[arm]-z[ref], axis=-1)[~v].sum()/denom)
                    rows[arm][r["scenario"]].append([known, debit, known-debit])
                if v[:, -1].any():
                    endpoints[arm][r["scenario"]].append(float((before-after)[v[:, -1], -1].mean()))
    output = {}
    for arm, cases in rows.items():
        means = {k: np.mean(v, axis=0).tolist() for k, v in sorted(cases.items())}
        summary = turn.effects.interval(list(means.values()))
        ref = result["all_windows"][arm]["lower_gain_vs_matched_raw_pp"]
        np.testing.assert_allclose(summary["site_mean"][2], ref["site_mean"], atol=1e-10)
        output[arm] = dict(pp=summary, by_scenario_pp=means,
                          paired_endpoint_gain_px=turn.effects.interval([np.mean(v) for v in endpoints[arm].values()]))
    evidence = dict(phase="frozen_forecast_attribution", columns=["known_gain", "missing_debit", "conservative_gain"],
        source=study.artifact(readout), code=study.artifact(Path(__file__)), totals=output, threshold_search=False)
    base.immutable(run.OUT/"debit_attribution.json", evidence)
    print(study.json.dumps({k: dict(pp=v["pp"], endpoint=v["paired_endpoint_gain_px"]) for k, v in output.items()}, indent=2))


if __name__ == "__main__":
    report()
