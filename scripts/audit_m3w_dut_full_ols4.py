"""Add OLS4 to the exact historical DUT population using bound aggregate receipts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_dut_matched_motion as study

np, old, method = study.np, study.old, study.method


def complete_motion(points):
    """Recount contiguous prefixes; forecast before matching future indices."""
    rows = dict(queries=0, targets=0, complete=0, known_steps=0, cv_ade_sum=0., cv_fde_sum=0.,
                ols4_ade_sum=0., ols4_fde_sum=0.)
    times = np.unique(points[:, 0])
    rows["queries"] = int((times >= times[0] + 7).sum())
    for agent in np.unique(points[:, 1]):
        p = points[points[:, 1] == agent]
        p = p[np.argsort(p[:, 0])]
        frame, xy = p[:, 0], p[:, 2:]
        if len(frame) < 8:
            continue
        if np.any(np.diff(frame) <= 0):
            raise ValueError("Duplicate DUT agent/frame")
        at = np.flatnonzero(frame[7:] - frame[:-7] == 7) + 7
        h = xy[at[:, None] + np.arange(-7, 1)]
        cv = h[:, -1:] + (h[:, -1:] - h[:, -2:-1]) * np.arange(1, 13)[None, :, None]
        ols = method.ols4(h)
        requested = frame[at, None] + np.arange(1, 13)
        index = np.searchsorted(frame, requested).clip(max=len(frame) - 1)
        ok = frame[index] == requested
        full = ok.all(1)
        y = xy[index[full]]
        rows["targets"] += len(at)
        rows["known_steps"] += int(ok.sum())
        rows["complete"] += int(full.sum())
        for k, pred in (("cv", cv), ("ols4", ols)):
            e = np.linalg.norm(pred[full] - y, axis=-1)
            rows[f"{k}_ade_sum"] += float(e.mean(1).sum())
            rows[f"{k}_fde_sum"] += float(e[:, -1].sum())
    return rows


def main():
    _, cfg, manifest, _ = study.register()
    old_path = study.ROOT / cfg["reports"] / "analysis.json"
    previous = study.read(old_path)
    ref = {v["recording"]: v for v in previous["records"]}
    if set(ref) != set(manifest["records"]):
        raise ValueError("Full DUT roster changed")
    output = {}
    for name, record in manifest["records"].items():
        path = study.ROOT / ref[name]["path"]
        if old.sha256(path) != ref[name]["sha256"]:
            raise ValueError("Frozen old recording receipt changed")
        old_row = study.read(path)
        if old_row["manifest_sha256"] != previous["manifest_sha256"] or old_row["recording"] != name:
            raise ValueError("Frozen full result identity changed")
        reader = old.RecordingWindows(study.ROOT / record["cache_path"])
        calc = complete_motion(reader.points)
        stats = old_row["stats"]
        first = next(v for v in stats.values() if v["baseline_controls"])
        native_cv = first["baseline_controls"]["constant_velocity_causal_fd"]["slices"]["all"]
        for f in ("queries", "targets", "complete", "known_steps"):
            if any(calc[f] != c[f] for c in stats.values()):
                raise ValueError(f"Full cohort differs: {name}/{f}")
        for metric in ("ade", "fde"):
            np.testing.assert_allclose(calc[f"cv_{metric}_sum"], native_cv[f"{metric}_sum"], rtol=1e-12, atol=1e-8)
        cached = {}
        for seed in study.SEEDS:
            m = stats[f"eqmotion_seed{seed}_bounded_fraction"]["arms"]
            cached[str(seed)] = {k: {f: m[k]["slices"]["all"][f] for f in ("n", "ade_sum", "fde_sum")}
                                 for k in ("uncontrolled", "strict")}
            if any(c["n"] != calc["complete"] for c in cached[str(seed)].values()):
                raise ValueError("Frozen EqMotion target counts differ")
        output[name] = dict(site=record["physical_scene"], motion=calc, cached_eqmotion=cached)
    sites = {}
    for site in sorted({r["site"] for r in output.values()}):
        parts = [r for r in output.values() if r["site"] == site]
        count = sum(r["motion"]["complete"] for r in parts)
        scores = {k: {m: sum(r["motion"][f"{k}_{m}_sum"] for r in parts) / count
                      for m in ("ade", "fde")} for k in ("cv", "ols4")}
        for k in ("uncontrolled", "strict"):
            scores[k] = {str(s): {m: sum(r["cached_eqmotion"][str(s)][k][f"{m}_sum"] for r in parts) / count
                                  for m in ("ade", "fde")} for s in study.SEEDS}
        sites[site] = scores
    totals = {k: sum(r["motion"][k] for r in output.values()) for k in ("queries", "targets", "complete", "known_steps")}
    old.immutable_json(study.OUT / "full_ols4.json", dict(
        result_source="fresh_run_motion_on_full_population_cached_verified_neural_errors",
        code=study.artifact(Path(__file__)), original=study.artifact(old_path),
        totals=totals, sites=sites, recordings=output, complete_case_only=True,
        new_neural_inference=False, forecast_pairs_not_reconstructible=True, confirmation_access=False))
    print(dict(totals=totals, sites=sites), flush=True)


if __name__ == "__main__":
    main()
