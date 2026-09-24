"""Post-readout descriptive harms; never changes a frozen policy or threshold."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from scripts.run_m3w_native_forecast import immutable_json


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    cfg = read("configs/m3w_cutoff_relative_risk_v1.json")
    private, public = ROOT/cfg["output"], ROOT/cfg["reports"]
    a = read(public/"analysis.json")
    outcomes = read("outputs/publication_readiness_2026_09/easy_moment_v1/analysis.json")
    done = read(private/"decisions_complete.json")
    assert file_digest(private/"decisions_complete.json") == a["decisions_sha256"]
    rows, statuses = {}, {}
    refs = {}
    for item in done["archives"]:
        assert file_digest(ROOT/item["path"]) == item["sha256"]
        r = read(item["path"])
        assert file_digest(ROOT/r["path"]) == r["sha256"]
        refs[r["view"], r["action"]] = r
        for q in r["queries"]:
            for kind in ("population", "selected"):
                status = q[kind]["status"]
                statuses[status] = statuses.get(status, 0)+1
    for action in cfg["actions"]:
        for j, policy in enumerate(cfg["policies"]):
            per_seed, harmed_ids, harms = {}, set(), []
            for seed in cfg["seeds"]:
                ref = next(r for r in outcomes["outcome_archives"] if r["path"].endswith(f"{action}_seed{seed}.npz"))
                assert file_digest(ROOT/ref["path"]) == ref["sha256"]
                with np.load(ROOT/ref["path"], allow_pickle=False) as z:
                    cv, candidate, zero = z["cv"], z["candidate_ade"], z["zero_CV"]
                bits = np.zeros(a["rows"], bool)
                by_scene = {}
                for site in cfg["sites"]:
                    r = refs[f"{site}_seed{seed}", action]
                    with np.load(ROOT/r["path"], allow_pickle=False) as z:
                        ids = z["ids"]; bits[ids] = z["choices"][:, j]
                    mask = bits[ids] & zero[ids] & (candidate[ids] > 0)
                    by_scene[site] = int(mask.sum())
                bad = bits & zero & (candidate > 0)
                assert int(bad.sum()) == a["summary"][action+"__"+policy]["seeds"][str(seed)]["zero_CV_harmed"]
                harmed_ids.update(np.flatnonzero(bad).tolist()); harms.extend(candidate[bad].tolist())
                per_seed[str(seed)] = dict(zero_cv_harmed=int(bad.sum()), zero_cv_total=int(zero.sum()),
                    by_scene=by_scene, selected_unknown=int((bits & ~np.isfinite(cv)).sum()))
            rows[action+"__"+policy] = dict(seeds=per_seed, unique_window_count=len(harmed_ids),
                repeated_window_seed_count=len(harms), observed_zero_cv_harm_ade_annotation_pixels={
                    "min":float(min(harms)), "median":float(np.median(harms)), "max":float(max(harms)),
                    "sum_over_repeated_window_seed_instances":float(sum(harms))} if harms else None)
    report = dict(result_source="fresh_run", role="post_readout_descriptive_not_registered_hypothesis_test",
        analysis_sha256=file_digest(public/"analysis.json"), code_sha256=file_digest(Path(__file__)),
        policy_changed=False, threshold_changed=False, external_readout=False,
        zero_cv_harms=rows, solver_status_counts=statuses)
    immutable_json(public/"postreadout_diagnostics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
