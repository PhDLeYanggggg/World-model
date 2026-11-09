"""Compare two fixed motion baselines on identical complete MOT20 pilot targets."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_mot20_camera_support as pilot
from scripts import evaluate_m3w_mot20_camera_support as evaluation


def query_errors(frames, references, rollout):
    histories, matches = {}, []
    for frame in frames:
        rows = np.asarray(frame["rows"]).reshape(-1, 7)
        hits, _ = pilot.match(rows, references.get(frame["camera_frame"], np.empty((0, 9))))
        matches.append(hits)
        for row in rows:
            histories.setdefault(int(row[0]), []).append([(row[1] + row[3]) / 2, row[4]])
    start = frames[0]["camera_frame"]
    paired = []
    for online, reference in sorted(set.intersection(*matches)):
        points = []
        for k in range(pilot.OBS, pilot.OBS + pilot.PRED):
            labels = references.get(pilot.frame_id(start, k), np.empty((0, 9)))
            target = labels[(labels[:, 1] == reference) & (labels[:, 6] != 0) & (labels[:, 7] == 1)]
            if len(target) != 1:
                break
            box = target[0]
            points.append([(box[2] + box[4]) / 2, box[5]])
        if len(points) != pilot.PRED:
            continue
        errors = np.linalg.norm(rollout(histories[online]) - np.asarray(points), axis=-1)
        paired.append(np.column_stack((errors.mean(axis=-1), errors[:, -1])))
    return np.asarray(paired).reshape(-1, 2, 2)


def summarize(queries):
    observed = [q for q in queries if len(q)]
    n = sum(len(q) for q in observed)
    counts = dict(complete_histories=n, scored_windows=len(observed))
    if n < 8 or len(observed) < 3:
        return dict(**counts, status="not_run_insufficient_complete_support")
    means = np.mean([q.mean(axis=0) for q in observed], axis=0)
    endpoint_oracle = np.mean([q[:, :, 1].min(axis=1).mean() for q in observed])
    return dict(**counts, status="complete_case_development_diagnostic", units="image pixels",
                equal_window_mean_ade_fde=dict(zip(("CV", "anchored_OLS4"), means.tolist(), strict=True)),
                per_target_endpoint_oracle_fde_px=float(endpoint_oracle),
                ols4_lower_fde_windows=sum(q[:, 1, 1].mean() < q[:, 0, 1].mean() for q in observed))


def main():
    from scripts import run_m3w_acme_hybrid_detector as hybrid

    path = pilot.PUBLIC / "registration.json"
    readout = pilot.PUBLIC / "analysis.json"
    reg, result = [json.loads(f.read_text()) for f in (path, readout)]
    key = hybrid.sha(path)
    gt = pilot.LOCAL / "gt.txt"
    evaluation_path = pilot.PUBLIC / "evaluation_registration.json"
    spec = json.loads(evaluation_path.read_text())
    if (reg["bindings"] != pilot.bindings(hybrid) or result["registration_sha256"] != key
            or result["evaluation_registration_sha256"] != hybrid.sha(evaluation_path)
            or spec["evaluator_sha256"] != hybrid.sha(Path(evaluation.__file__))
            or spec["motion_diagnostic_sha256"] != hybrid.sha(Path(__file__))
            or result["label_sha256"] != hybrid.sha(gt)):
        raise ValueError("The camera or label receipt changed")
    complete = [v["strict"]["complete"] for v in result["windows"]["causal_transport"]]
    if sum(complete) >= 8 and sum(n > 0 for n in complete) >= 3:
        labels = evaluation.zero_based_references(gt.read_bytes())
        errors = [query_errors(hybrid.old.reassociate(pilot.read_window(s, key)["frames"]),
                               labels, hybrid.old.gmu.rollout) for s in pilot.starts()]
        if [len(e) for e in errors] != complete:
            raise ValueError("Baseline and camera support populations differ")
        report = summarize(errors)
    else:
        report = dict(status="not_run_insufficient_complete_support", complete_histories=sum(complete),
                      scored_windows=0)
    report.update(result_source="fresh_run", code_sha256=hybrid.sha(Path(__file__)),
                  source_analysis_sha256=hybrid.sha(readout), independent_confirmation=False,
                  whole_roster_policy_test=False, no_training=True)
    hybrid.immutable(pilot.PUBLIC / "motion_diagnostic.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
