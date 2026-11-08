"""Change only causal association on exposed GMU camera detections."""

from collections import defaultdict, deque
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_acme_gmu_camera as initial
from scripts.run_m3w_acme_camera_repair import associate_labels, feet
from src.world_model.m3w_streaming_boxes import overlap


OUT = ROOT/"outputs/publication_readiness_2026_09/acme_gmu_camera_v1"


def last_position_and_transport(history, tick):
    last_time, last = history[-1]
    predicted = last.copy()
    if len(history) > 1:
        prior_time, prior = history[-2]
        v = (feet(last[None, :])[0]-feet(prior[None, :])[0])/(last_time-prior_time)
        limit = np.linalg.norm(last[2:4]-last[:2])
        v *= min(1., limit/max(np.linalg.norm(v), 1e-9))
        predicted[:4] += np.tile(v*(tick-last_time), 2)
    return last, predicted


def reassociate(frames):
    histories = defaultdict(lambda: deque(maxlen=2))
    counter = 0
    revised = []
    for t, frame in enumerate(frames):
        detections = np.asarray(frame["rows"], dtype=float).reshape(-1, 7)[:, 1:]
        active = [i for i, h in histories.items() if t-h[-1][0] <= 2]
        new_ids = {}
        if active and len(detections):
            last, predicted = map(np.stack, zip(*(last_position_and_transport(histories[i], t) for i in active)))
            sizes = np.maximum(np.linalg.norm(last[:, 2:4]-last[:, :2], axis=-1)[:, None],
                               np.linalg.norm(detections[:, 2:4]-detections[:, :2], axis=-1)[None, :])

            def costs(boxes):
                d = np.linalg.norm(feet(boxes)[:, None]-feet(detections)[None, :], axis=-1)/sizes.clip(1.)
                iou = overlap(boxes[:, :4], detections[:, :4])
                return np.where((iou >= .1) | (d <= 1.0), .65*(1-iou)+.35*d.clip(0., 1.), 1e6)

            matrix = np.minimum(costs(predicted), costs(last)+.1)
            i, j = linear_sum_assignment(matrix)
            new_ids = {b: active[a] for a, b in zip(i, j) if matrix[a, b] < 1e6}
        rows = []
        for j, detection in enumerate(detections):
            if j not in new_ids:
                new_ids[j] = counter
                counter += 1
            ident = new_ids[j]
            histories[ident].append((t, detection))
            rows.append([ident, *detection.tolist()])
        revised.append({**frame, "rows": rows})
    return revised


def all_frame_label_support(frames, labels, start):
    hits = []
    for k, f in enumerate(frames):
        tick = start+k*initial.STRIDE
        targets = [(agent, points[tick]) for agent, points in labels.items() if tick in points]
        pairs = associate_labels(np.asarray(f["rows"]).reshape(-1, 7), targets)
        hits.append({label for _, label in pairs})
    return len(set.intersection(*hits))


def main():
    register = OUT/"registration.json"
    base = OUT/"analysis.json"
    spec = json.loads(register.read_text())
    if initial.sha(initial.LABELS) != initial.LABEL_SHA:
        raise ValueError("Label snapshot changed")
    binding = initial.sha(register)
    initial.immutable(OUT/"reassociation_registration.json", dict(
        development_only=True, original_registration_sha256=binding,
        original_readout_sha256=initial.sha(base), code_sha256=initial.sha(Path(__file__)),
        changes="two-point causal foot transport with static association fallback",
        unchanged="detections, windows, label match, support thresholds, motion baselines",
    ))
    labels = initial.label_tables()
    results, predictions = [], []
    for s in spec["starts"]:
        saved = json.loads((initial.PRIVATE/f"window_{s:05d}.json").read_text())
        if saved["registration"] != binding:
            raise ValueError("Unregistered prefix cache")
        frames = saved["frames"]
        repaired = reassociate(frames)
        summary, bank = initial.associate_query(repaired, labels, s)
        summary["all_frame_hits_ignoring_ids_diagnostic"] = all_frame_label_support(frames, labels, s)
        results.append(summary)
        predictions.append(bank)
    n = sum(row["stable_matches"] for row in results)
    q = sum(row["stable_matches"] > 0 for row in results)
    passed = n >= initial.MIN_HISTORIES and q >= initial.MIN_QUERIES
    scores, complete = [], 0
    if passed:
        for s, bank in zip(spec["starts"], predictions, strict=True):
            per_window = []
            for lab, paths in bank:
                target = [labels[lab].get(s+(8+i)*initial.STRIDE) for i in range(12)]
                if any(p is None for p in target):
                    continue
                d = np.linalg.norm(paths-np.asarray(target), axis=-1)
                per_window.append(np.column_stack((d.mean(-1), d[:, -1])))
            if per_window:
                complete += len(per_window)
                scores.append(np.mean(per_window, axis=0))
    result = dict(
        windows=results, stable_histories=n, matched_queries=q, support_gate=passed,
        score_population="observed_complete_cases_development_only" if scores else "not_run",
        complete_futures=complete if passed else None, missing_futures=n-complete if passed else None,
        query_equal_ade_fde_px=dict(zip(("CV", "OLS4"), np.mean(scores, axis=0).tolist())) if scores else None,
        windows_with_scores=len(scores), independent_confirmation=False, deployment_or_risk_evaluation="not_run",
    )
    initial.immutable(OUT/"reassociation_analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}))


if __name__ == "__main__":
    main()
