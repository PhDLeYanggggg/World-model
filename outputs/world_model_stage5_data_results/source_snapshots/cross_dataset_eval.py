from __future__ import annotations

from typing import Dict, List


def cross_dataset_eval_plan(converted_datasets: List[str]) -> Dict:
    pairs = []
    for train in converted_datasets:
        for test in converted_datasets:
            if train != test:
                pairs.append({"train": train, "test": test, "status": "pending"})
    return {
        "converted_datasets": converted_datasets,
        "leave_one_dataset_out": "pending_until_3_real_datasets",
        "pairs": pairs,
        "note": "Stage 5-Data dry run creates the evaluation plan; real evaluation waits for more converted data.",
    }
