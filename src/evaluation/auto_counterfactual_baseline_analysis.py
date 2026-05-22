from __future__ import annotations


def counterfactual_baseline_questions() -> list[str]:
    return [
        "Would a stronger per-dataset causal baseline remove the apparent learned gain?",
        "Does the learned residual intervene mostly on easy cases where alpha should be low?",
        "Do scene/goal features improve hard/failure subsets after controlling for baseline FDE?",
    ]

