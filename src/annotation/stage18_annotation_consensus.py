from __future__ import annotations


def consensus_score(annotation):
    return float(annotation.get("self_checks", {}).get("consensus_score", 0.0))

