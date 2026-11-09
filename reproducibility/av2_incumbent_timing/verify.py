"""Offline verification of anonymous fixed-budget timing diagnostics."""

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODES = ("replay", "geometry", "history")
LEVELS = ("local_timing", "same_actor_and_peak", "same_actors", "all_recipients")


def read(name):
    return json.loads((ROOT/name).read_bytes())


def qualifies(row):
    risk = row["risk"]
    trajectory, pair = row["relative"] if row["relative"] is not None else (None, None)
    return bool(row["anchor_eligible"] and row["gain"] is not None and min(row["gain"]) > 0
                and all(x is not None and x <= .02 for x in risk)
                and trajectory is not None and trajectory >= -1e-8 and pair > 1e-10)


def verify():
    for name, value in read("manifest.json")["members"].items():
        if sha256((ROOT/name).read_bytes()).hexdigest() != value:
            raise ValueError(f"Changed file: {name}")
    evidence = read("evidence.json")
    groups = evidence["groups"]
    capacity = {k: sum(g["capacity"][k] for g in groups) for k in LEVELS}
    top = {m: sum(g["top"][m] for g in groups) for m in MODES}
    released = {m: len(evidence["released"][m]) for m in MODES}
    passed = {m: sum(qualifies(r) for r in evidence["released"][m]) for m in MODES}
    for g in groups:
        q = [g["capacity"][k] for k in LEVELS]
        if q != sorted(q) or q[-1] > g["anchor_eligible"] or g["anchor_eligible"] > g["views"]:
            raise ValueError("Capacity is not nested within eligible anchors")
        if any(g["top"][m] > q[0] for m in MODES):
            raise ValueError("A selection count exceeds local capacity")
    if any(passed[m] > top[m] for m in MODES):
        raise ValueError("A release count exceeds qualifying choices")
    return dict(two_slot_views=sum(g["views"] for g in groups), capacity=capacity,
                top=top, released=released, passed=passed, development_only=True)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
