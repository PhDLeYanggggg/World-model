"""Replay the anonymized paired coverage table with the standard library."""

from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from math import lcm
from pathlib import Path


def percentile(sorted_values, q):
    at = (len(sorted_values)-1)*q
    lo = int(at)
    hi = min(lo+1, len(sorted_values)-1)
    return sorted_values[lo]+(at-lo)*(sorted_values[hi]-sorted_values[lo])


def replay(payload):
    views = payload["cluster_views"]
    values = payload["coverage_count_changes"]
    n = len(views)
    if n != 6 or len(values) != n or sum(views) != payload["views"]:
        raise ValueError("Invalid development cluster roster")
    if payload["views"] != 3*payload["scenarios"]:
        raise ValueError("Require three paired seed views per scenario")
    common = lcm(*views)
    terms = []
    for k, term in enumerate(payload["terms"]):
        mass = [row[k]*common//v for row, v in zip(values, views, strict=True)]
        draws = sorted(map(sum, product(mass, repeat=n)))
        denom = n*common
        terms.append(dict(term=term, delta_pp=float(100*Fraction(sum(mass), denom)),
            exact_ci95_pp=[float(100*percentile(draws, q)/denom) for q in (Fraction(1, 40), Fraction(39, 40))],
            pooled_frozen=float(Fraction(payload["frozen_covered"][k], payload["views"])),
            pooled_selected=float(Fraction(payload["selected_covered"][k], payload["views"]))))
    return dict(scenarios=payload["scenarios"], views=payload["views"], cluster_bootstrap_draws=n**n,
                terms=terms, held_veto=payload["held_veto"])


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/"manifest.json").read_bytes())
    for name, digest in manifest["members"].items():
        if sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Digest mismatch: {name}")
    print(json.dumps(replay(json.loads((root/"evidence.json").read_bytes())), indent=2))


if __name__ == "__main__":
    main()
