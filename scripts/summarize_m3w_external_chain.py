"""Reduce frozen chain probe receipts without opening trajectories or outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/"outputs/publication_readiness_2026_09/external_policy_chain_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root, analysis):
    rows = []
    for ref in analysis["views"]:
        path = root/ref["path"]
        if digest(path) != ref["sha256"]:
            raise ValueError("Changed probe receipt")
        r = json.loads(path.read_text())
        if r["manifest_sha256"] != analysis["manifest_sha256"] or r["view"] != ref["view"]:
            raise ValueError("Wrong complete-chain identity")
        q = r["queries"]
        rows.append(dict(view=r["view"], queries=len(q),
            strict_interventions=sum(x["eligible"] for x in q),
            half_interventions=sum(x["switch_counts"]["half_joint"] for x in q),
            half_unary_reference_changed_agents=sum(x["budgets"]["half"]["unary_reference_changed_agents"] for x in q),
            half_joint_unary_changed_agents=sum(x["budgets"]["half"]["joint_unary_changed_agents"] for x in q),
            nonadditive_queries=sum(x["budgets"]["half"]["nonadditive_supported_edges"] > 0 for x in q)))
    if len(rows) != 12 or sum(x["queries"] for x in rows) != analysis["source_query_view_instances"]:
        raise ValueError("Incomplete chain view reduction")
    return dict(result_source="cached_verified_probe_reduction", views=rows,
        joint_effect_observed=any(x["half_joint_unary_changed_agents"] > 0 for x in rows),
        predictive_lift_evaluated=False, exhaustive_source_opportunity_audit=False,
        unique_query_count=analysis["source_queries"], independent_cluster_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = PUBLIC/"analysis.json"
    analysis = json.loads(path.read_text())
    replay = json.loads((PUBLIC/"replay.json").read_text())
    if not replay["all_checks_passed"] or replay["analysis_sha256"] != digest(path):
        raise ValueError("A complete exact replay is required")
    result = dict(analysis_sha256=digest(path), **summarize(ROOT, analysis))
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    target = PUBLIC/"mechanism_probe_summary.json"
    if target.exists() or args.verify:
        if target.read_text() != text:
            raise ValueError("Changed mechanism probe reduction")
    else:
        target.write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k != "views"}))


if __name__ == "__main__":
    main()
