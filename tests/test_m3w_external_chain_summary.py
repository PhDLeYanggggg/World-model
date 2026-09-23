import hashlib
import json

import pytest

from scripts.summarize_m3w_external_chain import summarize


def fixture(tmp_path):
    query = dict(eligible=2, switch_counts=dict(half_joint=1), budgets=dict(half=dict(
        unary_reference_changed_agents=0, joint_unary_changed_agents=0, nonadditive_supported_edges=0)))
    refs = []
    for i in range(12):
        p = tmp_path/f"view{i}.json"
        p.write_text(json.dumps(dict(manifest_sha256="frozen", view=str(i), queries=[query])))
        refs.append(dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), view=str(i)))
    return dict(views=refs, manifest_sha256="frozen", source_queries=1, source_query_view_instances=12)


def test_no_joint_effect_is_retained_without_predictive_claim(tmp_path):
    result = summarize(tmp_path, fixture(tmp_path))
    assert not result["joint_effect_observed"] and not result["predictive_lift_evaluated"]
    assert result["unique_query_count"] == 1
    assert sum(r["strict_interventions"] for r in result["views"]) == 24


def test_probe_tampering_fails_before_reduction(tmp_path):
    data = fixture(tmp_path)
    (tmp_path/"view0.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed"):
        summarize(tmp_path, data)
