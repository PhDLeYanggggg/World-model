from src.stage43_source_slice_repair import run_source_slice_repair


def test_stage43_k_source_slice_repair_passes_without_test_tuning():
    payload = run_source_slice_repair(bootstrap=20, batch_size=8192)
    gate = payload["stage43_k_gate"]
    assert gate["verdict"] == "stage43_k_source_slice_negative_repaired"
    assert gate["passed"] == gate["total"]
    assert payload["deployment_policy"]["test_tuned"] is False
    assert payload["deployment_policy"]["policy"]["allowed_families_source"] == "validation_only"


def test_stage43_k_repairs_negative_source_but_blocks_uniform_positive_claim():
    payload = run_source_slice_repair(bootstrap=20, batch_size=8192)
    dep = payload["deployment_policy"]
    assert dep["source_negative_count"] == 0
    assert dep["max_source_easy_degradation"] <= 0.02
    assert "TrajNet_mot" in dep["blocked_test_families"]
    assert payload["stage43_k_gate"]["source_safe_candidate"] is True
    assert payload["stage43_k_gate"]["uniform_positive_source_candidate"] is False
