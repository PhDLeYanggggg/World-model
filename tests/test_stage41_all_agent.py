from src import stage41_all_agent as s41a


def test_stage41_all_agent_constants() -> None:
    assert s41a.MAX_AGENTS >= 2
    assert s41a.TOKEN_K == 32
    assert s41a.OUT_DIR.name == "stage41_breakthrough"


def test_stage41_all_agent_dataset_shapes() -> None:
    report = s41a.build_all_agent_dataset()
    ds = s41a._ds("test")
    assert ds["agent_tokens"].ndim == 4
    assert ds["agent_tokens"].shape[1] == s41a.MAX_AGENTS
    assert ds["agent_tokens"].shape[2] == s41a.TOKEN_K
    assert ds["agent_mask"].shape[:2] == ds["agent_tokens"].shape[:2]
    assert report["no_leakage"]["future_endpoint_input"] is False
