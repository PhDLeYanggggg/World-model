from pathlib import Path

from src import stage41_m3w_neural_v1_package as pkg


def test_stage41_m3w_neural_v1_package_paths() -> None:
    assert pkg.OUT_DIR == Path("outputs/m3w_neural_v1")
    assert any("world_model_gate_stage41.json" in str(p) for p in pkg.SOURCE_PATHS)
    assert any("stage41_fresh_self_gated_endpoint_candidate.json" in str(p) for p in pkg.SOURCE_PATHS)


def test_stage41_m3w_neural_v1_current_facts_block_overclaim() -> None:
    text = "\n".join(pkg.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "不是 large-scale foundation" in text
    assert "raw-frame" in text
    assert "SMC 未启用" in text
