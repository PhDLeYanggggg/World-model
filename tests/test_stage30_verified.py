from pathlib import Path

import pytest

from src import stage30_m3w_verified as s30


def test_stage30_source_labels_and_ci_helper() -> None:
    ci = s30._mean_std_ci([0.1, 0.2, 0.3])
    assert ci["mean"] == pytest.approx(0.2)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert s30.OUT_DIR == Path("outputs/stage30_m3w_verified")
    assert s30.BASE_CURRENT_FACTS


def test_stage30_hash_missing_is_explicit() -> None:
    assert s30._hash_path(Path("definitely_missing_stage30_input")) == "missing"
