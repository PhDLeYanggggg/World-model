import csv
import io
import json

import pytest

from scripts.report_m3w_external_cost_bank import REPORT, render


def test_frozen_cost_logs_reconstruct_exactly_with_lf():
    analysis = json.loads((REPORT/"analysis.json").read_text())
    outputs = render(analysis)
    for name, value in outputs.items():
        assert (REPORT/name).read_bytes() == value.encode()
        assert "\r" not in value
    records = list(csv.DictReader(io.StringIO(outputs["training_loss.csv"])))
    assert len(records) == 234
    assert sum(row["head"] == "bounded_fraction" for row in records) == 186
    assert sum(row["head"] == "matched_fraction_forest" for row in records) == 48


def test_incomplete_cost_bank_cannot_render_as_complete():
    analysis = json.loads((REPORT/"analysis.json").read_text())
    analysis["models"][0]["fit"]["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        render(analysis)


def test_cost_report_retains_non_evaluation_and_negative_loss_qualifiers():
    analysis = json.loads((REPORT/"analysis.json").read_text())
    text = render(analysis)["training_losses.md"]
    assert "not a fixed validation set" in text
    assert "not directly comparable" in text
    assert "slightly higher than at sixteen trees" in text
    assert "not" in text and "downstream-lift" in text
