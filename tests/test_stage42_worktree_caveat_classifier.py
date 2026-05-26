from __future__ import annotations

import json

from src import stage42_worktree_caveat_classifier as cy


def test_strip_json_metadata_ignores_only_allowed_keys() -> None:
    before = {"a": 1, "generated_at_utc": "old", "nested": {"git_commit": "abc", "value": 2}}
    after = {"a": 1, "generated_at_utc": "new", "nested": {"git_commit": "def", "value": 2}}
    assert cy._strip_json_metadata(before, paper_size_allowed=False) == cy._strip_json_metadata(
        after, paper_size_allowed=False
    )
    changed = {"a": 99, "generated_at_utc": "new", "nested": {"git_commit": "def", "value": 2}}
    assert cy._strip_json_metadata(before, paper_size_allowed=False) != cy._strip_json_metadata(
        changed, paper_size_allowed=False
    )


def test_json_change_kind_distinguishes_metadata_and_substantive() -> None:
    head = json.dumps({"generated_at_utc": "old", "metric": 1.0})
    work = json.dumps({"generated_at_utc": "new", "metric": 1.0})
    assert cy._json_change_kind(cy.OUT_DIR / "x.json", head, work) == "metadata_only"
    changed = json.dumps({"generated_at_utc": "new", "metric": 2.0})
    assert cy._json_change_kind(cy.OUT_DIR / "x.json", head, changed) == "substantive_json_change"


def test_json_change_kind_allows_paper_size_only_for_claim_audit() -> None:
    path = cy.OUT_DIR / "paper_claim_evidence_audit_stage42.json"
    head = json.dumps({"paper_files": [{"path": "a", "size_bytes": 1}], "metric": 3})
    work = json.dumps({"paper_files": [{"path": "a", "size_bytes": 2}], "metric": 3})
    assert cy._json_change_kind(path, head, work) == "metadata_and_paper_size_only"


def test_md_change_kind_allows_paper_size_table_rows(tmp_path) -> None:
    path = tmp_path / "paper_claim_evidence_audit_stage42.md"
    diff = """diff --git a/x b/x
--- a/x
+++ b/x
@@ -1,2 +1,2 @@
-- generated_at_utc: `old`
+- generated_at_utc: `new`
-| `outputs/stage42_long_research/method_draft_stage42.md` | `True` | 7181 |
+| `outputs/stage42_long_research/method_draft_stage42.md` | `True` | 8428 |
"""
    assert cy._md_change_kind(path, diff) == "metadata_and_paper_size_only"


def test_gate_fails_on_substantive_stage42_dirty_file() -> None:
    payload = {
        "dirty_rows": [
            {
                "scope": "stage42",
                "path": "outputs/stage42_long_research/x.json",
                "classification": "substantive_json_change",
                "allowed_for_stage42_paper_freeze": False,
            }
        ],
        "summary": {"tracked_dirty_files": 1, "stage42_substantive_dirty_files": 1},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cy._gate(payload)
    assert gate["passed"] < gate["total"]
    assert not gate["gates"]["stage42_no_substantive_dirty_changes"]


def test_gate_passes_for_metadata_and_outside_scope_caveats() -> None:
    payload = {
        "dirty_rows": [
            {
                "scope": "stage42",
                "path": "outputs/stage42_long_research/x.json",
                "classification": "metadata_only",
                "allowed_for_stage42_paper_freeze": True,
            },
            {
                "scope": "outside_stage42_scope",
                "path": "outputs/reports/old.json",
                "classification": "outside_scope_or_unclassified",
                "allowed_for_stage42_paper_freeze": True,
            },
        ],
        "summary": {"tracked_dirty_files": 2, "stage42_substantive_dirty_files": 0},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cy._gate(payload)
    assert gate["verdict"] == "stage42_cy_worktree_caveat_classifier_pass"
    assert gate["passed"] == gate["total"]
