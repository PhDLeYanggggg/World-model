import copy
import json

import pytest

from scripts.audit_m3w_dronecrowd_grouping import (
    LEGACY_COMPLETED_RUN, LEGACY_CONTROLLER_SHA, checked_resume_identity, json_stable,
)


def identities():
    saved = {"run_id": LEGACY_COMPLETED_RUN, "config_sha256": "config", "opencv": "4.14.0",
             "code_sha256": {"scripts/audit_m3w_dronecrowd_grouping.py": LEGACY_CONTROLLER_SHA,
                             "src/evaluation/m3w_dronecrowd_grouping.py": "algorithm"}}
    current = copy.deepcopy(saved)
    current["run_id"] = "new_controller_identity"
    current["code_sha256"]["scripts/audit_m3w_dronecrowd_grouping.py"] = "fixed_controller"
    return saved, current


def test_same_version_resume_needs_no_compatibility_exception():
    saved, _ = identities()
    assert checked_resume_identity(saved, saved, False) == (LEGACY_COMPLETED_RUN, False)


def test_known_completed_run_can_replay_without_recomputing_pairs():
    saved, current = identities()
    assert checked_resume_identity(saved, current, True) == (LEGACY_COMPLETED_RUN, True)


@pytest.mark.parametrize("change", ["config", "algorithm", "runtime", "other_run", "not_complete"])
def test_changed_scientific_identity_or_unfinished_controller_cannot_resume(change):
    saved, current = identities()
    if change == "config":
        current["config_sha256"] = "changed"
    elif change == "algorithm":
        current["code_sha256"]["src/evaluation/m3w_dronecrowd_grouping.py"] = "changed"
    elif change == "runtime":
        current["opencv"] = "other"
    elif change == "other_run":
        saved["run_id"] = "unreviewed-run"
    with pytest.raises(ValueError):
        checked_resume_identity(saved, current, change != "not_complete")


def test_edge_tuples_survive_report_replay():
    value = {"prior_positive_edges_preserved": [("a", "b"), ("b", "c")]}
    normalized = json_stable(value)
    assert normalized == json.loads(json.dumps(value))
