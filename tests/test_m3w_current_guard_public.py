import copy
import json

import pytest

from scripts import report_m3w_current_guard_support as report


def test_complete_diagnostic_public_reconstruction():
    d = json.loads((report.PUBLIC / 'guard_support_diagnostic.json').read_text())
    s = report.summarize(d)
    assert s['groups'] == 72 and s['repeated_record_views'] == 444
    assert s['pointwise_blocked_views'] == 12
    assert s['native_joint_singleton_proof_views'] == 6
    assert s['all_views_totals']['true_signed_risk_safe_known_easy_rejected'] > 0
    assert s['all_views_totals']['known_signed_risk_unsafe_admitted_action_occurrences'] > 0
    assert not s['inference_used_future_labels'] and s['diagnostic_used_future_labels']
    assert not s['deployment_changed'] and not s['independent_confirmation']


def test_incomplete_or_duplicated_original_views_rejected():
    d = json.loads((report.PUBLIC / 'guard_support_diagnostic.json').read_text())
    missing = copy.deepcopy(d)
    missing['recordings'].pop()
    with pytest.raises(ValueError):
        report.summarize(missing)
    duplicate = copy.deepcopy(d)
    duplicate['recordings'][-1] = duplicate['recordings'][0]
    with pytest.raises(ValueError):
        report.summarize(duplicate)


def test_pointwise_block_is_not_automatically_joint_impossibility():
    d = json.loads((report.PUBLIC / 'guard_support_diagnostic.json').read_text())
    for row in d['recordings']:
        if (row['frozen_pointwise_guard_cannot_supply_known_easy_reference']
                and not row['frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof']):
            row['frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof'] = True
            break
    d['native_joint_singleton_proof_views'] += 1
    d['blocked_views'] = [r for r in d['recordings'] if r['frozen_pointwise_guard_cannot_supply_known_easy_reference']]
    with pytest.raises(ValueError, match='multi-target'):
        report.summarize(d)
