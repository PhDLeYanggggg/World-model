import copy

import pytest

from scripts import verify_m3w_recording_stability as audit
from src.world_model import m3w_recording_stability as api
from tests.test_m3w_selected_set_readout import fixture


def example():
    z, g, old, cfg = fixture()
    groups = [api.audit_head(z, old, g, cfg)]
    return groups, api.summarize(groups, cfg), cfg


def test_independent_summary_and_partition_reduction():
    g, s, c = example()
    assert audit.check_groups(g, s, c) > 80


@pytest.mark.parametrize('field', ['bootstrap', 'mass', 'retained', 'safety'])
def test_corrupt_accounting_rejected(field):
    g, s, c = example()
    if field == 'bootstrap':
        s['arms']['raw_pool']['utility_change_vs_parent']['CI95'][0] = 1.
    elif field == 'mass':
        g[0]['arms']['raw_pool']['removed']['selected_known_harm_mass'] += 1.
    elif field == 'retained':
        g[0]['arms']['raw_pool']['folds'][0]['retained'] += 1
    else:
        s['arms']['raw_pool']['consensus']['complete_support_pass'] += 1
    with pytest.raises(AssertionError):
        audit.check_groups(g, s, c)
