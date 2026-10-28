import copy

import pytest

from scripts import build_m3w_evidence_manuscript_v5 as api


@pytest.fixture(scope='module')
def sources():
    return api.load_sources()


def test_completed_loss_repair_retains_negative_primary_and_risk(sources):
    result = api.artifacts(*sources)
    paper = result['manuscript.md']
    assert 'scientific readout is pending' not in paper
    assert '-0.902716 [-1.775623, -0.060409]' in paper
    assert '-0.381278 [-0.994827, +0.126845]' in paper
    assert '71/72' in paper and '59/72' in paper
    assert '0.048558' in paper and '54 historical-exact' in paper
    assert paper.index('## LLM Usage Statement') < paper.index('## References')
    assert 'OpenAI ChatGPT and Codex' in paper
    assert 'Preparation note' not in paper
    assert 'not an anonymous or submission-ready manuscript' in paper
    assert '../evidence_manuscript_v4/figures/temporal' in paper


@pytest.mark.parametrize('change', ['promotion', 'risk', 'support', 'identity'])
def test_no_promotion_or_suppressed_diagnosis(sources, change):
    old, docs = copy.deepcopy(sources)
    if change == 'promotion': docs['summary']['advance_to_transfer_design'] = True
    elif change == 'risk': docs['diagnosis']['budget'] = .05
    elif change == 'support': docs['summary']['groups'] = 71
    else: docs['diagnosis']['identity_checks'] = 431
    with pytest.raises(ValueError): api.artifacts(old, docs)


def test_export_roles_and_pinned_sources(sources):
    result = api.artifacts(*sources)
    assert '"new_inference": false' in result['evidence.json']
    assert '"independent_confirmation": false' in result['evidence.json']
    assert '"numerical_results_source": "cached_verified"' in result['evidence.json']
    assert 'quadratic_matched_easy_deviance' in result['tables.md']
