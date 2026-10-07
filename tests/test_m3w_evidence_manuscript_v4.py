import copy

import pytest

from scripts import build_m3w_evidence_manuscript_v4 as api


@pytest.fixture(scope='module')
def sources():
    return api.load_sources()


def test_completed_scientific_evidence_not_fresh_export(sources):
    e=api.build(*sources)
    assert e['trained_temporal_cost_heads']==216 and e['completed_readout_views']==72
    assert e['assembly_source']=='fresh_run' and e['numerical_results_source']=='cached_verified'
    assert not e['new_training'] and not e['new_inference'] and not e['new_bootstrap']
    assert not e['submission_ready'] and not e['temporal_advance']
    assert e['train_sum_identity_assertions']==432
    assert len(e['temporal_comparisons'])==6


@pytest.mark.parametrize('change',['promote','missing_view','wrong_weight','selected_imputation','new_train','wrong_risk'])
def test_reject_evidence_role_and_support_drift(sources,change):
    old,docs=copy.deepcopy(sources)
    if change=='promote':docs['temporal_readout']['advance_to_transfer_design']=True
    elif change=='missing_view':docs['readout_complete']['groups'].pop()
    elif change=='wrong_weight':docs['temporal_readout']['contrasts']['rowmean_all_MSE']['mean']+=.1
    elif change=='selected_imputation':docs['temporal_readout']['contrasts']['none_original_selected_MSE']['mean']=0
    elif change=='new_train':docs['train_diagnostic']['new_optimizer_updates']=1
    else:docs['train_evidence']['budget_is_net_easy_ADE_degradation']=True
    with pytest.raises(ValueError):api.build(old,docs)


def test_negative_comparators_and_unknowns_preserved(sources):
    e=api.build(*sources);rows={r['comparator']:r for r in e['temporal_comparisons']}
    assert rows['rowmean']['all_MSE']['ci_high']<0
    for other in ('rowmean','none'):
        assert rows[other]['full_paired_lower_percent']['ci_high']<0
        assert rows[other]['matched_paired_lower_percent']['ci_high']<0
    assert e['temporal_safety']['temporal']['unknown_selected_occurrences']==1191
    assert e['temporal_safety']['temporal']['easy_upper_violations']==72
    assert [r['violations'] for r in e['train_diagnostic']]==[58,57,52]


def test_manuscript_assembles_all_six_tables(sources):
    template=(api.ROOT/api.OUTPUT/'manuscript.template.md').read_text()
    result=api.artifacts(*sources,template)
    assert '{{' not in result['manuscript.md']
    assert 'TRAIN resubstitution' in result['manuscript.md']
    assert '216 fits' in result['manuscript.md'] and '0.353836' in result['manuscript.md']
    assert 'not an anonymous or submission-ready manuscript' in result['manuscript.md']
    assert 'stage5c_executed' in result['evidence.json']
