import copy
import json

import pytest

from scripts.plot_m3w_selected_pool_source import build, load_source_rows, run


@pytest.fixture(scope='module')
def source():
    return load_source_rows()


def test_all_views_and_missingness_retained(source):
    rows,cfg=source; e=build(rows,cfg)
    assert len(e['points'])==72
    assert e['risk_pairs_defined']==30 and e['risk_pairs_undefined']==42
    assert e['bias_interval']['all_views']['mean'] is None
    assert e['bias_interval']['defined_only_descriptive']['CI95'][0]<0<e['bias_interval']['defined_only_descriptive']['CI95'][1]
    assert not e['transfer_outcomes_read'] and not e['new_training'] and not e['policy_changed']


def test_counterexamples_are_dependent_and_rejected(source):
    e=build(*source); cases=[r for r in e['points'] if r['new_known_violation']]
    assert len(cases)==2 and e['case_localities']==['eu-locality-074']
    assert e['case_source_screen_pass']==0
    assert {r['head_seed'] for r in cases}=={29,43}
    assert all(r['raw_risk']<.02<r['kept_risk'] and r['predicted_kept_risk']<.02 for r in cases)
    assert all(r['harm_retained']>r['reference_retained'] and r['kept_known']==2 for r in cases)


def test_missing_or_duplicate_view_fails(source):
    rows,cfg=source
    selected=[r for r in rows if r['role']=='source_oof' and r['mode']=='joint']
    with pytest.raises(AssertionError): build(selected[:-1],cfg)
    bad=copy.deepcopy(selected);bad[-1]=bad[0]
    with pytest.raises(AssertionError): build(bad,cfg)


def test_saved_figure_data_matches_all_source_values(source):
    expected=build(*source)
    saved=json.loads((run.PUBLIC/'source_figure_data.json').read_text())
    assert all(saved[k]==v for k,v in expected.items())
    assert saved['source_manifest_sha256']==run.digest(run.PUBLIC/'source_manifest.json')
    assert saved['source_replay_sha256']==run.digest(run.PUBLIC/'source_replay.json')
    assert saved['script_sha256']==run.digest(run.ROOT/'scripts/plot_m3w_selected_pool_source.py')


def test_svg_and_light_metrics_match_remote_receipt():
    receipt=json.loads((run.PUBLIC/'source_figure_receipt.json').read_text())
    for name in ('source_mechanism.svg','source_figure_data.json'):
        path=run.PUBLIC/name
        assert run.digest(path)==receipt['artifacts'][name]['sha256']
        assert path.stat().st_size==receipt['artifacts'][name]['bytes']
    assert not receipt['transfer_outcomes_read'] and not receipt['scientific_design_changed']
