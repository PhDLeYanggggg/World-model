from copy import deepcopy

import pytest

from scripts import report_m3w_train_census as report


def fixture():
    row = dict(identity=dict(group='g',source='s',seed=1),mode='budget_scaled',
        registration_sha256='reg',checkpoint_sha256='cp',optimizer_steps=0,
        independent_roles_read=False,deployment_changed=False,
        sampling_coverage=dict(sampler_exact=True),
        cached_development_policy={'value':1},cached_development_diagnostics={'value':2},
        census=dict(inference_replay_exact=True,inference_outcome_filter=False,
            new_training=False,independent_roles_read=False,full_TRAIN_rows=11,
            full=dict(known_rows=10,unknown_rows=1,selected_known_harm_mass=1.,
                selected_known_reference_mass=100.,known_all_risk=.01,
                selected_known_easy_harm_mass=.1,selected_known_easy_reference_mass=10.,
                known_easy_risk=.01)))
    key = ('g','s',1,'budget_scaled')
    docs = {key:dict(checkpoint=dict(sha256='cp'))}
    dev = {key[:3]:dict(result=dict(policies={'budget_scaled':{'value':1}},
                                   diagnostics={'budget_scaled':{'value':2}}))}
    return row,key,docs,dev


def test_receipt_verifies_mass_ratios_and_cached_comparison():
    row,key,docs,dev = fixture()
    report.check_rows([row],{key},'reg',docs,dev)


@pytest.mark.parametrize('mutation',['missing','duplicate','risk','future_filter','cached'])
def test_changed_evidence_fails(mutation):
    row,key,docs,dev = fixture()
    rows = [deepcopy(row)]
    if mutation == 'missing': rows = []
    elif mutation == 'duplicate': rows *= 2
    elif mutation == 'risk': rows[0]['census']['full']['known_easy_risk'] = 0
    elif mutation == 'future_filter': rows[0]['census']['inference_outcome_filter'] = True
    else: rows[0]['cached_development_policy'] = {'value':0}
    with pytest.raises(ValueError):
        report.check_rows(rows,{key},'reg',docs,dev)
