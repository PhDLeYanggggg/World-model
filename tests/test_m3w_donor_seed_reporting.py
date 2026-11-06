import pytest

from scripts.report_m3w_moving_donor_seeds import verdict


def analysis():
    return dict(folds=[{}]*72, contexts=[{}]*9, risk_head_seeds=[17,29,43], forecaster_seeds=[17],
        new_outer_C_H_readout=False, identity_certification=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False,
        replication_screen_pass=True)


def test_component_replication_is_not_deployment():
    assert verdict(analysis()) == 'three_risk_seed_donor_information_replicated_not_deployable'


@pytest.mark.parametrize('key', ['independent_confirmation','deployment_changed','new_outer_C_H_readout'])
def test_no_scope_promotion(key):
    a = analysis()
    a[key] = True
    with pytest.raises(ValueError):
        verdict(a)


def test_failed_replication_is_explicit():
    a = analysis()
    a['replication_screen_pass'] = False
    assert verdict(a) == 'risk_seed_replication_failed_not_deployable'
