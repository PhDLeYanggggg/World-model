import pytest

from scripts.report_m3w_moving_donor_augmentation import conclusion


def analysis():
    return dict(new_outer_C_H_readout=False, identity_certification=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False,
        folds=[{}]*24, contexts=[{}]*3, passing_contexts=3)


def test_passed_information_screen_is_not_deployment():
    assert conclusion(analysis()) == 'donor_augmentation_information_supported_not_deployable'


@pytest.mark.parametrize('key', ['new_outer_C_H_readout', 'identity_certification', 'deployment_changed', 'submission_ready'])
def test_overclaim_rejected(key):
    a = analysis()
    a[key] = True
    with pytest.raises(ValueError):
        conclusion(a)
