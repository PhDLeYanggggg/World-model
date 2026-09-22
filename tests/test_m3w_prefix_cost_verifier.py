import copy
import pytest
from scripts.verify_m3w_prefix_cost_readout import verify_contrasts
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast


def example():
    sites = ['a', 'b', 'c', 'd']
    def summary(gains):
        return {'ADE': {'by_scene': {s: {'gain_percent': g} for s, g in zip(sites, gains)}}}
    a, b = [1., 3., 2., 4.], [1., 1., 1., 1.]
    report = dict(summaries=dict(profile_guard=summary(a), control_terminal=summary(b)),
                  contrasts=dict(control_terminal=paired_scene_contrast(a, b), scalar_log_strict=paired_scene_contrast(a, b)))
    return report, {'summaries': {'strict_stop': summary(b)}}, sites


def test_real_nested_summary_contract():
    verify_contrasts(*example())


def test_changed_interval_rejected():
    report, scalar, sites = example()
    broken = copy.deepcopy(report)
    broken['contrasts']['control_terminal']['ci95_pp'][0] += 1
    with pytest.raises(AssertionError):
        verify_contrasts(broken, scalar, sites)
