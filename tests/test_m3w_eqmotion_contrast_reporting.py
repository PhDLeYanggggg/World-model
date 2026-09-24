"""Independent aggregate contrast checks; no dataset or checkpoints needed."""
import copy

import numpy as np
import pytest

from scripts.report_m3w_protected_eqmotion_controls import check_contrast


def example():
    sites = ['a','b','c','d']
    left = dict(expected_scenes=sites, by_scene={s:dict(gain_percent=i+2.) for i,s in enumerate(sites)})
    right = dict(expected_scenes=sites, by_scene={s:dict(gain_percent=i+1.) for i,s in enumerate(sites)})
    record = dict(unit='physical_scene', resamples=3000, seed=1,
                  scene_differences_pp=[1.,1.,1.,1.], mean_gain_difference_pp=1., ci95_pp=[1.,1.])
    return left,right,record


def test_paired_constant_difference():
    check_contrast(*example())


@pytest.mark.parametrize('field,value', [
    ('mean_gain_difference_pp',2.), ('ci95_pp',[0.,2.]),
    ('scene_differences_pp',[1.,1.,0.,2.]), ('unit','overlapping_windows'),
    ('resamples',20)])
def test_rejects_changed_contrast(field,value):
    left,right,report = example()
    report[field] = value
    with pytest.raises(AssertionError):
        check_contrast(left,right,report)


def test_rejects_scene_order_mismatch():
    left,right,report = example()
    right = copy.deepcopy(right)
    right['expected_scenes'] = list(reversed(right['expected_scenes']))
    with pytest.raises(AssertionError):
        check_contrast(left,right,report)


def test_rejects_nonfinite_site_gain():
    left,right,report = example()
    left['by_scene']['a']['gain_percent'] = np.nan
    with pytest.raises(AssertionError):
        check_contrast(left,right,report)
