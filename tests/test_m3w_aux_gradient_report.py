import copy
import pytest
from scripts.run_m3w_european_aux_gradient import validate_display_amendment
from scripts.report_m3w_european_aux_gradient import aggregate


def test_integer_assignment_ids_are_reportable():
    cfg = dict(repeats=8, probe_metrics=['cost4'])
    row = dict(producer=0, controller=1, pair='full', seed=17, states=[], initial=dict(shared=dict(cosine=None)))
    doc = aggregate([row], cfg)
    assert doc['initial_shared_not_estimable'] == 1
    assert not doc['separate_projection_training_warranted']


def amendment_fixture():
    paths = ['scripts/report_m3w_european_aux_gradient.py', 'scripts/run_m3w_european_aux_gradient.py']
    locked = dict(bindings={p: 'old' for p in paths}, parent='frozen')
    current = dict(bindings={p: 'new' for p in paths}, parent='frozen')
    amendment = dict(scientific_protocol_changed=False, changes={p: dict(before='old', after='new') for p in paths})
    return locked, current, amendment


def test_display_amendment_requires_exact_before_after_and_parent():
    a, b, m = amendment_fixture(); validate_display_amendment(a, b, m)
    changed = copy.deepcopy(b); changed['parent'] = 'changed'
    with pytest.raises(AssertionError): validate_display_amendment(a, changed, m)
    changed = copy.deepcopy(m); changed['changes']['configs/anything.json'] = dict(before='old', after='new')
    with pytest.raises(AssertionError): validate_display_amendment(a, b, changed)
    changed = copy.deepcopy(m); changed['scientific_protocol_changed'] = True
    with pytest.raises(AssertionError): validate_display_amendment(a, b, changed)
