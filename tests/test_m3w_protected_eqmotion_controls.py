import copy
import numpy as np
import pytest

from src.evaluation.m3w_protected_eqmotion_controls import check_reused_head, verify_same_population


def fixture():
    pr = dict(mean=np.zeros(2), std=np.ones(2), known=np.array([True, False]),
        weights=np.array([1., 0]), constant=np.array([.5, .2]), cost_scale=2., training_sites=['a', 'b'])
    cp = dict(identity=dict(inputs_sha256='x', labels_sha256='y'), settings=dict(steps=3, batch_size=2),
        seed=17, step=3, arm='bounded_fraction', preprocess=copy.deepcopy(pr), draws=np.array([6, 0]))
    receipt = dict(identity=copy.deepcopy(cp['identity']), fit=dict(complete=True))
    args = dict(settings=cp['settings'].copy(), seed=17, input_hash='x', label_hash='y', pr=pr,
                reference_draws=np.array([6, 0]), held='c')
    return cp, receipt, args


def test_exact_reused_head_accepted():
    cp, receipt, args = fixture()
    check_reused_head(cp, receipt, **args)


@pytest.mark.parametrize('bad', ['exposure', 'step', 'draws', 'scale', 'labels', 'preprocess', 'settings'])
def test_changed_or_exposed_head_rejected(bad):
    cp, receipt, args = fixture()
    if bad == 'exposure': cp['preprocess']['training_sites'].append('c')
    elif bad == 'step': cp['step'] = 2
    elif bad == 'draws': cp['draws'] = np.array([5, 1])
    elif bad == 'scale': cp['preprocess']['cost_scale'] = 3
    elif bad == 'labels': args['label_hash'] = 'wrong'
    elif bad == 'preprocess': cp['preprocess']['mean'][0] = 9
    elif bad == 'settings': cp['settings']['batch_size'] = 3
    with pytest.raises((ValueError, AssertionError)):
        check_reused_head(cp, receipt, **args)


def test_population_alignment_and_no_future_arrays():
    data = dict(geometry=np.zeros((2, 476)), scale=np.ones(2), sites=np.array(['a', 'b']),
                recordings=np.array(['a/0', 'b/0']))
    verify_same_population(data, copy.deepcopy(data))
    changed = copy.deepcopy(data); changed['recordings'] = changed['recordings'][::-1]
    with pytest.raises(AssertionError): verify_same_population(data, changed)
    changed = copy.deepcopy(data); changed['target'] = np.zeros((2, 12, 2))
    with pytest.raises(ValueError): verify_same_population(data, changed)
