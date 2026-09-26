import copy
import json
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


def test_complete_aggregate_is_strict_json_serializable():
    cfg = dict(repeats=8, probe_metrics=['cost4', 'easy_harm_all', 'easy_harm_positive'],
               bootstrap_seed=14, bootstrap_resamples=3000)
    geometry = dict(cosine=.2, norm_ratio=.3)
    gradient = dict(losses=dict(known_events=20), **{m: geometry for m in
        ('cost4_shared', 'cost4_full', 'easy_harm_positive_shared', 'easy_harm_positive_full')})
    rows = []
    for producer in range(3):
        for controller in range(3):
            if producer == controller: continue
            for seed in (17, 29, 43):
                for outer in 'abcd':
                    localities = [s for s in 'abcd' if s != outer]
                    repeat = dict(gradients=dict(true=gradient, shuffled=gradient), after={
                        name: dict(localities={s: {m: value for m in cfg['probe_metrics']} for s in localities})
                        for name, value in [('cost_only', 1.), ('true_aux', 1.1), ('shuffled_aux', 1.2),
                                            ('projected_true', .9), ('projected_shuffled', 1.05)]})
                    rows.append(dict(producer=producer, controller=controller, pair='full', seed=seed,
                        states=[dict(arm='cap_aux', repeats=[repeat]*8)], initial=dict(shared=dict(cosine=None))))
    result = aggregate(rows, cfg)
    assert result['separate_projection_training_warranted']
    assert json.loads(json.dumps(result, allow_nan=False))['repair_screen'] == result['repair_screen']
