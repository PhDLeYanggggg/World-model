import json
from unittest.mock import patch

import numpy as np
import pytest

from src.world_model import m3w_independent_risk_calibration as api
from src.world_model.m3w_multiaction_cost import select_actions


def roles(role='risk_calibration_reserved'):
    return dict(status='restricted_source_training_admitted_reserved_roles_closed',
                recordings=[dict(source_member='r.csv', role=role, training_access=False,
                                 predictive_readout_access=False, rows_sha256='x')])


def test_role_access_is_narrow():
    assert api.require_calibration_role(roles(), 'r.csv')['role'] == 'risk_calibration_reserved'
    for role in ('source_training', 'model_selection_reserved', 'confirmation_reserved'):
        with pytest.raises(PermissionError):
            api.require_calibration_role(roles(role), 'r.csv')
    with pytest.raises(ValueError):
        api.require_calibration_role(roles(), 'missing')


def test_coefficients_preserve_utility_and_match_native_independent():
    rng = np.random.default_rng(81)
    p = rng.uniform(.01, 5, (31, 7, 5)); p[..., 1] *= .01; p[..., 4] *= .003
    allowed = rng.random((31, 7)) > .2; original = p.copy()
    for m in api.MULTIPLIERS:
        q = api.coefficient_view(p, allowed, m)
        np.testing.assert_array_equal(q[..., :2], p[..., :2])
        np.testing.assert_array_equal(q[..., 2:4], p[..., 2:4] / m)
        a, _ = select_actions(q, allowed, arm='physical_neural')
        b, _ = select_actions(p, allowed, arm='physical_neural', harm_multiplier=m)
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(p, original)
    np.testing.assert_array_equal(api.coefficient_view(p, allowed, 1), p)
    with pytest.raises(ValueError):
        api.coefficient_view(p, allowed, 3)


def test_label_injection_rejected_before_any_array_load(tmp_path):
    receipt = dict(role='risk_calibration_inputs_only', source_member='r.csv', rows_sha256='x',
                   registration_sha256='registered', arrays={'future_xy': {}})
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    with patch.object(np, 'load', side_effect=AssertionError('must not load labels')):
        with pytest.raises(ValueError, match='input-only'):
            api.load_record(tmp_path, roles(), 'r.csv', 'registered')


def test_all_record_views_both_policies_required():
    curves = [dict(multiplier=m, policies={a: dict(required_record_views=18,
              supported_record_views=18 if m == 16 else 17) for a in ('independent', 'joint_matched')})
              for m in api.MULTIPLIERS]
    assert api.select_multiplier(curves, 18)['selected_multiplier'] == 16
    curves[4]['policies']['joint_matched']['supported_record_views'] = 0
    result = api.select_multiplier(curves, 18)
    assert result['selected_multiplier'] is None
    assert not result['calibration_supported']
    assert not result['empty_support_is_safe']
    with pytest.raises(ValueError):
        api.select_multiplier(curves[:-1], 18)


def test_original_accessors_remain_closed():
    from src.evaluation.m3w_european_squares_roles import require_source_training
    from src.data_unification.m3w_model_selection_inputs import require_input_role
    with pytest.raises(PermissionError):
        require_source_training(roles(), 'r.csv')
    with pytest.raises(PermissionError):
        require_input_role(roles(), 'r.csv')


def test_calibrated_joint_choice_matches_small_exhaustive_problem():
    import itertools
    p = np.zeros((2, 7, 5)); allowed = np.zeros((2, 7), bool)
    allowed[:, :2] = True
    p[:, 0] = [2., .001, 1., 1., .001]
    p[:, 1] = [3., .006, 1., 1., .006]
    q = api.coefficient_view(p, allowed, 4)
    parent, _ = select_actions(q, allowed, arm='physical_neural')
    out, _ = api.allocation.solve_frame(q, allowed, parent, exact_count=2, seconds=5.)
    scores = [api.allocation.terms(q, np.array(c))[0] for c in itertools.product(range(2), repeat=2)
              if api.allocation.feasible(q, allowed, np.array(c))]
    assert api.allocation.terms(q, out)[0] == max(scores)


def test_past_cache_roundtrip_has_no_label_interface(tmp_path):
    from tests.test_m3w_observation_quality import source
    from src.data_unification.m3w_model_selection_inputs import build_inputs
    from src.data_unification.m3w_selection_feature_bridge import pack_record
    raw = source(); inputs = build_inputs(raw, [84]); features = pack_record(inputs, raw)
    values = dict(**{'input_'+k: v for k, v in inputs.items()},
                  **{'feature_'+k: v for k, v in features.items()}, index_query_frames=np.array([84]))
    refs = {}
    import hashlib
    for key, value in values.items():
        path = tmp_path/(key+'.npy'); np.save(path, value, allow_pickle=False)
        refs[key] = dict(name=path.name, shape=list(value.shape), dtype=str(value.dtype),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path/'receipt.json').write_text(json.dumps(dict(role='risk_calibration_inputs_only',
        source_member='r.csv', rows_sha256='x', registration_sha256='registered', arrays=refs)))
    restored, bridge, queries = api.load_record(tmp_path, roles(), 'r.csv', 'registered')
    for k, v in inputs.items():
        np.testing.assert_array_equal(v, restored[k])
    for k, v in features.items():
        np.testing.assert_array_equal(v, bridge[k])
    assert queries.tolist() == [84]
