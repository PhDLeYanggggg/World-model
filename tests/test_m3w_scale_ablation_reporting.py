from scripts.report_m3w_moving_scale_ablation import summary


def test_secondary_success_cannot_replace_negative_main():
    contrasts = {a+'_minus_control': {k: {'mean': v} for k, v in
        zip(('known_pp', 'completion_lower_pp', 'completion_upper_pp'), values)}
        for a, values in [('continued_mse', (-1, -2, 0)), ('signed_cost_logistic', (3, 2, 4))]}
    r = summary(dict(contrasts=contrasts, expansion_supported=False,
        objective_expansion_screens={'continued_mse': False, 'signed_cost_logistic': True}))
    assert r['main_arm'] == 'continued_mse'
    assert not r['expansion_supported']
    assert not r['deployment_changed']
    assert not r['submission_ready']
    assert not r['completion_bounds_are_sampling_CI']
    assert not r['full_policy_coordinate_invariance']
    assert r['contrasts']['continued_mse']['known_pp'] == -1
