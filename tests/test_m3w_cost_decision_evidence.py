from scripts.report_m3w_cost_decision_evidence import policy_rows, render


def test_empty_policy_is_exported_as_undefined_not_safe():
    safety = dict(budget_record_views=0, required_record_views=2, budget_views=0,
        required_views=1, undefined_easy_record_views=2, selected=0, selected_unknown=0)
    data = dict(safety={'physical_neural': safety, 'floor': safety},
        accuracy={'physical_neural': {'point': 0., 'CI95': [0., 0.]}},
        unique_source_recordings=2)
    rows = policy_rows('study', {'scopes': {'fit': data, 'inner_held': data}})
    assert len(rows) == 2
    assert all(not r['full_record_roster_supported'] for r in rows)
    assert all(r['undefined_easy_record_head_views'] == 2 for r in rows)


def test_render_keeps_development_and_completion_bound_limitations():
    text = render({'policies': [], 'training': {}})
    assert 'known-label ADE' in text
    assert 'not independent' in text
    assert 'zero' in text and 'not a passed selected-risk gate' in text
    assert 'not a new neural trajectory lift' in text
