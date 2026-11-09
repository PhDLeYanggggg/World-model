from scripts.run_m3w_av2_selected_projection_risk import uniform_choice


def test_uniform_assignment_key_has_no_outcome_dependency_and_is_stable():
    keys = [f"query_{i}" for i in range(100)]
    first = [uniform_choice(k, 17, 6) for k in keys]
    assert first == [uniform_choice(k, 17, 6) for k in keys]
    assert set(first) == set(range(6))
    assert first != [uniform_choice(k, 29, 6) for k in keys]
    assert uniform_choice("query", 17, 1) == 0
