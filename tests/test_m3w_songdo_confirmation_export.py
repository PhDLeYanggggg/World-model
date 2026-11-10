from scripts.export_m3w_songdo_confirmation import metric, section


def example():
    return dict(point=None, ci95=None, status='required_component_not_estimable',
                defined_seed_views=6, required_seed_views=9,
                by_component={'private-id-a': 2., 'private-id-b': None, 'private-id-c': 3.},
                worst_estimable_source='private-id-c', worst_estimable_source_value=3.)


def test_missing_components_and_unnamed_order_are_retained():
    public = metric(example(), groups=True)
    assert public['component_points'] == [2., None, 3.]
    assert public['point'] is None and public['estimable_components'] == 2
    assert public['worst_estimable_source_value'] == 3.
    assert 'private-id' not in str(public)


def test_release_and_matched_risk_counts_are_not_interchanged():
    r = dict(metrics={'available_ade_native': example()}, worst_flight_seed_available_ade_native=7.,
             risk={'edited': 0}, original_risk={'edited': 9}, vs_cv={'lower_pp': example()})
    assert section(r, matched=True)['risk']['edited'] == 9
    assert section(r, matched=False)['risk']['edited'] == 0
    assert section(r, matched=False)['vs_cv']['lower_pp']['point'] is None
