from scripts.audit_m3w_dronecrowd_cross_archive import quarantine, select_views


def test_views_only_use_existing_frames():
    assert select_views([13, 1, 20, 4, 7]) == [1, 7, 20]


def test_cross_archive_match_quarantines_the_entire_external_component():
    groups = {'a':'g1', 'b':'g1', 'c':'g2', 'd':'g3'}
    assert quarantine([{'external':'b'}, {'external':'d'}], groups) == (['g1','g3'], ['a','b','d'])
    assert quarantine([], groups) == ([], [])
