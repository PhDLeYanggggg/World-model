import pytest

from scripts.summarize_m3w_past_motion_correspondence import paired_errors


def row(pair, error=None):
    return {'recording_id':'fit', 'agent_id':1, 'pair':pair, 'template_size':15,
            'status':'matched' if error is not None else 'out_of_image_support',
            'annotation_error_px':error}


def test_compares_same_support_and_does_not_count_missing_as_zero():
    a, b = [row(1,20),row(2,5)], [row(1),row(2,2)]
    r = paired_errors(a,b)
    assert r['jointly_supported'] == 1 and r['wider_missing'] == 1
    assert r['initial_mean_error_px'] == 5 and r['wider_mean_error_px'] == 2


def test_identity_change_or_duplicate_rejected():
    with pytest.raises(ValueError):
        paired_errors([row(1,1)], [row(2,1)])
    with pytest.raises(ValueError):
        paired_errors([row(1,1),row(1,2)], [row(1,1)])
