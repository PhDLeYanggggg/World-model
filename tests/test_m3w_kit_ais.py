import numpy as np
import pytest

from scripts.audit_m3w_kit_ais import parse, prefix, summarize_timing


def xml(*, second='00.5', duplicate=False):
    obj = '<object id="5"><box xc="8" yc="4" w="2" h="2"/><detection_score value="0.8"/></object>'
    row = lambda i, t, objects: (f'<frame number="{i}" file="C:/test/{i}.png" '
                                  f'utc="2020-Jan-01 11:00:{t}"><objectlist>{objects}</objectlist></frame>')
    return (f'<dataset name="dummy">{row(0, "00.0", obj)}{row(1, second, obj*int(1+duplicate))}</dataset>').encode()


def test_xml_frame_clock_and_box_geometry_are_preserved():
    frames, det = parse(xml(), 'dummy', detector=True)
    label_clock, gt = parse(xml(), 'dummy', detector=False)
    assert frames == label_clock
    assert frames[0][1] == '0.png'
    np.testing.assert_array_equal(det[0], [[7., 3., 9., 5., .8, 1.]])
    np.testing.assert_array_equal(gt[0], [[5., 7., 3., 9., 5.]])
    assert summarize_timing(frames) == dict(min=.5, median=.5, max=.5, gaps_over_1p5_median=0,
                                            run_lengths_between_large_gaps=[2])


def test_duplicate_id_and_nonmonotonic_clock_are_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        parse(xml(duplicate=True), 'dummy', detector=False)
    with pytest.raises(ValueError, match='monotonically'):
        parse(xml(second='00.0'), 'dummy', detector=True)


def test_prefix_excludes_future_detections_and_filters_by_fixed_score():
    det = np.array([[7., 3., 9., 5., .8, 1.], [20., 10., 22., 12., .4, 1.]])
    rows = [det]*8 + [np.zeros((1, 6))]
    a = prefix(rows, .6)
    assert len(a) == 8
    assert all(p.tolist() == [[0., 7., 3., 9., 5.]] for p in a)
