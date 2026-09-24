import io

import numpy as np
import pytest

from src.evaluation.m3w_european_squares_intake import COLUMNS, read_raw_csv as v1
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv


def source(with_name=True):
    fields = list(COLUMNS)
    values = [1, 2, 3, 4, 0, .8, 7, 'person', 10]
    if not with_name:
        fields.pop(7)
        values.pop(7)
    return ','.join(fields)+'\n'+','.join(map(str, values))+'\n'


def test_optional_text_does_not_change_numeric_rows():
    rows, names = read_raw_csv(io.StringIO(source(False)))
    expected, _ = v1(io.StringIO(source(True)))
    np.testing.assert_array_equal(rows, expected)
    assert names == {'not_provided': 1}
    assert rows['class_id'].tolist() == [0]


def test_full_schema_exactly_matches_frozen_v1():
    actual, names = read_raw_csv(io.StringIO(source(True)))
    expected, old_names = v1(io.StringIO(source(True)))
    np.testing.assert_array_equal(actual, expected)
    assert names == old_names


@pytest.mark.parametrize('field', ['confidence', 'tracker_id', 'frame_index'])
def test_no_other_column_is_optional(field):
    text = source(False).replace(field, 'unrecognized_field')
    with pytest.raises(ValueError, match='schema'):
        read_raw_csv(io.StringIO(text))


def test_duplicate_without_text_still_rejected():
    text = source(False)
    text += text.splitlines()[1]+'\n'
    with pytest.raises(ValueError, match='Duplicate'):
        read_raw_csv(io.StringIO(text))


def test_no_filtering_short_track_in_optional_schema():
    rows, _ = read_raw_csv(io.StringIO(source(False)), chunksize=1)
    assert len(rows) == 1
