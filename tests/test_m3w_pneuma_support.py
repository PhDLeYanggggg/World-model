from io import StringIO

import pytest

from src.data_unification.m3w_pneuma_support import HEADER, coverage, scan


def track(agent, *, first=0, last=39, holes=()):
    points = []
    for tick in range(first, last+1):
        if tick in holes:
            continue
        # The scanner must never decode positions past the prefix.
        lat = '37.98' if tick-first < 8 else 'opaque-future'
        points += [lat, '23.73', '0', '0', '0', f'{tick*.04:.6f}']
    return ';'.join([str(agent), 'Car', 'ignored', 'ignored', *points])


def test_future_availability_does_not_select_the_window():
    table = StringIO('\n'.join([';'.join(HEADER), track(1, holes=(8,)), track(2)]))
    row = coverage(scan(table), 1)
    assert row['past_selected_queries'] == 2
    assert row['selected_queries_all_agents_complete'] == 1
    assert row['selected_pair_steps'] == 24
    assert row['observed_pair_step_tokens'] == 23


def test_clock_gap_and_source_local_identity_are_checked():
    row = track(1, last=8).replace('0.320000', '0.321000')
    result = scan(StringIO('\n'.join([';'.join(HEADER), row])))
    assert result.diagnostics['off_grid'] == 1
    assert result.time[0].tolist() == list(range(8))
    with pytest.raises(ValueError, match='Duplicate'):
        scan(StringIO('\n'.join([';'.join(HEADER), row, row])))
