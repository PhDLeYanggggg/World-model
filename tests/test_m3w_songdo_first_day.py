import pytest
import numpy as np

from scripts import audit_m3w_songdo_first_day as audit
from scripts import build_m3w_songdo_first_day_vehicle_bank as bank
from scripts import fetch_m3w_songdo_first_day as fetch


def files():
    for s in 'ABCEFGHIJKLMNOPQRSTU':
        name = f'2022-10-04_{s}.zip'
        yield {'key': name, 'size': 25, 'checksum': 'md5:'+'c'*32,
               'links': {'self': f'{fetch.PUBLISHER}/files/{name}/content'}}


def test_publisher_inventory_excludes_other_dates_and_rejects_off_source_urls():
    rows = list(files())
    rows.append({'key': '2022-10-05_A.zip'})
    assert len(fetch.inventory(rows)) == 20
    rows[0]['links']['self'] = 'https://unrelated.test/file'
    with pytest.raises(ValueError, match='Unexpected publisher file metadata'):
        fetch.inventory(rows)


def test_sources_are_merged_transitively_before_any_role_assignment():
    assert audit.merge('ABCDE', [('A', 'B'), ('B', 'C')]) == [['A', 'B', 'C'], ['D'], ['E']]
    names = audit.expected_sessions('A')
    assert len(names) == 10
    assert '2022-10-04_A_AM1.csv' in names and '2022-10-04_A_PM5.csv' in names


def test_vehicle_bank_has_only_displacements_and_handles_empty_sources():
    net = bank.MODEL.motion.network(17).eval()
    h = np.arange(48, dtype=float).reshape(3, 8, 2)
    a = bank.forecasts(h, {17: [net]})
    b = bank.forecasts(h+150., {17: [net]})
    assert set(a) == {'cv', 'ols4', 'median4', 'self_17'}
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-9)
    assert all(p.shape == (0, 12, 2) for p in bank.forecasts(h[:0], {17: [net]}).values())


def test_near_prefix_screen_requires_spatial_and_temporal_overlap():
    xy = np.column_stack((np.arange(8), np.zeros(8)))[None].astype(float)
    t = (np.arange(8)*400)[None]
    assert audit.links.compare(xy, t, xy+0.2, t)['possible_duplicate_prefix_pairs'] == 1
    assert audit.links.compare(xy, t, xy+20., t)['possible_duplicate_prefix_pairs'] == 0
    assert audit.links.compare(xy, t, xy, t+10000)['possible_duplicate_prefix_pairs'] == 0
