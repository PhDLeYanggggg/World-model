import copy
import numpy as np
import pytest
from scripts import verify_m3w_selection_conditioned_analogue as v
from tests.test_m3w_pooled_analogue_verifier import fixture as old_fixture


def fixture():
    rows, _, expected = old_fixture()
    y = np.zeros((3, 5)); y[:, 2:4] = 1; y[0, [1, 4]] = .1; y[2] = np.nan
    out = np.full((3, 2), .02); p = np.full((3, 2), .5); c = np.full((3, 2), .04)
    d = v.run.api.decomposition(y, out, p, c, np.ones(3), 1., np.ones(3, bool))
    for r in rows:
        for mode in v.run.api.MODES: r['scores']['selected_'+mode] = copy.deepcopy(r['scores']['pooled_'+mode])
        ref = r['references'][0]
        ref.update(query_sha256='q', mean_sha256='m', std_sha256='s', unselected_donors=0,
                   donor_events=[16, 16], donor_fraction_sums=[1., 1.])
        r['full_references'] = copy.deepcopy(r['references'])
        r['decomposition'] = {'selected': copy.deepcopy(d), 'pooled': copy.deepcopy(d)}
    return rows, v.run.api.summarize(rows, expected), expected


def test_independent_arithmetic_and_fixed_normalizer():
    result = v.verify(*fixture())
    assert result['legacy_arm_target_checks'] == 1440
    assert result['decomposition_checks'] == 288
    assert not result['magnitude_screen_pass']


@pytest.mark.parametrize('mutation', ['normalizer', 'donor', 'loss', 'population', 'screen', 'drop'])
def test_tampering_is_rejected(mutation):
    rows, summary, expected = fixture()
    if mutation == 'normalizer': rows[0]['references'][0]['mean_sha256'] = 'new'
    if mutation == 'donor': rows[0]['references'][0]['unselected_donors'] = 1
    if mutation == 'loss': rows[0]['decomposition']['selected']['easy']['harm_SSE'] += 1
    if mutation == 'population': rows[0]['decomposition']['selected']['easy']['known'] += 1
    if mutation == 'screen': summary['magnitude_screen_pass'] = True
    if mutation == 'drop': rows.pop()
    with pytest.raises(ValueError): v.verify(rows, summary, expected)
