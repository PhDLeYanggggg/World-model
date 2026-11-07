from copy import deepcopy
from zipfile import ZipFile

import pytest

from scripts import build_m3w_decision_supplement_v20 as supplement


@pytest.fixture(scope='module')
def studies():
    return [supplement.main.verified_analysis(k, {}) for k in supplement.STUDIES]


def test_scalar_partition_and_causal_contrast_use_bound_results(studies):
    values = supplement.macros(*studies)
    assert r'\newcommand{\FeasibilityExcluded}{74}' in values
    assert r'\newcommand{\PrefixPassAgent}{3}' in values
    assert r'$-0.120\;[-0.391, +0.028]$' in values
    changed = list(studies)
    changed[1] = deepcopy(changed[1])
    changed[1]['feasible_witnesses'][0]['seed'] = 0
    with pytest.raises(ValueError, match='partition'):
        supplement.macros(*changed)


def test_source_package_has_exact_allowlist_and_rejects_edits(tmp_path):
    for name in supplement.PAYLOAD:
        (tmp_path/name).write_text(name)
    (tmp_path/'private_checkpoint.pt').write_text('not a review file')
    receipt = {'files': {k: supplement.base.sha(tmp_path/k) for k in supplement.PAYLOAD}}
    a, b = tmp_path/'first.zip', tmp_path/'second.zip'
    for out in (a, b):
        supplement.package(tmp_path, out, receipt)
    assert a.read_bytes() == b.read_bytes()
    with ZipFile(a) as z:
        assert set(z.namelist()) == {*supplement.PAYLOAD, 'README.txt'}
    (tmp_path/supplement.PAYLOAD[0]).write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        supplement.package(tmp_path, a, receipt)
