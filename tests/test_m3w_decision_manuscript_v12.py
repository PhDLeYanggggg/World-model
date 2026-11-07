from zipfile import ZipFile

import pytest

from scripts.build_m3w_decision_manuscript_v12 import (
    citation_metric,
    package,
    replace_once,
    sha,
)


def test_require_unique_insertion_marker():
    assert replace_once('aXb', 'X', 'new') == 'anewXb'
    for tex in ('ab', 'aXXb'):
        with pytest.raises(ValueError, match='insertion marker'):
            replace_once(tex, 'X', 'new')


def test_signed_contrast_preserves_direction_and_interval():
    assert citation_metric({'point': -0.3, 'interval95': [-0.7, 0.1]}, 3) == (
        '$-0.300$ [$-0.700$, $+0.100$]'
    )


def test_anonymous_source_bundle_is_allowlisted_and_deterministic(tmp_path):
    names = {'ablations.tex', 'decision_bottlenecks.pdf', 'evidence_values.tex',
             'manuscript.tex', 'protocol.tex'}
    for name in names | {'private_cache.npz', 'cvpr.sty', 'source_bindings.json'}:
        (tmp_path/name).write_text(name)
    bindings = {name: sha(tmp_path/name) for name in names}
    first, second = tmp_path/'first.zip', tmp_path/'second.zip'
    package(tmp_path, first, bindings)
    package(tmp_path, second, bindings)
    assert sha(first) == sha(second)
    with ZipFile(first) as archive:
        assert set(archive.namelist()) == names | {'README.txt'}
    (tmp_path/'protocol.tex').write_text('modified')
    with pytest.raises(ValueError, match='modified source: protocol.tex'):
        package(tmp_path, second, bindings)
