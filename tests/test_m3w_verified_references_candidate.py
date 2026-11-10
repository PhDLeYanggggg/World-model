import pytest

from scripts import build_m3w_verified_references_candidate as candidate


def test_revision_preserves_unrelated_manuscript_sections_and_checks_anchors():
    before = 'Protected abstract and text\n'
    after = '\nProtected disclosure\n'
    old = before+'\n'.join(a for a, _ in candidate.EDITS)+'\n'+candidate.BIB_END+after
    new = candidate.revise(old)
    assert new.startswith(before) and new.endswith(after)
    assert new.count(r'\bibitem{geotrax}') == 1
    assert new.count(r'\cite{songdo,geotrax}') == 1
    assert '2601.12500v3' in new
    with pytest.raises(ValueError, match='audited citation anchor'):
        candidate.revise(old+candidate.EDITS[0][0])
