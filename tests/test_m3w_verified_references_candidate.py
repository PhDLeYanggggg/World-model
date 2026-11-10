import json

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


def test_movement_revision_keeps_only_offline_scope():
    before = 'Protected abstract and text\n'
    after = '\nProtected disclosure\n'
    source = (before+r'\paragraph{Fixing the number of edits actually spent.}'+'\n'
              'does not establish an improvement by a past-only policy.'+after)
    evidence = json.loads(candidate.MOTION_EVIDENCE.read_bytes())
    result, markers = candidate.revise_motion(source, evidence)
    assert result.startswith(before) and result.endswith(after)
    assert {'60/606', '210/606', '242/606'} <= set(markers)
    assert 'not an online policy' in result
    evidence['confirmation_access'] = True
    with pytest.raises(ValueError, match='matched movement'):
        candidate.revise_motion(source, evidence)
