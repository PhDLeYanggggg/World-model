import pytest

from scripts import build_m3w_focused_candidate as candidate


def test_body_replacement_keeps_protected_text_byte_for_byte():
    start = '\\title{A title}\n\\begin{abstract}\nOriginal abstract.\n\\end{abstract}\n'
    end = candidate.BODY_END+'\nOriginal disclosure.\n'+candidate.BIB_START+'{99}\n'
    new = '\\section{Introduction}\nA focused argument.'
    original = start+candidate.BODY_START+'\nOld argument.\n'+end
    assert candidate.substitute_body(original, new) == start+new+'\n\n'+end


@pytest.mark.parametrize('body', [r'\input{other.tex}', r'\title{Changed}', r'\begin{abstract}'])
def test_body_rejects_replacements_outside_the_revision_scope(body):
    original = candidate.BODY_START+'\n'+candidate.BODY_END
    with pytest.raises(ValueError, match='outside the body revision scope'):
        candidate.substitute_body(original, '\\section{Introduction}\n'+body)


def test_verified_results_distinguish_caps_proposals_and_release():
    v = candidate.verified_values()
    assert (v['CompletionObserved'], v['CompletionFavorable']) == (36, 9)
    assert (v['SongdoJointReleased'], v['SongdoIndependentReleased'], v['SongdoIndependentRisk'],
            v['SongdoIndependentDual']) == (0, 15, 5, 4)
    assert float(v['WepCompletionLower']) < 0 < float(v['WepCompletionUpper'])
    assert float(v['SongdoLow']) < float(v['SongdoPoint']) < 0 < float(v['SongdoHigh'])
