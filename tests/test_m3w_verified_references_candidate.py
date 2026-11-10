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


def test_decision_ablation_revision_uses_frozen_counts_and_places_figure_early():
    figure = '\\begin{figure*}[t]\n\\label{fig:capacity-discrimination}\n\\end{figure*}\n\n'
    section = '\\subsection{Available gains need not be found from the past}'
    anchor = 'Reordering existing scores recovers little of the verified opportunity.'
    data, _ = candidate.decision_ablation_inputs()
    result, markers = candidate.revise_decision_ablations(f'First\n{section}\n{anchor}\n\n{figure}Last\n', data)
    assert result.startswith(f'First\n{figure}{section}') and result.endswith('Last\n')
    assert {'42/606', '54/606', '59/606', '55/606', '58/606'} <= set(markers)
    data[2]['arms']['conditional_utility']['edits'] += 1
    with pytest.raises(ValueError, match='same frozen comparison'):
        candidate.revise_decision_ablations(f'{section}\n{anchor}\n\n{figure}', data)


def test_exchange_frontier_keeps_witness_and_upper_counts_distinct():
    evidence = json.loads(candidate.FRONTIER_EVIDENCE.read_bytes())
    text = 'First\nsequence, not an online policy.\nLearning does not close this gap: Last'
    out, markers = candidate.revise_exchange_frontier(text, evidence)
    assert out.startswith('First\n') and out.endswith('Last')
    assert {'178/606', '109/606'} <= set(markers)
    assert 'at most one hindsight' in out and '68--69/606' in out
    evidence['external_outcome_access'] = True
    with pytest.raises(ValueError, match='matched-budget development frontier'):
        candidate.revise_exchange_frontier(text, evidence)
