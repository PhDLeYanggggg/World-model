"""Add the one-shot Songdo finding to a separate anonymous manuscript."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_related_work_candidate as parent


scope = parent.scope
DESTINATION = parent.DESTINATION.parent/'evidence_manuscript_v170_songdo'
RECEIPT = scope.main.PUBLIC/'evidence_manuscript_v170_songdo/receipt.json'
EVIDENCE = scope.main.PUBLIC/'songdo_calibrated_20261110'
OLD = r'''At fixed query counts, both path banks yield verified hindsight plans in \OLSMenuExpanded{} of \OLSMenuViews{} views. Even with risk removed and edits reallocated within recordings, OLS4 remains out of reach in \OLSMenuStillNegative{} of \OLSMenuExamined{} excluded views (supplement). Training a causal velocity spline yields no conservative whole-roster gain over OLS4 for the calibration-selected arm (supplement). On OpenDD, only \StreamSplineDoubleQueries{} of \StreamSplineActiveQueries{} edited queries allow within-query coupling; matched calibration does not improve risk (supplement). Independent confirmation remains sealed.'''
REFERENCE = r'''\bibitem{songdo}
R. Fonod, H. Cho, H. Yeo, and N. Geroliminis. Songdo Traffic v2.
\emph{Zenodo:17924857}, 2025.

'''


def paragraph(score):
    c = score['matched_joint_minus']['independent']['lower_pp']
    joint = score['released']['joint']['risk']
    independent = score['released']['independent']['risk']
    if (not score['no_fitting_on_confirmation'] or not score['original_2pct_unchanged']
            or [score[k] for k in ('calibration_flights', 'flights', 'source_components', 'matched_seed_views')]
            != [40, 160, 14, 480] or joint['edited'] != 0 or independent['edited'] <= 0
            or c['estimable_components'] != 14 or not c['ci95'][0] < 0 < c['ci95'][1]):
        raise ValueError('Expected the complete frozen Songdo finding')
    return (r'''On Songdo Traffic~\cite{songdo}, thresholds fit on 40 flights were
frozen before scoring 160 flights in 14 disjoint source components.
With two equal-radius edits, the lower joint-minus-independent gain is
'''+f"${c['point']:+.6f}$ pp (95\\% source bootstrap $[{c['ci95'][0]:+.6f},{c['ci95'][1]:+.6f}]$).\n"
            +f"Joint released 0/480 views; independent passed risk on {independent['original_risk_pass']}/{independent['edited']} releases.\n"
            +'''
DroneCrowd confirmation stays sealed.''')


def revise(text, score):
    end = parent.BIB_END
    if text.count(OLD) != 1 or text.count(end) != 1 or r'\bibitem{songdo}' in text:
        raise ValueError('Expected the unchanged prior candidate')
    new = paragraph(score)
    changed = text.replace(OLD, new).replace(end, REFERENCE+end)
    if changed.replace(new, OLD).replace(REFERENCE+end, end) != text:
        raise ValueError('Unrelated manuscript change')
    return changed


def verified_evidence():
    raw = (EVIDENCE/'calibration_receipt.json').read_bytes()
    freeze, digest = json.loads(raw), scope.main.digest(raw)
    result = json.loads((EVIDENCE/'confirmation_receipt.json').read_text())
    if (freeze['confirmation_outcomes_opened'] or result['binding']['published_calibration_receipt_sha256'] != digest
            or result['binding']['release_freeze_sha256'] != freeze['freeze_sha256']):
        raise ValueError('Songdo result does not bind the pre-outcome calibration')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', type=Path, default=DESTINATION)
    p.add_argument('--receipt', type=Path, default=RECEIPT)
    a = p.parse_args()
    original = scope.supplement.read_receipt('evidence_manuscript_v169_related_work')
    score = verified_evidence()
    scope.build(a.output, a.compiler.resolve(), a.receipt, original=parent.DESTINATION,
                expected=original['artifacts'], revise_source=partial(revise, score=score), changed_paragraphs=1,
                added_citations=('songdo',), prior_citation_count=original['bibliography_entries'],
                markers=('0/480 views', '17924857'))
