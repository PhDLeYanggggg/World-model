"""Clarify the missing-label capacity result in a separate anonymous draft."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_candidate as parent

scope = parent.scope
DESTINATION = parent.DESTINATION.parent/'evidence_manuscript_v171_completion'
RECEIPT = scope.main.PUBLIC/'evidence_manuscript_v171_completion/receipt.json'
EVIDENCE = scope.main.PUBLIC/'moving_completion_envelope_v1/receipt.json'
OLD = '''A better shortlist, an optimal solver, and an
apparently feasible predicted margin are distinct from actual
recording-level error control. The current evidence does not
justify replacing the baseline or a claim of metric geometry,
physical safety, or a transferable world model.'''


def revise(text, *, result):
    r = result['completion_envelope']['record']
    n = result['views']
    counts = tuple(r[k]['excluded_views'] for k in (
        'unfavorable_completion', 'observed_steps_only', 'favorable_completion'))
    if (n != 72 or not 0 <= counts[2] <= counts[1] <= counts[0] == 67
            or result['independent_confirmation'] or result['external_outcomes_read'] or result['risk_evaluated']):
        raise ValueError('Expected the verified development-only completion analysis')
    a, b, c = (f'{x}/{n}' for x in counts)
    new = f'''On Moving development, the displacement budget excludes positive
gain over OLS4 under the conservative score in {a} views, on
observed steps alone in {b}, and even under favorable missing
labels in {c}. These ceilings allow arbitrary paths and pool the
budget within a recording. The {a} result is not an exclusion
for every completion. None of these bounds certifies risk or
justifies replacing the baseline.'''
    if text.count(OLD) != 1 or new in text:
        raise ValueError('Expected the unchanged v170 limitations paragraph')
    return text.replace(OLD, new)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', type=Path, default=DESTINATION)
    p.add_argument('--receipt', type=Path, default=RECEIPT)
    args = p.parse_args()
    original = scope.supplement.read_receipt('evidence_manuscript_v170_songdo')
    evidence = json.loads(EVIDENCE.read_text())
    scope.build(args.output, args.compiler.resolve(), args.receipt, original=parent.DESTINATION,
                expected=original['artifacts'], revise_source=partial(revise, result=evidence), changed_paragraphs=1,
                prior_citation_count=original['bibliography_entries'],
                markers=tuple(f"{r['excluded_views']}/{evidence['views']}" for r in (
                    evidence['completion_envelope']['record'][k] for k in (
                        'unfavorable_completion', 'observed_steps_only', 'favorable_completion'))))
