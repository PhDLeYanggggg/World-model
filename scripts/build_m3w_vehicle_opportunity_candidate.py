"""Consolidate the vehicle experiments into an availability/nomination/selection comparison."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_action_value_candidate as previous

builder = previous.builder
OUT = previous.OUT.parent/'evidence_manuscript_v183_recording_screen'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_opportunity_v1'
START = r'\paragraph{Coordinating direction without changing recipients.}'
END = r'\begin{table}[t]'+'\n'+r'\centering\small'+'\n'+r'\begin{tabular}{@{}lrr@{}}'


def inputs():
    digest = builder.main.digest
    hashes = {k: digest((STUDY/f'{k}.json').read_bytes()) for k in ('registration', 'freeze', 'readout')}
    readout = json.loads((STUDY/'readout.json').read_bytes())
    if (any(readout[k] != hashes[k] for k in ('registration', 'freeze')) or readout['external_outcomes_read']
            or readout['independent_confirmation'] or not readout['original_2pct']
            or (readout['fits'], readout['recordings'], readout['totals']['views']) != (63, 493, 8874)):
        raise ValueError('Expected frozen recording-screen evaluation')
    return readout, hashes


def revise(source, d):
    if source.count(START) != 1 or source.count(END) != 1 or source.index(START) > source.index(END):
        raise ValueError('Expected one vehicle results block')
    n = d['totals']['nominee_strict']
    a, b = (d['counts'][k]['ego_count'] for k in ('margin', 'opportunity'))
    if a['count'] != b['count']:
        raise ValueError('Recording comparison lost the matched budget')
    diff = d['paired']['opportunity_ego_count']['lower_pp']
    c = d['counts']['opportunity']['calibrated']
    text = rf'''
\paragraph{{An available improvement need not be selectable.}}
On 493 OpenDD vehicle recordings (8,874 repeated views), we fix the
two recipients and their displacement from CV at every step. A
single rotation can pass risk and both reference checks, preserve
trajectory utility and improve relative motion in
{d['totals']['exists']:,} views. A learned angle nominator supplies
such an action in only {n:,} views even if we ignore its abstention
score. Better average trajectories do not solve the nomination
problem: a learned surrogate reduced available-step ADE from
6.142 to about 4.99 native units, but gave fewer qualifying angle
changes than preserving the original paths. Direct gain-and-harm
training beats a fixed angle on the same recipients
(+0.00239 pp), but not the ego-only control.

\paragraph{{Recording screen with the action held fixed.}}
We train a separate classifier on whether the legal angle menu
contains a qualifying action. It only changes \emph{{where}} to use
the frozen ego nominee. At the original {a['count']:,}-rotation
budget, action confidence and the existence screen realize
{a['strict']:,} and {b['strict']:,} strict successes, respectively.
The paired conservative trajectory difference is
\({diff['point']:+.6f}\) pp (95\% location interval
[{diff['ci95'][0]:+.6f}, {diff['ci95'][1]:+.6f}]).
The separate calibration rule releases {c['count']:,} rotations
for the existence screen; {c['positive_incremental']:,} meet
the held risk and positive gain checks. These are development
comparisons with identical proposals, not external confirmation.

'''
    before, rest = source.split(START, 1)
    _, after = rest.split(END, 1)
    return before+text+END+after, ('Recording screen', 'strict successes')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
