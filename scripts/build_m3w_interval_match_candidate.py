"""Add the equal-replacement-count comparison to an anonymous draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_external_completion_candidate as previous

builder = previous.builder
OUT = previous.OUT.parent/'evidence_manuscript_v191_equal_replacements'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_interval_matched_replacements_v1'
FIT = builder.main.PUBLIC/'opendd_vehicle_interval_angle_v1'


def inputs():
    paths = {'matched': STUDY/'readout.json', 'protocol': STUDY/'protocol.md',
             'fit': FIT/'readout.json', 'freeze': FIT/'freeze.json'}
    raw = {k: p.read_bytes() for k, p in paths.items()}
    hashes = {k: builder.main.digest(v) for k, v in raw.items()}
    m, f = (json.loads(raw[k]) for k in ('matched', 'fit'))
    if (m['freeze_sha256'] != hashes['freeze'] or m['scored_readout_sha256'] != hashes['fit']
            or m['protocol_sha256'] != hashes['protocol'] or f['freeze'] != hashes['freeze']
            or m['external_outcomes_read'] or m['deployment_changed'] or not m['original_2pct']
            or m['directions'] != f['directions'] or m['directions'] != 126):
        raise ValueError('Require unchanged development masks and outcome provenance')
    return (m, f), hashes


def revise(source, rows):
    m, f = rows
    marker = 'comparisons, not a release decision.'
    transition = 'The predicted screen reduces intervention without improving the'
    if source.count(marker) != 1 or source.count(transition) != 1:
        raise ValueError('Development text no longer matches')
    a = m['totals']['ego_count']
    c = f['counts']['interval']['ego_count']['calibrated_agreement']
    t = m['interval_minus_direct']['ego_count']['conservative_trajectory_pp']
    interval = rf'''

\paragraph{{Count-match the second replacement.}}
Interval supervision yields
{a['interval_original_qualified']} qualifying replacements from {a['interval_original_swaps']}
proposals, versus {a['direct_original_qualified']} from {a['direct_original_swaps']}
for the lower-bound target. At {a['matched_swaps']} replacements per arm,
these counts are {a['interval_qualified']} and {a['direct_qualified']};
a uniform subset of the interval proposals would yield
{a['interval_random_qualified']:.1f} in expectation.
The paired conservative trajectory difference is
\({t['point']*1e6:+.2f}\times10^{{-6}}\) pp
(95\% location bootstrap \([{t['ci95'][0]*1e6:+.2f}, {t['ci95'][1]*1e6:+.2f}]\times10^{{-6}}\)).
Calibration retains {c['swaps']} interval replacements with no increase
in strict successes. The count match uses the full development score
sequence and is not an online rule.'''
    revised = source.replace(marker, marker+interval, 1).replace(
        transition, 'On Moving development, the predicted screen reduces intervention without improving the', 1)
    return revised, ('Count-match the second replacement', 'in expectation')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
