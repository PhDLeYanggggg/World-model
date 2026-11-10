"""Add the crossed recipient/path comparison to the anonymous draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_attribution_candidate as parent

builder = parent.builder
OUT = parent.OUT.parent/'evidence_manuscript_v193_songdo_fixed_paths'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'songdo_fixed_recipient_paths_v1'


def inputs():
    raw = {k: (STUDY/f'{k}.json').read_bytes() for k in ('readout', 'freeze')}
    hashes = {k: builder.main.digest(v) for k, v in raw.items()}
    hashes['protocol'] = builder.main.digest((STUDY/'protocol.md').read_bytes())
    d, f = (json.loads(raw[k]) for k in ('readout', 'freeze'))
    if (d['proposal_freeze_sha256'] != hashes['freeze'] or f['binding'] != d['binding']
            or d['binding']['protocol'] != hashes['protocol'] or not f['before_attribution_scoring']
            or not d['original_diagonal_reproduced'] or d['new_cohort_opened']
            or d['calibration_outcomes_opened'] or d['training'] or d['deployment_changed']):
        raise ValueError('Require unchanged frozen Songdo controls')
    return d, hashes


def revise(source, d):
    marker = "This partitions the frozen forecasts' error, not a new release test."
    if source.count(marker) != 1:
        raise ValueError('The prior Songdo text changed')

    def entry(key):
        r = d['effects'][key]['lower_pp']
        a, b = r['ci95']
        return rf"\({r['point']*1e3:+.3f}\) [{a*1e3:+.3f}, {b*1e3:+.3f}]"

    extra = rf'''

\paragraph{{Hold the vehicles fixed.}}
As a retrospective control, each rule chooses paths for both original
recipient sets. The two edits use the same legal path bank and radius.
Joint minus independent path selection gives {entry('path_on_independent')}
on independent recipients and {entry('path_on_joint')} on joint recipients
(\(10^{{-3}}\) pp; 95\% source bootstrap intervals).
The path rules differ in {d['differing_paths']['path_on_independent']}
and {d['differing_paths']['path_on_joint']} of 480 views, respectively.
The observed-pair relative-motion gain is positive on independent
recipients, but the conservative relative-motion contrast is
negative on both recipient sets.
Both original proposals replay exactly. These crossed proposals do not
fit or change the original release policy.'''
    return source.replace(marker, marker+extra, 1), ('344 and 70 of 480 views', 'crossed proposals')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=parent.OUT, read_inputs=inputs, revise_source=revise)
