"""Add the frozen Songdo recipient attribution to the anonymous draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_interval_match_candidate as parent

builder = parent.builder
OUT = parent.OUT.parent/'evidence_manuscript_v192_songdo_attribution'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'songdo_recipient_attribution_v1'
ORIGINAL = builder.main.PUBLIC/'songdo_calibrated_20261110/confirmation_receipt.json'


def inputs():
    files = {'attribution': STUDY/'readout.json', 'protocol': STUDY/'protocol.md', 'confirmation': ORIGINAL}
    hashes = {k: builder.main.digest(p.read_bytes()) for k, p in files.items()}
    d, old = (json.loads(files[k].read_bytes()) for k in ('attribution', 'confirmation'))
    if (d['binding']['protocol_sha256'] != hashes['protocol'] or not d['primary_scores_reproduced']
            or d['new_cohort_opened'] or not d['no_fit'] or not d['original_2pct']
            or d['release_decision_changed'] or d['flights'] != 160 or d['flight_seed_views'] != 480
            or any(d['binding'][k] != old['binding'][k] for k in (
                'registration_sha256', 'primary_score_sha256', 'release_freeze_sha256', 'flight_score_hash_root'))):
        raise ValueError('Require retrospective attribution of the unchanged Songdo experiment')
    return d, hashes


def revise(source, d):
    marker = 'deploying the joint rule.'
    if source.count(marker) != 1:
        raise ValueError('Expected exactly one unchanged Songdo conclusion')
    u, f = (d['controls'][a] for a in ('independent', 'same_family'))

    def effect(row, part):
        return row[part]['lower_pp']['point']*1000.

    paragraph = rf'''

\paragraph{{Which Songdo recipients account for the difference?}}
For joint versus independent, recipients edited by only one arm
contribute \({effect(u, 'allocation'):+.3f}\times10^{{-3}}\) pp to the
conservative contrast; shared recipients contribute
\({effect(u, 'shared'):+.3f}\times10^{{-3}}\) pp.
Against the same-family control, the respective contributions are
\({effect(f, 'allocation'):+.3f}\times10^{{-3}}\) and
\({effect(f, 'shared'):+.3f}\times10^{{-3}}\) pp.
The small advantage over the restricted control is accounted for by
which vehicles were edited, not by improved paths for shared recipients.
This partitions the frozen forecasts' error, not a new release test.'''
    return source.replace(marker, marker+paragraph, 1), ('Which Songdo recipients', 'shared recipients')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=parent.OUT, read_inputs=inputs, revise_source=revise)
