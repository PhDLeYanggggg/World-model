"""Compile a separate candidate with the corrected observed-geometry control."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_direction_candidate as previous

OUT = previous.OUT.parent/'evidence_manuscript_v181_vehicle_context'
RECEIPT = previous.main.PUBLIC/OUT.name/'receipt.json'
STUDY = previous.main.PUBLIC/'opendd_vehicle_direction_context_v1'
ANCHOR = 'improves the forecast from the past.'


def inputs():
    read = (STUDY/'readout.json').read_bytes()
    digest = previous.main.digest
    d = json.loads(read)
    files = {n: digest((STUDY/f'{n}.json').read_bytes()) for n in ('registration', 'freeze')}
    if (any(d[k] != v for k, v in files.items()) or d['external_outcomes_read']
            or d['independent_confirmation'] or not d['original_2pct']
            or (d['fit_models'], d['updates'], d['unique_recordings'], d['views']) != (126, 25200, 493, 8874)):
        raise ValueError('Expected the complete frozen vehicle context comparison')
    return d, dict(context=digest(read), **files)


def revise(source, data):
    if source.count(ANCHOR) != 1:
        raise ValueError('Expected exactly one vehicle paragraph')
    c = data['counts']
    spacing = c['spaced_proxy']['dual']
    plain = c['centered_proxy']['dual']
    context, own = c['neighbor_model']['dual'], c['own_model']['dual']
    if c['neighbor_model']['rotated_views'] != c['own_at_neighbor']['rotated_views']:
        raise ValueError('Mismatched number of same-recipient rotations')
    a = data['available_ade']['own_minus_neighbor']
    gain = data['paired']['neighbor_model_minus_own_model']['lower_pp']
    same = data['paired']['neighbor_model_minus_own_at_neighbor']['lower_pp']
    cv, own_ade, peer_ade = (data['available_ade'][k]['point'] for k in ('cv', 'own', 'neighbor'))
    text = rf'''
The proxy's initial distance weights used actor-centered positions.
Restoring observed spacing changes the number of double-reference
and risk passes from {plain:,} to {spacing:,}. In a separate matched
ablation, two equally sized learned dynamics models are fit on
the same five donor locations per held/calibration pair.
Available-step ADE is {cv:.3f} for CV and {own_ade:.3f}/{peer_ade:.3f}
for the two models in native units. The paired ADE difference
(neighbor-free minus neighbor-aware) is \({a['point']:+.4f}\).
After using the forecasts for angle selection, the neighbor-aware
and neighbor-free rules give {context:,} and {own:,} passes,
respectively. Their paired conservative trajectory gain is
\({gain['point']:+.4f}\) pp (95\% location interval
[{gain['ci95'][0]:+.4f}, {gain['ci95'][1]:+.4f}]).
At the exact same {c['neighbor_model']['rotated_views']:,} rotating
actors the neighbor rule still loses \({abs(same['point']):.4f}\) pp
against ranking angles by the neighbor-free forecast's individual gain.
'''
    return source.replace(ANCHOR, ANCHOR+text, 1), ('Restoring observed spacing', f'{spacing:,}', f'{context:,}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    previous.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
