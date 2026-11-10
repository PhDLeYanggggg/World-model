"""Add the cross-dataset full-grid readouts to the anonymous manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_direction_candidate as builder
from reproducibility.m3w_reader_supplement_cross_grid.verify import check_wep

PARENT = builder.OUT.with_name('evidence_manuscript_v208_shared_seed')
OUT = PARENT.with_name('evidence_manuscript_v209_cross_grid')
PUBLIC = builder.main.PUBLIC/OUT.name
WEPCHECK = PUBLIC.parent/'wepdtof_full_grid_risk_v1/readout.json'
SONGCHECK = PUBLIC.parent/'songdo_full_grid_easy_risk_v1/readout.json'


def inputs():
    raw = {k: p.read_bytes() for k, p in dict(wepdtof=WEPCHECK, songdo=SONGCHECK).items()}
    d = {k: json.loads(v) for k, v in raw.items()}
    check_wep(d['wepdtof'])
    if (not d['songdo']['original_2pct_unchanged'] or d['songdo']['new_external_cohort_opened']
            or d['songdo']['fit'] or d['songdo']['deployment_changed']):
        raise ValueError('Expected the original Songdo risk sensitivity')
    return d, {k: builder.main.digest(v) for k, v in raw.items()}


def revise(text, d):
    wep = 'Refitting the OLS4 intercept from detections does not reliably help.'
    song = 'The result does not support\ndeploying the joint rule.'
    if text.count(wep) != 1 or text.count(song) != 1:
        raise ValueError('Expected the frozen v208 external paragraphs')
    w = d['wepdtof']['arms']['fixed_joint']
    s = d['songdo']['results']
    wr = (f"Of {w['edited_views']} edited joint views, {w['full_risk_pass']} pass risk for every free "
          f"completion of the twelve steps. Even the favorable accuracy bound\n"
          f"excludes beating both references in {w['full_pass_utility_excluded']} of those views.")
    sr = ('Allowing missing steps to change easy membership still gives a\n'
          f"guaranteed easy reference in {s['joint']['matched']['full_grid_easy_support']}/480 joint views "
          f"versus {s['independent']['matched']['full_grid_easy_support']}/480 independent views.\n"
          'The supplement gives the full-grid risk calculation; it does not change release.')
    text = text.replace(wep, wr, 1).replace(song, song+'\n'+sr, 1)
    return text, ('144/480', '229/480', 'favorable accuracy bound')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path)
    args = parser.parse_args()
    builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=PARENT,
                  read_inputs=inputs, revise_source=revise)
