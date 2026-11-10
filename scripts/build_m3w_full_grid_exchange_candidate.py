"""Add the full-grid OpenDD headroom result to v210."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_full_grid_candidate as prior
from reproducibility.m3w_reader_supplement_exchange.verify import check_exchange

OUT = prior.OUT.with_name('evidence_manuscript_v210_exchange_grid')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
READOUT = PUBLIC.parent/'opendd_full_grid_exchange_v1/readout.json'


def inputs():
    raw = READOUT.read_bytes()
    d = json.loads(raw)
    if (not d['original_2pct'] or not d['no_training'] or d['deployment_changed']
            or d['new_external_outcome_access'] or d['independent_confirmation']):
        raise ValueError('Expected the exposed OpenDD audit without training or deployment')
    check_exchange(d['summary'])
    return d, dict(opendd_exchange=prior.builder.main.digest(raw))


def revise(text, d):
    risk = 'create a positive selected denominator or a risk pass.'
    result = 'frozen scores on dual passes in these development fits.'
    if text.count(risk) != 1 or text.count(result) != 1:
        raise ValueError('Expected the v209 risk and development paragraphs')
    row = check_exchange(d['summary'])
    proof = ('\nWe also check risk on the full grid under free missing-position\n'
             'completions. The worst all-edited ratio places missing targets\n'
             'at CV. For the selected-easy ratio, sum positive harm minus\n'
             'the risk limit times CV error over mandatory easy tracks and\n'
             'take only positive excesses from optionally easy tracks. A\n'
             'complete easy edit guarantees reference support. The supplement\n'
             'derives these checks; they do not replace the release rule.')
    w = d['summary']['full_grid_contrasts']['aligned_minus_uniform']
    lo, hi = w['ci95_pp']
    add = (f"\nWith full-grid risk, {row['menu_or_keep'][1]:.0f}/606 views still have a passing\n"
           f"plan and {row['mask_oracle'][1]:.0f}/606 do when forced to switch on the learned mask.\n"
           f"The aligned ranker obtains {row['aligned'][1]:.0f}/606, versus {row['uniform'][1]:.2f} expected for\n"
           f"uniform exchange on that mask ({w['point_pp']:+.2f} pp, 95\\% location interval\n"
           f"[{lo:+.2f}, {hi:+.2f}]). Missing labels do not explain most of the gap.")
    text = text.replace(risk, risk+proof, 1).replace(result, result+add, 1)
    return text, ('175/606', '98/606', '44/606', '46.73')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    args = p.parse_args()
    prior.builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                        read_inputs=inputs, revise_source=revise)
