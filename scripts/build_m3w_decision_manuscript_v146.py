"""Bind the fixed-predicted-budget accuracy repair to the supplement."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v145 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE = parent.BASE
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v146'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    doc = parent.build(out, style, archive, 'supplement')
    src = PUBLIC.parent / 'moving_cv_fixed_budget_gain_v1'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json', 'capacity_readout.json')]
    reg, frozen, r, cap = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (frozen['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or not frozen['before_held_scoring'] or not reg['exact_count']
            or not r['hindsight_diagnostic_only'] or r['a_labels_opened']
            or r['independent_confirmation'] or not r['original_2pct']
            or cap['parent']['sha256'] != h(files[2]) or not cap['fractional_relaxation']):
        raise ValueError('Require frozen development evidence')
    arms = ('anchor', 'geometry', 'direct', 'factor', 'hindsight')
    if len({r['edits'][k] for k in arms}) != 1:
        raise ValueError('Mismatched edit counts')
    names = ('Anchor', 'Geometry', 'Direct', 'Factor', 'Hindsight')
    values = dict(Updates=f'{frozen["optimizer_steps"]:,}', Edits=f'{r["edits"]["anchor"]:,}',
                  Qualified=r['qualified'], Capacity=f'{cap["mean_upper_increment_ols4_pp"]:.4f}',
                  Queries=f'{cap["active_queries"]:,}')

    def ci(pair):
        v = r['summary']['effects'][pair]['completion_lower_pp']
        return f'{v["point"]:+.3f} [{v["interval95"][0]:+.3f}, {v["interval95"][1]:+.3f}]'

    for a, name in zip(arms, names, strict=True):
        k = r['summary']['risk'][a]
        values.update({name+'OLS': ci(f'{a}_vs_ols4'), name+'Safe': k['supported_views'],
                       name+'Easy': f'{k["worst_signed_easy_percent"]:+.2f}'})
        if a != 'anchor':
            values[name+'Change'] = ci(f'{a}_vs_anchor')
    fragment = PUBLIC / 'fixed_budget_gain.tex'
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\FixedGain{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): h(p) for p in (*files, fragment)})
    doc['files'].update({p.name: h(p) for p in (tex, defs)})
    (out / 'supplement_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', type=Path)
    parser.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    parser.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    args = parser.parse_args()
    out = DEFAULT / 'supplement'
    doc = build(out, args.style, args.archive)
    if args.source_zip:
        package_supplement(out, args.source_zip, doc)
