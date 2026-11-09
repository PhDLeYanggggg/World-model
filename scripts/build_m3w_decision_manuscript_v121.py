"""Build the supplement with the matched cost-target comparison."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v120 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v121'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive):
    report = PUBLIC.parent / 'moving_motion_choice_cost_v1'
    score, reg, freeze = (report / f'{n}.json' for n in ('readout', 'registration', 'freeze'))
    a = json.loads(score.read_text())
    digest = BASE.cal.digest
    if (a['registration'] != digest(reg) or a['choice_freeze'] != digest(freeze)
            or a['independent_confirmation'] or not a['original_2pct']):
        raise ValueError('Require bound development-only cost comparison')
    doc = parent.build(out, style, archive)
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    frag = PUBLIC / 'motion_choice_cost.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor))
    e = a['summary']['effects']
    macros = {}
    for k, r in (('PooledRank', e['pooled_vs_rank']), ('LeadPooled', e['lead_vs_pooled']),
                 ('PooledOLS', e['pooled_cost_vs_ols4'])):
        q = r['known_pp']
        l, u = q['interval95']
        macros[k] = (f'{q["point"]:+.3f} & [{l:+.3f}, {u:+.3f}] & '
                     f'[{r["completion_lower_pp"]["point"]:+.3f}, {r["completion_upper_pp"]["point"]:+.3f}]')
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\ChoiceCost{k}}}{{{v}}}' for k, v in macros.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): digest(p) for p in (score, reg, freeze, frag)})
    doc['files'].update({p.name: digest(p) for p in (tex, defs)})
    (out / 'supplement_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT / 'supplement')
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        PACK.supplement_package.package(a.output_dir, a.source_zip, doc)
