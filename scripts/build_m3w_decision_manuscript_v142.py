"""Bind risk-budgeted menu expansion and frozen-scorer transfer results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v141 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v142'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        return doc
    src = PUBLIC.parent / 'moving_cv_risk_frontier_v1'
    fragment = PUBLIC / 'risk_frontier.tex'
    files = [src / k for k in ('registration.json', 'readout.json', 'causal_registration.json',
        'causal_freeze.json', 'causal_readout.json', 'integer_readout.json')]
    a, r, b, f, c, integer = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (r['registration'] != h(files[0]) or b['parents'][0]['sha256'] != h(files[0])
            or f['registration'] != h(files[2]) or c['freeze'] != h(files[3])
            or integer['parent']['sha256'] != h(files[1]) or not integer['labels_hindsight_only']
            or not f['before_held_scoring'] or not r['labels_hindsight_only']
            or not a['original_2pct'] or not b['original_2pct']
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct']
                   for z in (r, c, integer))):
        raise ValueError('Require frozen development-only frontier results')
    risk = c['summary']['risk']
    if (risk['expanded']['switches'] != risk['independent_matched']['switches']
            or risk['original']['switches'] != risk['expanded_matched']['switches']):
        raise ValueError('Frontier transfer lost count matching')
    modes = ('original', 'expanded', 'full_rows')
    values = dict(Options=f'{r["stats"]["original"]:,}/{r["stats"]["expanded"]:,}',
        Verified='/'.join(str(integer['summary'][k].get('verified', 0)) for k in modes),
        Unresolved=integer['summary']['full_rows'].get('unresolved', 0),
        New=c['new_choices'], Qual=c['qualified'],
        Passes='/'.join(str(risk[k]['supported_views']) for k in modes[:2]),
        Worst=f'{risk["expanded"]["worst_signed_easy_percent"]:.2f}')

    def ci(v):
        p, lo, hi = (f'{x:+.3f}' for x in (v['point'], *v['interval95']))
        return f'{p} [{lo}, {hi}]'

    values['Coverage'] = ci(integer['effects']['expanded_minus_original'])
    values['Count'] = ci(c['summary']['effects']['expansion_at_count']['completion_lower_pp'])
    values['Joint'] = ci(c['summary']['effects']['joint_at_count']['completion_lower_pp'])
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\RiskFrontier{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): h(p) for p in (*files, fragment)})
    doc['files'].update({p.name: h(p) for p in (tex, defs)})
    (out / 'supplement_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT / ('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
