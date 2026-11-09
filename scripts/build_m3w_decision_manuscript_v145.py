"""Bind the coherent risk moments and matched-coverage controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v144 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v145'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        return doc
    src = PUBLIC.parent / 'moving_cv_risk_components_v1'
    fragment = PUBLIC / 'risk_components.tex'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json',
        'block_swap_readout.json', 'low_count_readout.json')]
    reg, f, r, swap, low = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or not f['before_held_scoring'] or not reg['frozen_gain'] or not low['exact_query_match']
            or any(x['parent']['sha256'] != h(files[2]) for x in (swap, low))
            or any(x['a_labels_opened'] or x['independent_confirmation'] or not x['original_2pct']
                   for x in (r, swap, low))):
        raise ValueError('Require frozen development-only risk-component evidence')
    risk, lower = r['summary']['risk'], low['summary']['risk']
    keys = ('direct', 'component', 'joint')
    if len({lower[k]['switches'] for k in keys}) != 1:
        raise ValueError('Low coverage counts differ')
    values = dict(Updates=f'{f["optimizer_steps"]:,}', Edits=f'{risk["direct_matched"]["switches"]:,}',
        LowEdits=f'{lower["joint"]["switches"]:,}', Qual=r['qualified'],
        Passes='/'.join(str(risk[k]['supported_views']) for k in ('direct_matched', 'component_matched')),
        LowPasses='/'.join(str(lower[k]['supported_views']) for k in keys))

    def ci(v):
        p, lo, hi = (f'{x:+.3f}' for x in (v['point'], *v['interval95']))
        return f'{p} [{lo}, {hi}]'

    values['High'] = ci(r['summary']['effects']['component_at_count']['completion_lower_pp'])
    values['HarmSwap'] = ci(swap['summary']['effects']['harm_only_vs_direct']['completion_lower_pp'])
    for name, k in (('Low', 'components_at_low_count'), ('Joint', 'joint_at_low_count')):
        values[name] = ci(low['summary']['effects'][k]['completion_lower_pp'])
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\RiskParts{k}}}{{{v}}}' for k, v in values.items()) + '\n')
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
