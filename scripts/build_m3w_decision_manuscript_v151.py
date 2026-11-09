"""Bind the streaming calibration and joint-opportunity results to the paper."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v150 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v151'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    src = PUBLIC.parent/'opendd_stream_spline_calibration_v1'
    files = [src/n for n in ('registration.json', 'freeze.json', 'readout.json', 'joint_opportunity.json')]
    f, r, a = [json.loads(p.read_text()) for p in files[1:]]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1]) or a['parent']['sha256'] != h(files[2])
            or not f['before_held_scoring'] or r['independent_confirmation'] or not a['label_aware_only']
            or any(x['confirmation_access'] or not x['original_2pct'] for x in (r, a))):
        raise ValueError('Require bound development-only streaming calibration and diagnostic')
    s, diag = r['summary'], a['summary']
    values = dict(Fits=len(f['count_heads']), Updates=f'{f["configured_updates"]:,}',
        Actual=f'{sum(z["actual"] for outer in f["roster"].values() for cell in outer.values() for z in cell):,}',
        Proxy=f'{sum(z["proxy"] for outer in f["roster"].values() for cell in outer.values() for z in cell):,}',
        DoubleQueries=diag['double_queries'], ActiveQueries=f'{diag["active_queries"]:,}',
        DoubleViews=diag['recording_views_with_double'],
        Restricted=diag['capacity']['double_slots_only']['qualified'], All=diag['capacity']['all_slots']['qualified'])
    for k, name in (('joint', 'Joint'), ('proxy_joint', 'Prior'), ('prefix_match', 'Prefix'), ('recording', 'Rec')):
        for label, field in (('Risk', 'risk_pass'), ('Dual', 'dual_positive')):
            values[name+label] = s['policies'][k][field]
    values['JointChoices'] = s['chosen']['joint']
    values['RecChoices'] = s['chosen']['recording']
    values['PriorChoices'] = s['chosen']['proxy_joint']

    def ci(v):
        lo, hi = v['interval95']
        return f'{v["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    values['Primary'] = ci(r['analysis']['effects']['joint_vs_proxy'])
    values['AgainstPrefix'] = ci(s['effects']['joint_vs_prefix_match'])
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out/'manuscript.tex', out/'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        anchor = ('On OpenDD, the spline improves direct CV forecasts but neither forced nor learned '
                  'replacement improves fixed-slot risk (supplement).')
        replacement = (r'On OpenDD, only \StreamSplineDoubleQueries{} of \StreamSplineActiveQueries{} edited queries '
                       r'allow within-query coupling; matched calibration does not improve risk (supplement).')
    else:
        tex, defs = out/'supplement.tex', out/'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC/'stream_calibration.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        replacement = frag.read_text()+'\n'+anchor
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, replacement))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\StreamSpline{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): h(p) for p in files})
    doc[key].update({p.name: h(p) for p in (tex, defs)})
    (out/receipt).write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
