"""Bind nested OpenDD spline selection to the working paper."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v149 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v150'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    src = PUBLIC.parent/'opendd_nested_spline_choice_v2'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json')]
    reg, f, r = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    repair = src/'receipt_repair_v2.json'
    files.append(repair)
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or f['receipt_repair'] != h(repair)
            or not f['before_held_scoring'] or not reg['nested_exclusion']
            or r['independent_confirmation'] or r['confirmation_access'] or not r['original_2pct']):
        raise ValueError('Require nested development-only spline selection')
    s = r['summary']
    values = dict(Producer=len(f['producer_fits']), Selector=len(f['selector_fits']),
        Updates=f'{sum(f["configured_updates"].values()):,}', Edits=f'{s["policies"]["anchor"]["edits"]:,}',
        Views=s['policies']['anchor']['views'], Undefined=s['policies']['anchor']['undefined'],
        JointRules=sum(v['joint'][0] is not None for v in f['calibration'].values()))
    arms = ('anchor', 'prefix_match', 'forced', 'gain', 'independent', 'joint', 'joint_same_rule')
    names = ('Anchor', 'Prefix', 'Forced', 'Gain', 'Independent', 'Joint', 'Shared')
    for k, n in zip(arms, names, strict=True):
        p = s['policies'][k]
        values.update({n+'Risk': p['risk_pass'], n+'Dual': p['dual_positive']})
        if k != 'prefix_match':
            values[n+'Uses'] = s['chosen'][k]

    def ci(v):
        lo, hi = v['interval95']
        return f'{v["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    for k, n in (('joint_vs_prefix_match', 'Primary'), ('joint_vs_anchor', 'AgainstAnchor'),
                 ('grouped_at_same_rule', 'Grouping')):
        values[n] = ci(s['effects'][k])
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out/'manuscript.tex', out/'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        marker = r'\section{Limitations and Independent Evidence}'
        first, last = tex.read_text().split(marker, 1)
        anchor = (r'On OpenDD, the matched comoving spline gains \OpenSplineMovingGain{} pp versus CV '
                  r'but does not improve the fixed-slot policy (supplement).')
        change = ('On OpenDD, the spline improves direct CV forecasts but neither forced nor learned '
                  'replacement improves fixed-slot risk (supplement).')
        content = first + marker + BASE.replace_once(last, anchor, change)
    else:
        tex, defs = out/'supplement.tex', out/'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC/'nested_spline_choice.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        content = BASE.replace_once(tex.read_text(), anchor, frag.read_text()+'\n'+anchor)
    tex.write_text(content)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(
        rf'\newcommand{{\NestedSpline{k}}}{{{v}}}' for k, v in values.items())+'\n')
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
