"""Bind single-query support and rare-event factorization evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v152 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v153'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    inputs = PUBLIC.parent
    src = inputs/'opendd_counterfactual_support_v1'
    second = inputs/'opendd_factored_counterfactual_support_v1'
    files = [src/k for k in ('registration.json','freeze.json','readout.json','opportunity.json')]
    files += [second/k for k in ('registration.json','freeze.json','readout.json')]
    _, f, r, a, reg2, f2, r2 = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration']!=h(files[0]) or r['freeze']!=h(files[1]) or a['parent']['sha256']!=h(files[2])
            or f2['registration']!=h(files[4]) or r2['freeze']!=h(files[5])
            or reg2['parents'][1]['sha256']!=h(files[2]) or not reg2['post_result_followup']
            or not all(z['before_held_scoring'] for z in (f,f2)) or not a['label_aware_only']
            or any(z['confirmation_access'] or not z['original_2pct'] for z in (r,a,r2))
            or any(z['independent_confirmation'] for z in (r,r2))):
        raise ValueError('Require paired development-only evidence')
    s, t = r['summary'],r2['summary']
    v = dict(Fits=len(f['fits']), FactorFits=len(f2['fits']), Oracle=a['summary']['oracle_dual'],
             Offered=a['summary']['whole_changed_in_recoverable_query'], Hit=a['summary']['whole_picked_qualified_option'])
    for k, label in (('local','Local'),('whole','Whole'),('prefix_match','Prefix'),
                     ('factor','Factor'),('factor_local','FactorLocal'),('factor_matched_gain','Matched')):
        v[label+'Risk'] = t['policies'][k]['risk_pass']
        v[label+'Dual'] = t['policies'][k]['dual_positive']
        v[label+'Loss'] = t['outcomes'][k]['loss']
        v[label+'Replaced'] = t['outcomes'][k]['replaced']
    def ci(d):
        lo, hi = d['interval95']
        return f'{d["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'
    v['WholeLocal'] = ci(s['effects']['whole_vs_local'])
    v['FactorLocalContrast'] = ci(t['effects']['factor_vs_factor_local'])
    v['LocalPrefix'] = ci(t['effects']['local_vs_prefix_match'])
    doc = previous.build(out,style,archive,part)
    if part == 'main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = ('The intervention menu and the ability to recognize useful recipients are distinct limitations '
                  '(supplement). On MovingDroneCrowd++ development data, conditional risk training also fails '
                  'to recover a conservative gain at matched budgets. Neither result warrants replacing the baseline.')
        change = ('The action menu and the ability to recognize useful recipients are distinct limitations. '
                  'Direct and factorized predictors of whole-record reference loss do not beat the corresponding '
                  'query-local guards at the same fixed slots (supplement). None warrants replacing the baseline.')
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'counterfactual_support.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        change = frag.read_text()+'\n'+anchor
    tex.write_text(BASE.replace_once(tex.read_text(),anchor,change))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\CFSupport{k}}}{{{z}}}' for k,z in v.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(inputs)):h(p) for p in files})
    doc[key].update({p.name:h(p) for p in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('main','supplement'))
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=BASE.v22.STYLE)
    p.add_argument('--archive',type=Path,default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part=='main' else 'supplement')
    doc = build(out,a.style,a.archive,a.part)
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out,a.source_zip,doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out,a.source_zip,doc)
