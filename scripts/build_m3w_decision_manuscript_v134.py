"""Bind the query budget by candidate menu factorial capacity diagnostic."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v133 as parent

BASE, PACK = parent.BASE,parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v134'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_budget_menu_factorial_v1'
    files = [src/k for k in ('registration_v2.json','readout.json','nested_verification.json')]
    reg,raw,r = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (raw['registration']!=h(files[0]) or r['parent']['sha256']!=h(files[1])
            or r['independent_confirmation'] or r['a_labels_opened']
            or not r['original_2pct'] or not r['labels_hindsight_only'] or r['views']!=72
            or reg['parents'][2]['sha256']!=h(parent.PUBLIC.parent/'moving_cv_slate_supervision_v1/fixed_menu_capacity.json')):
        raise ValueError('Require matching budget-menu evidence')
    values = {'Views':r['views']}
    for name,key in (('NarrowExact','narrow_exact'),('NarrowRelaxed','narrow_relaxed'),
                     ('FullExact','full_exact'),('FullRelaxed','full_relaxed')):
        c = r['counts'][key]
        values[name] = c.get('verified',0)
        values[name+'Status'] = ' & '.join(str(v) for v in (c.get('verified',0),
            c.get('risk_infeasible',0)+c.get('nonpositive_bound',0),c.get('undefined_support',0),c.get('unresolved',0)))
    for name,key in (('WidthFixed','expand_at_exact'),('BudgetNarrow','relax_with_narrow'),
                     ('WidthRelaxed','expand_at_relaxed'),('BudgetFull','relax_with_full')):
        z = r['contrasts'][key]
        l,u = z['interval95_pp']
        values[name] = f'{z["source_balanced_pp"]:+.2f} [{l:+.2f}, {u:+.2f}]'
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'Free counts expose \CoveragePoolUncapped{} dual-reference hindsight witnesses in the full menu.'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,
            r'The full bank has \CapacityFullExact/\CapacityViews{} hindsight witnesses at frozen counts versus \CapacityFullRelaxed/\CapacityViews{} with free counts.'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'budget_menu.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\Capacity{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)):h(p) for p in files})
    doc[key].update({p.name:h(p) for p in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('main','supplement'))
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=BASE.v22.STYLE)
    p.add_argument('--archive',type=Path,default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part=='main' else 'supplement')
    d = build(out,a.style,a.archive,a.part)
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out,a.source_zip,d['build_bindings'])
    elif a.source_zip:
        parent.parent.parent.package_supplement(out,a.source_zip,d)
