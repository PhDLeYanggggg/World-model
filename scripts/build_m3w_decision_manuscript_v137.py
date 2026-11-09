"""Bind equal-peak support learning and the arbitrary-path capacity envelope."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v136 as parent

BASE,PACK = parent.BASE,parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v137'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_fixed_recipient_support_v1'
    files = [src/k for k in ('registration.json','freeze.json','readout.json','arbitrary_envelope.json','replay_receipt.json')]
    reg,f,r,b,replay = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration']!=h(files[0]) or r['freeze']!=h(files[1]) or not f['before_held_scoring']
            or not b['labels_hindsight_only'] or b['parent']['sha256']!=reg['parents'][1]['sha256']
            or replay['parent']['sha256']!=h(files[2])
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct'] for z in (r,b))):
        raise ValueError('Require frozen support-learning and envelope receipts')
    risk,effect = r['summary']['risk'],r['summary']['effects']
    if len({risk[k]['switches'] for k in ('frozen','uniform','pooled','consensus')})!=1:
        raise ValueError('The main comparison changed the recipient count')

    def ci(v):
        l,u = v['interval95']
        point,low,high = (f'{x:+.3f}' if abs(x)>=.0005 else '0.000' for x in (v['point'],l,u))
        return f'{point} [{low}, {high}]'

    v = dict(Updates=f'{f["optimizer_steps"]:,}',Edits=f'{risk["uniform"]["switches"]:,}',
        Pooled=f'{replay["actual_path_changes"]["pooled"]:,}',Consensus=f'{replay["actual_path_changes"]["consensus"]:,}',
        Passes='/'.join(str(risk[k]['supported_views']) for k in ('frozen','uniform','pooled','consensus')),
        PoolEffect=ci(effect['pooled_vs_uniform']['completion_lower_pp']),
        SupportEffect=ci(effect['consensus_vs_uniform']['completion_lower_pp']),
        PoolPair=ci(r['geometry']['pooled_vs_uniform']['completion_lower_pixel']),
        SupportPair=ci(r['geometry']['consensus_vs_uniform']['completion_lower_pixel']),
        Qual=r['qualified'],MenuExcluded=r['fixed_recipient_capacity']['ols4_nonpositive'],
        HalfExcluded=b['summary']['0.5']['ols4_nonpositive'],FullExcluded=b['summary']['1.0']['ols4_nonpositive'],
        FreeExcluded=b['summary']['None']['ols4_nonpositive'])
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = 'The recipient/path factorial likewise shows no conservative lift over the frozen control (supplement).'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,
            r'Even arbitrary paths on frozen recipients cannot beat OLS4 in \FixedSupportFreeExcluded{}/72 views (supplement).'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'fixed_recipient.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\FixedSupport{k}}}{{{val}}}' for k,val in v.items())+'\n')
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
        parent.parent.parent.parent.parent.parent.package_supplement(out,a.source_zip,d)
