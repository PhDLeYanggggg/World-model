"""Bind the matched-count and matched-peak recipient-exchange ablation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v137 as parent

BASE,PACK = parent.BASE,parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v138'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_matched_recipient_exchange_v1'
    files = [src/k for k in ('registration.json','freeze.json','readout.json','peak_envelope.json','record_peak_envelope.json')]
    reg,f,r,q,pool = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration']!=h(files[0]) or r['freeze']!=h(files[1]) or not f['before_held_scoring']
            or q['parent']['sha256']!=h(files[2]) or pool['parent']['sha256']!=h(files[3])
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct'] for z in (r,q,pool))
            or reg['parents'][-1]['sha256']!=h(PUBLIC.parent/'moving_cv_fixed_recipient_support_v1/readout.json')):
        raise ValueError('Require fixed-anchor matched exchange receipts')
    risk,effect = r['summary']['risk'],r['summary']['effects']
    keys = ('half_anchor','geometry','learned','guard')
    if len({risk[k]['switches'] for k in keys})!=1:
        raise ValueError('Compared arms must be count matched')

    def ci(v):
        l,u = v['interval95']
        p,lo,hi = (f'{x:+.4f}' if abs(x)>=.00005 else '0.0000' for x in (v['point'],l,u))
        return f'{p} [{lo}, {hi}]'

    v = dict(Updates=f'{f["optimizer_steps"]:,}',Edits=f'{risk["half_anchor"]["switches"]:,}',
        Passes='/'.join(str(risk[k]['supported_views']) for k in keys),
        Worst='/'.join(f'{risk[k]["worst_signed_easy_percent"]:.2f}' for k in keys),
        Qual=r['qualified'],OLSLimit=r['one_exchange_upper']['ols4_excluded'])
    for k in keys[1:]:
        name = k.capitalize()
        v[name+'Count'] = f'{r["exchanges"][k]:,}'
        v[name+'Effect'] = ci(effect[f'{k}_vs_anchor']['completion_lower_pp'])
        v[name+'Pair'] = ci(r['geometry'][k]['completion_lower_pixel'])
    for name,key in (('Half','0.5'),('Base','1.0'),('Double','2.0'),('Four','4.0'),('Eight','8.0'),('Free','None')):
        v[name+'Limits'] = ' & '.join(str(x) for x in (q['summary'][key]['matched_count']['ols4_nonpositive'],
            q['summary'][key]['unrestricted_count']['ols4_nonpositive'],pool['summary'][key]['ols4_nonpositive']))
    v['RecordExcluded'] = pool['summary']['1.0']['ols4_nonpositive']
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'Even arbitrary paths on frozen recipients cannot beat OLS4 in \FixedSupportFreeExcluded{}/72 views (supplement).'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,
            r'Pooling the same peak and count budgets rules out OLS4 gains in \ExchangeMassRecordExcluded{}/72 views (supplement).'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'recipient_exchange.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\ExchangeMass{k}}}{{{x}}}' for k,x in v.items())+'\n')
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
        parent.parent.parent.parent.parent.parent.parent.package_supplement(out,a.source_zip,d)
