"""Bind the cross-domain transfer and relaxed budget envelope."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v153 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE,PACK = previous.BASE,previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v154'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root/'cross_domain_pass_transition_v1'
    files = [src/k for k in ('registration.json','freeze.json','readout.json','relaxed_envelope.json')]
    _,f,r,a = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration']!=h(files[0]) or r['freeze']!=h(files[1]) or a['parent']['sha256']!=h(files[2])
            or not f['before_scoring'] or r['independent_confirmation'] or not r['original_2pct']
            or not a['risk_ignored_in_bound'] or any(z['a_labels_opened'] or z['confirmation_access'] for z in (r,a))):
        raise ValueError('Require frozen exposed development transfer, no independent access')
    b = a['summary']['whole_query_upper_pp']
    v = dict(Fits=len(f['fits']),Updates=f'{f["configured_updates"]:,}',Excluded=b['ols4_nonpositive'],
        Views=b['views'],UpperMedian=f'{b["median"][1]:+.2f}')
    for k,name in (('opendd','Open'),('moving','Moving')):
        s = r[k]
        v[name+'Options'] = f'{s["options"]:,}'
        v[name+'Rescues'] = s['rescue_options']
        for arm,l in (('anchor','Anchor'),('transition','Model'),('oracle','Oracle')):
            p = s['policies'][arm]
            for key,lab in (('dual','Dual'),('risk','Risk'),('replaced','Swaps')):
                v[name+l+lab] = p[key]
        x = s['transition_effects']['matched_gain']
        lo,hi = x['interval95']
        v[name+'Matched'] = f'{x["point"]:+.5f} [{lo:+.5f}, {hi:+.5f}]'
    doc = previous.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        text = BASE.replace_once(tex.read_text(),
            'count policy succeeds in \\DirectCountCount{} views. At fixed recipients and displacement radii, learned shape choice\n'
            'improves over a linear edit but does not establish a gain over\n'
            'a simple observed-prefix rule. On 40 separate DroneCrowd calibration',
            r'count policy succeeds in \DirectCountCount{} views. On MovingDroneCrowd++ development data, even '
            r'an arbitrary one-query replacement cannot beat OLS4 in \CrossDomainExcluded{}/\CrossDomainViews{} '
            'views at the frozen displacement budget. On 40 separate DroneCrowd calibration')
        text = BASE.replace_once(text,'None warrants replacing the baseline.',
            'A zero-shot transition model adds no jointly passing Moving view. '
            'The bound separates restricted opportunity from ranking failure (supplement).')
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'cross_domain_transition.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor)
        marker = r'\end{thebibliography}'
        text = BASE.replace_once(text,marker,r'\bibitem{xu2022transfer} Y. Xu, L. Wang, Y. Wang, and Y. Fu. '
            r'Adaptive trajectory prediction via transferable GNN. In \emph{CVPR}, pages 6520--6531, 2022.'+'\n\n'+marker)
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\CrossDomain{k}}}{{{z}}}' for k,z in v.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(root)):h(p) for p in files})
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
