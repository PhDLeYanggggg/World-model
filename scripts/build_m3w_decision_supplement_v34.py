"""Bind the two support-preserving exchange experiments to the supplement."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_m3w_decision_supplement_v33 as prior

PUBLIC = prior.PUBLIC.parent/'evidence_manuscript_v34'
DEFAULT = prior.DEFAULT.parent/PUBLIC.name
base = prior.base


def evidence():
    out,digests = [],{}
    for name,extra in (('opendd_support_exchange_v1','oracle_headroom.json'),
                       ('opendd_direct_exchange_v1','choice_attribution.json')):
        files = [PUBLIC.parent/name/k for k in ('registration.json','analysis.json','execution.json',
                                               'replay_execution.json',extra)]
        digests.update({f'{name}/{p.name}':base.cal.digest(p) for p in files})
        reg,data,fresh,replay,side = [json.loads(p.read_text()) for p in files]
        sha = digests[f'{name}/analysis.json']
        if (data['registration']!=digests[f'{name}/registration.json'] or data['formal_confirmation']
            or data['deployment'] or data['primary_development_pass'] or not data['matched_query_budget']
            or not data['matched_all_recipients'] or reg['dronecrowd_confirmation_access']
            or fresh['new_heads_this_execution']!=21 or replay['new_heads_this_execution']!=0
            or fresh['fresh_exact_replay'] or not replay['fresh_exact_replay'] or fresh['pid']==replay['pid']
            or any(r['analysis_sha256']!=sha for r in (fresh,replay)) or side['original_analysis_sha256']!=sha):
            raise ValueError('Require exact replay of both development studies')
        out.append((data,side))
    (m,oracle),(d,attribution) = out
    values = {'HalfPass':m['summary']['half']['passes']}
    for name,v in (('Marginal',m['summary']['guarded']),('Direct',d['summary']['direct_utility']),
                   ('Guard',d['summary']['direct_guard']),('Event',d['summary']['event_guard'])):
        values.update({name+'Pass':v['passes'],name+'Gain':f'{v["vs_half"]["point"]:+.4f}',
                       name+'Actions':f'{v["active_queries"]:,}'})
    h = oracle['summary']['guarded_oracle']
    values.update(OracleGain=f'{h["lower_pp"]:+.4f}',OraclePass=h['passes'],
        OracleActions=f'{oracle["counts"]["queries_with_safe_positive_gain"]:,}',
        ActualGood=f'{attribution["choice_quality"]["direct_guard"]["actual_safe_positive"]:,}')
    if m['guarded_vs_matched']['required']-m['guarded_vs_matched']['defined']!=4 or attribution['decisions_changed']:
        raise ValueError('Coverage statement changed')
    return values,digests


def build(out,style,archive):
    values,bindings = evidence()
    doc = prior.build(out,style,archive)
    tex,defs = out/'supplement.tex',out/'supplement_values.tex'
    frag = PUBLIC/'supplement_support_exchange.tex'
    content = base.replace_once(tex.read_text(),r'\section*{LLM Usage Statement}',
                                frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}')
    ref = r'\bibitem{spo} A. N. Elmachtoub and P. Grigas. Smart \lq\lq Predict, then Optimize\rq\rq. \emph{Management Science}, 68(1):9--26, 2022.'
    tex.write_text(base.replace_once(content,r'\end{thebibliography}',ref+'\n\n'+r'\end{thebibliography}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Exchange{k}}}{{{v}}}'
        for k,v in values.items())+'\n')
    doc['source_bindings'].update(bindings)
    doc['source_bindings'][f'{PUBLIC.name}/{frag.name}'] = base.cal.digest(frag)
    for p in (tex,defs):
        doc['files'][p.name] = base.cal.digest(p)
    (out/'supplement_bindings.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=DEFAULT/'supplement')
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=base.v22.STYLE)
    p.add_argument('--archive',type=Path,default=base.v22.supp.base.REPLAY)
    a = p.parse_args()
    result = build(a.output_dir,a.style,a.archive)
    if a.source_zip:
        base.v22.supp.prior.package(a.output_dir,a.source_zip,result)
