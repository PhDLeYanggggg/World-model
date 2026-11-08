"""Attach complete-recording training and the fixed-radius lower bound."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_m3w_decision_supplement_v31 as prior

PUBLIC = prior.PUBLIC.parent/'evidence_manuscript_v32'
DEFAULT = prior.DEFAULT.parent/PUBLIC.name
base = prior.base


def evidence():
    name = 'opendd_flight_shape_risk_v1'
    files = [PUBLIC.parent/name/k for k in ('registration.json','analysis.json','execution.json',
                                            'replay_execution.json','fixed_radius_feasibility.json')]
    digest = {f'{name}/{p.name}':base.cal.digest(p) for p in files}
    reg, data, fresh, replay, bound = [json.loads(p.read_text()) for p in files]
    sha = digest[f'{name}/analysis.json']
    if (data['registration']!=digest[f'{name}/registration.json'] or data['formal_confirmation']
        or data['deployment'] or not data['original_unchanged'] or not data['exact_radius_matching']
        or reg['dronecrowd_confirmation_access'] or data['primary_recovery']
        or fresh['updates_this_execution']!=50400 or replay['updates_this_execution']!=0
        or fresh['fresh_exact_replay'] or not replay['fresh_exact_replay'] or fresh['pid']==replay['pid']
        or any(r['analysis_sha256']!=sha for r in (fresh,replay,bound))):
        raise ValueError('Require replayed matched-budget development evidence')
    values = {}
    for label, arm in (('Point','flight_point'),('Budget','flight_budget'),('Relative','flight_relative'),
                       ('Noncancel','flight_noncancel')):
        v = data['summary'][arm]
        values.update({label+'Gain':f'{v["comparisons"]["incumbent"]["lower_pp"]["point"]:+.3f}',
                       label+'Pass':v['passed']})
    status = bound['status_counts']
    if sum(status.values())!=606 or not bound['posthoc_outcome_aware_bound']:
        raise ValueError('Require full outcome-aware bound')
    t = data['transitions']['flight_relative']
    ci = data['summary']['flight_relative']['comparisons']['incumbent']['lower_pp']['ci95']
    values.update(Gained=t['gained_pass'],Lost=t['lost_pass'],
                  Interval='['+', '.join(f'{v:+.3f}' for v in ci)+']',
                  Undefined=status['undefined'],Debit=status['debit_only_impossible'],
                  Feasible=status['hindsight_feasible'])
    return values,digest


def build(out,style,archive):
    values, bindings = evidence()
    doc = prior.build(out,style,archive)
    tex, defs = out/'supplement.tex',out/'supplement_values.tex'
    frag = PUBLIC/'supplement_flight_risk.tex'
    content = base.replace_once(tex.read_text(),r'\section*{LLM Usage Statement}',
                                frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}')
    ref = r'\bibitem{cotter} A. Cotter, M. Gupta, H. Jiang, N. Srebro, K. Sridharan, S. Wang, B. Woodworth, and S. You. Training well-generalizing classifiers for fairness metrics and other data-dependent constraints. In \emph{ICML}, pages 1397--1405, 2019.'
    tex.write_text(base.replace_once(content,r'\end{thebibliography}',ref+'\n\n'+r'\end{thebibliography}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Flight{k}}}{{{v}}}' for k,v in values.items())+'\n')
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
