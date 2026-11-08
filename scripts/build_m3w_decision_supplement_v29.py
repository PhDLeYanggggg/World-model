"""Add matched recording support and risk feasibility to the supplement."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_m3w_decision_supplement_v28 as prior

PUBLIC = prior.PUBLIC.parent/'evidence_manuscript_v29'
DEFAULT = prior.DEFAULT.parent/PUBLIC.name
base = prior.base


def values():
    evidence,results = {},[]
    for name in ('opendd_velocity_calibration_v1','opendd_prefix_risk_v1','opendd_easy_support_v1'):
        root = PUBLIC.parent/name
        files = {k:root/k for k in ('registration.json','analysis.json','execution.json','replay_execution.json')}
        data = {k:json.loads(f.read_text()) for k,f in files.items()}
        sha = {k:base.cal.digest(f) for k,f in files.items()}
        start,end = data['execution.json'],data['replay_execution.json']
        result = data['analysis.json']
        if (result['scope']!='development' or result.get('official_confirmation',result.get('independent_confirmation',True))
            or result['registration']!=sha['registration.json'] or start['exact_replay'] or not end['exact_replay']
            or start['pid']==end['pid'] or any(r['analysis_sha256']!=sha['analysis.json'] for r in (start,end))):
            raise ValueError(f'Require fresh, replayed, development evidence: {name}')
        evidence.update({f'{name}/{k}':v for k,v in sha.items()})
        results.append(result['summary'])
    cal,prefix,support = results
    f = PUBLIC.parent/'opendd_velocity_calibration_v1/flight_feasibility.json'
    v = json.loads(f.read_text())
    if v['certifies_deployment'] or v['analysis_sha256']!=evidence['opendd_velocity_calibration_v1/analysis.json']:
        raise ValueError('Require the paired hindsight feasibility diagnostic')
    evidence['opendd_velocity_calibration_v1/flight_feasibility.json'] = base.cal.digest(f)
    counts = v['views']['uniform']
    u = cal['uniform']['flight']['pass_count']
    macros = dict(UniformPass=u,JointPass=prefix['joint']['flight']['pass_count'],
        UniformFailures=606-u,FeasibleFailed=counts['hindsight_singleton_feasible']-u,
        NoAffordable=counts['every_eligible_easy_edit_exceeds_budget'],NoReference=counts['no_full_easy_reference'],
        JointGap=f'{abs(prefix["joint_vs_matched_independent"]["lower_pp"]):.3f}',
        AP=f'{support["support_quality"]["average_precision"]["equal_site_mean"]:.3f}',
        Prevalence=f'{support["support_quality"]["prevalence"]["equal_site_mean"]:.3f}')
    for name,key in (('Local','local'),('Mass','mass_matched'),('Event','event_matched')):
        row = support[key]
        macros.update({name+'Pass':row['flight']['passes'],name+'Undefined':row['flight']['undefined_easy'],
                       name+'Gain':f'{row["cv_relative"]["lower_pp"]:+.3f}'})
    return macros,evidence


def build(out,style,archive):
    macros,bindings = values()
    doc = prior.build(out,style,archive)
    tex,defs = out/'supplement.tex',out/'supplement_values.tex'
    frag = PUBLIC/'supplement_recording_support.tex'
    tex.write_text(base.replace_once(tex.read_text(),r'\section*{LLM Usage Statement}',
        frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Record{k}}}{{{v}}}' for k,v in macros.items())+'\n')
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
