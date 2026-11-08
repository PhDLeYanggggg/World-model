"""Attach matched movement allocation and the distributed-budget control."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_m3w_decision_supplement_v32 as prior

PUBLIC = prior.PUBLIC.parent/'evidence_manuscript_v33'
DEFAULT = prior.DEFAULT.parent/PUBLIC.name
base = prior.base


def evidence():
    name = 'opendd_causal_amplitude_v1'
    files = [PUBLIC.parent/name/k for k in ('registration.json','analysis.json','execution.json',
        'replay_execution.json','amplitude_spread.json','half_radius_shapes.json')]
    digest = {f'{name}/{p.name}':base.cal.digest(p) for p in files}
    reg,data,fresh,replay,spread,half = [json.loads(p.read_text()) for p in files]
    sha = digest[f'{name}/analysis.json']
    if (data['registration']!=digest[f'{name}/registration.json'] or data['formal_confirmation']
        or data['deployment'] or not data['original_risk_unchanged'] or not data['radius_multiset_exact']
        or reg['dronecrowd_confirmation_access'] or fresh['new_heads_this_execution']!=63
        or replay['new_heads_this_execution']!=0 or fresh['fresh_exact_replay'] or not replay['fresh_exact_replay']
        or fresh['pid']==replay['pid'] or any(r['analysis_sha256']!=sha for r in (fresh,replay))
        or spread['original_analysis_sha256']!=sha or spread['decisions_changed']
        or not spread['query_mass_exact'] or half['spread_sha256']!=digest[f'{name}/amplitude_spread.json']
        or not half['exact_recipient_radius_match'] or half['decisions_changed']):
        raise ValueError('Require exact matched-budget controls and replay')
    values = {}
    for label,row in (('Full',data['summary']['incumbent']),('Joint',data['summary']['joint']),
                      ('Spread',spread['summary']['query_spread']),('Half',spread['summary']['fixed_half'])):
        values.update({label+'Edits':f'{row["edits"]:,}',label+'Pass':row['passed'],
                       label+'Undefined':row['undefined_selected_easy']})
    pairs = dict(JointIndependent=data['paired_joint']['matched_independent'],
        JointAdditive=data['paired_joint']['matched_additive'],SpreadJoint=spread['spread_vs_joint'],
        HalfShape=half['summary']['previous_point']['comparisons']['incumbent']['lower_pp'])
    if (not all('point' in v for v in pairs.values()) or
        half['summary']['previous_point']['passed'] != values['HalfPass']-5 or
        half['summary']['flight_noncancel']['passed'] != values['HalfPass']+4):
        raise ValueError('Manuscript contrasts must match verified readout')
    values.update({k:f'{v["point"]:+.3f}' for k,v in pairs.items()})
    return values,digest


def build(out,style,archive):
    values,bindings = evidence()
    doc = prior.build(out,style,archive)
    tex,defs = out/'supplement.tex',out/'supplement_values.tex'
    frag = PUBLIC/'supplement_causal_amplitude.tex'
    tex.write_text(base.replace_once(tex.read_text(),r'\section*{LLM Usage Statement}',
        frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Amplitude{k}}}{{{v}}}'
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
