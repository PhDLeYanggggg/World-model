"""Add the direct-forecast OpenDD development comparison to the supplement."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_m3w_decision_supplement_v27 as prior

PUBLIC = prior.PUBLIC.parent/'evidence_manuscript_v28'
DEFAULT = prior.DEFAULT.parent/PUBLIC.name
base = prior.base


def build(out,style,archive):
    home = PUBLIC.parent/'opendd_velocity_candidate_v1'
    names = ('analysis.json','registration.json','execution.json','replay_execution.json')
    hashes = {k:base.cal.digest(home/k) for k in names}
    result = json.loads((home/names[0]).read_text())
    fresh,replay = (json.loads((home/k).read_text()) for k in names[-2:])
    if (result['scope']!='development' or result['independent_confirmation'] or result['official_risk_pass']
            or result['registration']!=hashes[names[1]] or len(result['fits'])!=63
            or fresh['exact_replay'] or not replay['exact_replay'] or fresh['pid']==replay['pid']
            or any(r['phase']!='complete' or r['analysis_sha256']!=hashes[names[0]] for r in (fresh,replay))):
        raise ValueError('Require fresh and exactly replayed direct-forecast experiment')
    rows = result['summary']
    def effect(key):
        r = rows[key]
        return f'{r["lower_pp"]:+.3f} [' + ', '.join(f'{x:+.3f}' for x in r['ci95_pp']) + ']'

    effects = {'SelfDirect':'self','SelfQuarter':'self_quarter','PedQuarter':'pedestrian_quarter',
               'TrafficQuarter':'traffic_quarter','TrafficVsPed':'traffic_vs_pedestrian',
               'TrafficSupport':'traffic_vs_pedestrian_motor_supported','OracleAddSelf':'oracle_add_self'}
    values = {k:effect(v) for k,v in effects.items()}
    values['WorstEasy'] = f'{rows["self_quarter"]["worst_easy_pct"]:.2f}'
    values['TrafficCount'] = f'{sum(r["motor_rows"] for r in result["support"].values()):,}'
    doc = prior.build(out,style,archive)
    tex,defs = out/'supplement.tex',out/'supplement_values.tex'
    frag = PUBLIC/'supplement_velocity.tex'
    tex.write_text(base.replace_once(tex.read_text(),r'\section*{LLM Usage Statement}',
        frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Velocity{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({f'{home.name}/{k}':v for k,v in hashes.items()})
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
