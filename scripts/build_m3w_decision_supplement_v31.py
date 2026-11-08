"""Attach the fixed-radius training and recording harm attribution."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v30 as prior

PUBLIC = prior.PUBLIC.parent/'evidence_manuscript_v31'
DEFAULT = prior.DEFAULT.parent/PUBLIC.name
base = prior.base


def evidence():
    name = 'opendd_radius_shape_v1'
    files = [PUBLIC.parent/name/n for n in ('registration.json','analysis.json',
             'execution.json','replay_execution.json','risk_attribution.json')]
    digests = {f'{name}/{p.name}':base.cal.digest(p) for p in files}
    reg, read, start, replay, detail = [json.loads(p.read_text()) for p in files]
    digest = digests[f'{name}/analysis.json']
    if (read['registration'] != digests[f'{name}/registration.json'] or read['formal_confirmation']
        or read['deployment'] or not read['exact_radius_matching'] or reg['dronecrowd_confirmation_access']
        or start['fresh_exact_replay'] or not replay['fresh_exact_replay'] or start['pid']==replay['pid']
        or start['updates_this_execution']!=50400 or replay['updates_this_execution']!=0
        or any(r['analysis_sha256']!=digest for r in (start,replay))
        or detail['original_analysis_sha256']!=digest or detail['decisions_changed']):
        raise ValueError('Require the verified fixed-radius development study')
    v = read['summary']['easy_guard']['comparisons']['incumbent']['lower_pp']
    t = detail['transitions']
    old, new = (read['summary'][k]['passed'] for k in ('incumbent','easy_guard'))
    if (sum(t.values())!=606 or old-new!=t['lost_pass']-t['gained_pass']
        or detail['flipped_risk_components']['lost_pass']['completion_easy_risk_upper']!=t['lost_pass']):
        raise ValueError('Risk decomposition does not support manuscript wording')
    return dict(Gain=f'{v["point"]:+.3f}',Interval='['+', '.join(f'{x:+.3f}' for x in v['ci95'])+']',
                OldPass=old,NewPass=new,Lost=t['lost_pass'],Gained=t['gained_pass']),digests


def build(out, style, archive):
    values, bindings = evidence()
    doc = prior.build(out,style,archive)
    tex, defs = out/'supplement.tex',out/'supplement_values.tex'
    frag = PUBLIC/'supplement_radius_training.tex'
    tex.write_text(base.replace_once(tex.read_text(), r'\section*{LLM Usage Statement}',
        frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Radius{k}}}{{{v}}}'
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
