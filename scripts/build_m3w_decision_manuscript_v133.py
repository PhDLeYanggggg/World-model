"""Bind verified training slates and the held fixed-menu capacity check."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v132 as parent

BASE, PACK = parent.BASE,parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v133'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_slate_supervision_v1'
    files = [src/f'{k}.json' for k in ('teacher_registration','teacher_freeze','registration',
                                      'freeze','readout','fixed_menu_capacity')]
    treg,t,reg,f,r,c = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (t['registration']!=h(files[0]) or reg['parents'][0]['sha256']!=h(files[1])
            or f['registration']!=h(files[2]) or r['freeze']!=h(files[3])
            or c['parent']['sha256']!=h(files[4]) or not f['before_held_scoring']
            or not treg['no_calibration_or_held_labels_in_teacher'] or not c['labels_hindsight_only']
            or any(z['independent_confirmation'] or not z['original_2pct'] for z in (t,r,c))):
        raise ValueError('Require matching slate supervision, freeze and capacity evidence')
    risk,effect = r['summary']['risk'],r['summary']['effects']
    arms = ('frozen','gain_additive','feasible_additive','feasible_joint')
    if len({risk[k]['switches'] for k in arms})!=1:
        raise ValueError('Cannot report unequal edit counts as matched')

    def fmt(v):
        lo,hi = v['interval95']
        return f'{v["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    counts = c['counts']
    values = dict(Updates=f'{f["optimizer_steps"]:,}',Edits=f'{risk["frozen"]["switches"]:,}',
        Qual=r['qualified'],Views=r['summary']['views'],Witnesses='/'.join(str(z['verified']) for z in t['cells']),
        Passes='/'.join(str(risk[k]['supported_views']) for k in arms),
        Feasible=fmt(effect['feasible_supervision']['completion_lower_pp']),
        Joint=fmt(effect['joint_composition']['completion_lower_pp']),
        FixedVerified=counts.get('verified',0),FixedUnknown=counts.get('undefined_support',0),
        FixedExcluded=counts.get('risk_infeasible',0)+counts.get('nonpositive_bound',0),
        FixedUnresolved=counts.get('unresolved',0))
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = 'Better lead-specific annotation prediction likewise fails to improve matched conservative utility.'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,
            'Neither lead-specific annotation prediction nor recording-feasible query supervision improves matched conservative utility.'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'slate_supervision.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\SlateWitness{k}}}{{{v}}}' for k,v in values.items())+'\n')
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
        parent.parent.package_supplement(out,a.source_zip,d)
