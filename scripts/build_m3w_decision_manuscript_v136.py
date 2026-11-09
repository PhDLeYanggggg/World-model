"""Bind multi-plan recipient targets and equal-radius path comparisons."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v135 as parent

BASE, PACK = parent.BASE,parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v136'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_recipient_path_v1'
    files = [src/f'{k}.json' for k in ('teacher_registration','teacher_freeze','registration',
        'freeze','readout','radius_control')]
    treg,t,reg,f,r,a = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (not treg['train_only'] or not t['train_only'] or t['registration']!=h(files[0])
            or reg['parents'][0]['sha256']!=h(files[1]) or f['registration']!=h(files[2])
            or r['freeze']!=h(files[3]) or a['parent']['sha256']!=h(files[4])
            or not f['before_held_scoring'] or not a['diagnostic_only']
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct'] for z in (r,a))):
        raise ValueError('Require matched training-only and equal-radius receipts')
    risk = r['summary']['risk']
    if len({risk[f'{k}_{p}']['switches'] for k in ('cost','one','soft') for p in ('old','pair')})!=1:
        raise ValueError('Cannot report unequal interventions as matched')

    def ci(x):
        l,u = x['interval95']
        return f'{x["point"]:+.3f} [{l:+.3f}, {u:+.3f}]'

    effects,shape = r['summary']['effects'],a['summary']['effects']
    vals = dict(Plans=sum(z['verified_plans'] for z in t['cells']),
        Ambiguous=f'{sum(z["ambiguous_rows"] for z in t["cells"]):,}',
        Updates=f'{f["optimizer_steps"]:,}',Edits=f'{risk["cost_old"]["switches"]:,}',
        Soft=ci(effects['soft_recipient']['completion_lower_pp']),
        Pair=ci(effects['paired_path']['completion_lower_pp']),
        Shape=ci(shape['cost_shape']['completion_lower_pp']),
        ShapeKnown=ci(shape['cost_shape']['known_pp']),
        VersusFrozen=ci(shape['full_vs_frozen']['completion_lower_pp']),
        Ratio=f'{100*a["changed"]["cost"]["sum_radius_ratio"]:.2f}',
        Worst=f'{risk["cost_pair"]["worst_signed_easy_percent"]:.2f}',
        RadiusWorst=f'{a["summary"]["risk"]["cost_radius"]["worst_signed_easy_percent"]:.2f}',
        BaselineWorst=f'{risk["cost_old"]["worst_signed_easy_percent"]:.2f}',
        Qual=r['qualified'])
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = 'Neither lead-specific annotation prediction nor recording-feasible query supervision improves matched conservative utility.'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,anchor+
            r' The recipient/path factorial likewise shows no conservative lift over the frozen control (supplement).'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'recipient_path.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\FactorPath{k}}}{{{v}}}' for k,v in vals.items())+'\n')
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
        parent.parent.parent.parent.parent.package_supplement(out,a.source_zip,d)
