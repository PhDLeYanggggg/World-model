"""Bind certified edit budgets and the minimum-edit teacher ablation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v134 as parent

BASE, PACK = parent.BASE,parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v135'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_minimum_budget_v1'
    files = [src/f'{k}.json' for k in ('registration','teacher_freeze','capacity_certificates',
                                      'training_registration','training_freeze','training_readout','failure_diagnosis')]
    reg,t,cap,tr,f,r,fail = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (t['registration']!=h(files[0]) or not t['train_only'] or tr['parents'][0]['sha256']!=h(files[1])
            or f['registration']!=h(files[3]) or r['freeze']!=h(files[4]) or not f['before_held_scoring']
            or any(x['a_labels_opened'] or x['independent_confirmation'] or not x['original_2pct'] for x in (cap,r,fail))
            or fail['parent']['sha256']!=h(files[5]) or not fail['diagnostic_only']
            or cap['views']!=72 or cap['parents'][2]['sha256']!=h(PUBLIC.parent/'moving_cv_budget_menu_factorial_v1/nested_verification.json')):
        raise ValueError('Require matched capacity and model ablation receipts')
    risk = r['summary']['risk']
    effects = r['summary']['effects']
    if risk['cost']['switches']!=risk['teacher_matched']['switches']:
        raise ValueError('Teacher contrast is not count matched')

    def interval(x):
        l,u = x['interval95']
        return f'{x["point"]:+.3f} [{l:+.3f}, {u:+.3f}]'

    vals = dict(Updates=f'{f["optimizer_steps"]:,}',Matches=f'{risk["cost"]["switches"]:,}',
        OldTeacher=f'{sum(z["old_edits"] for z in t["cells"]):,}',NewTeacher=f'{sum(z["edits"] for z in t["cells"]):,}',
        Teachers=sum(z['recordings'] for z in t['cells']),Quarter=cap['capacity']['0.25']['verified'],
        Views=cap['views'],MedLow=f'{cap["median_lower_percent"]:.2f}',MedHigh=f'{cap["median_upper_percent"]:.2f}',
        Qual=r['qualified'],Passes='/'.join(str(risk[k]['supported_views']) for k in
        ('frozen','cost','teacher','teacher_matched','teacher_guarded')),
        Match=interval(effects['teacher_at_match']['completion_lower_pp']),
        MatchKnown=interval(effects['teacher_at_match']['known_pp']),
        Pair=interval(r['geometry']['completion_lower_pixel']),
        Fixed=interval(fail['summary']['effects']['fixed_recipients']['completion_lower_pp']),
        Clipped=interval(fail['summary']['effects']['clipped_radius']['completion_lower_pp']),
        RecipientShare=f'{100*fail["debit_recipient_change_pp"]/(fail["debit_recipient_change_pp"]+fail["debit_shared_recipient_pp"]):.1f}')
    for label,k in (('Five','0.05'),('Ten','0.1'),('QuarterRow','0.25'),('Half','0.5'),('ThreeQuarters','0.75')):
        z = cap['capacity'][k]
        vals[label] = ' & '.join(str(z[key]) for key in ('verified','excluded','undefined_support','unresolved'))
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'The full bank has \CapacityFullExact/\CapacityViews{} hindsight witnesses at frozen counts versus \CapacityFullRelaxed/\CapacityViews{} with free counts.'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,
            r'The full bank has \CapacityFullExact/\CapacityViews{} hindsight witnesses at frozen counts and \EditCertQuarter/\EditCertViews{} at a 25\% recording budget.'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'minimum_edit.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\EditCert{k}}}{{{v}}}' for k,v in vals.items())+'\n')
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
        parent.parent.parent.parent.package_supplement(out,a.source_zip,d)
