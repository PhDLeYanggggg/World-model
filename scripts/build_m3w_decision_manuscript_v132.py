"""Bind lead-aware missingness, asymmetric easy risk and late shape controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v131 as parent

BASE, PACK = parent.BASE,parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v132'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_lead_risk_v1'
    files = [src/f'{k}.json' for k in ('registration','freeze','readout','failure_replay','missing_disagreement',
                                      'late_taper_registration','late_taper')]
    reg,f,r,fail,missing,tr,t = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration']!=h(files[0]) or r['freeze']!=h(files[1]) or not f['before_held_scoring']
            or any(z['parent']['sha256']!=h(files[2]) for z in (fail,missing,tr))
            or t['registration']['sha256']!=h(files[5])
            or any(x['independent_confirmation'] or not x['original_2pct'] for x in (reg,r,fail,missing,tr,t))):
        raise ValueError('Require matching lead-risk, failure and taper evidence')
    risk,effect = r['summary']['risk'],r['summary']['effects']
    known = fail['risk_failures']

    def fmt(v):
        lo,hi = v['interval95']
        return f'{v["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'

    d = missing['summary']['lead_matched']
    values = dict(Updates=f'{f["optimizer_steps"]:,}',Edits=f'{risk["cond_lead_matched"]["switches"]:,}',
        Qual=r['qualified'],Views=r['summary']['views'],Brier=fmt(r['mask']['lead_brier_reduction']),
        Lead=fmt(effect['lead_matched']['completion_lower_pp']),
        Tail=fmt(effect['tail_matched']['completion_lower_pp']),
        TailLead=fmt(effect['lead_under_tail']['completion_lower_pp']),
        BasePass=risk['cond_lead_matched']['supported_views'],TailPass=risk['tail_lead_matched']['supported_views'],
        BaseWorst=f'{risk["cond_lead_matched"]["worst_signed_easy_percent"]:.2f}',
        TailWorst=f'{risk["tail_lead_matched"]["worst_signed_easy_percent"]:.2f}',
        BaseEasy=known['cond_lead_matched']['known_easy_over'],TailEasy=known['tail_lead_matched']['known_easy_over'],
        LateShare=f'{100*d["debit_by_lead_block_pp"][2]/sum(d["debit_by_lead_block_pp"]):.1f}',
        TaperCond=fmt(t['summary']['effects']['cond_lead_matched_taper_vs_uniform']['completion_lower_pp']),
        TaperTail=fmt(t['summary']['effects']['tail_lead_matched_taper_vs_uniform']['completion_lower_pp']))
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'missing-step bound (supplement).'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,anchor+
            r' Better lead-specific annotation prediction likewise fails to improve matched conservative utility.'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'lead_risk.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\LeadRiskAbl{k}}}{{{v}}}' for k,v in values.items())+'\n')
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
        parent.package_supplement(out,a.source_zip,d)
