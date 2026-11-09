"""Bind the lead-shape ablation and perfect-edit coverage lower bound."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v128 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v129'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    d = PUBLIC.parent/'moving_cv_lead_mixture_v1'
    files = [d/f'{k}.json' for k in ('registration','freeze','readout','hindsight_capacity','minimum_edits')]
    reg,freeze,r,cap,count = [json.loads(k.read_text()) for k in files]
    h = BASE.cal.digest
    if (freeze['registration']!=h(files[0]) or r['freeze']!=h(files[1]) or cap['parent']['sha256']!=h(files[2])
            or count['parent']['sha256']!=h(files[3]) or any(x['independent_confirmation'] for x in (reg,r,cap,count))
            or not all(x['original_2pct'] for x in (reg,r,cap,count))):
        raise ValueError('Require matching proposal and coverage evidence')
    risk, effect = r['summary']['risk'],r['summary']['effects']
    stats = count['stats']['cap8']

    def fmt(v):
        lo,hi = v['interval95']
        return f'{v["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'

    values = dict(Updates=f'{freeze["optimizer_steps"]:,}',Edits=f'{risk["lead"]["switches"]:,}',
        Views=r['summary']['views'],Frozen=risk['frozen']['supported_views'],Shared=risk['constant']['supported_views'],
        Lead=risk['lead']['supported_views'],Shape=fmt(effect['lead_radius_vs_constant']['completion_lower_pp']),
        LeadContrast=fmt(effect['lead_vs_constant']['completion_lower_pp']),
        LeadPair=fmt(r['geometry']['lead_vs_constant']['completion_lower_pixel']),
        SharedQual=r['qualified']['constant'],LeadQual=r['qualified']['lead'],
        CapacityBefore=cap['original'].get('verified',0),CapacityAfter=cap['proposed']['both_addback'].get('verified',0),
        NeedMore=stats['above_current'],Unreachable=stats['unreachable_or_no_reference'],
        Excluded=stats['views']-stats['at_or_below_current'],Extra=stats['median_extra_edits'])
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'\AllocationDualRecord{}/\AllocationViews{} dual-reference hindsight witnesses (supplement).'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,anchor+
            r' Even arbitrary perfect edits at that budget are excluded on '
            r'\LeadMixExcluded{}/\LeadMixViews{} views before checking risk.'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'lead_mixture.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\LeadMix{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(k.relative_to(PUBLIC.parent)):h(k) for k in files})
    doc[key].update({k.name:h(k) for k in (tex,defs)})
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
        PACK.supplement_package.package(out,a.source_zip,d)
