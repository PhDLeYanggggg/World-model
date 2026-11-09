"""Bind exact-count allocation, streaming, and hindsight capacity evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v127 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v128'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    exp = PUBLIC.parent/'moving_cv_record_allocation_v1'
    files = [exp/f'{k}.json' for k in ('registration','freeze','readout','geometry','hindsight_capacity')]
    reg, freeze, readout, geom, capacity = [json.loads(p.read_text()) for p in files]
    digest = BASE.cal.digest
    if (freeze['registration']!=digest(files[0]) or readout['freeze']!=digest(files[1])
            or any(v['parent']['sha256']!=digest(files[2]) for v in (geom,capacity))
            or not all(v['original_2pct'] for v in (reg,readout,capacity))
            or any(v['independent_confirmation'] for v in (reg,readout,geom,capacity))):
        raise ValueError('Require matching allocation, geometry and capacity evidence')
    r = readout['summary']['risk']
    counts = capacity['counts']

    def fmt(e):
        lo,hi = e['interval95']
        return f'{e["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'

    values = dict(Edits=f'{r["local"]["switches"]:,}', Views=readout['summary']['views'],
        Local=r['local']['supported_views'], Gain=r['offline_gain']['supported_views'],
        Risk=r['offline_risk']['supported_views'],OnlineLocal=r['online_local']['supported_views'],
        OnlineCarry=r['online_carry']['supported_views'],OnlineLocalEdits=r['online_local']['switches'],
        OnlineCarryEdits=r['online_carry']['switches'],PastCredits=readout['online_spent_past_credits'],
        Qualified=readout['qualified'], CVQuery=counts['cv_only_query_exact'].get('verified',0),
        CVRecord=counts['cv_only_record_exact'].get('verified',0),
        DualQuery=counts['dual_query_exact'].get('verified',0), DualRecord=counts['dual_record_exact'].get('verified',0),
        CarryEffect=fmt(readout['summary']['effects']['online_carry_vs_local']['completion_lower_pp']),
        RiskEffect=fmt(readout['summary']['effects']['offline_risk_vs_local']['completion_lower_pp']),
        PairEffect=fmt(geom['effects']['offline_risk_vs_local']['completion_lower_pixel']))
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'Allowing more candidates through screening does not qualify calibration either (supplement).'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,anchor+
            r' Freeing the timing at the same recording edit count yields only '
            r'\AllocationDualRecord{}/\AllocationViews{} dual-reference hindsight witnesses (supplement).'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'recording_allocation.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\Allocation{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)):digest(p) for p in files})
    doc[key].update({p.name:digest(p) for p in (tex,defs)})
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
