"""Bind broader predicted coverage and the risk/utility attribution."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v129 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v130'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_pooled_coverage_v1'
    files = [src/f'{k}.json' for k in ('registration','freeze','readout','support_attribution',
                                      'hindsight_capacity','uncapped_continuation','support_funnel')]
    reg,f,r,a,c,d,u = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration']!=h(files[0]) or r['freeze']!=h(files[1])
            or any(v['parent']['sha256']!=h(files[2]) for v in (a,c))
            or d['parent']['sha256']!=h(files[4]) or u['parent']['sha256']!=h(files[3])
            or any(v['independent_confirmation'] or not v['original_2pct'] for v in (reg,r,a,c,d,u))):
        raise ValueError('Require matching pooled, attribution and capacity results')
    risk = r['summary']['risk']
    real = a['summary']['risk']
    donors = a['query_donors']
    cap = c['verified_counts']

    def fmt(v):
        lo,hi = v['interval95']
        return f'{v["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    vals = dict(Edits=f'{risk["pool_full"]["switches"]:,}',Pass=risk['pool_full']['supported_views'],
        Views=r['summary']['views'],MarginalEdits=f'{risk["marginal"]["switches"]:,}',MarginalPass=risk['marginal']['supported_views'],
        Qual=r['qualifying'],EstimatedDonor=donors['predicted'],TrueDonor=f'{donors["actual"]:,}',
        Queries=f'{a["query_views"]:,}',RiskOnlyEdits=f'{real["true_risk"]["switches"]:,}',
        RiskGainEdits=f'{real["true_risk_gain"]["switches"]:,}',TruePass=real['true_risk_gain']['supported_views'],
        TrueRiskCV=fmt(a['summary']['effects']['true_risk_vs_cv']['completion_lower_pp']),
        TrueBothCV=fmt(a['summary']['effects']['true_risk_gain_vs_cv']['completion_lower_pp']),
        TrueBothOLS=fmt(a['summary']['effects']['true_risk_gain_vs_ols4']['completion_lower_pp']),
        CapEight=cap['cap8'].get('verified',0),CapWide=cap['cap32'].get('verified',0),
        UncappedInitial=cap['uncapped'].get('verified',0),
        Uncapped=d['final_uncapped'].get('verified',0),Unresolved=d['final_uncapped'].get('unresolved',0),
        Probability=f'{u["query_views"]["probability_only"]:,}',
        BothCosts=f'{u["query_views"]["both_costs_only"]:,}')
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = (r'A separate fitting-only admission check qualifies \AdmissionQualified{}/\AdmissionCalibration{} '
            r'calibrators; even hindsight withholding of fixed nominees recovers only \AdmissionDual{}/72 '
            r'dual-reference views. Allowing more candidates through screening does not qualify calibration either '
            r'(supplement). Freeing the timing at the same recording edit count yields only '
            r'\AllocationDualRecord{}/\AllocationViews{} dual-reference hindsight witnesses (supplement). '
            r'Even arbitrary perfect edits at that budget are excluded on '
            r'\LeadMixExcluded{}/\LeadMixViews{} views before checking risk.')
        text = (r'On a separate fitting-only roster, even arbitrary perfect edits cannot beat both motion '
            r'references at the sparse budget in \LeadMixExcluded{}/\LeadMixViews{} views. Free counts expose '
            r'\CoveragePoolUncapped{} dual-reference hindsight witnesses in the full menu. The causal pooled '
            r'controller identifies support in only \CoveragePoolEstimatedDonor{} of \CoveragePoolTrueDonor{} queries '
            r'where a seed action exists with outcomes, and qualifies no calibration setting (supplement).')
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,text))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'pooled_coverage.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\CoveragePool{k}}}{{{v}}}' for k,v in vals.items())+'\n')
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
    doc = build(out,a.style,a.archive,a.part)
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out,a.source_zip,doc['build_bindings'])
    elif a.source_zip:
        PACK.supplement_package.package(out,a.source_zip,doc)
