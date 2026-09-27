"""Post-readout paired objective-fit/transport diagnostics; no policy tuning."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_excess as run
from scripts.report_m3w_fixed_floor_tail import ci


def main():
    cfg=json.loads((ROOT/run.CONFIG).read_text()); d=json.loads((run.PRIVATE/'details.json').read_text())
    idx={(r['group'],r['site'],r['policy']):r['metric'] for r in d['quality']}
    rows=[]
    for r in d['quality']:
        if r['policy']!='excess': continue
        a=r['metric']; b=idx[(r['group'],r['site'],'mse')]; m={}
        for key in ('all_fit_MSE','easy_fit_MSE','all_normalized_excess_MSE','easy_normalized_excess_MSE'):
            m[key+'_difference']=a[key]-b[key]
        rows.append(dict(site=r['site'],metric=m))
    sites=sorted(set(r['site'] for r in rows))
    paired={k:run.inter.paired_localities(rows,sites,k,cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in rows[0]['metric']}
    selected=[r for r in d['rows'] if r['policy']=='excess']
    empty=dict(Counter(r['site'] for r in selected if r['metric']['selected_positive_harm_ratio'] is None))
    worst=max((r for r in selected if r['metric']['selected_positive_harm_ratio'] is not None),key=lambda r:r['metric']['selected_positive_harm_ratio'])
    doc=dict(result_source='fresh_post_readout_diagnosis_cached_verified_readout',policy_changed=False,
        paired_signed_MSE_differences=paired,empty_views_by_site=empty,
        worst_view=dict(site=worst['site'],group=worst['group'],harm_ratio=worst['metric']['selected_positive_harm_ratio'],
            known_selected_rows=round(worst['metric']['known_intervention_rate']*worst['metric']['known_rows'])),
        independent_roles_read=False,diagnostic_not_primary=True)
    run.immutable_json(run.PUBLIC/'failure_localization.json',doc)
    lines=['# Fixed-Floor Objective Failure Localization','','## Material Passport','',
        'Fresh post-readout analysis of frozen development scores. No model, threshold, data-role or primary change.',
        'Unadjusted post-hoc paired differences below are diagnostic, not extra confirmatory hypotheses.','',
        '| Direct-excess minus moment-MSE | Difference [95% locality interval] |','|---|---:|']
    for k,m in paired.items(): lines.append('| '+k+' | '+ci(m)+' |')
    lines+=['','Negative squared-error differences favor direct supervision. Fitting references are repeated context summaries, not independent held outcomes.',
        'Lower fitting error without lower held error points to transport/generalization or selection calibration; it does not prove an irreducible data limit.','',
        '## Sparse Coverage and Worst View','',
        'Empty views by locality: '+json.dumps(empty)+'.',
        'Worst defined view: '+json.dumps(doc['worst_view'])+'.',
        'The worst ratio is reported with its selected-row count; rare interventions are not silently excluded.',
        'Unknown-label interventions remain in decisions but cannot supply outcome evidence.','',
        '## Failure Taxonomy','',
        '- Runtime/optimization failure: not supported;108 complete fits, fixed budgets, checkpointed finite losses.',
        '- Objective fit: improved on fitting sources; this is not sufficient downstream evidence.',
        '- Source transport: all-risk score error does not improve on held sources under the matched objective contrast.',
        '- Conditional selection: predicted acceptable scores still select over-budget outcomes in79/216views.',
        '- Coverage:14empty views across three localities leave the fixed-roster harm primary undefined.',
        '- Ranking: equal-count ADE interval crosses zero; changing the objective does not demonstrate better allocation.',
        '- Easy net error: preserved, but separate from positive-harm safety.',
        '- Interpretation: direct-loss output components are unidentified score bases, not calibrated harm/reference moments.','',
        'Next: before another risk-head loss variant, localize the train-to-held gap by fitting-support, motion/reference-error scale and label completeness; measure whether useful safe actions occupy feature-supported regions. Use fixed bins derived only from fitting data, include unknown support and leave reserved sources closed.',
        'Then choose a representation/source-support repair only if that diagnosis provides a concrete discriminating hypothesis. Do not pursue more threshold sweeps or declare development calibration as external success.',
        'Image-local detector silver, obs8/pred12 rawstride12; no metric/seconds/physical-safety/true3D/foundation claim. Stage5C/SMC remain disabled.']
    (run.PUBLIC/'failure_analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(doc))


if __name__=='__main__': main()
