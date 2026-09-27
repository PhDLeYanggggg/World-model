"""Post-readout failure localization only; never changes the frozen policy."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_tail as run


def main():
    _,data,_,_,_,_=run.load()
    details=json.loads((run.PRIVATE/'details.json').read_text())
    summary=json.loads((run.PUBLIC/'summary.json').read_text())
    policies={}
    for policy in ('mse','tail4','mse_matched_count'):
        rows=[r for r in details['rows'] if r['policy']==policy]
        ratios=[r['metric']['selected_positive_harm_ratio'] for r in rows if r['metric']['selected_positive_harm_ratio'] is not None]
        q=[r['metric'] for r in details['quality'] if r['policy']==policy]
        def finite_median(key):
            a=[r[key] for r in q if r[key] is not None]
            return float(np.median(a)) if a else None
        policies[policy]=dict(dependent_views=len(rows),defined_risk_views=len(ratios),
            descriptive_defined_view_risk_median=float(np.median(ratios)),
            maximum_defined_view_risk=float(max(ratios)),
            undefined=[dict(group=r['group'],site=r['site'],rows=r['metric']['rows'],
                            intervention_rate=r['metric']['intervention_rate']) for r in rows if r['metric']['selected_positive_harm_ratio'] is None],
            unknown_interventions=sum(r['metric']['unknown_interventions'] for r in rows),
            descriptive_defined_view_reference_bias_median=finite_median('predicted_over_actual_reference'),
            descriptive_defined_view_predicted_harm_median=finite_median('predicted_selected_harm_ratio'),
            descriptive_defined_view_actual_harm_median=finite_median('actual_selected_harm_ratio'))
    counts=dict(query_views=0,nonempty_query_views=0,rankable_query_views=0,
                different_rank_selection_query_views=0,tail_interventions=0,common_interventions=0)
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text())
    for ref in freeze['decisions']:
        rec=json.loads((ROOT/ref['path']).read_text())
        assert run.artifact(ROOT/rec['arrays']['path'])==rec['arrays']
        with np.load(ROOT/rec['arrays']['path'],allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
        ids=a['ids']; groups=defaultdict(list)
        for i in range(len(ids)):
            j=ids[i]; groups[(str(data['sites'][j]),str(data['recordings'][j]),int(data['frames'][j]))].append(i)
        for ix in groups.values():
            k=int(a['tail4'][ix].sum()); available=int(a['eligible'][ix].sum())
            assert k==int(a['mse_matched_count'][ix].sum())
            counts['query_views']+=1; counts['nonempty_query_views']+=int(k>0)
            counts['rankable_query_views']+=int(0<k<available)
            counts['different_rank_selection_query_views']+=int(np.any(a['tail4'][ix]!=a['mse_matched_count'][ix]))
        counts['tail_interventions']+=int(a['tail4'].sum())
        counts['common_interventions']+=int((a['tail4']&a['mse_matched_count']).sum())
    doc=dict(result_source='fresh_post_readout_diagnosis_cached_verified_frozen_scores',
        policy_changed=False,primary_unchanged=True,policies=policies,query_counts=counts,
        independent_units=12,defined_view_medians_are_descriptive_only=True)
    run.immutable_json(run.PUBLIC/'failure_localization.json',doc)
    lines=['# Post-Readout Failure Localization','','## Material Passport','',
        'Fresh analysis of frozen development readout and decisions. No refitting, threshold change, candidate selection or independent-role opening.',
        'Primary fixed-roster intervals remain unchanged. Defined-view medians below exclude empty views and are ONLY descriptive, not repaired primary estimates.','',
        '## Empty Coverage','',
        f"Tail-weighted and matched-count policies have {len(policies['tail4']['undefined'])} empty held views; all are at eu-locality-112.",
        'The registered harm-ratio primary is undefined, not zero. Locality-average intervention may still be positive because other dependent views intervene.','',
        '| Policy | Defined /216 views | Median observed harm % | Maximum observed harm % | Median predicted harm % | Median predicted/actual reference | Unknown-label interventions |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def num(x,m=1): return 'not_applicable' if x is None else f'{m*x:.4f}'
    for p,m in policies.items():
        lines.append(f"| {p} | {m['defined_risk_views']} | {num(m['descriptive_defined_view_risk_median'],100)} | {num(m['maximum_defined_view_risk'],100)} | {num(m['descriptive_defined_view_predicted_harm_median'],100)} | {num(m['descriptive_defined_view_reference_bias_median'])} | {m['unknown_interventions']} |")
    lines+=['','Counts above repeat windows across fitted heads and seeds; they are not independent people or scenes.','',
        '## Does Count Matching Permit a Ranking Test?','',*[f'- {k}: {v}' for k,v in counts.items()],'',
        'Rankable means0<K<eligible current-query agents. If K=0 or K=all eligible, matching fixes the selection and cannot test ordering.',
        'A weak matched contrast can reflect little within-query freedom as well as weak ranking. Neither explanation establishes better selection.','',
        '## Diagnosis','',
        '1. All216 fitting monitors improve: optimization ran, but training loss is not conditional risk calibration.',
        '2. Bounded nonnegative outputs remove the former negative-output clipping failure, yet MSE predicts selected harm below its observed value and inflates the reference denominator.',
        '3. Tail weighting shrinks coverage and net gain. The equal-count ADE interval crosses zero; no stable ordering advantage is demonstrated.',
        '4. Empty views invalidate the registered fixed-roster harm primary;95 tail views still exceed the2% budget. Net easy preservation does not certify positive-harm safety.',
        '5. This contrast freezes the utility head and forecasting bank, so it does not isolate errors of a newly trained trajectory model or prove representation impossibility.','',
        'Next: directly model fixed-floor signed budget excess using the SAME repaired producer chain and non-clipped utility floor. Prespecify a matched objective-only contrast and test cancellation/denominator bias; do not open reserved sources or sweep held thresholds.',
        'Only registered future fitting experiments can test that next hypothesis. Existing signed-excess experiments used a different CV-reference/producer setting, so they are relevant negative controls, not equivalent evidence.',
        'Image-local detector silver, obs8/pred12 rawstride12. No metric/seconds/physical-safety/true3D/foundation claim. Stage5C/SMC remain disabled.']
    (run.PUBLIC/'failure_localization.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(doc))


if __name__=='__main__': main()
