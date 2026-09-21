"""Render all fixed cost-head fit diagnostics, with no policy selection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/publication_readiness_2026_09/cost_head_fit_forensics_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/cost_head_fit_forensics_v1'


def main():
    import numpy as np
    rows, verifications, bindings, cohorts = [], [], {}, []
    for family in ('transformer','eqmotion'):
        path = OUT/(family+'.json')
        completion = json.loads((PRIVATE/family/'completion.json').read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != completion['report_sha256']:
            raise ValueError('Diagnosis changed')
        verification = json.loads((OUT/(family+'_verification.json')).read_text())
        if verification['analysis_sha256'] != digest:
            raise ValueError('Verification belongs to another diagnosis')
        report=json.loads(path.read_text())
        rows.extend(report['results'])
        for seed in (17,29,43):
            keys=[]
            for fold in (0,1,2):
                relative=f'data/stage_cvpr2027_experiments/8to12_{family}_v6/seed{seed}_ridge/fold_{fold}.npz'
                cache=ROOT/relative
                if hashlib.sha256(cache.read_bytes()).hexdigest()!=report['source_bindings'][relative]:
                    raise ValueError('Original cohort cache changed')
                with np.load(cache,allow_pickle=False) as a:
                    identities=json.loads(str(a['identities_json']))
                keys.extend(tuple(r[k] for k in ('recording_id','agent_id','frame_id','horizon_raw')) for r in identities)
            if len(keys)!=11966 or len(set(keys))!=11966:
                raise ValueError('Expected original unique OOF cohort')
            cohorts.append(keys)
        verifications.append(verification)
        bindings[family] = digest
    if any(keys!=cohorts[0] for keys in cohorts):
        raise ValueError('Family/seed cost cohorts differ')
    if len(rows) != 12 or {(r['family'],r['seed'],r['head']) for r in rows} != {
            (f,s,h) for f in ('transformer','eqmotion') for s in (17,29,43) for h in ('ridge','neural_cost')}:
        raise ValueError('Keep all fixed heads')
    lines = ['# Complete Cost-Head Fit Results', '',
        'Post-hoc in-sample diagnosis. Each head uses the same 11,966 query identities',
        'and 306 causal/predicted-rollout features. OOF refers to the trajectory',
        'producer; these rows trained the cost head itself. No independent calibration,',
        'test selection, deployment or new fit. All costs retain past-normalized ADE.', '',
        '## Whole Fit Population', '',
        '| Family | Seed | Head | True mean harm | Predicted mean harm | Harm MSE | Mean-label reference MSE |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        m=r['full_fit']
        lines.append(f"| {r['family']} | {r['seed']} | {r['head']} | {m['target_harm_mean']:.8g} | "
            f"{m['predicted_harm_mean']:.8g} | {m['harm_mse']:.8g} | {r['mean_label_reference']['harm_mse']:.8g} |")
    lines.extend(['', '## Fixed Per-Agent Eligibility', '',
        'Conservative: predicted gain >=0.02 and predicted harm <=0.05. Moderate:',
        'gain >=0.01 and harm <=0.1. These are original rules, not searched thresholds.',
        'Eligibility is not a scene-solver decision or intervention rate. Positive gain',
        'means benefit minus harm. All nonempty groups are retained, including the small',
        'positive EqMotion seed17/neural/conservative result.', '',
        '| Family | Seed | Head | Rule | Rows | True harm | Predicted harm | True net gain | Predicted net gain |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    flat=[]
    for r in rows:
        for policy in ('conservative','moderate'):
            m=r['conditional']['eligible_'+policy]
            lines.append(f"| {r['family']} | {r['seed']} | {r['head']} | {policy} | {m['rows']} | "
                f"{m['target_harm_mean']:.8g} | {m['predicted_harm_mean']:.8g} | "
                f"{m['actual_net_gain_mean']:.8g} | {m['predicted_net_gain_mean']:.8g} |")
        for recording, groups in [('all_fit',r['conditional']),*r['per_recording'].items()]:
            for name,m in groups.items():
                flat.append(dict(family=r['family'],seed=r['seed'],head=r['head'],recording=recording,subset=name,**m))
    lines.extend(['', '## Ridge Projection And Target Concentration', '',
        '| Family | Seed | Negative raw harm rows | Share (%) | Actually harmed among these (%) | Mean actual harm | Top 1% squared harm-label mass (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in rows:
        if r['head']!='ridge':continue
        m=r['conditional']['ridge_harm_clipped_to_zero']
        lines.append(f"| {r['family']} | {r['seed']} | {m['rows']} | {100*m['rows']/r['full_fit']['rows']:.5f} | "
            f"{100*m['positive_harm_fraction']:.5f} | {m['target_harm_mean']:.8g} | "
            f"{100*r['top_harm_squared_target_mass_fraction']:.5f} |")
    lines.extend(['', 'The six target populations are shared by their ridge/neural heads. The largest',
        '1% contains 120 rows per population. Squared label mass is not the final loss',
        'contribution, and does not by itself prove a causal training defect. Projection',
        'of a negative value to zero increases it; removing projection is not a remedy.',
        'Per-recording results and empty subsets are retained in `all_fit_slices.csv`.', ''])
    (OUT/'complete_results.md').write_text('\n'.join(lines))
    with (OUT/'all_fit_slices.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(flat)
    groups=[r['conditional']['eligible_'+p] for r in rows for p in ('conservative','moderate')]
    summary=dict(source_bindings=bindings,heads=12,rows_per_head=11966,
        exact_common_cohort_across_six_family_seed_pairs=True,
        eligibility_comparisons=len(groups),negative_eligible_net_gain_count=sum(g['actual_net_gain_mean']<0 for g in groups),
        eligible_harm_underprediction_count=sum(g['target_harm_mean']>g['predicted_harm_mean'] for g in groups),
        heads_better_than_mean_label_harm_mse=sum(r['full_fit']['harm_mse']<r['mean_label_reference']['harm_mse'] for r in rows),
        global_harm_overprediction_heads=sum(r['full_fit']['predicted_harm_mean']>r['full_fit']['target_harm_mean'] for r in rows),
        top_one_percent_squared_harm_label_mass_range=[min(r['top_harm_squared_target_mass_fraction'] for r in rows),
                                                    max(r['top_harm_squared_target_mass_fraction'] for r in rows)],
        exact_normalization_checks=sum(r['normalization_exact'] for r in rows),
        raw_batch_checks=sum(len(v['original_batch_checks']) for v in verifications),
        raw_rows_replayed=sum(v['exact_replayed_rows'] for v in verifications),
        independent_scalar_summaries=sum(v['independent_scalar_summaries'] for v in verifications),
        new_training=False,policy_selection=False,primary_metric_changed=False,independent_confirmation=False,
        new_forecast_inference='verification_of_old_fit_batches_only',stage5c_executed=False,smc_enabled=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
