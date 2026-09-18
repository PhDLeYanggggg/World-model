"""Descriptive analysis of all fixed fits; no threshold or model selection."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def gain(error,reference):
    return float(100*(1-np.mean(error)/np.mean(reference))) if np.mean(reference)>1e-12 else None


def summarize_arm(trials):
    slices={}
    names=sorted(set.intersection(*(set(t['slices']) for t in trials)))
    for name in names:
        ts=[t['slices'][name] for t in trials if t['slices'][name]['rows']>0]
        if not ts:
            continue
        slices[name]=dict(paired_cells=len(ts),gain_percent=gain([s['primary_ADE'] for s in ts],[s['reference_ADE'] for s in ts]),
            mean_ADE=float(np.mean([s['primary_ADE'] for s in ts])),
            mean_reference_ADE=float(np.mean([s['reference_ADE'] for s in ts])),
            mean_harm=float(np.mean([s['mean_harm_over_reference'] for s in ts])),
            scope='equal_available_seed_site_cells_not_pooled_windows')
    held=[t['vs_CV'] for t in trials]
    return dict(slices=slices,
        easy_relative_range=[float(min(m['easy_degradation_percent'] for m in held)),float(max(m['easy_degradation_percent'] for m in held))],
        easy_absolute_harm_range=[float(min(m['easy_absolute_harm'] for m in held)),float(max(m['easy_absolute_harm'] for m in held))],
        train_gain_range=[float(min(t['train_equal_scene_gain_percent'] for t in trials)),float(max(t['train_equal_scene_gain_percent'] for t in trials))],
        held_gain_range=[float(min(m['improvement_percent'] for m in held)),float(max(m['improvement_percent'] for m in held))],
        finite_prediction_failures=sum(m['forecast_nonfinite'] for m in held))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args(); reg=json.loads(args.registration.read_text())
    reports=ROOT/reg['reports']; path=reports/'report.json'; report=json.loads(path.read_text())
    if not report['complete'] or len(report['trials'])!=36:
        raise ValueError('All fixed fits required')
    arms={a:summarize_arm([t for t in report['trials'] if t['arm']==a]) for a in report['summary']}
    legacy=[t for t in report['trials'] if t['arm']=='legacy_sdd_aux']
    guards=dict(gain_before=gain([t['vs_CV']['primary_ADE'] for t in legacy],[t['vs_CV']['reference_ADE'] for t in legacy]),
        gain_guard_only=gain([t['no_anchor_guard_only_counterfactual']['primary_ADE'] for t in legacy],[t['vs_CV']['reference_ADE'] for t in legacy]),
        reason='Post hoc attribution diagnostic only, not a fitted or deployable policy')
    loss_checks={}
    for arm in reg['arms']:
        ts=[t for t in report['trials'] if t['arm']==arm]
        loss_checks[arm]={}
        for phase in ('source','main'):
            logs=[l for t in ts for l in t['fit']['losses'] if l['source']==(phase=='source')]
            loss_checks[arm][phase]=dict(logged_batches=len(logs),
                gradient_norm_quantiles=np.quantile([l['gradient_norm'] for l in logs],[0,.5,.95,1]).tolist(),
                logged_clipping_fraction=float(np.mean([l['gradient_norm']>5 for l in logs])),
                note='Sparse logged batches only; internal and primary loss magnitudes are not comparable')
    result=dict(result_source='fresh_run_analysis_of_fresh_and_cached_verified_fixed_fits',
        report_sha256=file_digest(path),registration_sha256=file_digest(args.registration),
        arms=arms,guard_only=guards,logged_gradients=loss_checks,
        no_threshold_search=True,no_model_selected=True,independent_confirmation=False)
    json_write(reports/'analysis.json',result)
    lines=['# Conditioning Contrasts and Failure Slices','',
        'Descriptive results from all fixed exposed-site folds; no selection or deployment.','',
        '| Arm | Gain vs CV | Descriptive site CI | Per-seed gains | Easy degradation range |',
        '| --- | ---: | --- | --- | --- |']
    for a,s in report['summary'].items():
        cv=s['vs_CV']; easy=arms[a]['easy_relative_range']
        lines.append(f"| {a} | {cv['gain_percent']:.6f}% | {cv['exploratory_scene_ci95_percent']} | {cv['per_seed_gain_percent']} | {easy} |")
    lines+=['','Three repeatedly exposed site clusters are not independent confirmation. Easy percentages can be very large when CV error is near zero; absolute harm is retained below.','',
        '| Arm | Absolute easy harm range | Train gain range (%) | Static-start gain (%) | Static-stay harm | Hard gain (%) |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for a,s in arms.items():
        def fmt(v):
            return 'undefined' if v is None else f'{v:.6f}'
        z=s['slices']
        lines.append(f"| {a} | {s['easy_absolute_harm_range']} | {s['train_gain_range']} | {fmt(z['static_moves']['gain_percent'])} | {fmt(z['static_stays']['mean_harm'])} | {fmt(z['hard_train_q75']['gain_percent'])} |")
    lines+=['',f'Legacy gain {guards["gain_before"]:.8f}%; adding only the no-anchor CV guard gives {guards["gain_guard_only"]:.8f}%.',
        'This guard counterfactual is post hoc attribution, not a selected policy.',
        'Native-coordinate aggregate metrics mix local units and are diagnostic only; the primary normalized metric is unchanged.','']
    (reports/'contrasts.md').write_text('\n'.join(lines))
    stream=io.StringIO(); fields=['trial','arm','seed','fold','step','source','loss','gradient_norm','loss_rows','valid_label_rows','no_anchor_rows','loss_coordinate']
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader()
    for t in report['trials']:
        if t['arm'] in reg['arms']:
            for row in t['fit']['losses']:
                writer.writerow(dict(trial=t['trial'],arm=t['arm'],seed=t['seed'],fold=t['fold'],**row))
    (reports/'loss_trace.csv').write_text(stream.getvalue())
    print(json.dumps(result))


if __name__=='__main__':
    main()
