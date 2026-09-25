"""Report every fixed producer replacement, with uncertainty and no winner selection."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_m3w_european_protected_motion import compact_metric
from scripts.report_m3w_european_cv_reference import value, ci

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_producer_transport_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(a):
    views = {}
    for key,v in a['views'].items():
        views[key] = dict(ADE_vs_CV={s:compact_metric(m) for s,m in v['ADE_vs_CV'].items()},
            raw_ADE_vs_CV=compact_metric(v['raw_ADE_vs_CV']),FDE_vs_CV=compact_metric(v['FDE_vs_CV']),
            zero_CV=v['zero_CV'],safety_observed_pass=v['safety_observed_pass'],switch_rate=v['switch_rate'],
            opportunity=v['ledger']['summary'],gain_capture_fraction=v['ledger']['gain_capture_fraction'])
    return dict(result_source=a['result_source'],views=views,
        small_vs_full={key:dict(raw_small_vs_full=compact_metric(v['raw_small_vs_full']),
            policy_small_vs_full={s:compact_metric(m) for s,m in v['policy_small_vs_full'].items()})
            for key,v in a['small_vs_full'].items()},
        new_training=False,threshold_refit=False,reserved_roles_opened=False,deployment_changed=False,
        interpretation='per_pair_eight_excluded_sources_shared_twelve_source_roster_not_independent_trials',
        stage5c_executed=False,smc_enabled=False)


def raw_forecast_audit(a):
    """Account for the registered raw FDE contrast using unchanged prediction banks."""
    import numpy as np
    from scripts import run_m3w_european_producer_transport as runner
    reg, preg, _, data, identity, _ = runner.load()
    if identity != a['identity']:
        raise ValueError('Raw forecast audit identity mismatch')
    rows = {}
    for name, candidate, fold, seed, design in runner.previous.jobs(preg, data, identity['previous_identity']):
        if candidate != 'neural':
            continue
        ids = design['held_ids']
        sites = data['sites'][ids]
        full = identity['previous_identity']['producer_identity']['frozen_final_producers'][f'single{fold}_seed{seed}']
        predictions = dict(full4=runner.previous.read_predictions(full['prediction'], ids),
            damping097=runner.baseline_numpy(data['history'][ids], 3)-data['origin'][ids,None])
        for half in (0, 1):
            key = f'fold{fold}_half{half}_seed{seed}'
            art = json.loads((runner.PRIVATE/'predictions'/(key+'.json')).read_text())
            predictions[f'half{half}'] = runner.previous.read_predictions(art, ids)
        for variant, prediction in predictions.items():
            ade, fde = runner.native_errors(prediction.astype(float)+data['origin'][ids,None],
                data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            metrics = {}
            for field, cost, baseline in [('ADE', ade, data['baseline_ade'][ids,1]),
                                          ('FDE', fde, data['baseline_fde'][ids,1])]:
                m = runner.paired_scene_metrics(cost, baseline, sites, expected_scenes=sorted(set(sites)),
                    dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                    bootstrap_resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
                if field == 'ADE' and m != a['views'][f'fold{fold}_seed{seed}_{variant}_all']['raw_ADE_vs_CV']:
                    raise ValueError('Raw accounting does not reproduce frozen ADE')
                metrics[field] = compact_metric(m, scenes=True)
            rows[f'fold{fold}_seed{seed}_{variant}'] = metrics
    runner.assert_identity(identity)
    out = dict(result_source='fresh_run_raw_metric_accounting_cached_verified_predictions',
        analysis_sha256=sha(PUBLIC/'analysis.json'), raw_predictor_views=36, metrics=rows,
        new_training=False, new_predictions=False, deployment_changed=False)
    runner.immutable_json(PUBLIC/'raw_forecast_metrics.json', out)
    lines = ['# Raw Forecast Accuracy', '',
        'Registered raw ADE/FDE accounting, without either intervention gate. All 36 predictor views are retained.',
        'Each view uses eight excluded development localities; the 3,000-resample intervals are conditional.', '',
        '| Fold/seed/producer | ADE gain vs CV (%) | ADE CI | FDE gain vs CV (%) | FDE CI | ADE supported | FDE supported |',
        '|---|---:|---|---:|---|---:|---:|']
    for key, row in rows.items():
        lines.append(f"| {key} | {value(row['ADE'])} | {ci(row['ADE'])} | {value(row['FDE'])} | {ci(row['FDE'])} | {row['ADE']['supported_rows']} | {row['FDE']['supported_rows']} |")
    lines += ['', 'Source-development detector tracks, image pixels, obs8/pred12 rawstride12 only.',
        'Different ADE/FDE label support is retained. No metric, seconds, physical-safety or deployment claim.', '']
    (PUBLIC/'raw_forecast_results.md').write_text('\n'.join(lines))


def main():
    a = json.loads((PUBLIC/'analysis.json').read_text())
    v = json.loads((PUBLIC/'verification.json').read_text())
    b = json.loads((PUBLIC/'bank_replay.json').read_text())
    if not v['all_passed'] or v['analysis_sha256'] != sha(PUBLIC/'analysis.json') or b['identity'] != a['identity']:
        raise ValueError('Complete verified diagnostic required')
    if len(b['checks'])!=18 or any(not r['exact'] or r['replay_rows']!=4096 for r in b['checks']):
        raise ValueError('All frozen producers must replay exactly')
    for row in a['lineages']:
        if set(row['trained'])&set(row['readout']):
            raise ValueError('Producer training exposure')
        for field in ('checkpoint','prediction'):
            r = row[field]
            if sha(ROOT/r['path'])!=r['sha256']:
                raise ValueError('Changed inference artifact')
    summary = compact(a)
    summary['analysis_sha256'] = v['analysis_sha256']
    (PUBLIC/'summary_metrics.json').write_text(json.dumps(summary,separators=(',',':'),allow_nan=False)+'\n')
    lines = ['# Fixed Producer Comparisons','',
        'Every row uses eight source localities excluded from its full head/producer chain. All twelve localities are opened development sources.',
        'Same head and2% predicted risk rule; intervention counts may differ. No model or threshold is selected from these results.',
        '3,000 locality-bootstrap resamples per comparison are conditional and overlapping,not independent replications. Raw contrasts repeat across event targets;only18 are distinct.','',
        '| Fold/seed/candidate/event | Raw ADE gain vs CV (%) | Controlled ADE gain (%) | Conditional CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harmed | Switch (%) |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for key,r in a['views'].items():
        e = r['ADE_vs_CV']['easy']['worst_scene_gain_percent']
        easy = 'undefined' if e is None else f'{-e:.6f}'
        lines.append(f"| {key} | {value(r['raw_ADE_vs_CV'])} | {value(r['ADE_vs_CV']['all'])} | {ci(r['ADE_vs_CV']['all'])} | {value(r['FDE_vs_CV'])} | {value(r['ADE_vs_CV']['hard'])} | {easy} | {r['zero_CV']['harmed_rows']}/{r['zero_CV']['rows']} | {100*r['switch_rate']:.6f} |")
    lines += ['', '## Small Versus Full Producer on Identical Rows','',
        'Positive values favor the small producer. This changes producer identity and its fitting roster/normalizer/floor together,not only data quantity.','',
        '| Pair | Subset | Gain (%) | Conditional CI |','|---|---|---:|---|']
    for key,r in a['small_vs_full'].items():
        for subset,m in [('raw',r['raw_small_vs_full']),*r['policy_small_vs_full'].items()]:
            lines.append(f'| {key} | {subset} | {value(m)} | {ci(m)} |')
    lines += ['','Image pixels,obs8/pred12 rawstride12,detector tracks. Not t50,seconds,metric,human-gold,physical safety,true3D or foundation.',
        'No new training,calibration,threshold search,reserved confirmation,deployment,Stage5C or SMC.','']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    lines = ['# Opportunity and Gate Attribution','',
        'Costs below are hindsight accounting. Only past geometry and frozen model scores determine gate decisions.','',
        '| View | Oracle benefit/CV (%) | Captured benefit/CV (%) | Paid harm/CV (%) | Missed utility benefit/CV (%) | Missed risk-veto benefit/CV (%) | Gain capture (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,v in a['views'].items():
        s = v['ledger']['summary']
        def f(name):
            x=s[name]['equal_locality']
            return 'undefined' if x is None else f'{x:.6f}'
        capture=v['ledger']['gain_capture_fraction']
        c='undefined' if capture is None else f'{100*capture:.6f}'
        lines.append(f"| {key} | {f('oracle_gain')} | {f('captured_gain')} | {f('switched_harm')} | {f('missed_nonpositive_utility')} | {f('missed_risk_veto')} | {c} |")
    lines += ['','Oracle benefit is not an inference policy or proof of learnability. Locality contributions use identical native-unit CV denominators.',
        'Source-development detector tracks only; no metric/seconds,physical-safety,independent-confirmation or deployment claim.','']
    (PUBLIC/'opportunity_attribution.md').write_text('\n'.join(lines))
    errors={k:v['score_errors'] for k,v in a['views'].items()}
    (PUBLIC/'score_errors.json').write_text(json.dumps(errors,separators=(',',':'),allow_nan=False)+'\n')
    lines = ['# Gain and Harm Score Reliability', '',
        'Means below weight the defined localities equally, after dividing each MAE by that locality subset mean CV error.',
        'Selected event-harm ratios are positive harm / event-reference mass on selected rows, not net easy degradation.',
        'Unknown labels, zero-event denominators and zero-selected localities are not silently treated as zero.', '',
        '| View | Population gain MAE/CV | Population harm MAE/CV | Selected gain MAE/CV | Selected harm MAE/CV | Defined population/selected localities | Selected event ratio >2% / defined | Worst selected event ratio (%) |',
        '|---|---:|---:|---:|---:|---|---|---:|']
    for key, sites in errors.items():
        fields = {}
        for subset in ('population', 'selected'):
            rows = [r[subset]['utility']['mae_over_mean_cv'] for r in sites.values()
                if r[subset]['rows'] and r[subset]['utility']['mae_over_mean_cv'] is not None]
            fields[subset] = rows
        ratios = [100*r['selected']['risk']['actual_mean'][1]/r['selected']['risk']['actual_mean'][0]
            for r in sites.values() if r['selected']['rows'] and r['selected']['risk']['actual_mean'][0] > 0]
        values = [f'{sum(r[i] for r in fields[s])/len(fields[s]):.6f}' if fields[s] else 'undefined'
            for s in ('population', 'selected') for i in (0, 1)]
        worst = f'{max(ratios):.6f}' if ratios else 'undefined'
        lines.append(f"| {key} | {' | '.join(values)} | {len(fields['population'])}/{len(fields['selected'])} | {sum(r>2 for r in ratios)}/{len(ratios)} | {worst} |")
    lines += ['', 'Full per-locality predicted/actual means, biases and errors are in score_errors.json.',
        'This is post-hoc score diagnosis. It neither changes decisions nor certifies future risk.', '']
    (PUBLIC/'score_reliability.md').write_text('\n'.join(lines))
    raw_forecast_audit(a)
    print(json.dumps(dict(views=len(a['views']),comparisons=len(a['small_vs_full']),checkpoint_replays=18,new_training=False)))


if __name__=='__main__':
    main()
