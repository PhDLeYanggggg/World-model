"""All fixed contrasts for the strong-base cap auxiliary experiment."""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_strong_cap_auxiliary as run
from scripts.report_m3w_european_cap_auxiliary_cost import aggregates, PRIMARY
from scripts.diagnose_m3w_european_cap_auxiliary_cost import diagnose


def main():
    cfg, _ = run.registration(); run.check_freeze()
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    freeze = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    assert len(rows) == 36 and sum(len(r['folds']) for r in rows) == 144
    result = aggregates(rows, cfg)
    result['gates']['strong_control_reconstructed'] = all(c['tolerance_pass'] for c in freeze['controls'])
    result['control_exact_views'] = sum(c['exact'] for c in freeze['controls'])
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', result)
    fits, endpoints = [], []
    for ref in freeze['receipts']:
        r = json.loads((ROOT/ref['path']).read_text()); h, f = r['identity'], r['fit']
        row = dict(tag=h['input']['tag'], seed=h['seed'], arm=h['arm'], fit=f); fits.append(row)
        endpoints.append(dict(tag=row['tag'], arm=row['arm'], seed=row['seed'], updates=f['step'],
            first_cost_loss=f['trace'][0]['cost_loss'], last_cost_loss=f['trace'][-1]['cost_loss'],
            first_cap_BCE=f['trace'][0]['true_cap_BCE'], last_cap_BCE=f['trace'][-1]['true_cap_BCE'],
            fit_seconds=f['seconds']))
    run.immutable_json(run.PUBLIC/'training_metrics.json', fits)
    run.immutable_json(run.PUBLIC/'fit_held_diagnosis.json', diagnose(rows, fits))
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(heads=len(fits),
        updates=sum(r['fit']['step'] for r in fits), summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),
        unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        known_zero_envelope_draws=sum(r['fit']['zero_envelope_rows_sampled'] for r in fits),
        parameters=sorted(set(r['fit']['parameters'] for r in fits)), threads=4, interop=1, workers=0,
        real_torch=True, remote_jobs_submitted=0, excludes_loading_inference_preflight_and_verification=True))
    with (run.PUBLIC/'training_loss_endpoints.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoints[0])); writer.writeheader(); writer.writerows(endpoints)
    lines = ['# Strong-Base Cap Auxiliary Results', '', '## Material Passport',
        'fresh_run:432 native-Torch heads/864000 updates, original-control reconstruction and144 source-held readouts.',
        'cached_verified:source forecasts, original controls and row-locality-excluded event producers.',
        'not_run:new forecasting, policy evaluation, independent selection/calibration/confirmation.', '',
        '| Contrast / family | Primary positive / negative / overlap / missing | Point range (%) |', '|---|---|---|']
    for key, pairs in result['summary'].items():
        for pair, metrics in pairs.items():
            m = metrics[PRIMARY]
            lines.append(f"| {key}/{pair} | {[m[k] for k in ('positive','negative','overlap','not_estimable')]} | {m['point_range']} |")
    lines += ['', '## All Primary Intervals', '', '| Contrast / family / assignment | Point (%) | 95% locality CI |', '|---|---:|---|']
    for key, pairs in result['contrasts'].items():
        for pair, assignments in pairs.items():
            for assignment, metrics in assignments.items():
                m = metrics[PRIMARY]
                lines.append(f"| {key}/{pair}/{assignment} | {m.get('point','not_estimable')} | {m.get('CI','not_estimable')} |")
    lines += ['', 'Three seeds averaged within locality,3000 paired resamples of four localities per assignment.',
        'Assignments overlap; previously exposed source development. No multiplicity-adjusted or independent claim.',
        'Original native inputs,GELU64,four-cost loss,all-known support and site-balanced sampling are retained.',
        'Only the auxiliary objective changes among new arms. Reference outputs remain frozen for held cost comparisons.',
        'The event is producer-relative cap exceedance,not generic easy membership. Inner/outer producer transport remains.',
        'Full/motion changes forecasts and populations,not a matched feature ablation. No threshold/model selection.',
        'Training diagnosis uses dependent views and is not causal proof. Legacy original fitting subset is unavailable.', '',
        '```json', json.dumps(result['gates'], indent=2), '```', '',
        'Obs8/pred12 annotation steps,detector pixels. No metric/seconds,physical-safety,human-gold,true3D or foundation claims.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=result['gates'], control_exact_views=result['control_exact_views'],
        full={k: v['full'][PRIMARY] for k,v in result['summary'].items()}), indent=2))


if __name__ == '__main__': main()
