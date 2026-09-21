"""Summarize fixed-endpoint training receipts without reading held predictions."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Existing training summary differs: '+str(path))
    else:
        path.write_text(text)


def main():
    cfg = json.loads((ROOT/'configs/m3w_native_forecast_v1.json').read_text())
    private, public = ROOT/cfg['output'], ROOT/cfg['reports']
    identity = json.loads((private/'identity.json').read_text())
    for relative, sha in identity['source_bindings'].items():
        if digest(ROOT/relative) != sha:
            raise ValueError('Changed training dependency: '+relative)
    rows, traces, hashes = [], [], {}
    for site in cfg['sites']:
        for objective in cfg['objectives']:
            for seed in cfg['seeds']:
                trial = f'{site}_{objective}_seed{seed}'
                path = private/'trials'/trial/'complete.json'
                receipt = json.loads(path.read_text())
                fit, spec = receipt['fit'], receipt['identity']
                if (spec['identity'] != identity or spec['held_site'] != site
                        or spec['objective'] != objective or spec['seed'] != seed
                        or not fit['complete'] or fit['step'] != cfg['training']['steps']
                        or fit['held_rows_sampled'] != 0
                        or digest(ROOT/receipt['checkpoint']) != receipt['checkpoint_sha256']):
                    raise ValueError('Invalid completed endpoint: '+trial)
                trace = fit['losses']
                expected = [1]+list(range(cfg['training']['heartbeat_every'],
                                         cfg['training']['steps']+1, cfg['training']['heartbeat_every']))
                if [r['step'] for r in trace] != expected:
                    raise ValueError('Incomplete or duplicated loss trace: '+trial)
                if fit['total_draws'] != fit['step']*cfg['training']['batch_size']:
                    raise ValueError('Incorrect total optimizer draws')
                eligible = sum(v['rows'] for v in spec['normalizers'].values())
                rows.append(dict(trial=trial, held_site=site, objective=objective, seed=seed,
                    steps=fit['step'], fit_seconds=fit['seconds'], parameters=fit['parameters'],
                    training_eligible_rows=eligible, unique_training_rows=fit['unique_training_rows'],
                    unique_eligible_fraction=fit['unique_training_rows']/eligible,
                    total_draws=fit['total_draws'], held_training_draws=fit['held_rows_sampled'],
                    first_ten_logged_batch_loss_mean=statistics.mean(r['loss'] for r in trace[:10]),
                    last_ten_logged_batch_loss_mean=statistics.mean(r['loss'] for r in trace[-10:])))
                traces.extend(dict(trial=trial, held_site=site, objective=objective, seed=seed, **r) for r in trace)
                hashes[str(path.relative_to(ROOT))] = digest(path)
    result = dict(result_source='fresh_run_verified_training_receipt_reduction',
        config_sha256=digest(ROOT/'configs/m3w_native_forecast_v1.json'),
        implementation_sha256=digest(Path(__file__)), receipt_hashes=hashes,
        completed_fits=len(rows), optimizer_updates=sum(r['steps'] for r in rows),
        optimizer_row_draws=sum(r['total_draws'] for r in rows),
        summed_fit_seconds=sum(r['fit_seconds'] for r in rows),
        held_training_draws=sum(r['held_training_draws'] for r in rows),
        training_only=True, held_predictions_read=False,
        loss_scope='logged_training_minibatches_not_full_epoch_or_validation_loss', trials=rows)
    write_once(public/'training_diagnostics.json', json.dumps(result, indent=2, allow_nan=False)+'\n')
    csv_stream = io.StringIO(newline='')
    writer = csv.DictWriter(csv_stream, fieldnames=list(traces[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(traces)
    write_once(public/'training_loss.csv', csv_stream.getvalue())
    lines = ['# Real Training Loss and Sampling Coverage', '',
        'All registered endpoints are verified before this reduction. No held predictions are read.',
        'The trace contains sampled training minibatches, not epoch loss or validation performance.',
        'Old and native objectives assign different sample weights; their loss values are not directly comparable.',
        'Full-population sampling means every eligible training row can be drawn, not that every row was drawn.', '',
        f"Completed fits: {len(rows)}; optimizer updates: {result['optimizer_updates']:,}; row draws: {result['optimizer_row_draws']:,}.",
        f"Summed fit time: {result['summed_fit_seconds']/60:.2f} minutes, excluding setup and later evaluation.", '',
        '| Held Site | Objective | Seed | Unique / Eligible Training Rows | Coverage (%) | Early / Late Logged Batch Loss |',
        '| --- | --- | ---: | --- | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['held_site']} | {r['objective']} | {r['seed']} | {r['unique_training_rows']:,} / {r['training_eligible_rows']:,} | {100*r['unique_eligible_fraction']:.2f} | {r['first_ten_logged_batch_loss_mean']:.5f} / {r['last_ten_logged_batch_loss_mean']:.5f} |")
    lines += ['', 'Early/late values average the first/last ten logged minibatches. They are not evidence of held-scene improvement.',
              'Across-fit draw counts are not independent sample counts. All training held-site draw counts are zero.',
              'Checkpoint/sampler replay and final held-source metrics are separate verification steps.', '']
    write_once(public/'training_diagnostics.md', '\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials', 'receipt_hashes')}, indent=2))


if __name__ == '__main__':
    main()
