"""Export registered fixed-batch loss endpoints without reading held labels."""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_exceedance as run
import numpy as np


def main():
    freeze = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    rows = []
    for ref in freeze['receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        d = json.loads((ROOT/ref['path']).read_text()); fit = d['fit']; inp = d['identity']['input']
        assert fit['complete'] and fit['step'] == 2000 and fit['unknown_rows_sampled'] == 0
        trace = fit['trace']; assert trace[0]['step'] == 0 and trace[-1]['step'] == 2000
        rows.append(dict(tag=inp['tag'], arm=d['identity']['arm'], seed=d['identity']['seed'],
                         pair='motion_only' if '_motion_only_' in inp['tag'] else 'full',
                         parameters=fit['parameters'], fitting_prior=d['fitting_prior'],
                         start_BCE=trace[0]['fixed_BCE'], end_BCE=trace[-1]['fixed_BCE'],
                         start_Brier=trace[0]['fixed_Brier'], end_Brier=trace[-1]['fixed_Brier'],
                         fit_seconds=fit['seconds']))
    assert len(rows) == 288
    with (run.PUBLIC/'training_loss_endpoints.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for pair in ['full', 'motion_only']:
        for arm in ['linear', 'mlp']:
            q = [r for r in rows if (r['pair'], r['arm']) == (pair, arm)]
            summary[pair+'__'+arm] = dict(heads=len(q), fixed_batch_BCE_decreased=sum(r['end_BCE'] < r['start_BCE'] for r in q),
                median_start_BCE=float(np.median([r['start_BCE'] for r in q])),
                median_end_BCE=float(np.median([r['end_BCE'] for r in q])),
                summed_fit_seconds=sum(r['fit_seconds'] for r in q))
    run.immutable_json(run.PUBLIC/'training_summary.json', dict(heads=len(rows), effective_updates=576000,
        unknown_rows_sampled=0, summary=summary, fixed_batch_losses_not_generalization=True,
        held_labels_read=False, loss_endpoints=run.artifact(run.PUBLIC/'training_loss_endpoints.csv')))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
