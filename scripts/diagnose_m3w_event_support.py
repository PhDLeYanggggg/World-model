"""Count source-fitting event support before specifying a hurdle-risk refit."""
import json
import os
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_geometric_cost as geo
from scripts.run_m3w_native_forecast import immutable_json

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_hurdle_risk_v1'


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, previous, data, identity, originals = geo.load()
    rows = []
    for name, candidate, fold, seed, design in geo.old.jobs(previous, data, identity['old_identity']):
        a = geo.old.assemble(candidate, fold, seed, design, data, identity['old_identity'])
        ids = design['train_ids']
        cv = np.asarray(data['baseline_ade'][ids, 1])
        harm = a['y'][:, 1]
        d = geo.rollout_envelope(a['b'][ids], a['p'][ids])
        for event in ('all', 'easy'):
            y, pr = geo.old.task_data(a, design, data, event)
            known = pr['known']
            e = known if event == 'all' else known & (cv > 0) & (cv <= design['easy_cut'])
            j = e & (harm > 0)
            for site in sorted(set(data['sites'][ids])):
                s = data['sites'][ids] == site
                n, ne, nh = int((s & known).sum()), int((s & e).sum()), int((s & j).sum())
                ratios = harm[s & j]/d[s & j]
                rows.append(dict(head=name+'_'+event, locality=str(site), known=n,
                    unknown=int((s & ~known).sum()), event=ne, positive_event_harm=nh,
                    zero_reference=int((s & known & (cv == 0)).sum()),
                    expected_event_rows_per_256=256*ne/n,
                    expected_positive_rows_per_256=256*nh/n,
                    harm_probability_given_event=nh/ne if ne else None,
                    positive_harm_envelope_quantiles=np.quantile(ratios,[0,.5,.9,1]).tolist() if nh else None))
        print(json.dumps(dict(group=name, fit_rows=len(ids), held_outcomes_read=False)), flush=True)
    geo.assert_identity(identity)
    immutable_json(PUBLIC/'fitting_support.json', dict(result_source='fresh_run_fitting_only_support_counts',
        identity=identity, rows=rows, held_outcomes_read=False, new_training=False,
        script_sha256=geo.old.digest(Path(__file__))))
    print(json.dumps(dict(slices=len(rows), training_only=True)), flush=True)


if __name__ == '__main__':
    main()
