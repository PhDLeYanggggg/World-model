"""Registered-protocol endpoint diagnostic; never a checkpoint selection signal."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_prefix_actor as run


def calculate():
    reg = run.read(run.REG); run.read(run.PUBLIC/'forecast_freeze.json')
    data = run.source.load_data(); result = []
    for item in reg['controls']:
        key = item['job']['key']
        with run.np.load(run.checked(item['forecast']), allow_pickle=False) as z:
            ix = z['local_ids']; old = z['relative_prediction'].astype(float)
            run.np.testing.assert_array_equal(z['global_ids'], data['global_ids'][ix])
        ref = run.read(run.PUBLIC/'forecasts'/f'{key}.json')
        with run.np.load(run.checked(ref['forecast']), allow_pickle=False) as z:
            new = z['prediction'].astype(float)
            run.np.testing.assert_array_equal(z['ids'], data['global_ids'][ix])
        floor = run.baseline_numpy(data['history'][ix], 1)-data['origin'][ix, None]
        target = data['target_eval'][ix]-data['origin'][ix, None]
        known = data['valid'][ix, -1]
        errors = [run.np.linalg.norm(v[:, -1]-run.np.where(known[:, None], target[:, -1], 0), axis=1) for v in (old, new, floor)]
        for site in item['job']['donor_sites']:
            at = data['sites'][ix] == site; support = at&known; q = data['queries'][ix][at]
            _, inv, counts = run.np.unique(q, return_inverse=True, return_counts=True)
            w = run.np.zeros(len(ix)); w[at] = 1./counts[inv]; w /= w.sum()
            denominator = float(w[support]@errors[2][support])
            def pct(value):
                return 100*float(value)/denominator if denominator > 0 else None
            result.append(dict(source=site, rows=int(at.sum()), endpoint_label_rows=int(support.sum()),
                endpoint_unknown_rows=int((at&~known).sum()),
                control_endpoint_FDE_gain_percent=pct(w[support]@(errors[2][support]-errors[0][support])),
                treatment_endpoint_FDE_gain_percent=pct(w[support]@(errors[2][support]-errors[1][support])),
                paired_endpoint_FDE_pp=pct(w[support]@(errors[0][support]-errors[1][support]))))
    return dict(result_source='fresh_run', sources=result, available_endpoint_only=True,
        no_checkpoint_or_policy_selection=True, primary_unchanged=True)


def main():
    value = calculate(); path = run.PUBLIC/'endpoint_diagnostic.json'
    if path.exists():
        run.exact(run.read(path), value)
    else:
        run.once(path, value)
    print('Endpoint diagnostic complete; unknown endpoints retained and never zero-labelled.')


if __name__ == '__main__':
    main()
