"""Post-hoc observed ADE attribution at matched learned-motion recipients."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_gated_motion_residual as m
from src.evaluation.m3w_joint_path_audit import bootstrap
from src.evaluation.m3w_selection_forecast_readout import errors

run, prev, np = m.run, m.prev, m.np
SLICES = ('observed_all', 'observed_easy', 'observed_hard', 'observed_other')


def main():
    cut, _ = m.register()
    original = prev.receipt(m.PUBLIC, 'replay')
    forecasts = prev.receipt(m.PUBLIC, 'infer')['forecasts']
    choices = prev.receipt(prev.PUBLIC, 'freeze')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    groups = sorted(set(d['sites']))
    cube = np.zeros((len(groups), len(prev.SEEDS), len(SLICES), 4))
    row_counts = np.zeros(len(SLICES), dtype=int)
    for f in choices['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        preds = {}
        for r in forecasts:
            if r['seed'] == seed:
                with np.load(run.checked(r['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    preds[r['arm']] = z['prediction'].astype(float)
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0]
        for _, (ix, _) in records.items():
            floor, truth, mask = cv[ix], y[ix], valid[ix]
            gain = run.model.selected_paths(selected[ix], bank[ix], floor)
            switched = (gain != floor).any(axis=(1, 2))
            local = [preds[a][ix] for a in m.ARMS]
            if not all((np.linalg.norm(p[switched]-floor[switched], axis=2).max(1) > 1e-6).all() for p in local):
                raise ValueError('This diagnostic requires the verified full matched substitution set')
            _, inv, size = np.unique(d['queries'][ix], return_inverse=True, return_counts=True)
            w = 1./size[inv]
            ref, old, ungated, gated = (errors(p, truth, mask)[0] for p in (floor, gain, *local))
            observed = mask.any(1) & switched
            easy = (ref > 0.) & (ref <= cut['easy'])
            hard = ref > cut['hard']
            g, s = groups.index(d['sites'][ix[0]]), prev.SEEDS.index(seed)
            for j, keep in enumerate((observed, observed & easy, observed & hard, observed & ~easy & ~hard)):
                row_counts[j] += int(keep.sum())
                cube[g, s, j] += [w[keep] @ x[keep] for x in (ref, old-ungated, old-gated, ungated-gated)]
    if row_counts[0] != row_counts[1:].sum():
        raise ValueError('Observed subset partition mismatch')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    report = {}
    for j, name in enumerate(SLICES):
        x = cube[:, :, j]
        denom = x[:, :, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            v = 100.*x[:, :, 1:]/denom[..., None]
        report[name] = dict(observed_switched_rows=int(row_counts[j]),
            missing_source_seed_cells=int((denom <= 0.).sum()),
            within_switched_subset_pp={k: bootstrap(v[:, :, i], draw) for i, k in
                enumerate(('ungated_vs_gain', 'gated_vs_gain', 'gated_vs_ungated'))})
    out = m.PUBLIC/'matched_slice_analysis.json'
    run.once(out, run.stable(dict(slices=report, matched_switches=original['both_head_substitutions'],
        result_source='fresh_posthoc_observed_matched_development_attribution',
        normalization='observed_CV_error_mass_within_same_switched_subset',
        original_policy_or_thresholds_changed=False, external_access=False, independent_confirmation=False)))


if __name__ == '__main__':
    main()
