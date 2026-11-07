"""Post-hoc fixed-speed heading contrast on the same exposed development roster."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_damped_motion_capacity as motion
from src.evaluation.m3w_damped_motion_options import fixed_recipient_paths, options

run, prev, np = motion.run, motion.prev, motion.np
STRAIGHT, ROTATE = 'fixed_speed_straight', 'fixed_speed_turn'
PAIRS = {
    'turn_vs_fixed_speed': (STRAIGHT, ROTATE, motion.BASE),
    'fixed_speed_vs_gain': (motion.CONTROL, STRAIGHT, motion.BASE),
    'fixed_speed_turn_vs_gain': (motion.CONTROL, ROTATE, motion.BASE),
}


def main():
    cut, _, original = motion.register()
    prior = prev.receipt(motion.PUBLIC, 'replay')
    frozen = prev.receipt(prev.PUBLIC, 'freeze')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    rows, turned = [], 0
    for f in frozen['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choices = z['choice'][0]
        for rec, (ix, _) in records.items():
            floor, base = cv[ix], bank[ix]
            ols = base[:, run.NAMES.index(motion.OLS)]
            s, t, has_speed, has_turn = options(d['history'][ix], ols, floor)
            s, t = s[:, 0], t[:, 0]
            gain = run.model.selected_paths(choices[ix], base, floor)
            speed, _ = fixed_recipient_paths(gain, floor, s, t, has_speed, has_turn)
            switched = (gain != floor).any(axis=(1, 2))
            use = switched & has_turn & (s != floor).any(axis=(1, 2)) & (t != floor).any(axis=(1, 2))
            a = np.where(use[:, None, None], s, speed)
            b = np.where(use[:, None, None], t, speed)
            for p in (a, b):
                np.testing.assert_array_equal((p != floor).any(axis=(1, 2)), switched)
            delta = lambda p: np.diff(p, axis=1, prepend=np.zeros_like(p[:, :1]))
            np.testing.assert_allclose(np.linalg.norm(delta(a), axis=2), np.linalg.norm(delta(b), axis=2),
                                       atol=1e-8, rtol=1e-10)
            turned += int((a != b).any(axis=(1, 2)).sum())
            out = run.parent.readout.score_recording(
                {motion.BASE: floor, motion.CONTROL: gain, STRAIGHT: a, ROTATE: b},
                PAIRS, y[ix], valid[ix], d['queries'][ix], rec, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(out, seed=seed, source=roster[rec]))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    control = grouped['risk'][motion.CONTROL]
    if control != original['grouped']['risk'][motion.CONTROL] or control != prior['past_only']['risk'][motion.CONTROL]:
        raise ValueError('Native gain control did not replay')
    out = motion.PUBLIC/'matched_heading_analysis.json'
    run.once(out, run.stable(dict(result_source='fresh_posthoc_speed_matched_heading_development_diagnostic',
        grouped=grouped, changed_heading_rows=turned, past_only=True, original_risk_unchanged=True,
        external_access=False, independent_confirmation=False, deployment=False)))


if __name__ == '__main__':
    main()
