"""Verify saved recovery states while retaining the extra stop-point log."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_producer_crossfit as run
from scripts.run_m3w_moving_source_forecast import exact


def compare(a, b):
    if a.keys() != b.keys():
        raise ValueError('Different checkpoint fields')
    exact({k: v for k, v in a.items() if k not in ('seconds', 'losses')},
          {k: v for k, v in b.items() if k not in ('seconds', 'losses')})
    direct = {v['step']: v for v in b['losses']}
    resumed = {v['step']: v for v in a['losses']}
    if len(direct) != len(b['losses']) or len(resumed) != len(a['losses']):
        raise ValueError('Repeated log step')
    if not set(direct) <= set(resumed):
        raise ValueError('Missing direct trace steps')
    for step in direct:
        exact(resumed[step], direct[step])
    return sorted(set(resumed)-set(direct))


def main():
    import fcntl
    import os
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, reg = run.register()
        run.require_committed()
        beat = run.runtime(cfg, 'pilot_recovery_verification')
        data = run.load_data()
        entries = run.designs(data, reg)
        entry = entries[0]
        home = run.PRIVATE/'pilot'
        a = run.torch.load(home/'resumed/checkpoint.pt', map_location='cpu', weights_only=False)
        b = run.torch.load(home/'direct/checkpoint.pt', map_location='cpu', weights_only=False)
        extra = compare(a, b)
        if extra != [cfg['pilot_split']] or a['step'] != cfg['pilot_total'] or a['identity'] != entry[3]:
            raise ValueError('Unexpected pilot or logging difference')
        model = run.SourceForecaster(reg['architecture'], entry[1]['baseline_index'])
        ids = entry[1]['train_ids'][:16]
        model.load_state_dict(a['model'])
        pa = run.predict(model, data, ids, 16)
        model.load_state_dict(b['model'])
        pb = run.predict(model, data, ids, 16)
        run.np.testing.assert_array_equal(pa, pb)
        if not run.np.isfinite(pa).all():
            raise ValueError('Nonfinite recovery outputs')
        run.once(run.PUBLIC/'pilot.json', run.stable(dict(real_Torch_CPU_training=True,
            pid=os.getpid(), resume_exact=True, update=cfg['pilot_total'], new_outer_readout=False,
            seconds=a['seconds'], estimate_six_full_fit_seconds=a['seconds']/cfg['pilot_total']*24000,
            pilot_not_scientific_success=True, prefix=run.artifact(home/'resumed/checkpoint.pt'),
            direct=run.artifact(home/'direct/checkpoint.pt'), verifier=run.artifact(Path(__file__)),
            exact_fields=[k for k in a if k not in ('seconds', 'losses')],
            common_trace_exact=True, extra_stop_log_steps=extra,
            initial_verifier_exit=1, initial_failure='extra_intermediate_stop_log_not_scientific_state',
            additional_training_updates=0, preserved_original_runner=True)))
        beat(state='recovery_exact_except_expected_extra_stop_log', extra_steps=extra)
        beat(state='phase_complete')


if __name__ == '__main__':
    main()
