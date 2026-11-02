"""Direct full refit of the actually disk-interrupted model; no policy change."""
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_selection_harm_deviance as run


def main():
    cfg, producers, ids = run.admission()
    if not (run.PUBLIC/'training_freeze.json').exists():
        raise ValueError('Complete resumed training required first')
    finish = run.PUBLIC/'resource_resume_verification.json'
    if finish.exists(): raise ValueError('Completed recovery verification immutable')
    beat, guard, receipt = run.runtime(cfg, 'actual_resource_resume_verification')
    row = next(r for r in producers if r['trial'] == 'single1_seed17'); arm = 'easy_deviance'
    source = run.prior.source_inputs(json.loads(run.prior.frozen.REG.read_text()))
    data, _ = run.fitting_data(row, source, ids)
    path = run.fit_path(row, arm, 'resource_direct_refit')
    direct = run.one(data, row, arm, cfg, beat, guard, family='resource_direct_refit', resume=path.exists())
    resumed = run.prior.api.read_checkpoint(run.fit_path(row, arm))
    run.prior.assert_recovery(direct, resumed)
    run.once(finish, dict(**receipt(), real_disk_interruption_2000_to8192_full_state_exact=True,
        scientific_model_optimizer_sampler_draws_trace_and_RNG_exact=True,
        elapsed_seconds_excluded_from_scientific_equality=True,
        direct_checkpoint=run.artifact(path), resumed_checkpoint=run.artifact(run.fit_path(row, arm)),
        additional_reproduction_updates=direct['step'], policy_changed=False, confirmation_access=False))
    print(json.dumps(dict(actual_interruption_full_recovery_exact=True, updates=direct['step'])))


if __name__ == '__main__':
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); main()
