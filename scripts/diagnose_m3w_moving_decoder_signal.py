"""Fitting-only decoder precision/link diagnosis of already frozen cost heads."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as runner
from src.evaluation.m3w_moving_decoder_signal import probe, signal, THRESHOLDS
from src.world_model.m3w_multiaction_cost import select_actions
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_decoder_signal_v1'


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evaluate', 'verify'))
    args = parser.parse_args()
    if not runner.read(runner.PUBLIC/'verification.json')['all36cost_predictions_and_policy_scores_exact']:
        raise ValueError('Require complete original cost study before post-readout diagnosis')
    cfg = runner.read(runner.CONFIG)
    start, guard, beat = runner.runtime(cfg, 'fitting_decoder_signal')
    data, jobs, refs = runner.admission(cfg, beat)
    models = {}
    for ref in runner.read(runner.PUBLIC/'training_freeze.json')['fits']:
        row = runner.read(runner.checked(ref))
        models[row['job']['key'], row['arm']] = runner.checked(row['checkpoint'])
    rows = []
    for job in jobs:
        fitting = runner.view(data, job, refs, training=True)
        fitting.pop('learning', None)
        ids, causal = fitting['ids'], fitting['causal']
        for arm in cfg['arms']:
            state = runner.api.base.base.read_checkpoint(models[job['key'], arm])
            diagnostic = probe(state, causal)
            original, _ = select_actions(diagnostic['float32'], causal['allowed'], arm='physical_neural')
            precise, _ = select_actions(diagnostic['same_float32_inputs_float64'], causal['allowed'], arm='physical_neural')
            # Labels are supplied only after both frozen causal choice vectors.
            target = supported_bound_targets(fitting['paths'],
                data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids],
                job['floor_index'], job['easy_cut'])
            scores = signal(diagnostic['raw_action6'], diagnostic['float32'],
                diagnostic['same_float32_inputs_float64'], target['moments'], data['valid'][ids].any(1), original)
            rows.append(dict(job=job['key'], arm=arm, role='critic_fitting_only', rows=len(ids),
                checkpoint_sha256=runner.sha(models[job['key'], arm]),
                float32_float64_choice_disagreements=int((original != precise).sum()), scores=scores))
            del state, diagnostic, target
        del fitting, causal
        guard()
        beat(state='decoder_fitting_job_complete', job=job['key'], views=len(rows))
    value = dict(result_source='fresh_post_readout_fitting_only_diagnostic_not_independent_confirmation',
        parent_training_sha256=runner.sha(runner.PUBLIC/'training_freeze.json'),
        parent_verification_sha256=runner.sha(runner.PUBLIC/'verification.json'),
        bindings={str(p.relative_to(ROOT)):runner.sha(p) for p in (Path(__file__),
            ROOT/'src/evaluation/m3w_moving_decoder_signal.py', ROOT/'tests/test_m3w_moving_decoder_signal.py')},
        thresholds=THRESHOLDS, rows=rows, new_training=False,
        held_role_metrics_computed=False, independent_roles_opened=False,
        new_policy_or_threshold_selection=False, stage5c_executed=False, smc_enabled=False)
    runner.once(PUBLIC/'analysis.json', runner.stable(value))
    runner.once(PUBLIC/(args.phase+'_execution.json'), dict(pid=runner.os.getpid(),
        seconds=time.monotonic()-start, peak_RSS_bytes=runner.resource.getrusage(runner.resource.RUSAGE_SELF).ru_maxrss,
        all_saved_scalars_exact=args.phase == 'verify', real_training=False))
    beat(state='decoder_signal_complete', replay=args.phase == 'verify')


def main():
    with (runner.PRIVATE/'process.lock').open('a') as lock:
        runner.fcntl.flock(lock, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
        run()


if __name__ == '__main__':
    main()
