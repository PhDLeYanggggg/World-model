"""C-triggered loss-mask adaptation before any new held-out outcome evaluation."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_record_excess_cost as teacher
from src.world_model import m3w_adaptive_record_cost as api
from src.world_model.m3w_harm_refinement import select_actions
from src.evaluation.m3w_selected_cost_audit import selected_costs

source = teacher.source
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_adaptive_record_cost_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/moving_adaptive_record_cost_v1'
CONFIG = ROOT / 'configs/m3w_moving_adaptive_record_cost_v1.json'
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC / 'protocol.md', ROOT / 'src/world_model/m3w_adaptive_record_cost.py',
         ROOT / 'tests/test_m3w_adaptive_record_cost.py')


def register(beat):
    spec = source.read(CONFIG)
    if (spec['training'] != source.read(teacher.CONFIG)['training'] or spec['loss_weight'] != 1.
            or spec['positive_harm_budget'] != .02 or spec['easy_net_degradation_budget'] != .02
            or spec['new_primary_neural_cost_heads'] != 18 or spec['primary_optimizer_updates'] != 36000
            or any(spec[k] for k in ('threshold_search', 'new_data_roles', 'new_forecast_training',
                'confirmation_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Change only the selected-action mask in the existing extra loss')
    audit = teacher.PUBLIC / 'fitting_regions'
    decision = source.read(audit / 'analysis.json')
    verified = source.read(audit / 'verification.json')
    if (not decision['C_only_adaptive_followup_trigger'] or not decision['trigger_frozen_before_new_Hprediction']
            or not verified['all72cells_and36C_cost_arrays_exact']
            or verified['analysis_sha256'] != source.sha(audit / 'analysis.json')):
        raise ValueError('Require the exact C-only trigger, never a failed Hscore')
    _, cfg, data, jobs, refs, point_controls = teacher.register(beat)
    frozen = {source.read(source.checked(r))['job']['key']: source.read(source.checked(r))
        for r in source.read(teacher.PUBLIC / 'training_freeze.json')['fits']}
    if len(frozen) != 18:
        raise ValueError('All18actual matched frozen-teacher models required')
    source.once(REG, source.stable(dict(configuration=spec,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_runtime_registration=source.artifact(teacher.RUNTIME_REG),
        parent_freeze=source.artifact(teacher.PUBLIC / 'training_freeze.json'),
        C_trigger=source.artifact(audit / 'analysis.json'), C_trigger_verification=source.artifact(audit / 'verification.json'),
        controls={k: r['checkpoint'] for k, r in frozen.items()}, jobs=jobs, forecasts=refs,
        C_only_trigger_no_H_model_selection=True, new_neural_cost_training=True, new_trajectory_training=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False)))
    return spec, cfg, data, jobs, refs, point_controls, frozen


def checkpoint(job, arm, kind='fits'):
    return PRIVATE / kind / (job['key'] + '_' + arm + '.pt.gz')


def identity(job, arm):
    return dict(registration_sha256=source.sha(REG), job=job, record_loss_arm=arm,
        feature_arm='local_history', amplitude_arm='amplitude_bank',
        amplitude_schema='floor_then_six_fixed_fractions_true_CV_separate')


def one(p, job, arm, spec, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(p['learning'], loss_arm=arm, seed=job['seed'], settings=spec['training'],
        identity=identity(job, arm), path=checkpoint(job, arm, kind), resume=resume, stop_at=stop_at,
        heartbeat=lambda **v: beat(job=job['key'], **v), guard=guard)


def pilot(spec, cfg, data, jobs, refs, controls, frozen, beat, guard, resume):
    cells = []
    for fold in range(3):
        job = next(j for j in jobs if j['critic_fold'] == fold)
        p, _, _ = teacher.fitting(data, job, refs, controls[job['key']])
        for arm in api.ARMS:
            path = checkpoint(job, arm, 'pilot_resume')
            step = api.base.base.read_checkpoint(path)['step'] if path.exists() else 0
            if path.exists() and not resume:
                raise ValueError('Explicit unchanged adaptive pilot resume required')
            if step < spec['pilot_stop']:
                one(p, job, arm, spec, beat, guard, kind='pilot_resume', resume=path.exists(), stop_at=spec['pilot_stop'])
            a = one(p, job, arm, spec, beat, guard, kind='pilot_resume', resume=True, stop_at=spec['pilot_target'])
            path = checkpoint(job, arm, 'pilot_direct')
            b = one(p, job, arm, spec, beat, guard, kind='pilot_direct', resume=resume and path.exists(), stop_at=spec['pilot_target'])
            api.exact_scientific_state(a, b)
            if a['selection_draw_hash'] != b['selection_draw_hash']:
                raise ValueError('Adaptive selection stream changed on resume')
            if arm == 'frozen_teacher':
                path = checkpoint(job, arm, 'pilot_frozen_original_loop')
                original = api.frozen.fit(p['learning'], loss_arm='record_excess', seed=job['seed'], settings=spec['training'],
                    identity=identity(job, arm), path=path, resume=resume and path.exists(), stop_at=spec['pilot_target'],
                    heartbeat=lambda **v: beat(job=job['key'], **v), guard=guard)
                api.exact_scientific_state(original, b)
            cells.append(dict(job=job['key'], arm=arm, steps=b['step'], seconds=b['seconds'], budget=b['budget'],
                parameters=b['parameters'], exact_resume=True, selection_stream_exact=True,
                frozen_loop_exact=True if arm == 'frozen_teacher' else None))
        del p, a, b
        source.gc.collect()
        guard()
    projection = 6*sum(r['seconds']/r['steps']*r['budget'] for r in cells if r['arm'] == 'adaptive_selection')
    source.once(PUBLIC / 'pilot.json', source.stable(dict(cells=cells, all6resume_and_selection_streams_exact=True,
        all3frozen_teacher_loop_controls_exact=True, actual_pilot_optimizer_updates=1500,
        projected_primary_fit_seconds_not_completion=projection)))


def train(spec, cfg, data, jobs, refs, controls, frozen, beat, guard, resume):
    if not source.read(PUBLIC / 'pilot.json')['all3frozen_teacher_loop_controls_exact']:
        raise ValueError('Complete matched-loop pilot before primary adaptive fitting')
    receipts = []
    for job in jobs:
        p, _, _ = teacher.fitting(data, job, refs, controls[job['key']])
        old = api.base.base.read_checkpoint(source.checked(frozen[job['key']]['checkpoint']))
        path = checkpoint(job, 'adaptive_selection')
        state = one(p, job, 'adaptive_selection', spec, beat, guard, resume=resume and path.exists())
        for key in ('initial_model', 'parameters', 'budget', 'sampler', 'draw_hash', 'fit_ids', 'preprocess', 'settings', 'seed', 'torch_rng'):
            api.base.base.exact(state[key], old[key])
        if state['step'] != 2000:
            raise ValueError('Keep the complete unchanged update budget')
        if job == jobs[0]:
            path = checkpoint(job, 'frozen_teacher', 'full_refit')
            reference = one(p, job, 'frozen_teacher', spec, beat, guard, kind='full_refit', resume=resume and path.exists())
            api.exact_scientific_state(reference, old)
            path2 = checkpoint(job, 'adaptive_selection', 'full_refit')
            replica = one(p, job, 'adaptive_selection', spec, beat, guard, kind='full_refit', resume=resume and path2.exists())
            api.exact_scientific_state(replica, state)
            if replica['selection_draw_hash'] != state['selection_draw_hash']:
                raise ValueError('Full adaptive selection stream failed exact refit')
            source.once(PUBLIC / 'first_full_refits.json', dict(frozen_teacher_matches_original=True,
                adaptive_refit_and_selection_stream_exact=True, frozen_checkpoint=source.artifact(path), adaptive_checkpoint=source.artifact(path2)))
            del reference, replica
        prediction = api.parent.predict(state, p['causal'])
        frozen_prediction = api.parent.predict(old, p['causal'])
        if source.array_hash(frozen_prediction) != frozen[job['key']]['fitting_predictions_sha256']:
            raise ValueError('Frozen teacher inference changed')
        regions = {name: select_actions(value, p['causal']['allowed'], arm='physical_neural')[0]
            for name, value in (('frozen_teacher', frozen_prediction), ('adaptive_selection', prediction))}
        ids = p['ids']
        truth = api.parent.targets(p, data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids], job['easy_cut'])
        rows = [dict(model=name, region=region, scores=selected_costs(value, truth['moments'], truth['observed_future'], choice, p['causal']['groups']))
            for name, value in (('frozen_teacher', frozen_prediction), ('adaptive_selection', prediction)) for region, choice in regions.items()]
        path = PUBLIC / 'fits' / (job['key'] + '.json')
        source.once(path, source.stable(dict(job=job, checkpoint=source.artifact(checkpoint(job, 'adaptive_selection')),
            cached_matched_control=frozen[job['key']]['checkpoint'], step=state['step'], seconds=state['seconds'], parameters=state['parameters'],
            all_initialization_source_normalization_sampler_budget_bindings_match=True,
            fitting_predictions_sha256=source.array_hash(prediction), trace=state['trace'], selection_draw_hash=state['selection_draw_hash'],
            C_region_cells=rows, new_neural_cost_training=True, new_trajectory_training=False)))
        receipts.append(source.artifact(path))
        beat(state='adaptive_cost_fit_frozen', job=job['key'], complete=len(receipts), total=18)
        del p, old, state, prediction, frozen_prediction, regions, truth, rows
        source.gc.collect()
        guard()
    source.once(PUBLIC / 'training_freeze.json', dict(fits=receipts, complete_new_adaptive_cost_heads=18,
        primary_optimizer_updates=36000, cached_frozen_teacher_controls=18,
        all_adaptive_and_frozen_models_before_new_Hprediction=True, new_trajectory_training=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (teacher.PUBLIC / 'evaluate_execution.json').exists() or any((teacher.PUBLIC / 'held').glob('*.json')):
        raise ValueError('C-driven adaptive study must freeze before new parent Houtcomes')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source.PRIVATE = PRIVATE
    teacher.lineage.PRIVATE = PRIVATE
    with (PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        spec, cfg = source.read(CONFIG), source.read(teacher.PARENT.CONFIG)
        start, guard, beat = teacher.lineage.runtime(cfg, spec, args.phase)
        spec, cfg, data, jobs, refs, controls, frozen = register(beat)
        if args.phase == 'register':
            return
        for path in (*FILES, REG):
            body = source.subprocess.run(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if source.hashlib.sha256(body).hexdigest() != source.sha(path):
                raise ValueError('Commit unchanged adaptive definition before fitting')
        receipt = PUBLIC / (args.phase + '_execution.json')
        if receipt.exists():
            raise ValueError('Completed adaptive phase is immutable')
        if args.phase == 'pilot':
            pilot(spec, cfg, data, jobs, refs, controls, frozen, beat, guard, args.resume)
        else:
            train(spec, cfg, data, jobs, refs, controls, frozen, beat, guard, args.resume)
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_neural_cost_training=True, new_trajectory_training=False, new_H_prediction=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
