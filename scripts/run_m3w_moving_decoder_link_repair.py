"""Complete matched warm-continuation control; original tasks/risks unchanged."""
import argparse
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as source
from src.world_model import m3w_moving_decoder_link_repair as api

np = source.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_decoder_link_repair_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_decoder_link_repair_v1'
PARENT_PUBLIC, PARENT_PRIVATE = source.PUBLIC, source.PRIVATE
CONFIG = ROOT/'configs/m3w_moving_decoder_link_repair_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'src/world_model/m3w_moving_decoder_link_repair.py',
    ROOT/'tests/test_m3w_moving_decoder_link_repair.py')
POLICIES = ('floor', 'uncontrolled_neural', 'independent', 'frame_uniform', 'joint_matched', 'joint_free')
MATCHED = ('initial_model', 'initial_optimizer', 'parameters', 'budget', 'sampler',
    'draw_hash', 'fit_ids', 'preprocess', 'parent_query_counts', 'torch_rng')


def register(cfg, beat):
    if (cfg['arms'] != list(api.ARMS) or cfg['additional_updates_each'] != 2000
            or cfg['auxiliary_coefficient'] != 1. or cfg['new_continuations'] != 36
            or cfg['parent_feature_arm'] != 'local_history'
            or any(cfg[k] for k in ('new_forecast_training', 'threshold_search',
                'calibration_opened', 'confirmation_opened', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Keep one fixed objective change and original task/risk/source roles')
    if not source.read(PARENT_PUBLIC/'verification.json')['all36cost_predictions_and_policy_scores_exact']:
        raise ValueError('Complete verified parent policy study required')
    signal = ROOT/'outputs/publication_readiness_2026_09/moving_decoder_signal_v1'
    if not source.read(signal/'verify_execution.json')['all_saved_scalars_exact']:
        raise ValueError('Finish the fitting-only mechanism check first')
    old_cfg = source.read(source.CONFIG)
    for k,v in cfg['training'].items():
        if old_cfg['training'][k] != v:
            raise ValueError('No optimizer/capacity/sampler change with the objective control')
    data, jobs, refs = source.admission(old_cfg, beat)
    warm = {}
    for ref in source.read(PARENT_PUBLIC/'training_freeze.json')['fits']:
        receipt = source.read(source.checked(ref))
        if receipt['arm'] == 'local_history':
            source.checked(receipt['checkpoint'])
            warm[receipt['job']['key']] = receipt['checkpoint']
    if len(warm) != 18:
        raise ValueError('All18frozen source-disjoint parents required')
    value = source.stable(dict(configuration=cfg, bindings={str(p.relative_to(ROOT)):source.sha(p) for p in FILES},
        parent_registration=source.artifact(source.REG),
        parent_training=source.artifact(PARENT_PUBLIC/'training_freeze.json'),
        parent_verification=source.artifact(PARENT_PUBLIC/'verification.json'),
        fitting_signal=source.artifact(signal/'analysis.json'),
        fitting_signal_replay=source.artifact(signal/'verify_execution.json'),
        jobs=jobs, warm=warm, new_primary_updates=72000, parent_updates_not_recounted=True,
        exposed_source_development_not_independent_test=True,
        original24group_primary='not_estimable_not_passed',
        unchanged_original2percent_risks=True, stage5c_executed=False, smc_enabled=False))
    source.once(REG, value)
    return data, jobs, refs, warm


def check_committed():
    for path in (*FILES, REG):
        value = subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(value).hexdigest() != source.sha(path):
            raise ValueError('Commit the complete definition before fitting')


def state_path(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def fit(data, job, arm, cfg, warm, beat, guard, *, kind='fits', resume=False, stop_at=None):
    parent = api.base.base.read_checkpoint(source.checked(warm[job['key']]))
    return api.fit(data['learning'], parent, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job, warm=warm[job['key']]),
        path=state_path(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v:beat(job=job['key'], **v))


def pilot(data, jobs, refs, warm, cfg, beat, guard, resume):
    cells = []
    for fold in range(3):
        job = next(j for j in jobs if j['critic_fold'] == fold)
        fitting = source.view(data, job, refs, training=True)
        states = []
        for arm in api.ARMS:
            path = state_path(job, arm, 'pilot_resume')
            if path.exists() and not resume:
                raise ValueError('Explicit pilot resume required')
            step = api.base.base.read_checkpoint(path)['step'] if path.exists() else 0
            if step < 50:
                fit(fitting, job, arm, cfg, warm, beat, guard, kind='pilot_resume',
                    resume=path.exists(), stop_at=50)
            a = fit(fitting, job, arm, cfg, warm, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
            b = fit(fitting, job, arm, cfg, warm, beat, guard, kind='pilot_direct',
                resume=resume and state_path(job, arm, 'pilot_direct').exists(), stop_at=100)
            source.recovered(a, b)
            states.append(a)
            cells.append(dict(job=job['key'], arm=arm, step=a['step'], seconds=a['seconds'],
                full_parent_plus_additional_step=a['total_parent_plus_new_updates'], exact_resume=True))
        for key in MATCHED:
            api.base.base.exact(states[0][key], states[1][key])
        del fitting, states, a, b
        gc.collect()
    source.once(PUBLIC/'pilot.json', source.stable(dict(cells=cells, real_Torch_training=True,
        full36fit_seconds_projection_not_actual=120*sum(r['seconds'] for r in cells), held_policy_scores_read=False)))


def train(data, jobs, refs, warm, cfg, beat, guard, resume):
    source.read(PUBLIC/'pilot.json')
    receipts = []
    for job in jobs:
        fitting = source.view(data, job, refs, training=True)
        states = []
        for arm in api.ARMS:
            state = fit(fitting, job, arm, cfg, warm, beat, guard,
                resume=resume and state_path(job, arm).exists())
            if state['step'] != 2000:
                raise ValueError('Complete additional2kbudget required')
            if job == jobs[0]:
                replay = fit(fitting, job, arm, cfg, warm, beat, guard, kind='full_refit',
                    resume=resume and state_path(job, arm, 'full_refit').exists())
                source.recovered(state, replay)
                source.once(PUBLIC/('full_refit_'+arm+'.json'), dict(scientific_state_exact=True,
                    additional_updates=2000, checkpoint=source.artifact(state_path(job, arm, 'full_refit'))))
                del replay
            path = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(path, source.stable(dict(job=job, arm=arm, checkpoint=source.artifact(state_path(job, arm)),
                additional_updates=state['step'], total_parent_plus_new_updates=4000,
                seconds=state['seconds'], parameters=state['parameters'], draw_hash=state['draw_hash'],
                minimum_additional_query_visits=int((state['sampler']['counts']-state['parent_query_counts']).min()),
                initial_trace=state['trace'][0], final_trace=state['trace'][-1])))
            receipts.append(source.artifact(path))
            states.append(state)
            beat(state='continuation_complete', job=job['key'], arm=arm, completed=len(receipts), total=36)
        for key in MATCHED:
            api.base.base.exact(states[0][key], states[1][key])
        del fitting, states, state
        gc.collect()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_continuations=36,
        new_primary_updates=72000, warm_updates_not_recounted=True, all18pairs_exact=True,
        independent_roles_opened=False, real_Torch_training=True))


def evaluate(data, jobs, refs, warm, cfg, beat, guard, verify=False):
    frozen = source.read(PUBLIC/'training_freeze.json')
    if frozen['complete_continuations'] != 36 or not frozen['all18pairs_exact']:
        raise ValueError('Finish all matched continuations before reading held policies')
    fits = {}
    for ref in frozen['fits']:
        row = source.read(source.checked(ref))
        fits[row['job']['key'], row['arm']] = row
    records, receipts = [], []
    arms = ('frozen_parent', *api.ARMS)
    for job in jobs:
        held = source.view(data, job, refs)
        ids, causal, paths = held['ids'], held['causal'], held['paths']
        floor = paths[:, job['floor_index']]
        choices, policies, hashes = {}, {}, {}
        for arm in arms:
            ref = warm[job['key']] if arm == 'frozen_parent' else fits[job['key'], arm]['checkpoint']
            state = api.base.base.read_checkpoint(source.checked(ref))
            predicted = source.api.predict(state, causal)
            hashes[arm] = source.array_hash(predicted)
            if arm == 'frozen_parent':
                source.PRIVATE = PARENT_PRIVATE
                try:
                    choice, _ = source.allocated(predicted, causal, held, job, 'local_history', cfg, beat, verify=True)
                finally:
                    source.PRIVATE = PRIVATE
            else:
                choice, _ = source.allocated(predicted, causal, held, job, arm, cfg, beat, verify=verify)
            choices[arm] = choice
            policies[arm] = source.policy_paths(paths, floor, choice)
            del predicted, state
        rows = []
        for site in sorted(set(held['sites'])):
            take = held['sites'] == site
            target = data['target_eval'][ids[take]]-data['origin'][ids[take], None]
            valid = data['valid'][ids[take]]
            metrics = {arm:{key:source.readout.policy(path[take], floor[take], paths[take, 1], target,
                valid, held['recordings'][take], easy_cut=job['easy_cut'], hard_cut=job['hard_cut'])
                for key,path in policies[arm].items()} for arm in arms}
            paired = {key:source.readout.paired(policies['continue_original'][key][take],
                policies['decoder_auxiliary'][key][take], floor[take], target, valid) for key in POLICIES}
            groups = source.query_groups(held['recordings'][take], held['frames'][take], ids[take])
            utility = {arm:source.matched_utility(policies[arm]['independent'][take], policies[arm]['joint_matched'][take],
                floor[take], target, valid, choices[arm]['independent'][take], choices[arm]['joint_matched'][take], groups)
                for arm in arms}
            rows.append(dict(site=str(site), job=job['key'], producer_fold=job['producer_fold'], seed=job['seed'],
                policies=metrics, paired_auxiliary_minus_continuation=paired, matched_utility=utility))
        path = PUBLIC/'groups'/(job['key']+'.json')
        source.once(path, source.stable(dict(job=job, rows=rows, cost_hashes=hashes)))
        records.extend(rows)
        receipts.append(source.artifact(path))
        beat(state='held_job_replayed' if verify else 'held_job_evaluated', job=job['key'], groups=len(records))
        del held, causal, policies, paths, choices
        gc.collect()
        guard()
    sites = sorted(set(data['sites']))
    expected = {(site,p,s) for site in sites for p in range(3) for s in cfg['seeds']
        if p != next(j['evaluation_fold'] for j in jobs if site in j['evaluation_sites'])}
    if len(records) != 138 or {(r['site'],r['producer_fold'],r['seed']) for r in records} != expected:
        raise ValueError('All23supported groups need both excluded producers and all3seeds')
    interval = lambda getter:source.group_interval(records, getter, sites, cfg)
    contrasts = {key:{metric:interval(lambda r,k=key,m=metric:r['paired_auxiliary_minus_continuation'][k][m])
        for metric in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for key in POLICIES}
    safety, accuracy, lower = {}, {}, {}
    for arm in arms:
        safety[arm], accuracy[arm], lower[arm] = {}, {}, {}
        for key in POLICIES:
            accuracy[arm][key] = interval(lambda r,a=arm,k=key:r['policies'][a][k]['known_ADE_gain_floor_percent'])
            lower[arm][key] = interval(lambda r,a=arm,k=key:r['policies'][a][k]['completion_lower_gain_floor_percent'])
            rs = [v for r in records for v in r['policies'][arm][key]['recordings'].values()]
            easy = [r['known_easy_net_degradation'] for r in rs if r['known_easy_net_degradation'] is not None]
            safety[arm][key] = dict(required_record_views=len(rs), supported_record_views=sum(v['supported'] for v in rs),
                known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk']>.02 for v in rs),
                known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk']>.02 for v in rs),
                undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in rs),
                worst_known_easy_net_degradation_percent=100*max(easy) if easy else None,
                full_original_risk_pass=bool(rs and all(v['supported'] for v in rs)))
    summary = source.stable(dict(original24group_primary='not_estimable_not_passed',
        supported23group_development_secondary=contrasts['joint_free']['completion_lower_pp'],
        paired_auxiliary_minus_continuation=contrasts, safety=safety, known_accuracy=accuracy, completion_lower=lower,
        actual_seeds=cfg['seeds'], group_receipts=receipts, source_views=138,
        held_outcomes_previously_exposed=True, independent_confirmation=False, new_deployment=False,
        stage5c_executed=False, smc_enabled=False))
    source.once(PUBLIC/'summary.json', summary)
    if verify:
        source.once(PUBLIC/'verification.json', dict(all36fresh_cost_predictions_and_policy_scalars_exact=True,
            saved_solver_bindings_and_original_predicted_budgets_verified=True,
            new_bitwise_MILP_reoptimization_not_claimed=True, summary_sha256=source.sha(PUBLIC/'summary.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source.PRIVATE = PRIVATE
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg = source.read(CONFIG)
        start, guard, beat = source.runtime(cfg, args.phase)
        data, jobs, refs, warm = register(cfg, beat)
        if args.phase == 'register':
            return
        check_committed()
        if (PUBLIC/(args.phase+'_execution.json')).exists():
            raise ValueError('Completed phase is immutable')
        if args.phase == 'pilot':
            pilot(data, jobs, refs, warm, cfg, beat, guard, args.resume)
        elif args.phase == 'train':
            train(data, jobs, refs, warm, cfg, beat, guard, args.resume)
        else:
            evaluate(data, jobs, refs, warm, cfg, beat, guard, verify=args.phase == 'verify')
        source.once(PUBLIC/(args.phase+'_execution.json'), dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_Torch_training=args.phase in ('pilot', 'train'), independent_roles_opened=False,
            stage5c_executed=False, smc_enabled=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
