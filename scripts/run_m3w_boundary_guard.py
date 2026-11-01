"""Train matched coherent critics with/without two-sided budget-boundary loss."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_current_bank_deferral as prior
from src.world_model import m3w_boundary_guard as api
from src.evaluation import m3w_boundary_guard as evaluation
from src.evaluation.m3w_metric_bundle import pack

NAME = 'european_boundary_guard_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
CONFIG = ROOT / 'configs/m3w_boundary_guard_v1.json'; REG = PUBLIC / 'registration.json'
sha, once, write = prior.sha, prior.once, prior.write
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_boundary_guard.py', ROOT/'src/evaluation/m3w_boundary_guard.py',
         ROOT/'tests/test_m3w_boundary_guard.py', ROOT/'tests/test_m3w_boundary_guard_evidence.py', PUBLIC/'protocol.md')


def admission(register=False):
    pc, values, fits, own_fitted = prior.admission()
    cfg = json.loads(CONFIG.read_text())
    if (cfg['arms'] != list(api.ARMS) or cfg['seeds'] != [17, 29, 43] or cfg['risk_budget'] != .02
            or cfg['boundary_loss_weight'] != 1. or cfg['normalized_numerical_epsilon'] != api.EPSILON
            or cfg['parameters_per_head'] != 31119
            or cfg['training'] != json.loads(prior.parent.CONFIG.read_text())['training']
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Unchanged capacity, complete budget, 2% contract and source roles required')
    parent_files = {n: sha(prior.PUBLIC/n) for n in ('registration.json', 'training_freeze.json', 'summary.json', 'complete.json', 'verification.json', 'metric_bundle.json', 'guard_support_diagnostic.json')}
    deps = ('src/world_model/m3w_own_decision_cost.py', 'src/world_model/m3w_reference_cost_refit.py',
            'src/world_model/m3w_coupled_benefit_risk.py', 'src/world_model/m3w_current_bank_deferral.py',
            'src/evaluation/m3w_current_guard_support.py', 'scripts/run_m3w_current_bank_deferral.py')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={p: sha(ROOT/p) for p in deps}, parent_files=parent_files,
        contexts=json.loads(prior.REG.read_text())['contexts'], seeds=cfg['seeds'], fits=36,
        parameters_each=31119, primary='boundary_minus_moment_full_completion_lower',
        factor='unit_weight_BCE_of_log_harm_over_detached_predicted_original2percent_reference',
        both_training_query_losses='fixed_original_causal_initial_choices',
        frozen_ranking='complete_current_bank_TDEF_final_scores',
        warm_state='same_completed_own_choice_critic_fresh_AdamW',
        source_role='exposed_SOURCE_fitting_resubstitution_only', source_forecaster_seed=43,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        new_trajectory_training=False, stage5c_executed=False, smc_enabled=False))
    reg = json.loads(REG.read_text())
    if reg['parent_files'] != parent_files:
        raise ValueError('Parent readout changed')
    for rel, digest in dict(reg['bindings'], **reg['dependencies']).items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered boundary definition changed')
        if not register:
            prior.parent.previous.parent.parent.source.frozen.BASE.engine.committed(ROOT/rel)
    if not register:
        prior.parent.previous.parent.parent.source.frozen.BASE.engine.committed(REG)
    score_fitted = {}
    for ref in json.loads((prior.PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Frozen score receipt changed')
        row = json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen ranking weights changed')
        score_fitted[row['context'], row['seed'], row['arm']] = row
    return cfg, values, fits, own_fitted, score_fitted


def runtime(cfg, phase):
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1); start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume the unchanged checkpoint')))
    def guard():
        if time.monotonic() - start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h phase boundary; resume unchanged budget')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('10 GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_checkpoint_cap_bytes']:
            raise OSError('1 GiB private checkpoint cap')
    def beat(**event):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), phase=phase, **event)
        write(PRIVATE/'heartbeat.json', value); print(json.dumps(value), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), threads=4, interop_threads=1, workers=0, resource_probing=False)
    return start, beat, guard


def inputs(values, fits, own_fitted, score_fitted, beat):
    old = prior.parent.aggregate_groups(prior.parent.NAME)
    scored = prior.parent.aggregate_groups(prior.NAME)
    for context, seed, data, ranges, packets, _, _, _ in prior.parent.inputs(values, fits, beat):
        ref = own_fitted[context, seed, 'own_choice']; warm = api.parent.parent.read_checkpoint(ROOT/ref['checkpoint']['path'])
        api.parent.parent.exact(warm['input_hashes'], data['input_hashes'])
        risk = api.predict(warm, data['causal'])
        score_ref = score_fitted[context, seed, 'tdef']; state = api.parent.parent.read_checkpoint(ROOT/score_ref['checkpoint']['path'])
        score_data = prior.api.training_data(data, packets, ranges)
        prior.api.base.exact(state['input_hashes'], score_data['input_hashes'])
        score = prior.api.predict(state, data['causal'])
        for name in packets:
            key = f'{context}_{name}_seed{seed}_fit.json'
            if (prior.api.base.digest(risk[ranges[name]]) != old[key]['prediction_sha256']['own_choice']
                    or prior.api.base.digest(score[ranges[name]]) != scored[key]['score_sha256']['tdef']):
                raise ValueError('Frozen cost or TDEF causal inference changed')
        yield context, seed, data, ranges, packets, warm, ref, risk, score


def checkpoint(context, seed, arm, phase='heads'):
    return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data, warm, ref, context, seed, arm, cfg, phase, beat, guard, **kwargs):
    return api.fit(data, warm=warm, arm=arm, settings=cfg['training'],
        identity=dict(context=context, boundary_arm=arm, registration_sha256=sha(REG), warm_checkpoint=ref['checkpoint']),
        path=checkpoint(context, seed, arm, phase), heartbeat=lambda **event: beat(context=context, seed=seed, **event), guard=guard, **kwargs)


def pilot(resume=False):
    cfg, *args = admission(); start, beat, guard = runtime(cfg, 'pilot')
    if (PUBLIC/'pilot.json').exists():
        raise ValueError('Complete pilot immutable')
    context, seed, data, _, _, warm, ref, _, score = next(inputs(*args, beat))
    states, cells = [], []
    for arm in api.ARMS:
        path = checkpoint(context, seed, arm, 'pilot')
        if not path.exists():
            fit(data, warm, ref, context, seed, arm, cfg, 'pilot', beat, guard, stop_at=50)
        recovered = fit(data, warm, ref, context, seed, arm, cfg, 'pilot', beat, guard, stop_at=100, resume=True)
        other = checkpoint(context, seed, arm, 'pilot_direct')
        direct = fit(data, warm, ref, context, seed, arm, cfg, 'pilot_direct', beat, guard, stop_at=100, resume=resume and other.exists())
        api.assert_replay(recovered, direct)
        left, right = api.predict(recovered, data['causal']), api.predict(direct, data['causal'])
        api.np.testing.assert_array_equal(left, right)
        prior.api.base.exact(prior.api.choices(score, data['causal']['allowed'], left), prior.api.choices(score, data['causal']['allowed'], right))
        if not any(not api.torch.equal(v, direct['initial_model'][k]) for k, v in direct['model'].items()):
            raise ValueError('Real optimizer updates required')
        cells.append(dict(arm=arm, seconds=direct['seconds'], scientific_recovery_exact=True,
            complete_causal_prediction_and_choice_recovery_exact=True, first_trace=direct['trace'][0], final_trace=direct['trace'][-1]))
        states.append(direct)
    api.assert_matched(*states)
    once(PUBLIC/'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_optimizer_updates=400, engineering_pass=True, cells=cells,
        projected_main_fit_seconds=sum(c['seconds'] for c in cells)/100*104160,
        projection_excludes_preparation_refits_and_readout=True, new_policy_outcomes_scored=False, registration_sha256=sha(REG)))
    beat(state='real400update_boundary_pilot_complete')


def train(resume=False):
    cfg, *args = admission(); start, beat, guard = runtime(cfg, 'train')
    if (PUBLIC/'training_freeze.json').exists():
        raise ValueError('Complete training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Actual recovery/resource pilot required')
    refs = []; refit_updates = 0; first = sorted(json.loads(REG.read_text())['contexts'])[0]
    for context, seed, data, _, _, warm, ref, _, _ in inputs(*args, beat):
        states = []
        for arm in api.ARMS:
            path = checkpoint(context, seed, arm)
            state = fit(data, warm, ref, context, seed, arm, cfg, 'heads', beat, guard, resume=resume and path.exists())
            if state['step'] != state['budget'] or state['trainable_parameters'] != 31119 or state['sampler']['counts'].min() < 4:
                raise ValueError('Complete 36-head budget/capacity/coverage required')
            for k in ('draw_hash', 'budget', 'step', 'unknown_rows_drawn', 'row_draws'):
                api.parent.parent.exact(state[k], warm[k])
            if context == first and seed == 17:
                other_path = checkpoint(context, seed, arm, 'full_refit')
                other = fit(data, warm, ref, context, seed, arm, cfg, 'full_refit', beat, guard, resume=resume and other_path.exists())
                api.assert_replay(state, other); refit_updates += state['step']
                once(PUBLIC/f'first_full_refit_{arm}.json', dict(scientific_state_exact=True, updates=state['step'], wall_time_excluded=True,
                    checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(other_path)))
            receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
            once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'], budget=state['budget'], parameters=state['trainable_parameters'],
                checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size), seconds=state['seconds'],
                input_hashes=state['input_hashes'], draw_hash=state['draw_hash'], minimum_query_visits=int(state['sampler']['counts'].min()),
                source_query_counts=state['source_query_counts'], first_trace=state['trace'][0], final_trace=state['trace'][-1],
                unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_supervised_labels=0, row_draws=state['row_draws'],
                warm_checkpoint=ref['checkpoint'], prior_query_draws_exact=True, registration_sha256=sha(REG)))
            refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt))); states.append(state)
            beat(state='boundary_head_complete', fits=len(refs), total=36, context=context, seed=seed, arm=arm)
        api.assert_matched(*states)
    once(PUBLIC/'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=refs, primary_optimizer_updates=sum(json.loads((ROOT/r['path']).read_text())['step'] for r in refs),
        exact_full_refit_updates=refit_updates, registration_sha256=sha(REG)))
    beat(state='all36_boundary_heads_frozen')


def evaluate(verify=False, resume=False):
    cfg, *args = admission(); start, beat, _ = runtime(cfg, 'verify' if verify else 'eval')
    if (PUBLIC/'complete.json').exists() and not verify:
        raise ValueError('Complete readout immutable')
    trained = {}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('New fit receipt changed')
        r = json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/r['checkpoint']['path']) != r['checkpoint']['sha256']:
            raise ValueError('Frozen boundary weights changed')
        trained[r['context'], r['seed'], r['arm']] = r
    if len(trained) != 36:
        raise ValueError('All 36 new heads required')
    old = prior.parent.aggregate_groups(prior.NAME); rows = []; refs = []; ids = set(); records = set()
    for context, seed, data, ranges, packets, _, _, risk, score in inputs(*args, beat):
        states = {a: api.parent.parent.read_checkpoint(ROOT/trained[context, seed, a]['checkpoint']['path']) for a in api.ARMS}
        api.assert_matched(*states.values())
        for state in states.values():
            api.parent.parent.exact(state['input_hashes'], data['input_hashes'])
        predictions = {a: api.predict(s, data['causal']) for a, s in states.items()}
        job = next(j for j in args[0][1][4] if j['key'] == context.split('_controller')[0])
        for name, packet in sorted(packets.items()):
            key = f'{context}_{name}_seed{seed}_fit.json'; path = PUBLIC/'groups'/key; progress = PRIVATE/'progress'/(key+'.pt.gz'); parent_row = old[key]
            part = {a: p[ranges[name]] for a, p in predictions.items()}; selected = {}; ranks = {}
            for arm in api.ARMS:
                selected[arm+'_guarded'], ranks[arm+'_guarded'] = prior.api.choices(score[ranges[name]], packet['allowed'], part[arm])['guarded']
            saved = prior.api.base.read_checkpoint(prior.PRIVATE/'progress'/(key+'.pt.gz'))
            for target, origin in (('frozen_tdef_guarded', 'tdef_guarded'), ('frozen_own_joint', 'frozen_own_joint'), ('tdef_raw', 'tdef_raw')):
                selected[target] = saved['choices'][origin]; ranks[target] = saved['ranks'][origin]
                if prior.api.base.digest(selected[target]) != parent_row['action_sha256'][origin]:
                    raise ValueError('Frozen original action changed')
            feasible = all(prior.api.parent.allocation.base.feasible(part[a], packet['allowed'], selected[a+'_guarded']) for a in api.ARMS)
            if not feasible:
                raise ValueError('Original predicted risk rules must pass before outcome scoring')
            payload = dict(choices=selected, ranks=ranks, prediction_sha256={a: prior.api.base.digest(p) for a, p in part.items()})
            if verify:
                prior.api.base.exact(prior.api.base.read_checkpoint(progress), payload)
            elif not (path.exists() and resume):
                prior.api.base.save_checkpoint(progress, payload)
            result = evaluation.readout(packet, selected, ranks, part, job['design']['easy_cut'], args[0][1][3]['origin'][packet['ids']])
            for target, origin in (('frozen_tdef_guarded', 'tdef_guarded'), ('frozen_own_joint', 'frozen_own_joint'), ('tdef_raw', 'tdef_raw')):
                if result['policies'][target] != parent_row['result']['policies'][origin]:
                    raise ValueError('Original full policy score changed')
                for rec, v in result['recordings'].items():
                    if v[target] != parent_row['result']['recordings'][rec][origin]:
                        raise ValueError('Original record score changed')
            row = dict(context=context, source=name, seed=seed, result=result, matched_training=True, original_control_exact=True,
                guarded_predicted_feasible=feasible, minimum_gradient_query_visits=min(int(s['sampler']['counts'].min()) for s in states.values()),
                ids_sha256=prior.api.base.digest(packet['ids']), prediction_sha256=payload['prediction_sha256'],
                action_sha256={a: prior.api.base.digest(c) for a, c in selected.items()}, registration_sha256=sha(REG),
                training_freeze_sha256=sha(PUBLIC/'training_freeze.json'), parent_complete_sha256=sha(prior.PUBLIC/'complete.json'))
            if verify or (path.exists() and resume):
                if row != json.loads(path.read_text()):
                    raise ValueError('New cost inference and original scores differ on replay')
            else:
                once(path, row)
            rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            ids.update(map(int, packet['ids'])); records.update((name, str(r)) for r in packet['recordings'])
            beat(state='boundary_group_complete', groups=len(rows), total=72, context=context, source=name, seed=seed)
    if len(ids) != 149510 or len(records) != 74:
        raise ValueError('Original full cohort required')
    summary = evaluation.summarize(rows, json.loads(REG.read_text())['contexts'], cfg)
    summary.update(unique_target_rows=len(ids), unique_recordings=len(records))
    receipt = dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()):
            raise ValueError('Canonical summary differs')
        once(PUBLIC/'verification.json', dict(**receipt, all72_new_cost_inference_and_action_scores_exact=True, original_full_and_record_controls_exact=True,
            optimization_replayed=False, summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(**receipt, groups=refs, summary_sha256=sha(PUBLIC/'summary.json'), training_freeze_sha256=sha(PUBLIC/'training_freeze.json')))
        pack(ROOT, PUBLIC)
    beat(state='all72_boundary_verified' if verify else 'all72_boundary_complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'eval', 'verify')); parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    for directory in (PUBLIC, PRIVATE, PRIVATE/'progress'):
        directory.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register': admission(True)
        elif args.phase == 'pilot': pilot(args.resume)
        elif args.phase == 'train': train(args.resume)
        else: evaluate(args.phase == 'verify', args.resume)
