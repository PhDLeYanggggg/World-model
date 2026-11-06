"""Actual matched producer-loss repair, restricted to producer-A OOF donors."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_producer_crossfit as source
from src.training import m3w_prefix_forecast as method
from src.world_model.m3w_european_source_forecast import SourceForecaster, baseline_numpy
from src.world_model.m3w_native_forecast import predict
from scripts.run_m3w_moving_source_forecast import exact, same_scientific_checkpoint

np, torch = source.np, source.torch
read, sha, once, stable, artifact, checked = source.read, source.sha, source.once, source.stable, source.artifact, source.checked
NAME = 'moving_prefix_actor_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs'/f'm3w_{NAME}.json'; REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), Path(method.__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'tests/test_m3w_prefix_forecast.py', PUBLIC/'verification_amendment.md')


def register():
    cfg = read(CONFIG); _, parent = source.register()
    preliminary = read(PUBLIC/'registration.json')
    for path in (CONFIG, Path(method.__file__), PUBLIC/'protocol.md'):
        if sha(path) != preliminary['files'][str(path.relative_to(ROOT))]:
            raise ValueError('Verification-only amendment cannot change the model/loss/protocol')
    analysis = read(checked(artifact(ROOT/cfg['parent_analysis'])))
    if (cfg['arms'] != list(method.ARMS) or cfg['early_prefixes'] != list(method.PREFIXES)
            or cfg['full_weight'] != .5 or cfg['early_weight'] != .5 or cfg['seed'] != 17
            or cfg['fresh_full_models'] != 7 or cfg['steps_per_model'] != 4000
            or cfg['expected_OOF_rows'] != 125540 or cfg['original_risk_budget'] != .02
            or cfg['recovery_split'] != 100 or cfg['recovery_total'] != 200
            or any(cfg[k] for k in ('warm_start', 'new_own_B_readout', 'new_outer_C_H_readout',
                'identity_certification', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed actor-loss comparison and original rules required')
    if len(analysis['forecasts']) != 6 or not analysis['teacher_exposure_audit_pass']:
        raise ValueError('Six verified disjoint donor banks required')
    references = []
    for item in analysis['forecasts']:
        checked(item['checkpoint']); checked(item['forecast'])
        references.append(dict(job=item['job'], checkpoint=item['checkpoint'], forecast=item['forecast']))
    cuts_path = ROOT/'outputs/publication_readiness_2026_09/moving_role_separated_cost_v1/registration_v3.json'
    cuts = {}
    for fold in (0, 1):
        jobs = [n['job'] for n in read(cuts_path)['jobs'] if n['job']['producer_fold'] == fold]
        values = {(j['easy_cut'], j['hard_cut']) for j in jobs}
        if len(values) != 1:
            raise ValueError('Frozen producer-level diagnostic cuts required')
        easy, hard = values.pop(); cuts[str(fold)] = dict(easy=easy, hard=hard)
    value = stable(dict(config=cfg, training=parent['training'], architecture=parent['architecture'],
        jobs=parent['jobs'], controls=references, cuts=cuts,
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in
            (Path(source.__file__), ROOT/'src/world_model/m3w_native_forecast.py',
             ROOT/'src/world_model/m3w_european_source_forecast.py', ROOT/'src/world_model/m3w_eqmotion_adapter.py')},
        parents=[artifact(ROOT/cfg[k]) for k in ('parent_registration', 'parent_population', 'parent_analysis', 'diagnosis')],
        diagnostic_cuts=artifact(cuts_path), author_core=parent['author_core'],
        preliminary_registration=artifact(PUBLIC/'registration.json'), verification_amendment_only=True,
        role='producer_A_OOF_component_training_not_independent_confirmation',
        torch=str(torch.__version__), numpy=np.__version__, machine=platform.machine(), original_rules_unchanged=True))
    once(REG, value); return cfg, value


def committed(reg):
    for p in (*FILES, REG):
        content = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
            capture_output=True, check=True).stdout
        if hashlib.sha256(content).hexdigest() != sha(p):
            raise ValueError('Commit exact definitions before real-data fitting')
    for p, digest in reg['dependencies'].items():
        if sha(ROOT/p) != digest:
            raise ValueError('Parent implementation changed')


def runtime(cfg, phase):
    torch.set_num_threads(cfg['torch_threads']); torch.set_num_interop_threads(cfg['interop_threads'])
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume saved atomic state')))
    start = time.monotonic()
    def beat(**values):
        now = time.monotonic()-start
        size = sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
        if now > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('Resume unchanged study after12h boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes'] or size > cfg['artifact_cap_bytes']:
            raise OSError('Keep registered disk reserve/private cap')
        event = dict(phase=phase, pid=os.getpid(), seconds=now,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), private_bytes=size,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **values)
        source.atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), device='cpu', threads=4, interop=1, workers=0, resource_probing=False)
    return start, beat


def entries(data, reg):
    original = source.designs(data, read(ROOT/reg['config']['parent_registration']))
    if len(original) != 6:
        raise ValueError('Full six-model roster required')
    return original


def identity(entry, arm):
    if arm == 'available12':
        return entry[3]
    return stable(dict(parent_identity=entry[3], registration_sha256=sha(REG), arm=arm,
        loss='half_available12_half_mean_supported_prefix1_3_6_else_full'))


def fit_one(data, entry, reg, home, arm, beat, resume=False, stop_at=None):
    torch.manual_seed(17)
    model = SourceForecaster(reg['architecture'], entry[1]['baseline_index'])
    result = method.fit(model, data, entry[1], arm=arm, seed=17, settings=reg['training'],
        identity=identity(entry, arm), directory=home, resume=resume, stop_at=stop_at,
        heartbeat=lambda **x: beat(job=entry[0]['key'], arm=arm, **x))
    state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
    if state['draws'][entry[1]['held_ids']].any():
        raise ValueError('Teacher exposure violation')
    return result, state


def pilot(cfg, reg, data, jobs, beat):
    if (PUBLIC/'pilot.json').exists():
        raise ValueError('Pilot complete; verify cached receipt instead of re-running')
    a, b = PRIVATE/'pilot/resumed', PRIVATE/'pilot/direct'
    if not (a/'checkpoint.pt').exists():
        fit_one(data, jobs[0], reg, a, 'prefix_balanced', beat, stop_at=cfg['recovery_split'])
    result, sa = fit_one(data, jobs[0], reg, a, 'prefix_balanced', beat, True, cfg['recovery_total'])
    _, sb = fit_one(data, jobs[0], reg, b, 'prefix_balanced', beat, (b/'checkpoint.pt').exists(), cfg['recovery_total'])
    same_scientific_checkpoint(sa, sb)
    once(PUBLIC/'pilot.json', stable(dict(result_source='fresh_run', actual_Torch_training=True,
        resume100plus100_equals200=True, exact_model_optimizer_draws_RNG_trace=True,
        parameters=result['parameters'], prefix=artifact(a/'checkpoint.pt'), total_step=200,
        seconds=result['seconds'], projected_six_fit_seconds=result['seconds']/200*24000,
        new_own_B_readout=False, pilot_is_not_scientific_success=True)))


def control(reg, data, jobs, beat, resume):
    if (PUBLIC/'control_refit.json').exists():
        checked(read(PUBLIC/'control_refit.json')['checkpoint']); return
    entry = jobs[0]; home = PRIVATE/'control_refit'
    result, new = fit_one(data, entry, reg, home, 'available12', beat, resume)
    old = torch.load(checked(reg['controls'][0]['checkpoint']), map_location='cpu', weights_only=False)
    if new.keys() != old.keys():
        raise ValueError('Control checkpoint schema changed')
    for k in new:
        if k not in ('seconds', 'losses'):
            exact(new[k], old[k])
    a, b = ({r['step']: r for r in v['losses']} for v in (old, new))
    if set(b)-set(a) or set(a)-set(b)-{20, 40}:
        raise ValueError('Unexpected trace mismatch')
    for k in b:
        exact(a[k], b[k])
    if not result['complete']:
        raise ValueError('Control must finish full4000updates')
    once(PUBLIC/'control_refit.json', stable(dict(result_source='fresh_run', actual_updates=4000,
        all_model_optimizer_sampler_RNG_draw_state_exact=True, common_trace_exact=True,
        original_pilot_only_extra_log_steps=sorted(set(a)-set(b)), checkpoint=artifact(home/'checkpoint.pt'),
        cached_control=reg['controls'][0]['checkpoint'], result=result)))
    beat(state='full_control_exact')


def train(reg, data, jobs, beat, resume):
    pilot_ref = read(PUBLIC/'pilot.json'); control_ref = read(PUBLIC/'control_refit.json')
    checked(pilot_ref['prefix']); checked(control_ref['checkpoint'])
    for i, entry in enumerate(jobs):
        key = entry[0]['key']; dest = PUBLIC/'fits'/f'{key}.json'; home = PRIVATE/'fits'/key
        if dest.exists():
            checked(read(dest)['checkpoint']); beat(state='completed_fit_verified', job=key); continue
        if i == 0 and not (home/'checkpoint.pt').exists():
            home.mkdir(parents=True, exist_ok=True); shutil.copyfile(checked(pilot_ref['prefix']), home/'checkpoint.pt')
        result, state = fit_one(data, entry, reg, home, 'prefix_balanced', beat, resume or i == 0)
        old = torch.load(checked(reg['controls'][i]['checkpoint']), map_location='cpu', weights_only=False)
        for k in ('draws', 'sampler_rng', 'train_ids', 'factors'):
            exact(state[k], old[k])
        if not result['complete'] or state['step'] != 4000:
            raise ValueError('Final registered checkpoint required')
        once(dest, stable(dict(result_source='fresh_run', job=entry[0], checkpoint=artifact(home/'checkpoint.pt'),
            result=result, identical_control_draws_and_factors=True, held_rows_sampled=0,
            identity=identity(entry, 'prefix_balanced'))))
    beat(state='six_treatment_fits_complete', total_final_updates=24000, full_control_refit_updates=4000)


def forecasts(reg, data, jobs, beat, replay=False):
    outputs = []
    for entry in jobs:
        key = entry[0]['key']; receipt = read(PUBLIC/'fits'/f'{key}.json')
        saved = torch.load(checked(receipt['checkpoint']), map_location='cpu', weights_only=False)
        if saved['identity'] != identity(entry, 'prefix_balanced') or saved['step'] != 4000:
            raise ValueError('Registered final predictor only')
        model = SourceForecaster(reg['architecture'], entry[1]['baseline_index']); model.load_state_dict(saved['model'])
        ix = entry[2]; rows = []
        for start in range(0, len(ix), 4096):
            rows.append(predict(model, data, ix[start:start+4096], 128))
            beat(state='predicting_fitting_donors_only', job=key, rows=min(start+4096, len(ix)))
        prediction = np.concatenate(rows)
        if not np.isfinite(prediction).all():
            raise FloatingPointError('Finite forecast required')
        home = PRIVATE/'forecasts'/f'{key}.npz'; out = PUBLIC/'forecasts'/f'{key}.json'
        if home.exists() or out.exists():
            checked(read(out)['forecast'])
            with np.load(home, allow_pickle=False) as old:
                np.testing.assert_array_equal(old['prediction'], prediction)
                np.testing.assert_array_equal(old['ids'], data['global_ids'][ix])
        else:
            if replay:
                raise ValueError('Replay requires completed forecast bank')
            home.parent.mkdir(parents=True, exist_ok=True)
            pending = home.with_suffix('.pending')
            with pending.open('wb') as f:
                np.savez(f, prediction=prediction, ids=data['global_ids'][ix], local_ids=ix)
            pending.replace(home)
            once(out, dict(result_source='fresh_run', job=entry[0], forecast=artifact(home),
                checkpoint=receipt['checkpoint'], rows=len(ix), no_future_target_input=True, teacher_disjoint=True))
        outputs.append(read(out))
    if sum(x['rows'] for x in outputs) != reg['config']['expected_OOF_rows']:
        raise ValueError('Full donor population required')
    if not replay:
        once(PUBLIC/'forecast_freeze.json', dict(banks=outputs, all_predictions_frozen_before_evaluation=True))


def statistics(old, new, floor, target, valid, weights, easy_cut, hard_cut):
    # All inputs have already been generated and frozen; targets are label-only.
    distance = [np.linalg.norm(np.where(valid[..., None], v-target, 0), axis=-1) for v in (old, new, floor)]
    result = []
    cv_full = np.divide(distance[2].sum(1), valid.sum(1), out=np.full(len(valid), np.nan), where=valid.any(1))
    easy = (cv_full > 0)&(cv_full <= easy_cut); hard = cv_full >= hard_cut
    for k in (1, 3, 6, 12):
        count = valid[:, :k].sum(1); known = count > 0
        loss = [np.divide(d[:, :k].sum(1), count, out=np.full(len(valid), np.nan), where=known) for d in distance]
        a, b, cv = loss; w = weights
        den = float(w[known]@cv[known]); ee = known&easy; hh = known&hard
        eden = float(w[ee]@cv[ee]); hden = float(w[hh]@cv[hh])
        delta = float(w[known]@(a[known]-b[known]))
        unknown = float(w[~known]@np.linalg.norm(new-old, axis=-1)[:, :k].max(1)[~known])
        def pct(value, denominator):
            return 100*float(value)/denominator if denominator > 0 else None
        row = dict(prefix=k, rows=len(valid), known_rows=int(known.sum()), wholly_unknown_rows=int((~known).sum()),
            partial_rows=int((known&(count < k)).sum()), complete_rows=int((count == k).sum()),
            paired_known_pp=pct(delta, den), paired_completion_lower_pp=pct(delta-unknown, den),
            paired_completion_upper_pp=pct(delta+unknown, den), known_easy_rows=int(ee.sum()), known_hard_rows=int(hh.sum()))
        for name, cost in (('control', a), ('treatment', b)):
            row[name] = dict(known_gain_percent=pct(w[known]@(cv[known]-cost[known]), den),
                known_positive_harm_percent=pct(w[known]@np.maximum(cost[known]-cv[known], 0), den),
                known_easy_positive_harm_percent=pct(w[ee]@np.maximum(cost[ee]-cv[ee], 0), eden),
                known_easy_signed_degradation_percent=pct(w[ee]@(cost[ee]-cv[ee]), eden),
                known_hard_gain_percent=pct(w[hh]@(cv[hh]-cost[hh]), hden))
        result.append(row)
    return result


def evaluate(reg, data, jobs, beat, replay=False):
    read(PUBLIC/'forecast_freeze.json'); sources = []; recordings = []
    for i, entry in enumerate(jobs):
        key = entry[0]['key']; ix = entry[2]
        with np.load(checked(reg['controls'][i]['forecast']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['global_ids'], data['global_ids'][ix]); old = z['relative_prediction'].astype(float)
        ref = read(PUBLIC/'forecasts'/f'{key}.json')
        with np.load(checked(ref['forecast']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], data['global_ids'][ix]); new = z['prediction'].astype(float)
        floor = baseline_numpy(data['history'][ix], 1)-data['origin'][ix, None]
        y = data['target_eval'][ix]-data['origin'][ix, None]; mask = data['valid'][ix]
        cut = reg['cuts'][str(entry[0]['producer_fold'])]
        for grouping, dest, name in ((data['sites'][ix], sources, 'source'), (data['recordings'][ix], recordings, 'recording')):
            for group in sorted(set(grouping)):
                at = grouping == group; q = data['queries'][ix][at]
                _, inverse, count = np.unique(q, return_inverse=True, return_counts=True)
                weight = 1./count[inverse]; weight /= weight.sum()
                rows = statistics(old[at], new[at], floor[at], y[at], mask[at], weight, cut['easy'], cut['hard'])
                dest.append(dict(job=key, producer_fold=entry[0]['producer_fold'], **{name: str(group)},
                    rows=int(at.sum()), queries=len(count), summary=rows))
        beat(state='paired_labels_scored', job=key, rows=len(ix))
    if len(sources) != 14 or sum(x['rows'] for x in sources) != 125540:
        raise ValueError('Every original donor row and source required')
    full = [x['summary'][-1] for x in sources]
    lower = np.array([r['paired_completion_lower_pp'] for r in full])
    known = np.array([r['paired_known_pp'] for r in full])
    rng = np.random.default_rng(reg['config']['bootstrap_seed'])
    samples = rng.integers(0, len(full), (reg['config']['bootstrap_resamples'], len(full)))
    ci = np.quantile(lower[samples].mean(1), [.025, .975])
    fold_means = {str(f): float(np.mean([r['summary'][-1]['paired_known_pp'] for r in sources if r['producer_fold'] == f])) for f in (0, 1)}
    screen = bool(lower.mean() > 0 and ci[0] > 0 and (known > 0).sum() >= 8 and min(fold_means.values()) > 0)
    output = stable(dict(result_source='fresh_run', sources=sources, recordings=recordings,
        total_rows=125540, primary=dict(equal_source_paired_completion_lower_pp=float(lower.mean()),
            equal_source_paired_known_pp=float(known.mean()), lower_exploratory_bootstrap_CI=ci.tolist(),
            positive_source_known_contrasts=int((known > 0).sum()), fold_known_means_pp=fold_means,
            actor_component_progression_screen_pass=screen),
        bootstrap_not_independent_confirmation=True, one_forecaster_seed=True,
        new_controller_training=False, new_own_B_readout=False, new_outer_C_H_readout=False,
        original_rules_unchanged=True, safety_certified=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if replay:
        exact(read(PUBLIC/'analysis.json'), output)
    else:
        once(PUBLIC/'analysis.json', output)
    beat(state='complete_component_readout', progression_screen_pass=screen)


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('register', 'pilot', 'control', 'train', 'forecast', 'evaluate', 'replay'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, reg = register()
        if args.phase == 'register':
            print('Registered seven actual full fits:one control and six prefix actors.'); return
        committed(reg); start, beat = runtime(cfg, args.phase); data = source.load_data(); jobs = entries(data, reg)
        before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase == 'pilot': pilot(cfg, reg, data, jobs, beat)
        elif args.phase == 'control': control(reg, data, jobs, beat, args.resume)
        elif args.phase == 'train': train(reg, data, jobs, beat, args.resume)
        elif args.phase == 'forecast': forecasts(reg, data, jobs, beat)
        elif args.phase == 'evaluate': evaluate(reg, data, jobs, beat)
        else:
            forecasts(reg, data, jobs, beat, replay=True); evaluate(reg, data, jobs, beat, replay=True)
            exact(before, {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        receipt = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, actual_Torch=True)
        dest = PRIVATE/'replays' if args.phase == 'replay' else PUBLIC/'executions'
        once(dest/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json', receipt); beat(state='phase_complete')


if __name__ == '__main__':
    main()
