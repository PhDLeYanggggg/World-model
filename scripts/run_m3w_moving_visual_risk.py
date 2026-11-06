"""Actual matched visual cost learning; fixed original complete own-B readout."""
import argparse
import fcntl
from functools import lru_cache
import hashlib
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
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_role_separated_cost as parent
from scripts import run_m3w_moving_simple_controls as controls
from scripts import run_m3w_moving_past_background as images
from scripts import run_m3w_moving_past_background_own as own_images
from src.data_unification import m3w_visual_context_packet as context_method
from src.world_model import m3w_visual_risk_cost as method

np, torch = parent.np, parent.torch
read, sha, stable, once, checked, artifact = parent.read, parent.sha, parent.stable, parent.once, parent.checked, parent.artifact
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_visual_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_visual_risk_v1'
CONFIG = ROOT/'configs/m3w_moving_visual_risk_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', Path(method.__file__), Path(context_method.__file__),
    ROOT/'tests/test_m3w_visual_risk_cost.py', ROOT/'tests/test_m3w_visual_context_packet.py')
OWN_INPUT_KEYS = ('x', 'envelope', 'scales', 'sites', 'query', 'ids', 'row', 'action')
OLD_INPUT_KEYS = ('ids', 'local_ids', 'sites', 'recordings', 'frames', 'paths', 'floor', 'cv', 'active', 'proposal_self')
POLICIES = parent.POLICIES


def measurement_bundle():
    h = hashlib.sha256(); counts = {}
    for key, home, regpath in (('A', images.PRIVATE, images.REG), ('own', own_images.PRIVATE, own_images.REG)):
        count = 0
        for folder in ('rasters', 'pairs'):
            for p in sorted((home/folder).glob('*/*.json')):
                v = read(p)
                if v['registration_sha256'] != sha(regpath): raise ValueError('Changed image-measurement binding')
                if folder == 'rasters': checked(v['cache'])
                h.update(str(p.relative_to(ROOT)).encode()); h.update(sha(p).encode()); count += 1
        counts[key] = count
    if counts != dict(A=1851+1817, own=996+1231): raise ValueError('Retain the full measurement bundle')
    return dict(sha256=h.hexdigest(), receipt_counts=counts)


def register():
    cfg = read(CONFIG); _, old = parent.register()
    a, b = read(images.PUBLIC/'measurement.json'), read(own_images.PUBLIC/'measurement.json')
    if (not a['measurement_engineering_screen_pass'] or not b['measurement_engineering_screen_pass']
            or cfg['arms'] != list(method.ARMS) or cfg['fresh_models'] != 24 or cfg['full_exact_refits'] != 24
            or cfg['risk_head_seeds'] != [17, 29, 43] or cfg['risk_budget'] != .02
            or cfg['steps_per_model'] != method.SETTINGS['steps'] or cfg['expected_evaluation_rows'] != 62466
            or any(cfg[k] for k in ('threshold_search', 'new_outer_C_H_readout', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed matched four-arm comparison and original complete rules required')
    source_reg = read(parent.source.PUBLIC/'registration.json')
    producer = {str(f): sorted({s for j in source_reg['jobs'] if j['producer_fold'] == f for s in j['producer_sites']}) for f in (0, 1)}
    for node in old['jobs']:
        if set(producer[str(node['job']['producer_fold'])])&set(node['part']['names']):
            raise ValueError('A-only training overlaps its own-B prediction sources')
    value = stable(dict(config=cfg, settings=method.SETTINGS, jobs=old['jobs'], producer_sources=producer,
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in (Path(parent.__file__), Path(parent.method.__file__),
            Path(parent.allocation.__file__), Path(images.method.__file__))},
        parents=[artifact(p) for p in (parent.REG, parent.PUBLIC/'A_moments.json', parent.PUBLIC/'own.json',
            parent.full.PUBLIC/'population.json', parent.full.PUBLIC/'inference_freeze_v2.json',
            images.REG, images.PUBLIC/'measurement.json', own_images.REG, own_images.PUBLIC/'measurement.json')],
        measured_inputs=measurement_bundle(), role='exposed_own_B_model_development_not_confirmation',
        original_rules_unchanged=True, own_B_targets_in_training=False, forecaster_seeds=[17]))
    once(REG, value)
    once(PUBLIC/'input_admission.json', dict(registration_sha256=sha(REG),
        version='past_image_derived_features_only_v1', original_admission_files_edited=False,
        training_role='supervised_training_existing_producer_A_only',
        prediction_role='already_exposed_own_B_development_inputs_not_independent_test',
        modalities=['past_background_image_motion_proxy', 'native_image_bounds'],
        future_targets_allowed_as_inputs=False, future_masks_allowed_as_inputs=False,
        current_head_masks_only=True, new_outer_C_H_readout=False,
        original_risk_support_NA_rules_unchanged=True, stage5c_executed=False, smc_enabled=False))
    return cfg, value


def committed(reg):
    for p in (*FILES, REG, PUBLIC/'input_admission.json'):
        blob = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != sha(p): raise ValueError('Commit exact visual model before fitting')
    for p, digest in reg['dependencies'].items():
        if sha(ROOT/p) != digest: raise ValueError('Changed parent definition')


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic state')))
    start, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('Resume unchanged study after12h')
        if now-last[0] > 1:
            last[0] = now
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Keep5GiBreserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['artifact_cap_bytes']: raise OSError('Private cap exceeded')
    def beat(**values):
        guard(); event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
        parent.source.atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(parent.json.dumps(event)+'\n')
        print(parent.json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), device='cpu', threads=4, interop=1, workers=0, resource_probing=False)
    return start, guard, beat


@lru_cache(maxsize=8192)
def flow(node, frame):
    found = []
    for home in (images.PRIVATE, own_images.PRIVATE):
        p = home/'pairs'/node/f'{frame:06d}.json'
        if p.exists(): found.append(read(p)['measurement'])
    if not found or any(v != found[0] for v in found[1:]): raise ValueError('Missing or inconsistent past transform')
    return found[0]


@lru_cache(maxsize=8192)
def size(node, frame):
    for home in (images.PRIVATE, own_images.PRIVATE):
        p = home/'rasters'/node/f'{frame:06d}.json'
        if p.exists(): return read(p)['native_size']
    raise ValueError('No verified native image-size metadata')


def packet(data, ix, paths):
    admission = read(ROOT/'outputs/publication_readiness_2026_09/moving_development_admission_v1/admission.json')
    return context_method.packet(data['history'][ix], paths, data['recordings'][ix], data['frames'][ix],
        sorted(admission['recordings']), flow, size)


def own_input(node):
    key = node['job']['key']
    ref = next(x['cache'] for x in read(parent.PUBLIC/'own.json')['banks'] if x['job'] == key)
    with np.load(checked(ref), allow_pickle=False) as z: own = {k: z[k].copy() for k in OWN_INPUT_KEYS}
    ref = read(parent.full.PUBLIC/'inference_v2'/f'{key}.json')['cache']
    with np.load(checked(ref), allow_pickle=False) as z: old = {k: z[k].copy() for k in OLD_INPUT_KEYS}
    np.testing.assert_array_equal(own['ids']//7, old['ids'][own['row']])
    return own, old


def align(table_ids, keys, values):
    order = np.argsort(keys); pos = np.searchsorted(keys[order], table_ids)
    if (pos == len(order)).any(): raise ValueError('Missing exact context ID')
    np.testing.assert_array_equal(keys[order[pos]], table_ids)
    return values[order[pos]]


def prepare(reg, guard, beat):
    data = parent.source.load_data(); receipts = []
    forecasts = read(parent.source.PUBLIC/'analysis.json')['forecasts']
    for fold in (0, 1):
        ids, motion, quality = [], [], []
        for item in forecasts:
            if item['job']['producer_fold'] != fold: continue
            cached = parent.load_cache(item['forecast']); ix = cached['local_ids']
            floor = parent.donor.baseline_numpy(data['history'][ix], 1)-data['origin'][ix, None]
            neural = cached['relative_prediction'].astype(float)
            paths = np.stack([floor, *[floor+a*(neural-floor) if a != 1 else neural for a in parent.donor.FRACTIONS]], 1)
            c = packet(data, ix, paths); guard()
            ids.append((data['global_ids'][ix, None]*7+np.arange(7)).reshape(-1))
            motion.append(c['motion'].reshape(-1, 56)); quality.append(c['quality'].reshape(-1, 56))
            beat(state='A_causal_context', job=item['job']['key'])
        keys = np.concatenate(ids); added = parent.added_data(fold)
        path = PRIVATE/'context'/f'A{fold}.npz'
        parent.full.save_arrays(path, dict(ids=added['ids'],
            motion=align(added['ids'], keys, np.concatenate(motion)),
            quality=align(added['ids'], keys, np.concatenate(quality))))
        if set(added['sites']) != set(reg['producer_sources'][str(fold)]): raise ValueError('A-only source mismatch')
        receipts.append(dict(bank=f'A{fold}', cache=artifact(path), rows=len(added['ids'])))
    del data
    pop = read(parent.full.PUBLIC/'population.json')
    data = own_images.past_members(checked(pop['cache']))
    for node in reg['jobs']:
        own, old = own_input(node); ix = old['local_ids']
        np.testing.assert_array_equal(data['global_ids'][ix], old['ids'])
        c = packet(data, ix, old['paths']); path = PRIVATE/'context'/f'{node["job"]["key"]}.npz'
        parent.full.save_arrays(path, dict(ids=own['ids'], motion=c['motion'][own['row'], own['action']],
            quality=c['quality'][own['row'], own['action']]))
        receipts.append(dict(bank=node['job']['key'], cache=artifact(path), rows=len(own['ids'])))
        beat(state='own_B_past_only_context', job=node['job']['key'], rows=len(ix), own_targets_opened=False)
    once(PUBLIC/'context.json', dict(banks=receipts, exact_original_ID_joins=True, own_B_future_targets_opened=False))


def context(name, ids):
    ref = next(x['cache'] for x in read(PUBLIC/'context.json')['banks'] if x['bank'] == name)
    value = parent.load_cache(ref); np.testing.assert_array_equal(value['ids'], ids)
    return value


def training_data(fold, arm):
    data = parent.added_data(fold); c = context(f'A{fold}', data['ids'])
    return dict(data, context=context_method.select(c, arm))


def identity(fold, seed, arm):
    return dict(registration_sha256=sha(REG), producer_fold=fold, seed=seed, arm=arm, training='A_only')


def fit_path(fold, seed, arm, kind='fits'):
    return PRIVATE/kind/f'A{fold}_seed{seed}_{arm}.pt.gz'


def pilot(guard, beat):
    d = training_data(0, 'motion_geometry'); ident = identity(0, 17, 'motion_geometry')
    p, q = PRIVATE/'pilot/resume.pt.gz', PRIVATE/'pilot/direct.pt.gz'
    if not p.exists(): method.fit(d, ident, p, arm='motion_geometry', seed=17, stop_at=128, guard=guard, heartbeat=beat)
    a = method.fit(d, ident, p, arm='motion_geometry', seed=17, stop_at=256, resume=True, guard=guard, heartbeat=beat)
    b = method.fit(d, ident, q, arm='motion_geometry', seed=17, stop_at=256, resume=q.exists(), guard=guard, heartbeat=beat)
    method.exact(a, b)
    once(PUBLIC/'pilot.json', stable(dict(actual128plus128_vs256_exact=True, parameters=a['parameters'],
        fit_rows=a['fit_rows'], seconds_direct256=b['seconds'], prefix=artifact(p), pilot_is_not_model_success=True)))


def train(reg, guard, beat, resume, verify=False):
    pilot_ref = read(PUBLIC/'pilot.json'); results = []; count = 0
    for fold in (0, 1):
        for seed in (17, 29, 43):
            first = None
            for arm in method.ARMS:
                path = fit_path(fold, seed, arm, 'verify' if verify else 'fits')
                if not verify and fold == 0 and seed == 17 and arm == 'motion_geometry' and not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(checked(pilot_ref['prefix']), path)
                d = training_data(fold, arm)
                state = method.fit(d, identity(fold, seed, arm), path, arm=arm, seed=seed,
                    resume=resume or path.exists(), guard=guard, heartbeat=lambda **kw: beat(fold=fold, seed=seed, **kw))
                if not state['complete'] or state['step'] != 4096: raise ValueError('Full fixed fit required')
                if verify: method.exact(method.base.read_checkpoint(fit_path(fold, seed, arm)), state)
                else:
                    once(PUBLIC/'fits'/f'A{fold}_seed{seed}_{arm}.json', stable(dict(fold=fold, seed=seed, arm=arm,
                        checkpoint=artifact(path), parameters=state['parameters'], steps=state['step'], seconds=state['seconds'],
                        fit_rows=state['fit_rows'], trace=state['trace'], fitting_sources=state['preprocess']['train_sources'], result_source='fresh_run')))
                if first is None: first = state
                else:
                    for k in ('initial_model', 'generator', 'visits', 'parameters', 'step'):
                        method.base.exact(first[k], state[k])
                    for k in first['preprocess']:
                        if k not in ('context_mean', 'context_std', 'fit_context_hash'):
                            method.base.exact(first['preprocess'][k], state['preprocess'][k])
                    if first['trace'][0]['total'] != state['trace'][0]['total']: raise ValueError('Initial functions differ')
                results.append(dict(fold=fold, seed=seed, arm=arm, checkpoint=artifact(path)))
                count += 1; beat(state='full_refit_exact' if verify else 'fit_complete', complete=count, total=24)
    once(PUBLIC/('verification.json' if verify else 'training.json'), dict(models=24, updates=24*4096,
        all24_full_refits_exact=verify, matched_initial_functions_and_draws=True, models_info=results))


def predict_one(node, arm, seed):
    own, old = own_input(node); c = context_method.select(context(node['job']['key'], own['ids']), arm)
    fold = node['job']['producer_fold']; state = method.base.read_checkpoint(fit_path(fold, seed, arm))
    if set(state['preprocess']['train_sources'])&set(old['sites']): raise ValueError('Own-B source used in fitting')
    out = method.predict(state, own['x'], own['envelope'], c); n = len(old['ids'])
    prediction = dict(moments=np.zeros((n, 7, 5)), conditional=np.zeros((n, 7, 5)), probability=np.zeros((n, 7)))
    for key in ('moments', 'conditional'):
        prediction[key][own['row'], own['action']] = out[key]*own['scales'][:, None]
    prediction['probability'][own['row'], own['action']] = out['observation_probability']
    return prediction, own, old


def infer(cfg, reg, beat, resume=False, replay=False):
    if not read(PUBLIC/'verification.json')['all24_full_refits_exact']: raise ValueError('Complete real refits first')
    refs = []
    for node in reg['jobs']:
        for seed in (17, 29, 43):
            for arm in method.ARMS:
                key = f'{node["job"]["key"]}_{arm}_seed{seed}'; dest = PUBLIC/'inference'/f'{key}.json'
                if resume and dest.exists() and not replay:
                    checked(read(dest)['cache']); refs.append(artifact(dest)); continue
                prediction, own, old = predict_one(node, arm, seed); p = prediction['moments']
                allowed = np.zeros(p.shape[:2], bool); allowed[own['row'], own['action']] = True
                proposal = np.zeros_like(allowed); at = old['active']; proposal[at, old['proposal_self'][at]] = True
                proposal_only, _ = parent.allocation.select_actions(p, proposal, arm='physical_neural')
                path = PRIVATE/'inference'/f'{key}.npz'
                if replay:
                    cached = parent.load_cache(read(dest)['cache'])
                    for k, v in prediction.items(): np.testing.assert_array_equal(v, cached[k])
                    np.testing.assert_array_equal(proposal_only, cached['proposal_only'])
                    independent, _ = parent.allocation.select_actions(p, allowed, arm='physical_neural')
                    np.testing.assert_array_equal(independent, cached['independent'])
                    for group in parent.allocation.query_groups(old['recordings'], old['frames'], old['ids']):
                        for name in parent.allocation.ARMS:
                            if not parent.allocation.feasible(p[group], allowed[group], cached[name][group]): raise ValueError('Predicted budget changed')
                        if (cached['joint_matched'][group] >= 0).sum() != (independent[group] >= 0).sum(): raise ValueError('Matched count changed')
                else:
                    cp = PRIVATE/'allocation'/f'{key}.pt.gz'; cp.parent.mkdir(parents=True, exist_ok=True)
                    restored = method.base.read_checkpoint(cp) if resume and cp.exists() else None
                    choices, _, info = parent.allocation.allocate(p, allowed, old['recordings'], old['frames'], old['ids'],
                        seconds=cfg['solver_seconds_per_query'], heartbeat=lambda **kw: beat(state='allocation', job=key, **kw),
                        checkpoint=lambda state: method.base.save_checkpoint(cp, state), resume_state=restored)
                    parent.full.save_arrays(path, dict(**prediction, proposal_only=proposal_only, **choices))
                    once(dest, stable(dict(cache=artifact(path), allocation=info, arm=arm, seed=seed,
                        job=node['job']['key'], own_B_targets_opened=False, all_choices_before_labels=True)))
                refs.append(artifact(dest)); beat(state='predictions_replayed' if replay else 'choices_frozen', job=key)
    if not replay: once(PUBLIC/'inference_freeze.json', dict(contexts=refs, all36_choices_before_readout=True))


def evaluate(cfg, reg, beat, replay=False):
    if not read(PUBLIC/'inference_freeze.json')['all36_choices_before_readout']: raise ValueError('Freeze all choices first')
    data = parent.full.load_data(); rows = []; contrasts = {k: [] for k in ('primary', 'bounds', 'full_bundle', 'fixed_control')}
    for node in reg['jobs']:
        own, old = own_input(node); job = node['job']; ix = old['local_ids']
        target = data['target_eval'][ix]-data['origin'][ix, None]; valid = data['valid'][ix]
        fixed = controls.method.choices(data['history'][ix], old['paths'], old['floor'], 1, 'last_velocity_nonzero')
        fixed_paths = parent.full.method.paths_for(old['paths'], old['floor'], fixed)
        for seed in (17, 29, 43):
            banks = {arm: parent.load_cache(read(PUBLIC/'inference'/f'{job["key"]}_{arm}_seed{seed}.json')['cache']) for arm in method.ARMS}
            paths = {arm: {name: parent.full.method.paths_for(old['paths'], old['floor'], banks[arm][name]) for name in POLICIES} for arm in method.ARMS}
            for site in node['part']['names']:
                at = old['sites'] == site
                for arm in method.ARMS:
                    policies = {name: parent.full.method.original.policy(p[at], old['floor'][at], old['cv'][at], target[at], valid[at],
                        old['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, p in paths[arm].items()}
                    rows.append(dict(job=job['key'], source=site, seed=seed, arm=arm, policies=policies))
                def pair(a, b): return parent.full.method.original.paired(a[at], b[at], old['floor'][at], target[at], valid[at])
                for name, a, b in (('primary', paths['quality_geometry']['joint_free'], paths['motion_geometry']['joint_free']),
                    ('bounds', paths['no_image']['joint_free'], paths['image_bounds']['joint_free']),
                    ('full_bundle', paths['no_image']['joint_free'], paths['motion_geometry']['joint_free']),
                    ('fixed_control', fixed_paths, paths['motion_geometry']['joint_free'])):
                    contrasts[name].append(dict(source=site, seed=seed, **pair(a, b)))
        beat(state='original_complete_readout', job=job['key'])
    summary = {}
    for arm in method.ARMS:
        rr = [r for r in rows if r['arm'] == arm]; summary[arm] = {}
        for name in POLICIES:
            pp = [r['policies'][name] for r in rr]
            utility = {label: parent.interval([dict(source=r['source'], value=r['policies'][name][key]) for r in rr], 'value', cfg)
                for label, key in (('known', 'known_ADE_gain_floor_percent'), ('lower', 'completion_lower_gain_floor_percent'))}
            summary[arm][name] = dict(safety=parent.full.method.safety(pp), utility=utility,
                hard_gain_percent=float(np.mean([p['subsets']['hard']['gain_percent'] for p in pp])),
                easy_gain_percent=float(np.mean([p['subsets']['easy']['gain_percent'] for p in pp])),
                switch_rate=float(np.mean([p['switch_rate'] for p in pp])))
    intervals = {name: {k: parent.interval(v, k, cfg) for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name, v in contrasts.items()}
    seed_means = {str(s): float(np.mean([r['completion_lower_pp'] for r in contrasts['primary'] if r['seed'] == s])) for s in (17, 29, 43)}
    screen = intervals['primary']['completion_lower_pp']['interval95'][0] > 0 and sum(v > 0 for v in seed_means.values()) >= 2
    result = stable(dict(result_source='fresh_run', rows=rows, policies=summary, contrasts=intervals, contrast_rows=contrasts,
        primary_seed_means=seed_means, motion_component_screen_pass=bool(screen),
        source_components=8, unique_recordings=24, required_record_views_per_policy=72,
        risk_head_seeds=[17, 29, 43], forecaster_seeds=[17], fresh_models=24,
        training_support='A_only_not_old50percent_A_B', independent_confirmation=False,
        original_rules_unchanged=True, new_outer_C_H_readout=False, identity_certification=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if replay: method.base.exact(read(PUBLIC/'analysis.json'), result)
    else: once(PUBLIC/'analysis.json', result)


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('register', 'prepare', 'pilot', 'train', 'verify', 'infer', 'evaluate', 'replay'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, reg = register()
        if args.phase == 'register': print('Registered24matched A-only visual cost models and24complete refits; original rules.'); return
        committed(reg); start, guard, beat = runtime(cfg, args.phase)
        before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase == 'prepare': prepare(reg, guard, beat)
        elif args.phase == 'pilot': pilot(guard, beat)
        elif args.phase in ('train', 'verify'): train(reg, guard, beat, args.resume, verify=args.phase == 'verify')
        elif args.phase == 'infer': infer(cfg, reg, beat, args.resume)
        elif args.phase == 'evaluate': evaluate(cfg, reg, beat)
        else:
            infer(cfg, reg, beat, replay=True); evaluate(cfg, reg, beat, replay=True)
            method.base.exact(before, {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        dest = PRIVATE/'replays' if args.phase == 'replay' else PUBLIC/'executions'
        once(dest/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json', dict(pid=os.getpid(), phase=args.phase,
            seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            timed_MILP_search_recomputed=False if args.phase == 'replay' else None))
        beat(state='phase_complete')


if __name__ == '__main__':
    main()
