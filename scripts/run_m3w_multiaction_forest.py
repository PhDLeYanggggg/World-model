"""Frozen seven-action expected-cost forest control, with real recovery and replay."""
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
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
import torch
from scripts import run_m3w_static_scene_cost as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_multiaction_forest as api
from src.evaluation import m3w_multiaction_cost_readout as readout
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets

NAME = 'european_multiaction_forest_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_multiaction_forest_v1.json'; REG = PUBLIC/'registration.json'
sha, once, json_write = parent.sha, parent.once, parent.json_write
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_multiaction_forest.py',
    ROOT/'src/evaluation/m3w_multiaction_forest_readout.py', ROOT/'tests/test_m3w_multiaction_forest.py',
    PUBLIC/'protocol.md', PUBLIC/'compatibility.md', ROOT/'src/world_model/m3w_source_forest.py')


def load():
    cfg = json.loads(CONFIG.read_text())
    values = parent.admission()
    if cfg['seeds'] != [17, 29, 43] or cfg['arms'] != list(parent.parent.api.ARMS) or cfg['risk_budget'] != .02:
        raise ValueError('Fixed seeds, banks and unchanged risk budget required')
    legacy = ROOT/'outputs/publication_readiness_2026_09/european_source_forest_v1'
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        parent_verification_sha256=sha(parent.PUBLIC/'verification.json'),
        old_binary_reference_sha256={p.name:sha(p) for p in (legacy/'results.md', legacy/'training_freeze.json')},
        contexts=sorted(values[2]['contexts']), arms=cfg['arms'], seeds=cfg['seeds'], fits=54, trees_per_fit=128,
        primary_total_trees=6912, sklearn=sklearn.__version__, fields=398, tree_inputs=406,
        objective=api.OBJECTIVE, original_query_draws_per_context_seed=256000,
        normalization='physical_fitting_only', sampler_matches_static_scene_neural=True,
        estimator_and_objective_not_one_factor_equivalent=True, independent_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    return cfg, values


def admission():
    cfg, values = load()
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel) != digest:raise ValueError('Registered forest asset changed: '+rel)
        parent.parent.committed(ROOT/rel)
    parent.parent.committed(REG)
    return cfg, values


def runtime(cfg):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic tree block')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.joblib')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Private forest checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h healthy-runtime boundary; resume the same scientific roster')
    def beat(**event):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    guard(); return started, beat, guard


def context_data(context, items, values, beat, *, held=False):
    _, _, _, source, jobs, identity, _, packed, polygons = values
    return parent.scene_packets(parent.parent.prepare_context(context, items, source, jobs, identity, beat,
        include_held=held), source, packed, polygons)


def checkpoint(context, seed, arm, *, folder='heads', direct=False):
    return PRIVATE/folder/f'{context}_seed{seed}_{arm}{"_direct" if direct else ""}.joblib'


def matched_data(packets, context, seed, neural, settings):
    data = parent.training_data(packets)
    counts, draw_hash = api.replay_draws(data, settings, seed)
    for arm in parent.parent.api.ARMS:
        prior = parent.parent.api.read_checkpoint(ROOT/neural[context, seed, arm]['checkpoint']['path'])
        parent.parent.api.exact(data['preprocess'], prior['preprocess'])
        parent.parent.api.exact(data['input_hashes'], prior['input_hashes'])
        if draw_hash != prior['draw_hash']:raise ValueError('Original neural query sampler replay differs')
    return data, counts, draw_hash


def pilot(resume=False):
    cfg, values = admission(); finish = PUBLIC/'pilot.json'
    if finish.exists():raise ValueError('Completed pilot immutable')
    started, beat, guard = runtime(cfg); neural = parent.models()
    context, items = sorted(values[2]['contexts'].items())[0]
    packets = context_data(context, items, values, beat)['fit']
    data, counts, draw_hash = matched_data(packets, context, 17, neural, values[0]['training']); del packets
    refs = []
    for arm in cfg['arms']:
        before = time.monotonic(); matrix = api.training_matrix(data, arm, counts); matrix_seconds = time.monotonic()-before
        kwargs = dict(preprocess=data['preprocess'], arm=arm, settings=cfg['estimator'], seed=17,
            identity=dict(context=context, registration_sha256=sha(REG)), draw_hash=draw_hash, heartbeat=beat, guard=guard)
        path = checkpoint(context, 17, arm, folder='pilot')
        if not path.exists():api.fit(matrix, **kwargs, path=path, stop_at=cfg['pilot_resume_at'])
        state = api.fit(matrix, **kwargs, path=path, resume=True, stop_at=cfg['pilot_trees'])
        direct_path = checkpoint(context, 17, arm, folder='pilot', direct=True)
        direct = api.fit(matrix, **kwargs, path=direct_path, resume=resume and direct_path.exists(), stop_at=cfg['pilot_trees'])
        api.exact_forest(state, direct)
        refs.append(dict(arm=arm, trees=state['trees'], nodes=state['nodes'], seconds=state['seconds'],
            matrix_seconds=matrix_seconds, matrix_bytes=matrix['matrix_bytes'], candidates=state['positive_candidate_rows'],
            sampled_queries=state['positive_query_rows'], checkpoint_bytes=path.stat().st_size,
            input_hashes=matrix['input_hashes'], exact_resume=True))
        del state, direct, matrix
    once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, fits=refs,
        engineering_pass=True, full_fitting_data=True, exact_resume_all_banks=True, pilot_total_trees=96,
        projected_primary_fit_seconds=sum(r['seconds'] for r in refs)*8*18,
        projected_primary_checkpoint_bytes=sum(r['checkpoint_bytes'] for r in refs)*8*18,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, held_scored=False))
    beat(state='pilot_complete')


def train(resume=False):
    cfg, values = admission(); finish = PUBLIC/'training_freeze.json'
    if finish.exists():raise ValueError('Completed54fit training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Real recovery pilot required')
    started, beat, guard = runtime(cfg); neural = parent.models(); refs = []
    replay = PUBLIC/'full128_replay.json'
    first_context = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        packets = context_data(context, items, values, beat)['fit']
        data = parent.training_data(packets); del packets
        for seed in cfg['seeds']:
            counts, draw_hash = api.replay_draws(data, values[0]['training'], seed)
            for arm in cfg['arms']:
                prior = parent.parent.api.read_checkpoint(ROOT/neural[context, seed, arm]['checkpoint']['path'])
                parent.parent.api.exact(data['preprocess'], prior['preprocess'])
                parent.parent.api.exact(data['input_hashes'], prior['input_hashes'])
                if draw_hash != prior['draw_hash']:raise ValueError('Original query draw hash differs')
                before = time.monotonic(); matrix = api.training_matrix(data, arm, counts); prep_seconds = time.monotonic()-before
                path = checkpoint(context, seed, arm)
                kwargs = dict(preprocess=data['preprocess'], arm=arm, settings=cfg['estimator'], seed=seed,
                    identity=dict(context=context, registration_sha256=sha(REG)), draw_hash=draw_hash, heartbeat=beat, guard=guard)
                state = api.fit(matrix, **kwargs, path=path, resume=resume and path.exists())
                if context == first_context and seed == 17 and arm == 'physical_only':
                    if not replay.exists():
                        direct_path = checkpoint(context, seed, arm, folder='full_replay', direct=True)
                        direct = api.fit(matrix, **kwargs, path=direct_path, resume=resume and direct_path.exists())
                        api.exact_forest(state, direct)
                        once(replay, dict(pid=os.getpid(), full128_scientific_state_and_trees_exact=True,
                            context=context, seed=seed, arm=arm, primary_checkpoint_sha256=sha(path),
                            direct_checkpoint_sha256=sha(direct_path), primary_seconds=state['seconds'],
                            direct_seconds=direct['seconds'], held_scored=False, registration_sha256=sha(REG)))
                        del direct
                    else:
                        check = json.loads(replay.read_text())
                        if not check['full128_scientific_state_and_trees_exact'] or check['primary_checkpoint_sha256'] != sha(path):
                            raise ValueError('First128tree replay binding changed')
                receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                value = dict(context=context, seed=seed, arm=arm, trees=state['trees'], nodes=state['nodes'],
                    seconds=state['seconds'], matrix_seconds=prep_seconds,
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    draw_hash=draw_hash, matrix_bytes=state['matrix_bytes'], candidates=state['positive_candidate_rows'],
                    sampled_queries=state['positive_query_rows'], matrix_input_hashes=state['input_hashes'],
                    first_trace=state['trace'][0], final_trace=state['trace'][-1],
                    query_sampler_features_and_normalizers_match=True, sklearn=sklearn.__version__,
                    registration_sha256=sha(REG), result_source='fresh_native_CPU_ExtraTrees_current_seven_action_contract')
                if receipt.exists():
                    old = json.loads(receipt.read_text()); value['matrix_seconds'] = old['matrix_seconds']
                once(receipt, value)
                refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                beat(state='fit_complete', fits=len(refs), total=54, context=context, seed=seed, arm=arm,
                    trees=state['trees'], nodes=state['nodes'])
                del state, matrix, prior
        del data
    if len(refs) != 54:raise ValueError('All54final128tree models required')
    once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, fits=refs, count=54, primary_trees=6912,
        registration_sha256=sha(REG), inner_held_scored=False,
        full128_replay_sha256=sha(replay), peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        private_checkpoint_bytes=sum(p.stat().st_size for p in PRIVATE.rglob('*.joblib')),
        new_Torch_optimizer_updates=0, independent_roles_opened=False))
    beat(state='training_complete', forests=54, primary_trees=6912)


def models():
    parent.parent.committed(PUBLIC/'training_freeze.json')
    frozen = json.loads((PUBLIC/'training_freeze.json').read_text())
    if frozen['count'] != 54 or frozen['inner_held_scored'] or frozen['registration_sha256'] != sha(REG):
        raise ValueError('Complete committed54fitfreeze required')
    result = {}
    for ref in frozen['fits']:
        row = json.loads((ROOT/ref['path']).read_text()); key = row['context'], row['seed'], row['arm']
        if sha(ROOT/ref['path']) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen endpoint changed')
        if key in result or row['trees'] != 128:raise ValueError('Unique128tree endpoints required')
        result[key] = row
    return result


def evaluate(verify=False, resume=False):
    cfg, values = admission(); learned, neural = models(), parent.models(); pc = values[1]
    finish = PUBLIC/'complete.json'
    if finish.exists() and not verify:raise ValueError('Completed readout immutable; use verify')
    started, beat, guard = runtime(cfg); rows, refs = [], []
    old_groups = aggregate_groups(parent.NAME)
    old_hashes = {Path(ref['path']).name:ref['sha256'] for ref in json.loads((parent.PUBLIC/'complete.json').read_text())['groups']}
    _, origin_refs = parent.parent.old.direct.ORIGINS.models(values[2])
    for context, items in sorted(values[2]['contexts'].items()):
        scopes = context_data(context, items, values, beat, held=True)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            states = {a:joblib.load(ROOT/learned[context, seed, a]['checkpoint']['path']) for a in cfg['arms']}
            neural_states = {a:parent.parent.api.read_checkpoint(ROOT/neural[context, seed, a]['checkpoint']['path']) for a in cfg['arms']}
            origin = parent.parent.old.base.core.read_checkpoint(ROOT/origin_refs[context, seed, 'self']['checkpoint']['path'])
            for scope, packets in scopes.items():
                originals = parent.parent.old.direct.origin_predictions(origin, {s:p['args'] for s,p in packets.items()})
                for source, packet in sorted(packets.items()):
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{scope}.json'
                    if path.exists() and not (verify or resume):raise ValueError('Explicit partial readout resume required')
                    old = old_groups[path.name]; choices, ranks, predicted = {}, {}, {}
                    for arm in cfg['arms']:
                        p = api.predict(states[arm], packet['x'], packet['scene'], packet['envelope'], packet['scales'])
                        predicted[arm] = p
                        for multiplier in (1., *pc['sensitivity_harm_multipliers']):
                            key = arm if multiplier == 1 else arm+'_harm'+str(int(multiplier))
                            choices[key], ranks[key] = parent.parent.api.select_actions(p, packet['allowed'], arm=arm, harm_multiplier=multiplier)
                        p = parent.api.predict(neural_states[arm], packet['x'], packet['scene'], packet['envelope'], packet['scales'])
                        key = arm+'_neural_parent'; predicted[key] = p
                        choices[key], ranks[key] = parent.parent.api.select_actions(p, packet['allowed'], arm=arm)
                        if api.fingerprint(choices[key]) != old['action_sha256'][arm]:raise ValueError('Static neural choices differ')
                    op = originals[source]
                    take = parent.parent.old.base.actions(op, parent.parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                    choices['frozen_origin_binary'] = np.where(take, 6, -1); ranks['frozen_origin_binary'] = op[:, 0]-op[:, 1]
                    choices['floor'] = np.full(len(take), -1, np.int64); ranks['floor'] = np.zeros(len(take))
                    result = readout.group_readout(packet, choices, ranks, predicted)
                    for arm in cfg['arms']:
                        if result['policies'][arm+'_neural_parent'] != old['result']['policies'][arm]:
                            raise ValueError('Static neural metric replay differs')
                    result['forest_minus_neural'] = {}
                    for arm in cfg['arms']:
                        left, right = choices[arm+'_neural_parent'], choices[arm]
                        aa, bb = readout.matched_choices(left, right, ranks[arm+'_neural_parent'], ranks[arm],
                            packet['recordings'], packet['frames'], packet['ids'])
                        result['forest_minus_neural'][arm] = dict(full=readout.paired_bounds(packet, left, right),
                            matched=readout.paired_bounds(packet, aa, bb), matched_count=int((aa >= 0).sum()),
                            matched_role='utility_only_not_safety')
                    result['subsets'] = subsets(packet, choices, job['design']['easy_cut'])
                    value = dict(context=context, source=source, seed=seed, scope=scope, result=result,
                        action_sha256={a:api.fingerprint(c) for a,c in choices.items()},
                        prediction_sha256={a:api.fingerprint(p) for a,p in predicted.items()},
                        neural_parent_group_sha256=old_hashes[path.name], registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_source_internal_ExtraTrees_cached_verified_neural_comparator', independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):raise ValueError('Exact forest readout replay differs')
                    else:once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='readout_progress', context=context, views=len(rows), total=144)
        del scopes
    summary = readout.summarize(rows, pc)
    for scope in summary['scopes']:
        selected = [r for r in rows if r['scope'] == scope]
        summary['scopes'][scope]['forest_minus_neural'] = {arm:{mode+'_'+bound:readout.interval(selected,
            lambda r,a=arm,m=mode,b=bound:r['forest_minus_neural'][a][m][b], pc)
            for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')} for arm in cfg['arms']}
        summary['scopes'][scope]['subsets'] = pooled_subsets(selected)
    summary.update(estimator='ExtraTreesRegressor', sklearn=sklearn.__version__, primary_comparator_family='static_scene_neural_cost_head',
        matching='same_query_draw_counts_features_normalizers_not_same_objective_decoder_or_compute',
        new_Torch_training=False, primary_forest_fits=54, primary_trees=6912)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or json.loads(finish.read_text())['groups'] != refs:
            raise ValueError('Exact forest summary differs')
        once(PUBLIC/'verification.json', dict(exact_replay=True, groups=144, summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), registration_sha256=sha(REG), independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='readout_verified' if verify else 'readout_complete', full_budget=summary['primary_full_roster_budget_supported'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register':load()
        elif args.phase == 'pilot':pilot(args.resume)
        elif args.phase == 'train':train(args.resume)
        else:evaluate(args.phase == 'verify', args.resume)


if __name__ == '__main__':main()
