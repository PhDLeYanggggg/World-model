"""Registered real-data public regression-deferral control on fixed Moving forecasts."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_source_separated_critics as old
from src.world_model import m3w_moving_deferral as api
from src.world_model.m3w_multiaction_cost import select_actions
from src.world_model.m3w_source_separated_critics import matched_self

source = old.source
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_public_deferral_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/moving_public_deferral_v1'
CONFIG = ROOT / 'configs/m3w_moving_public_deferral_v1.json'
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC / 'protocol.md',
    ROOT / 'src/world_model/m3w_moving_deferral.py', ROOT / 'src/world_model/m3w_regression_deferral_loss.py',
    ROOT / 'tests/test_m3w_moving_deferral.py', ROOT / 'tests/test_m3w_moving_deferral_runner.py')
POLICIES = ('physical_floor', 'point_proposer', 'point_veto', 'point_matched',
    'public_router', 'public_veto', 'public_matched')
CONTRASTS = ('public_minus_point', 'public_veto_minus_point_veto')


def register(beat):
    spec = source.read(CONFIG)
    if (spec['training'] != source.read(old.CONFIG)['training']
            or spec['new_heads'] != 54 or spec['primary_updates'] != 108000
            or spec['parameters'] != 27969 or spec['risk_budget'] != .02 or spec['easy_net_budget'] != .02
            or any(spec[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Complete matched budget and original scientific rules required')
    _, cfg, data, jobs, refs, parts = old.register(beat)
    verification = source.read(old.PUBLIC / 'verification.json')
    if verification['analysis_sha256'] != source.sha(old.PUBLIC / 'analysis.json'):
        raise ValueError('Frozen negative parent must remain verified')
    frozen = {}
    for ref in source.read(old.PUBLIC / 'training_freeze.json')['fits']:
        r = source.read(source.checked(ref))
        if r['replica'] == 0:
            frozen[r['job'], r['part']] = r
    if len(frozen) != 54:
        raise ValueError('All original point proposers/critics required')
    source.once(REG, source.stable(dict(configuration=spec,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(old.REG), parent_freeze=source.artifact(old.PUBLIC / 'training_freeze.json'),
        jobs=jobs, partitions=parts, forecasts=refs, original_rules_unchanged=True,
        original24group_primary='not_estimable_not_passed', exposed_SOURCE_only=True,
        method='Mao2024_Eq3_task_adaptation_not_original_experiment_or_theorem_reproduction',
        new_H_readout=False, independent_confirmation=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    return spec, cfg, data, jobs, refs, parts, frozen


def checkpoint(job, part, kind='fits'):
    return PRIVATE / kind / (job['key'] + f'_part{part}.pt.gz')


def one(p, job, part, spec, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(p['learning'], seed=job['seed'], settings=spec['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job, part=part,
            feature_arm='local_history', no_new_H=True), path=checkpoint(job, part, kind),
        heartbeat=lambda **v: beat(job=job['key'], part=part, **v), guard=guard, resume=resume, stop_at=stop_at)


def pilot(spec, cfg, data, jobs, refs, parts, frozen, beat, guard, resume):
    rows = []
    for fold in range(3):
        job = next(j for j in jobs if j['critic_fold'] == fold)
        p = old.packet(data, job, refs, parts[job['key']][0], training=True)
        path = checkpoint(job, 0, 'pilot_resume')
        step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if path.exists() and not resume:
            raise ValueError('Explicit pilot resume required')
        if step < 50:
            one(p, job, 0, spec, beat, guard, kind='pilot_resume', resume=path.exists(), stop_at=50)
        a = one(p, job, 0, spec, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        path = checkpoint(job, 0, 'pilot_direct')
        b = one(p, job, 0, spec, beat, guard, kind='pilot_direct', resume=resume and path.exists(), stop_at=100)
        api.exact_state(a, b)
        source.np.testing.assert_array_equal(api.predict(a, p['causal']), api.predict(b, p['causal']))
        if not any(not api.torch.equal(b['model'][k], b['initial_model'][k]) for k in b['model']):
            raise ValueError('Real optimizer update required')
        rows.append(dict(job=job['key'], steps=100, seconds=b['seconds'], parameters=b['parameters'],
            exact_resume=True, budget=b['budget']))
        del p, a, b; source.gc.collect(); guard()
    source.once(PUBLIC / 'pilot.json', source.stable(dict(cells=rows, all3_resume_exact=True,
        actual_updates=600, projected_primary_seconds_not_completion=18*sum(r['seconds']/100*r['budget'] for r in rows),
        validation_readout=False)))


def train(spec, cfg, data, jobs, refs, parts, frozen, beat, guard, resume):
    if not source.read(PUBLIC / 'pilot.json')['all3_resume_exact']:
        raise ValueError('Real resume pilot first')
    receipts = []
    for job in jobs:
        for part in parts[job['key']]:
            k = part['part']; p = old.packet(data, job, refs, part, training=True)
            path = checkpoint(job, k)
            state = one(p, job, k, spec, beat, guard, resume=resume and path.exists())
            if state['step'] != 2000 or state['budget'] != 2000 or state['parameters'] != spec['parameters']:
                raise ValueError('Full registered budget and exact head required')
            if not receipts:
                repeat_path = checkpoint(job, k, 'full_refit')
                repeat = one(p, job, k, spec, beat, guard, kind='full_refit', resume=resume and repeat_path.exists())
                api.exact_state(state, repeat)
                source.once(PUBLIC / 'first_full_refit.json', dict(exact=True, checkpoint=source.artifact(repeat_path)))
                del repeat
            parent = api.base.read_checkpoint(source.checked(frozen[job['key'], k]['checkpoint']))
            for key in ('source_sites', 'fit_ids', 'preprocess', 'sampler', 'draw_hash'):
                api.base.exact(state[key], parent[key])
            rowpath = PUBLIC / 'fits' / (path.name.removesuffix('.pt.gz')+'.json')
            source.once(rowpath, source.stable(dict(job=job['key'], part=k, seed=job['seed'],
                checkpoint=source.artifact(path), steps=state['step'], budget=state['budget'],
                parameters=state['parameters'], seconds=state['seconds'], trace=state['trace'],
                fit_sources=state['source_sites'], fit_ids_sha256=source.array_hash(state['fit_ids']),
                parent_sampling_and_preprocess_exact=True, own_part_preprocessing=True, validation_readout=False)))
            receipts.append(source.artifact(rowpath))
            beat(state='public_deferral_fit_frozen', job=job['key'], part=k, complete=len(receipts), total=54)
            del p, state, parent; source.gc.collect(); guard()
    source.once(PUBLIC / 'training_freeze.json', dict(fits=receipts, complete_heads=54,
        primary_updates=108000, all_before_validation=True, parent_preprocessing_and_sampling_exact=True))


def causal_choices(point, logits, critic, causal):
    own, point_utility = select_actions(point, causal['allowed'], arm='physical_neural')
    public, public_utility = api.choose(logits, causal['allowed'])
    choices = dict(physical_floor=source.np.full(len(own), -1, source.np.int64),
        point_proposer=own, public_router=public)
    for name, proposal, utility in (('point', own, point_utility), ('public', public, public_utility)):
        ix = source.np.flatnonzero(proposal >= 0)
        costs = critic[ix, proposal[ix]]
        keep = ((costs[:, 2] > 0) & (costs[:, 3] > 0)
            & (costs[:, 1] <= .02*costs[:, 2]) & (costs[:, 4] <= .02*costs[:, 3]))
        veto = source.np.full(len(own), -1, source.np.int64); veto[ix[keep]] = proposal[ix[keep]]
        choices[name+'_veto'] = veto
        choices[name+'_matched'] = matched_self(proposal, utility, veto, causal['groups'], causal['ids'])
    return choices


def evaluate(spec, cfg, data, jobs, refs, parts, frozen, beat, guard, *, verify):
    freeze = source.read(PUBLIC / 'training_freeze.json')
    if freeze['complete_heads'] != 54 or not freeze['all_before_validation']:
        raise ValueError('Freeze all new models before validation')
    fitted = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); fitted[r['job'], r['part']] = r
    records, receipts = [], []
    for job in jobs:
        for part in parts[job['key']]:
            v = part['part']; others = sorted(set(range(3))-{v})
            p = old.packet(data, job, refs, part, training=False); c, ids = p['causal'], p['ids']
            routers, points, hashes = {}, {}, {}
            for k in others:
                for name, mapping, predictor, output in (
                    ('point', frozen, old.amplitude.predict, points),
                    ('public', fitted, api.predict, routers)):
                    state = api.base.read_checkpoint(source.checked(mapping[job['key'], k]['checkpoint']))
                    if (set(state['fit_ids']) & set(ids) or set(state['preprocess']['source_names']) & set(part['names'])):
                        raise ValueError('Every fitted field must exclude validation')
                    output[k] = predictor(state, c)
                    hashes[f'{name}{k}'] = source.array_hash(output[k])
            pending = []
            for proposer in others:
                critic = next(k for k in others if k != proposer)
                chosen = causal_choices(points[proposer], routers[proposer], points[critic], c)
                paths = {a: old.rollout(p['paths'], p['floor'], x) for a, x in chosen.items()}
                pending.append((proposer, critic, chosen, paths))
            # Targets are first read after all causal decisions have been fixed.
            target, valid = data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids]
            for proposer, critic, chosen, paths in pending:
                key = job['key']+f'_val{v}_propose{proposer}'
                cache = PRIVATE / 'choices' / (key+'.npz'); cache.parent.mkdir(exist_ok=True)
                cached = dict(chosen, ids=ids)
                if cache.exists():
                    with source.np.load(cache, allow_pickle=False) as previous:
                        for name, value in cached.items():
                            source.np.testing.assert_array_equal(previous[name], value)
                elif verify:
                    raise ValueError('Missing original causal decision cache')
                else:
                    source.np.savez_compressed(cache, **cached)
                rows = []
                for site in sorted(set(p['sites'])):
                    at = p['sites'] == site
                    q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                    policies = {a: source.readout.policy(x[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                        p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for a, x in paths.items()}
                    matched, full = {}, {}
                    for name in ('point', 'public'):
                        a, b = name+'_matched', name+'_veto'
                        matched[name] = source.matched_utility(paths[a][at], paths[b][at], p['floor'][at], target[at],
                            valid[at], chosen[a][at], chosen[b][at], q)
                        full[name] = source.readout.paired(paths[a][at], paths[b][at], p['floor'][at], target[at], valid[at])
                    contrasts = {name: source.readout.paired(paths[a][at], paths[b][at], p['floor'][at], target[at], valid[at])
                        for name, a, b in (('public_minus_point', 'point_proposer', 'public_router'),
                            ('public_veto_minus_point_veto', 'point_veto', 'public_veto'))}
                    rows.append(dict(site=str(site), job=job['key'], seed=job['seed'], validation_part=v,
                        proposer_part=proposer, critic_part=critic, policies=policies,
                        nonempty_matched=matched, secondary_full_strategy=full, direct_contrasts=contrasts))
                path = PUBLIC / 'groups' / (key+'.json')
                source.once(path, source.stable(dict(rows=rows, prediction_hashes=hashes,
                    choice_hashes={a: source.array_hash(x) for a, x in chosen.items()},
                    decision_cache=source.artifact(cache), no_new_H=True)))
                receipts.append(source.artifact(path)); records.extend(rows)
                beat(state='public_deferral_replay' if verify else 'public_deferral_validation', complete=len(receipts), total=108)
            del p, routers, points, pending; source.gc.collect(); guard()
    if len(records) != 276 or len(receipts) != 108:
        raise ValueError('All original required source/orientation views retained')
    sites = sorted({r['site'] for r in records})
    ci = lambda getter: source.group_interval(records, getter, sites, cfg)
    measures = ('known_pp', 'completion_lower_pp', 'completion_upper_pp')
    matched = {a: {m: ci(lambda r, a=a, m=m: r['nonempty_matched'][a]['contrast'][m]) for m in measures} for a in ('point', 'public')}
    full = {a: {m: ci(lambda r, a=a, m=m: r['secondary_full_strategy'][a][m]) for m in measures} for a in ('point', 'public')}
    direct = {a: {m: ci(lambda r, a=a, m=m: r['direct_contrasts'][a][m]) for m in measures} for a in CONTRASTS}
    safety = {}
    for arm in POLICIES:
        views = [v for r in records for v in r['policies'][arm]['recordings'].values()]
        if len(views) != 696:
            raise ValueError('All recording views retained')
        safety[arm] = dict(required=696, supported=sum(v['supported'] for v in views),
            known_all_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
            known_easy_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
            undefined_selected_easy=sum(v['completion_easy_risk_upper'] is None for v in views),
            full_original_risk_pass=all(v['supported'] for v in views))
    analysis = source.stable(dict(nonempty_matched=matched, secondary_full_strategy=full, direct_contrasts=direct,
        safety=safety, switch_percent={a: ci(lambda r, a=a: 100*r['policies'][a]['switch_rate']) for a in POLICIES},
        unavailable_primary_views={a: sum(r['nonempty_matched'][a]['contrast']['known_pp'] is None for r in records) for a in ('point', 'public')},
        group_receipts=receipts, source_units=len(sites), new_H_readout=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    source.once(PUBLIC / 'analysis.json', analysis)
    primary = matched['public']['completion_lower_pp']
    source.once(PUBLIC / 'decision.json', dict(development_criteria_pass=bool(primary['CI95'] is not None and primary['CI95'][0] > 0
        and safety['public_veto']['full_original_risk_pass']), deployment_changed=False, submission_ready=False,
        original24group_primary='not_estimable_not_passed', new_H_readout=False, stage5c_executed=False, smc_enabled=False))
    if verify:
        source.once(PUBLIC / 'verification.json', dict(all108_policy_blocks_and_prediction_hashes_exact=True,
            analysis_sha256=source.sha(PUBLIC / 'analysis.json'), no_new_H=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        start, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        values = register(beat)
        if args.phase == 'register':
            return
        for path in (*FILES, REG):
            body = source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if source.hashlib.sha256(body).hexdigest() != source.sha(path):
                raise ValueError('Commit definition before real fitting')
        receipt = PUBLIC / (args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase immutable; do not repeat training')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](*values, beat, guard, args.resume)
        else:
            evaluate(*values, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss, real_torch=True,
            new_H_readout=False, new_trajectory_training=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
