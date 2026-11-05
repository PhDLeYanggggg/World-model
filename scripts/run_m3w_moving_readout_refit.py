"""Frozen-encoder readout repair on the unchanged three SOURCE pilot contexts."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_direct_tail_risk as parent
from src.world_model import m3w_frozen_readout_refit as api

source, old = parent.source, parent.old
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_readout_refit_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_readout_refit_v1'
CONFIG = ROOT/'configs/m3w_moving_readout_refit_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'src/world_model/m3w_frozen_readout_refit.py', ROOT/'tests/test_m3w_frozen_readout_refit.py')
POLICIES = parent.POLICIES + ('readout_veto', 'readout_matched')


def register(beat):
    spec = source.read(CONFIG)
    _, cfg, data, jobs, refs, parts, points, routers, critics = parent.register(beat)
    if (spec['new_readout_fits'] != 3 or spec['learned_parameters_per_fit'] != 130
            or spec['new_encoder_updates'] != 0 or spec['new_adam_updates'] != 0 or spec['rcond'] != 1e-10
            or spec['risk_budget'] != .02 or spec['easy_net_budget'] != .02
            or any(spec[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed readout intervention only, original scientific rules unchanged')
    if source.read(parent.PUBLIC/'verification.json')['analysis_sha256'] != source.sha(parent.PUBLIC/'analysis.json'):
        raise ValueError('Verified frozen previous pilot required')
    fitted = {}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        row = source.read(source.checked(ref)); fitted[row['job'], row['arm']] = row
    source.once(REG, source.stable(dict(configuration=spec, jobs=jobs,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(parent.REG), parent_freeze=source.artifact(parent.PUBLIC/'training_freeze.json'),
        parent_analysis=source.artifact(parent.PUBLIC/'analysis.json'),
        original_rules_unchanged=True, no_new_H=True, exposed_SOURCE_pilot=True,
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)))
    return spec, cfg, data, jobs, refs, parts, points, routers, critics, fitted


def path_for(job, kind='fits'):
    return PRIVATE/kind/(job['key']+'.pt.gz')


def train(spec, cfg, data, jobs, refs, parts, points, routers, critics, fitted, beat, guard, resume):
    receipts = []
    for job in jobs:
        p, _ = parent.parent.training_packet(data, job, refs, parts[job['key']][0], 1, points, routers)
        warm_ref = fitted[job['key'], 'mean']['checkpoint']
        warm = api.base.read_checkpoint(source.checked(warm_ref))
        if not all(int(v['step']) == 2000 for v in warm['optimizer']['state'].values()):
            raise ValueError('Real completed warm optimizer required')
        identity = dict(registration_sha256=source.sha(REG), warm_checkpoint=warm_ref, job=job)
        path = path_for(job)
        state = api.refit(p['learning'], warm, identity=identity, path=path, resume=resume and path.exists(),
            heartbeat=lambda **v: beat(job=job['key'], **v), guard=guard)
        if not receipts:
            repeat_path = path_for(job, 'independent_refit')
            repeated = api.refit(p['learning'], warm, identity=identity, path=repeat_path,
                resume=resume and repeat_path.exists(), heartbeat=lambda **v: beat(job=job['key'], repeat=True, **v), guard=guard)
            api.exact_refit(state, repeated)
            source.once(PUBLIC/'first_exact_refit.json', dict(exact=True, checkpoint=source.artifact(repeat_path)))
        rowpath = PUBLIC/'fits'/(job['key']+'.json')
        source.once(rowpath, source.stable(dict(job=job['key'], checkpoint=source.artifact(path),
            warm_checkpoint=warm_ref, seconds=state['seconds'], diagnostics=state['diagnostics'],
            learned_parameters=130, new_encoder_updates=0, new_adam_updates=0, validation_readout=False)))
        receipts.append(source.artifact(rowpath))
        beat(state='readout_refit_frozen', complete=len(receipts), total=3)
        del p, warm, state; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_readouts=3,
        all_before_validation=True, first_full_refit_exact=True, new_encoder_updates=0,
        new_adam_updates=0, fresh_readout_solves=3, extra_reproduction_solves=1))


def flow(truth, proposal, before, after):
    proposal, before, after = map(source.np.asarray, (proposal, before, after))
    for chosen in (before, after):
        if ((chosen >= 0) & (chosen != proposal)).any():
            raise ValueError('Only admission changes, not a new trajectory or amplitude')
    proposed = proposal >= 0
    masks = dict(retained=proposed & (before >= 0) & (after >= 0),
        newly_rejected=proposed & (before >= 0) & (after < 0),
        newly_admitted=proposed & (before < 0) & (after >= 0),
        still_rejected=proposed & (before < 0) & (after < 0))
    if not source.np.array_equal(sum(m.astype(int) for m in masks.values()), proposed.astype(int)):
        raise ValueError('Every proposal must appear exactly once in the flow table')
    result = {}
    for name, mask in masks.items():
        known = mask & truth['observed_future']; rows = source.np.flatnonzero(known)
        mass = truth['moments'][rows, proposal[rows]].sum(0)
        result[name] = dict(known_rows=int(known.sum()), unknown_rows=int((mask & ~truth['observed_future']).sum()),
            known_moments=mass.tolist(), known_net_benefit=float(mass[0]-mass[1]),
            known_all_budget_excess=float(mass[1]-.02*mass[2]),
            known_easy_budget_excess=float(mass[4]-.02*mass[3]),
            diagnostic_not_risk_certification=True)
    return result


def evaluate(spec, cfg, data, jobs, refs, parts, points, routers, critics, fitted, beat, guard, *, verify):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_readouts'] != 3 or not freeze['all_before_validation']:
        raise ValueError('Every new readout must be frozen before held outcomes')
    refits = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); refits[r['job']] = r
    records, receipts, fit_rows = [], [], []
    for job in jobs:
        for part_id in (0, 2):
            part = parts[job['key']][part_id]
            p = old.packet(data, job, refs, part, training=False); c, ids = p['causal'], p['ids']
            chosen, prediction, hashes = parent.decisions(p, job, part, points, routers, critics, fitted, part_id == 2)
            model = api.base.read_checkpoint(source.checked(refits[job['key']]['checkpoint']))
            if part_id == 2:
                parent.api.parent.assert_excluded(model, c, part['names'])
            prediction['readout'] = api.predict(model, c); hashes['readout'] = source.array_hash(prediction['readout'])
            chosen['readout_veto'], chosen['readout_matched'] = parent.api.choose(
                prediction['router'], prediction['all_action'], prediction['readout'], c)
            # Both policies are committed before this source's outcome/validity read.
            target = data['target_eval'][ids]-data['origin'][ids, None]
            valid = data['valid'][ids]
            truth = old.amplitude.targets(p, target, valid, job['easy_cut'])
            paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in chosen.items()}
            rows = []
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                policies = {name: source.readout.policy(x[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                    p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, x in paths.items()}
                q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                pair = source.readout.paired(paths['mean_veto'][at], paths['readout_veto'][at], p['floor'][at], target[at], valid[at])
                matched = source.matched_utility(paths['readout_matched'][at], paths['readout_veto'][at],
                    p['floor'][at], target[at], valid[at], chosen['readout_matched'][at], chosen['readout_veto'][at], q)
                local_truth = dict(moments=truth['moments'][at], observed_future=truth['observed_future'][at])
                reliability = {name: parent.reliability(prediction[name][at], local_truth,
                    dict(proposals=chosen['public_router'][at], admitted=chosen[name+'_veto'][at])) for name in ('mean', 'readout')}
                rows.append(dict(job=job['key'], site=str(site), policies=policies,
                    readout_minus_mean=pair, matched=matched, reliability=reliability,
                    action_flow=flow(local_truth, chosen['public_router'][at], chosen['mean_veto'][at], chosen['readout_veto'][at])))
            if part_id == 0:
                fit_rows.extend(rows)
                del p, prediction, paths, truth; source.gc.collect(); guard()
                continue
            key = job['key']+'_val2_propose1'
            previous = source.read(parent.PUBLIC/'groups'/(key+'.json'))
            for name in parent.POLICIES:
                if source.array_hash(chosen[name]) != previous['choice_hashes'][name]:
                    raise ValueError('All parent decisions must match')
            for name, value in previous['prediction_hashes'].items():
                if hashes[name] != value:
                    raise ValueError('Parent predictions changed')
            for row in rows:
                previous_row = next(r for r in previous['rows'] if r['site'] == row['site'])
                for name in parent.POLICIES:
                    if row['policies'][name] != previous_row['policies'][name]:
                        raise ValueError('Old metric readout changed')
            cache = PRIVATE/'choices'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(readout_prediction=prediction['readout'], readout_veto=chosen['readout_veto'],
                readout_matched=chosen['readout_matched'], ids=ids)
            if cache.exists():
                with source.np.load(cache, allow_pickle=False) as saved:
                    for name, value in payload.items():
                        source.np.testing.assert_array_equal(saved[name], value)
            elif verify:
                raise ValueError('Original inference cache missing')
            else:
                source.np.savez_compressed(cache, **payload)
            output = PUBLIC/'groups'/(key+'.json')
            source.once(output, source.stable(dict(rows=rows, prediction_hashes=hashes,
                choice_hashes={k: source.array_hash(v) for k, v in chosen.items()},
                cache=source.artifact(cache), parent_controls_exact=True)))
            records.extend(rows); receipts.append(source.artifact(output))
            beat(state='readout_refit_evaluation', complete=len(receipts), total=3)
            del p, prediction, paths, truth; source.gc.collect(); guard()
    source.once(PUBLIC/'fitting_readout.json', source.stable(fit_rows))
    sites = sorted({r['site'] for r in records})

    def macro(rows, getter):
        by_site = {}
        for s in sorted({r['site'] for r in rows}):
            values = [getter(r) for r in rows if r['site'] == s]
            by_site[s] = float(source.np.mean(values)) if all(v is not None for v in values) else None
        return dict(by_source=by_site, mean=float(source.np.mean(list(by_site.values())))
            if all(v is not None for v in by_site.values()) else None, CI95=None, descriptive_pilot=True)

    def summarize(rows):
        return {name: dict(known=macro(rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
            switch_percent=macro(rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']),
            hard=macro(rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
            safety=parent.safety(rows, name)) for name in POLICIES}

    contexts = []
    for job in jobs:
        rows = [r for r in records if r['job'] == job['key']]
        lower = macro(rows, lambda r: r['readout_minus_mean']['completion_lower_pp'])['mean']
        before, after = parent.safety(rows, 'mean_veto'), parent.safety(rows, 'readout_veto')
        any_switch = any(r['policies']['readout_veto']['switch_rate'] > 0 for r in rows)
        expand = lower is not None and lower > 0 and any_switch and all(after[k] <= before[k]
            for k in ('known_all_violations', 'known_easy_violations'))
        contexts.append(dict(job=job['key'], direct_lower_pp=lower, original_safety=before,
            refit_safety=after, expansion_condition=bool(expand)))
    result = dict(held_policy=summarize(records), fit_policy=summarize(fit_rows),
        held_readout_minus_mean={m: macro(records, lambda r, m=m: r['readout_minus_mean'][m])
            for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')},
        fit_readout_minus_mean={m: macro(fit_rows, lambda r, m=m: r['readout_minus_mean'][m])
            for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')},
        per_context=contexts, expansion_supported=all(v['expansion_condition'] for v in contexts),
        sources=len(sites), source_views=len(records), group_receipts=receipts,
        unavailable_matched_source_views=sum(r['matched']['contrast']['known_pp'] is None for r in records),
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        no_new_H=True, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'analysis.json', source.stable(result))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_predictions_choices_and_readouts_exact=True,
            all_parent_controls_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        start, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        values = register(beat)
        if args.phase == 'register':
            return
        for path in (*FILES, REG):
            body = source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(body).hexdigest() != source.sha(path):
                raise ValueError('Commit exact experiment definition before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase receipt is immutable')
        if args.phase == 'train':
            train(*values, beat, guard, args.resume)
        else:
            evaluate(*values, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch_representation=True, new_encoder_training=False, no_new_H=True))
        beat(state='complete')


if __name__ == '__main__':
    main()
