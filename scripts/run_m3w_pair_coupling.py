"""Fixed-bank source-excluded component test of analytic pair coupling."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_prefix_actor as base
from src.world_model import m3w_pair_coupling as model

np = base.np
read, sha, checked, artifact, once, stable = base.read, base.sha, base.checked, base.artifact, base.once, base.stable
HOME = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = HOME/'moving_pair_coupling_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_pair_coupling_v1'
PARENT = HOME/'moving_producer_crossfit_v1'
REG = PUBLIC/'registration.json'
FILES = [Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_pair_coupling.py']
PARENTS = [PARENT/name for name in ('registration.json', 'population.json', 'analysis.json')]
PARENTS += [HOME/'moving_camera_frame_v1/registration_v2.json', Path(base.__file__), ROOT/'src/world_model/m3w_european_source_forecast.py']


def register():
    value = stable(dict(files={str(p.relative_to(ROOT)): sha(p) for p in FILES}, parents=[artifact(p) for p in PARENTS],
        role='previously_exposed_A_source_excluded_component_not_policy', forecast_seed=17,
        ridge=.001, neighbors=4, max_switched=2, risk_budget_unchanged=.02,
        bootstrap_draws=3000, bootstrap_seed=73107, independent_confirmation=False,
        stage5c_executed=False, smc_enabled=False, deployment_changed=False, submission_ready=False))
    once(REG, value)
    return value


def groups(query):
    order = np.argsort(query, kind='stable')
    return np.split(order, np.flatnonzero(query[order][1:] != query[order][:-1])+1)


def arrays(names):
    refs = read(PARENT/'population.json')['files']
    return {k: np.load(checked(refs[k]), mmap_mode='r', allow_pickle=False) for k in names}


def design(data, ref):
    with np.load(checked(ref['forecast']), allow_pickle=False) as bank:
        ix, neural = bank['local_ids'], bank['relative_prediction'].astype(float)
        np.testing.assert_array_equal(data['global_ids'][ix], bank['global_ids'])
    sites = data['sites'][ix]; job = ref['job']
    if set(sites) != set(job['donor_sites']) or set(sites)&set(job['fit_sites']) or not ref['teacher_exposure_disjoint']:
        raise ValueError('Original frozen producer must exclude all donors in this design')
    history, ids, query = (data[k][ix] for k in ('history', 'global_ids', 'queries'))
    floor = base.baseline_numpy(history, 1)-data['origin'][ix, None]
    subsets = groups(query)
    if any(len(set(sites[g])) != 1 for g in subsets): raise ValueError('A query cannot cross sources')
    return ix, sites, history, ids, floor, neural, subsets


def fit_predict(d, labels):
    ix, sites, history, ids, floor, neural, subsets = d
    names = sorted(set(sites)); moments = {s: [np.zeros((12, 9, 9)), np.zeros((12, 9)), 0] for s in names}
    for q in subsets:
        g = model.geometry(history[q], floor[q], neural[q], ids[q]); state = moments[sites[q[0]]]
        gram, rhs = model.moments(g, labels['target_eval'][ix[q]], labels['valid'][ix[q]])
        state[0] += gram; state[1] += rhs; state[2] += 1
    coef = {}
    for held in names:
        # Fit only other donor sources from the same frozen forecaster design.
        fit = [m for k, m in moments.items() if k != held]
        a, b = (sum(v[i]/v[2] for v in fit)/len(fit) for i in range(2))
        coef[held] = np.linalg.solve(a+.001*np.eye(9), b[..., None])[..., 0]
    choices = np.zeros((len(ix), 2), bool)
    for q in subsets:
        g = model.geometry(history[q], floor[q], neural[q], ids[q]); u, joint = model.costs(g, coef[sites[q[0]]], len(q))
        choices[q] = np.stack(model.choose(u, *g[:2], joint, ids[q]), axis=1)
    return dict(ix=ix, choices=choices, coefficients=np.stack(list(coef.values())), sources=np.asarray(names))


def score(d, frozen, labels, cuts):
    ix, sites, history, ids, floor, neural, subsets = d
    choice = frozen['choices']; y, valid = (labels[k][ix] for k in ('target_eval', 'valid'))
    agg = {s: dict(queries=0, scored_pair_queries=0, requested_pair_steps=0, observed_pair_steps=0,
        changed_queries=0, switched_rows=0, squared=np.zeros(2), distance=np.zeros(2)) for s in set(sites)}
    row_weights = np.zeros(len(ix))
    for q in subsets:
        a = agg[sites[q[0]]]; g = model.geometry(history[q], floor[q], neural[q], ids[q])
        counts = choice[q].sum(0)
        if not (counts == min(2, len(q)-1)).all(): raise ValueError('Changed intervention count')
        row_weights[q] = 1/len(q); a['queries'] += 1; a['changed_queries'] += bool(np.any(choice[q, 0] != choice[q, 1]))
        a['switched_rows'] += int(counts[0]); a['requested_pair_steps'] += 12*len(g[0])
        metrics = [model.error(g, y[q], valid[q], choice[q, k]) for k in range(2)]
        n = metrics[0][2]; a['observed_pair_steps'] += n
        if n:
            a['scored_pair_queries'] += 1
            a['squared'] += np.array([m[0] for m in metrics])/n
            a['distance'] += np.array([m[1] for m in metrics])/n
    paths = [np.where(choice[:, k, None, None], neural, floor) for k in range(2)]
    y = y-history[:, -1, None]
    output = []
    for s, a in sorted(agg.items()):
        at = sites == s; w = row_weights[at]; n = a['scored_pair_queries']
        output.append(dict(source=s, **{k:v for k,v in a.items() if k not in ('squared', 'distance')},
            pair_squared_available=(a['squared']/n).tolist() if n else None,
            pair_norm_available=(a['distance']/n).tolist() if n else None,
            agent=base.statistics(paths[0][at], paths[1][at], floor[at], y[at], valid[at], w/w.sum(), cuts['easy'], cuts['hard'])))
    return output


def summarize(rows):
    draw = np.random.default_rng(73107).integers(0, len(rows), size=(3000, len(rows)))
    def paired(values):
        if any(v is None for v in values): return None
        d = np.asarray(values, float)
        return dict(point=float(d.mean()), interval95=np.quantile(d[draw].mean(1), [.025, .975]).tolist(),
            positive_sources=int((d > 0).sum()))
    return dict(pair_squared_reduction=paired([None if r['pair_squared_available'] is None else
        r['pair_squared_available'][0]-r['pair_squared_available'][1] for r in rows]),
        pair_norm_reduction=paired([None if r['pair_norm_available'] is None else
        r['pair_norm_available'][0]-r['pair_norm_available'][1] for r in rows]),
        agent_gain_pp=paired([r['agent'][-1]['paired_known_pp'] for r in rows]),
        agent_completion_lower_pp=paired([r['agent'][-1]['paired_completion_lower_pp'] for r in rows]))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('register', 'run', 'replay')); args = parser.parse_args()
    reg = register()
    if args.phase == 'register': return
    for path in [*FILES, REG]:
        committed = subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT)
        if committed != path.read_bytes(): raise ValueError('Commit the component test before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = time.monotonic()
        def beat(**status):
            if shutil.disk_usage(ROOT).free < 2*1024**3: raise OSError('Keep 2 GiB free; completed design files are resumable')
            if time.monotonic()-start > 43200: raise TimeoutError('Resume from completed design files')
            event = stable(dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **status))
            base.source.atomic(PRIVATE/'heartbeat.json', event); print(event, flush=True)
        beat(state='started', machine=platform.machine(), BLAS_threads=4, workers=0)
        data = arrays(('global_ids', 'history', 'origin', 'queries', 'sites'))
        labels = arrays(('target_eval', 'valid'))
        refs = read(PARENT/'analysis.json')['forecasts']; cuts = read(PARENTS[3])['cuts']
        indices, hashes = [], []
        for ref in refs:
            d = design(data, ref); key = ref['job']['key']; p = PRIVATE/(key+'.npz'); mark = PRIVATE/(key+'.json')
            if not mark.exists() or args.phase == 'replay':
                result = fit_predict(d, labels)
                if args.phase == 'replay':
                    with np.load(checked(read(mark)['array']), allow_pickle=False) as saved:
                        for k, v in result.items(): np.testing.assert_array_equal(saved[k], v)
                else:
                    pending = p.with_suffix('.pending')
                    with pending.open('wb') as f: np.savez_compressed(f, **result)
                    pending.replace(p); once(mark, dict(array=artifact(p), registration_sha256=sha(REG)))
            saved = read(mark)
            if saved['registration_sha256'] != sha(REG): raise ValueError('Different registration in checkpoint')
            hashes.append(saved['array']); indices.append(d[0]); beat(state='frozen_design', key=key)
        np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(125540))
        freeze = dict(banks=hashes, all_choices_fixed_before_scoring=True, new_forecast_training=False)
        once(PUBLIC/'choice_freeze.json', freeze)
        rows = []
        for ref, cache in zip(refs, hashes):
            d = design(data, ref)
            with np.load(checked(cache), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ix'], d[0]); rows.extend(score(d, z, labels, cuts[str(ref['job']['producer_fold'])]))
            beat(state='scored_design', key=ref['job']['key'])
        if len(rows) != 14 or len({r['source'] for r in rows}) != 14: raise ValueError('Retain every source')
        result = stable(dict(rows=125540, sources=rows, paired=summarize(rows), original_policy_gate_not_replaced=True,
            original_full_recording_gate_run=False, original_risk_budgets_enforced_by_this_diagnostic=False,
            split='within_frozen_design_leave_one_donor_source_out', new_independent_outcomes=False,
            stage5c_executed=False, smc_enabled=False, submission_ready=False))
        once(PUBLIC/'analysis.json', result)
        beat(state='complete', scalar_sha256=sha(PUBLIC/'analysis.json'))


if __name__ == '__main__':
    main()
