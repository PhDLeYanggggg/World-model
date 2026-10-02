"""TRAIN-only iteration-cap diagnosis; no validation prediction or policy."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_cost_aligned_positive_harm as run
import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--newton-first-tree', action='store_true')
    args = parser.parse_args()
    run.parent.core.torch.set_num_threads(4); run.parent.core.torch.set_num_interop_threads(1)
    assert run.registration() == json.loads((run.PUBLIC/'registration.json').read_text())
    if args.newton_first_tree:
        from src.world_model import m3w_cost_harm_newton
        run.api = m3w_cost_harm_newton
    docs = run.parent.parent.docs()
    _, _, data, jobs, oid, _, _, _ = run.parent.inner.old.load()
    q, _ = run.prior.prior.past_quality(data)
    c = next(iter(run.parent.parent.contexts(data, jobs, oid))); site = run.parent.inner.sources(c)[0]
    group = c['name']+'_fit_'+site; source = docs[group, 17]
    assert run.sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
    state = joblib.load(ROOT/source['checkpoint']['path'])
    _, ids, x, env, y, _, _ = run.parent.inner.training_arrays(c, data, site)
    tr, _, partition = run.parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
    x, env, y, q = x[tr], env[tr], y[tr], q[ids[tr]]; tid = ids[tr]
    f = run.api.positive.base.forest; pr = state['preprocess']; known = np.isfinite(y).all(1)
    z, _ = f.causal_inputs(x, env, pr)
    w, _ = f.core.weights(data['sites'][tid], data['recordings'][tid], data['frames'][tid], known)
    mean = w@q; std = np.sqrt(w@((q-mean)**2)).clip(1e-6)
    qq = run.api.positive.base.standardize(q, mean, std)[known]
    scales = run.api.harm_scales(pr['scale'], pr['rms'][5:]); events = []; start = time.monotonic()
    for tree_id, tree in enumerate(state['model'].estimators_):
        leaves = tree.apply(z[known]); fixed = tree.tree_.value[leaves, :, 0]/f.FACTORS*pr['rms']*pr['scale']
        effective = run.api.score_equivalent_targets(y[known], fixed[:, :5], pr['rms'][5:])
        failed = False
        for cap in ((128,) if args.newton_first_tree else (128, 512, 2048)):
            event = dict(pid=os.getpid(), group=group, tree=tree_id, cap=cap, rows=int(known.sum()))
            print(json.dumps(dict(event, state='training_only_solver_probe')), flush=True)
            before = time.monotonic()
            try:
                fit = run.api.cost_leaf(leaves, qq, y[known][:, run.api.HARM], w[known], scales,
                    squared_target=effective, max_iter=cap)
                event.update(status='converged', maximum_gradient=fit['maximum_gradient'],
                    max_iterations=int(fit['iterations'].max()), loss=fit['loss'])
            except RuntimeError as exc:
                event.update(status='failed', error=str(exc)); failed = True
            event['seconds'] = time.monotonic()-before; events.append(event); print(json.dumps(event), flush=True)
            if event['status'] == 'converged': break
        if failed or args.newton_first_tree:
            report = dict(result_source='fresh_run_train_only_solver_diagnosis', events=events,
                validation_evaluated=False, new_complete_models=0, seconds=time.monotonic()-start,
                partition=partition, training_ids_hash=run.parent.base.inter.array_hash(tid),
                objective_changed=False, risk_budget_changed=False)
            name = 'solver_newton_diagnosis.json' if args.newton_first_tree else 'solver_diagnosis.json'
            run.once(run.PUBLIC/name, report)
            return
    raise RuntimeError('Original real failure did not reproduce')


if __name__ == '__main__': main()
