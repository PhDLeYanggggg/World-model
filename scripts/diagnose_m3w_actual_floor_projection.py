"""SOURCE-only zero-update geometric cost check, never deployment evaluation."""
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_selection_cost_forest as run
from src.evaluation.m3w_selection_cost_fit_objective import score
from src.evaluation.m3w_actual_floor_projection import project

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_actual_floor_projection_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_actual_floor_projection_v1'
np = run.np


def main():
    _,producers,ids = run.admission()
    verify = json.loads((ROOT/'outputs/publication_readiness_2026_09/european_selection_bank_support_v1/verification.json').read_text())
    if not verify['all_record_support_counts_exact'] or not verify['all9_old_mean_cost_inferences_exact']:
        raise ValueError('Complete SOURCE support/geometry reconstruction required')
    finish = PUBLIC/'complete.json'
    if finish.exists(): raise ValueError('Completed SOURCE projection diagnostic immutable')
    # The independent diagnostic uses its own heartbeat, never the live forest process.
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    import os,resource,time
    start = time.monotonic()
    source = run.prior.source_inputs(json.loads(run.prior.frozen.REG.read_text()))
    models = run.parent.fitted(); rows = []
    for row in producers:
        case = run.transport.source_case(row,source,ids); c,y = case['causal'],case['target']
        state = run.prior.api.read_checkpoint(run.checked(models[row['trial'],'mean_absolute']['checkpoint']))
        data = run.prior.api.training_data(c,y)
        run.prior.api.exact(data['preprocess'],state['preprocess'])
        counts = state['sampler']['counts']; known = data['known']; legal = c['allowed']&known[:,None]
        bank = json.loads((run.prior.PRIVATE/'source_banks'/row['trial']/'complete.json').read_text())
        neural = np.load(run.checked(bank['prediction']),mmap_mode='r',allow_pickle=False)
        origin = source['origin'][c['ids']]
        physical = np.stack([run.prior.baseline_numpy(source['history'][c['ids']],k)-origin[:,None] for k in range(6)],1)
        paths = np.concatenate((physical,neural[:,None].astype(float)),1)
        radius = np.linalg.norm(paths-paths[:,row['baseline_index'],None],axis=-1).max(-1)
        p = run.prior.api.predict(state,c)
        q = project(p,radius,state['preprocess']['rms'])
        native = np.where(known[:,None,None],y/c['scales'][:,None,None],0.)
        denominator = c['scales'][:,None,None]*state['preprocess']['rms']
        before = np.square((p-y)/denominator).sum(-1)[legal]
        after = np.square((q-y)/denominator).sum(-1)[legal]
        if np.any(after > before+1e-9*np.maximum(1,before)):
            raise ValueError('Native feasible five-cost distance increased')
        scores = {}
        for target_name,target in (('native_normalized',native),('original_float32_training',data['truth'].numpy())):
            scores[target_name] = {name:score(value,target,c['allowed'],known,c['scales'],state['preprocess']['rms'],
                    c['groups'],counts,c['sites']) for name,value in (('original',p),('projected',q))}
        rows.append(dict(trial=row['trial'],SOURCE_sites=sorted(set(c['sites'])),scores=scores,
            legal_known_action_views=int(legal.sum()),prediction_sha256=run.api.fingerprint(p),
            projected_sha256=run.api.fingerprint(q),actual_radius_sha256=run.api.fingerprint(radius),
            all_native_five_cost_errors_nonincreasing=True,reference_predictions_unchanged=bool(np.array_equal(p[:,:,2:4],q[:,:,2:4]))))
        heartbeat = dict(pid=os.getpid(),state='SOURCE_projection_scored',producers=len(rows),total=9)
        run.write(PRIVATE/'heartbeat.json',heartbeat); print(json.dumps(heartbeat),flush=True)
        del data,case,c,y,state,physical,paths,p,q,native,neural
    run.once(finish,dict(result_source='fresh_SOURCE_zero_update_cost_projection_not_policy_success',
        pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rows=rows,plan_sha256=run.sha(PUBLIC/'plan.md'),script_sha256=run.sha(Path(__file__)),
        projection_sha256=run.sha(ROOT/'src/evaluation/m3w_actual_floor_projection.py'),
        new_training=False,policy_evaluation=False,development_labels_read=False,calibration_or_confirmation_access=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(complete=True,seconds=time.monotonic()-start)))


if __name__ == '__main__':
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
