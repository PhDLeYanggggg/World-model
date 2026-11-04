"""Fitting-only clipping diagnostic, without new held scores or training."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as source
from src.evaluation.m3w_input_clipping_audit import profile
from src.world_model.m3w_multiaction_cost import select_actions
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

np = source.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_input_clipping_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_input_clipping_v1'
PARENT = source.PUBLIC
CONFIG = source.CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('evaluate','verify'))
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    source.PRIVATE = PRIVATE
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg = source.read(CONFIG)
        start,guard,beat = source.runtime(cfg,'fitting_input_clipping')
        verified = source.read(PARENT/'verification.json')
        if not verified['all36cost_predictions_and_policy_scores_exact']:
            raise ValueError('Complete verified parent cost bank required')
        data,jobs,refs = source.admission(cfg,beat)
        parents = {}
        for ref in source.read(PARENT/'training_freeze.json')['fits']:
            receipt = source.read(source.checked(ref))
            if receipt['arm'] == 'local_history':
                parents[receipt['job']['key']] = receipt['checkpoint']
        if len(parents) != 18:
            raise ValueError('All18frozen local-history heads required')
        result = []
        for job in jobs:
            packet = source.view(data,job,refs,training=True)
            packet.pop('learning',None)
            causal,ids = packet['causal'],packet['ids']
            state = source.api.base.base.read_checkpoint(source.checked(parents[job['key']]))
            predicted = source.api.predict(state,causal)
            choice,_ = select_actions(predicted,causal['allowed'],arm='physical_neural')
            target = supported_bound_targets(packet['paths'],
                data['target_eval'][ids]-data['origin'][ids,None], data['valid'][ids],
                job['floor_index'],job['easy_cut'])
            pr = state['preprocess']
            # Shared context slots are zeroed by the local arm; audit only its380active inputs.
            n = source.api.COMMON_DIMENSION
            regions = dict(all_neural=np.ones(len(ids),bool),frozen_parent_independent=choice == 6)
            values = profile(causal['x'][:,6,:n],pr['mean'][:n],pr['std'][:n],pr['clip'],
                regions,target['observed_future'],target['moments'][:,6])
            physical = profile(causal['x'][:,:6,:n].reshape(-1,n),pr['mean'][:n],pr['std'][:n],pr['clip'],
                dict(all_physical=np.ones(6*len(ids),bool)),np.repeat(target['observed_future'],6),
                target['moments'][:,:6].reshape(-1,5))
            result.append(dict(job=job['key'],neural=values,physical=physical,
                original_physical_target_rms=pr['rms'].tolist(),
                fitting_ids_sha256=source.hashlib.sha256(ids.tobytes()).hexdigest()))
            beat(state='fitting_clipping_checked',job=job['key'],complete=len(result),total=18)
            del packet,causal,state,predicted,choice,target
            source.gc.collect()
            guard()
        value = source.stable(dict(rows=result,critic_fitting_only=True,
            result_source='fresh_fitting_only_post_readout_diagnosis_not_independent_test',
            parent_summary=source.artifact(PARENT/'summary.json'),
            parent_verification=source.artifact(PARENT/'verification.json'),
            bindings={str(p.relative_to(ROOT)):source.sha(p) for p in (Path(__file__),
                ROOT/'src/evaluation/m3w_input_clipping_audit.py', ROOT/'tests/test_m3w_input_clipping_audit.py')},
            new_training=False,new_held_scores=False,new_policy_or_threshold=False,
            stage5c_executed=False,smc_enabled=False))
        source.once(PUBLIC/'analysis.json',value)
        source.once(PUBLIC/(args.phase+'_execution.json'),dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            all_scalars_exact=args.phase == 'verify',new_training=False))
        beat(state='complete',verify=args.phase == 'verify')


if __name__ == '__main__':
    main()
