"""Reconstruct frozen actor query draws without reading trajectories or future labels."""
import json
from pathlib import Path
import platform
import sys

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required before Torch')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_free_allocation_actor as parent


def reconstruct(keys,*,seed,steps,batch_size):
    np,torch=parent.api.np,parent.api.torch
    sources=sorted({k[0] for k in keys})
    if not keys or steps<=0 or batch_size%len(sources):raise ValueError('Original equal-source query design required')
    groups=[np.flatnonzero(np.array([k[0] for k in keys])==s) for s in sources]
    rng=torch.Generator().manual_seed(seed+7919);counts=np.zeros(len(keys),np.int64)
    for _ in range(steps):
        for g in groups:
            chosen=g[torch.randint(len(g),(batch_size//len(groups),),generator=rng).numpy()]
            np.add.at(counts,chosen,1)
    result=[]
    for source,g in zip(sources,groups):
        c=counts[g]
        result.append(dict(source=source,queries=len(g),draws=int(c.sum()),never_drawn=int((c==0).sum()),
            seen_fraction=float((c>0).mean()),draws_per_query_quantiles=np.quantile(c,[0,.25,.5,.75,1]).tolist()))
    return result,rng.get_state()


def main():
    parent.api.torch.set_num_threads(4);parent.api.torch.set_num_interop_threads(1)
    freeze=json.loads((parent.PUBLIC/'training_freeze.json').read_text());rows=[]
    for ref in freeze['fits']:
        path=ROOT/ref['path'];receipt=json.loads(path.read_text())
        if parent.sha(path)!=ref['sha256'] or parent.sha(ROOT/receipt['checkpoint']['path'])!=receipt['checkpoint']['sha256']:
            raise ValueError('Original complete training evidence required')
        state=parent.api.parent.read_checkpoint(ROOT/receipt['checkpoint']['path'])
        values,rng=reconstruct(state['query_keys'],seed=state['seed'],steps=state['step'],batch_size=state['settings']['query_batch_size'])
        parent.api.parent.exact(rng,state['sampler_rng'])
        rows.append(dict(context=receipt['context'],seed=receipt['seed'],arm=receipt['arm'],sources=values,
            checkpoint_sha256=receipt['checkpoint']['sha256'],final_sampler_rng_exact=True))
    if len(rows)!=36:raise ValueError('All36 frozen actors required')
    paired={}
    for r in rows:
        key=r['context'],r['seed']
        if key in paired and paired[key]!=r['sources']:raise ValueError('Context arms must share all query draws')
        paired[key]=r['sources']
    cells=[r for v in paired.values() for r in v];large=[r for r in cells if r['queries']>=1000]
    result=dict(result_source='fresh_posthoc_sampler_reconstruction_from_cached_verified_frozen_checkpoints',
        script_sha256=parent.sha(Path(__file__)),training_freeze_sha256=parent.sha(parent.PUBLIC/'training_freeze.json'),
        fits=36,distinct_paired_samplers=18,sources_per_sampler=4,rows=rows,
        large_source_views=len(large),large_source_never_drawn=sum(r['never_drawn'] for r in large),
        large_source_queries=sum(r['queries'] for r in large),
        large_source_seen_fraction_range=[min(r['seen_fraction'] for r in large),max(r['seen_fraction'] for r in large)],
        all_final_sampler_rng_exact=True,all_context_arm_draws_equal=True,
        gradient_queries_only_not_full_risk_monitoring=True,full_risk_passes_cover_all_queries=True,
        final_rng_match_is_not_complete_row_draw_hash_replay=True,
        private_input_scope='frozen_actor_checkpoints_only_no_trajectory_or_future_label_reads',
        new_training=False,independent_roles_opened=False,deployment_changed=False)
    target=ROOT/'outputs/publication_readiness_2026_09/european_frozen_cost_support_v1/actor_query_coverage.json'
    parent.once(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
