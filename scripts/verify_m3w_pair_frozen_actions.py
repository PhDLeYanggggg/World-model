"""Replay all causal bindings,geometry and scores without reoptimizing timed MILPs.

This is explicitly not proof of identical fresh optimization under a wall-clock
cutoff. It cannot replace a failed registered fresh-decoder replay gate.
"""
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_pair_action_allocation as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups


def verify(diagnostic=None):
    cfg,values,store,banks,models,controls=run.admission(); start,beat=run.runtime(cfg)
    complete=json.loads((run.PUBLIC/'complete.json').read_text())
    if (run.PUBLIC/'frozen_action_verification.json').exists(): raise ValueError('Completed action replay immutable')
    if complete['summary_sha256']!=run.sha(run.PUBLIC/'summary.json'): raise ValueError('Complete summary changed')
    old,main=aggregate_groups(run.source.frozen.source.parent.NAME),aggregate_groups(run.source.frozen.NAME)
    original=aggregate_groups(run.unit.NAME); reported=aggregate_groups(run.NAME)
    rows=[]; caches=[]; all_ids=set(); recordings=set()
    for context,items in sorted(values[2]['contexts'].items()):
        packets,changed=run.previous.context_packets(context,items,values,store,banks,beat)
        job=next(j for j in values[4] if j['key']==context.split('_controller')[0])
        for seed in cfg['seeds']:
            prediction,ranges=run.infer(context,seed,packets,changed,values,store,models,controls,old,main,beat)
            for name,packet in sorted(packets.items()):
                key=f'{context}_{name}_seed{seed}_fit.json'; expected=reported[key]; prior=original[key]
                p=prediction[ranges[name]]; origin=values[3]['origin'][packet['ids']]
                parent=run.unit.api.parent.read_checkpoint(run.unit.PRIVATE/'progress'/(key+'.pt.gz'))['choices']['set_joint']
                cp=run.PRIVATE/'progress'/(key+'.pt.gz'); saved=run.unit.api.parent.read_checkpoint(cp)
                groups=run.api.base.query_groups(packet['recordings'],packet['frames'],packet['ids'])
                if saved['completed_queries']!=len(groups): raise ValueError('Only complete frozen choices can be replayed')
                choices,ranks,info=run.decode(packet,p,origin,parent,cfg,resume_state=saved)
                masses=dict(before=0.,after=0.); edges_n=0
                for g in groups:
                    edges,table=run.api.geometry(packet['rollouts'][g],packet['floor_rollout'][g],origin[g],
                        packet['scales'][g],packet['ids'][g],neighbors=cfg['past_nearest_neighbors'])
                    before=run.api.pair_value(edges,table,parent[g])
                    after=run.api.pair_value(edges,table,choices['pair_joint'][g]); scale=float(packet['scales'][g].sum())
                    old_objective=-run.api.base.terms(p[g],parent[g])[0]/scale+cfg['pair_weight']*before
                    new_objective=-run.api.base.terms(p[g],choices['pair_joint'][g])[0]/scale+cfg['pair_weight']*after
                    if new_objective>old_objective+1e-10: raise ValueError('Frozen action objective violates original acceptance')
                    edges_n+=len(edges); masses['before']+=before*len(edges); masses['after']+=after*len(edges)
                if edges_n!=info['graph_edges'] or any(masses[k]!=info['graph_pair_mass_'+k] for k in masses):
                    raise ValueError('Fresh complete causal graph arithmetic differs')
                result=run.evaluation.readout(packet,choices,ranks,info,job['design']['easy_cut'],origin)
                if (result!=expected['result'] or result['policies']['unary_joint']!=prior['result']['policies']['set_joint']
                        or run.unit.api.parent.digest(p)!=expected['prediction_sha256']
                        or run.unit.api.parent.digest(p)!=prior['prediction_sha256']['selected_set']
                        or run.unit.api.parent.digest(packet['ids'])!=expected['ids_sha256']
                        or any(run.unit.api.parent.digest(choices[a])!=expected['action_sha256'][a] for a in run.api.NAMES)):
                    raise ValueError('Original inputs,predictions,frozen actions or full scores differ')
                if diagnostic is not None:
                    diagnostic(key,expected,packet,origin,choices,values[3]['target_eval'][packet['ids']],values[3]['valid'][packet['ids']])
                rows.append(expected); all_ids.update(map(int,packet['ids']))
                recordings.update((name,str(r)) for r in packet['recordings'])
                caches.append(dict(group=key,action_cache_sha256=run.sha(cp)))
                run.write(run.PRIVATE/'frozen_action_replay_progress.json',dict(pid=os.getpid(),groups=len(rows),
                    total_groups=72,complete_sha256=run.sha(run.PUBLIC/'complete.json'),caches=caches,
                    optimization_replayed=False,whole_pipeline_OOF=False))
                beat(state='frozen_action_group_verified',groups=len(rows),total=72,context=context,source=name,seed=seed)
            del prediction
        del packets,changed
    if len(all_ids)!=149510 or len(recordings)!=74: raise ValueError('Complete original cohort required')
    rebuilt=run.evaluation.summarize(rows,json.loads(run.REG.read_text())['contexts'],cfg)
    rebuilt.update(unique_target_rows=len(all_ids),unique_recordings=len(recordings))
    if rebuilt!=json.loads((run.PUBLIC/'summary.json').read_text()): raise ValueError('Full summary arithmetic differs')
    receipt=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        all72_original_predictions_and_controls_exact=True,all72_frozen_actions_exact=True,
        all72_causal_graphs_and_full_scores_exact=True,summary_sha256=run.sha(run.PUBLIC/'summary.json'),
        complete_sha256=run.sha(run.PUBLIC/'complete.json'),action_caches=caches,
        optimization_replayed=False,fresh_timed_decoder_reproduction_certified=False,
        registered_reoptimization_gate_replacement=False,independent_confirmation=False)
    run.once(run.PUBLIC/'frozen_action_verification.json',receipt)
    beat(state='all72_frozen_action_scores_verified_not_fresh_optimization')


if __name__=='__main__':
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify()
