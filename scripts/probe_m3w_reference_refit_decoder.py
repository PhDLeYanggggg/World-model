"""Measure real frozen-cost allocation and recovery without policy outcome scores."""
import copy
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_reference_cost_refit as run


def main():
    if (run.PUBLIC/'decoder_pilot.json').exists():raise ValueError('Completed decoder resource probe is immutable')
    cfg,values=run.admission();start,beat,_=run.runtime(cfg)
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text())
    if len(freeze['fits'])!=36:raise ValueError('All36 frozen cost models required before policy work')
    gen=run.inputs(values,beat);context,seed,data,ranges,packets,_,_,_=next(gen)
    states={a:run.api.parent.read_checkpoint(run.checkpoint(context,seed,a)) for a in run.api.ARMS}
    run.api.assert_matched(*states.values())
    name=sorted(packets)[0];packet=packets[name]
    predicted={a:run.api.predict(s,data['causal'])[ranges[name]] for a,s in states.items()}
    key=f'{context}_{name}_seed{seed}_fit.json'
    control=run.api.parent.read_checkpoint(run.parent.PRIVATE/'progress'/(key+'.pt.gz'))['choices']['zero_joint']
    groups=run.decoder.base.query_groups(packet['recordings'],packet['frames'],packet['ids'])
    choose=sorted(set(range(min(12,len(groups))))|set(sorted(range(len(groups)),key=lambda i:(-len(groups[i]),i))[:6]))
    ix=run.api.np.sort(run.api.np.concatenate([groups[i] for i in choose]))
    args=({a:v[ix] for a,v in predicted.items()},packet['allowed'][ix],packet['recordings'][ix],packet['frames'][ix],packet['ids'][ix],control[ix])
    saved=[]
    class Stop(Exception):pass
    def interrupt(v):saved.append(copy.deepcopy(v));raise Stop()
    try:run.decoder.allocate(*args,seconds=cfg['solver_seconds_per_query'],checkpoint=interrupt)
    except Stop:pass
    recovered=run.decoder.allocate(*args,seconds=cfg['solver_seconds_per_query'],resume_state=saved[0])
    before=time.monotonic();direct=run.decoder.allocate(*args,seconds=cfg['solver_seconds_per_query']);seconds=time.monotonic()-before
    run.api.parent.exact(recovered,direct)
    run.once(run.PUBLIC/'decoder_pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
        direct_decoder_seconds=seconds,queries=len(choose),rows=len(ix),maximum_query_agents=max(len(groups[i]) for i in choose),
        rough112596query_decoder_projection_seconds=seconds/len(choose)*112596,
        estimate_is_one_source_size_biased_not_completed_readout=True,exact_interrupted_query_recovery=True,
        diagnostics=direct[2],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        new_policy_outcome_scores_read=False,registration_sha256=run.sha(run.REG),training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json')))
    beat(state='actual_decoder_probe_complete_no_outcome_scores')


if __name__=='__main__':
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
