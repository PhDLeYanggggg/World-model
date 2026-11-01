"""Real causal runtime/recovery probe; no learned-policy outcome scoring."""
import copy
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_free_allocation_actor as run


def main():
    if (run.PUBLIC/'decoder_pilot.json').exists():raise ValueError('Completed resource pilot is immutable')
    cfg,values,store,banks,models,controls=run.admission();start,beat,_=run.runtime(cfg)
    old,main,original=run.resources();context,items=sorted(values[2]['contexts'].items())[0]
    states={a:run.api.parent.read_checkpoint(run.checkpoint(context,17,a)) for a in run.api.ARMS};run.api.assert_matched(*states.values())
    packets,changed=run.previous.previous.context_packets(context,items,values,store,banks,beat)
    data,ranges,zeros=run.prepare(context,17,packets,changed,store,models,controls,old,main,original,beat,preprocess=states[run.api.ARMS[0]]['preprocess'])
    name=sorted(packets)[0];packet=packets[name];c=data['causal'];p=c['prediction'][ranges[name]]
    utilities={a:run.api.predict(s,c)[ranges[name]] for a,s in states.items()}
    groups=run.decoder.base.query_groups(packet['recordings'],packet['frames'],packet['ids'])
    choose=sorted(set(range(min(12,len(groups))))|set(sorted(range(len(groups)),key=lambda i:(-len(groups[i]),i))[:6]))
    ix=run.np.sort(run.np.concatenate([groups[i] for i in choose]));u={a:v[ix] for a,v in utilities.items()};zero={a:v[ix] for a,v in zeros[name].items()}
    args=(p[ix],packet['allowed'][ix],u,packet['recordings'][ix],packet['frames'][ix],packet['ids'][ix],zero)
    saved=[]
    class Stop(Exception):pass
    def interrupt(v):saved.append(copy.deepcopy(v));raise Stop()
    try:run.decoder.allocate(*args,seconds=cfg['solver_seconds_per_query'],checkpoint=interrupt)
    except Stop:pass
    recovered=run.decoder.allocate(*args,seconds=cfg['solver_seconds_per_query'],resume_state=saved[0])
    began=time.monotonic();direct=run.decoder.allocate(*args,seconds=cfg['solver_seconds_per_query']);seconds=time.monotonic()-began
    run.api.parent.exact(recovered,direct)
    run.once(run.PUBLIC/'decoder_pilot.json',dict(pid=__import__('os').getpid(),seconds=time.monotonic()-start,
        direct_decoder_seconds=seconds,queries=len(choose),rows=len(ix),maximum_query_agents=max(len(groups[i]) for i in choose),
        rough112596query_decoder_projection_seconds=seconds/len(choose)*112596,
        estimate_is_one_source_size_biased_not_completed_readout=True,exact_interrupted_query_recovery=True,
        diagnostics=direct[2],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        new_outcome_scores_read=False,registration_sha256=run.sha(run.REG),training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json')))
    beat(state='real_decoder_resource_probe_complete_no_outcome_scores')


if __name__=='__main__':main()
