"""Separate label-aware opening availability from score and timing failures."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_easy_joint_anchor as study
from src.evaluation.m3w_selection_forecast_readout import policy

run,np,prior = study.run,study.np,study.prior


def inspect(c,b,y,valid,groups,easy,hard,rec):
    event,atoms,full_easy = study.model.labels(c,b,y,valid,easy)
    candidate = event[...,0] & (atoms[...,5]>0.)
    legal = np.linalg.norm(b-c[:,None],axis=-1).max(2)>0.
    first = next((i for i,g in enumerate(groups) if len(g)>1 and legal[g].any()),None)
    positive = [i for i,g in enumerate(groups) if len(g)>1 and candidate[g].any()]
    if not positive:
        return dict(category='none',queries=len(groups),legal_opening_exists=first is not None)
    j = positive[0]
    g = groups[j]
    i,a = np.argwhere(candidate[g])[0]
    pred = c.copy()
    pred[g[i]] = b[g[i],a]
    r = policy(pred,c,c,y,valid,np.full(len(c),rec),easy_cut=easy,hard_cut=hard)['recordings'][rec]
    if not r['supported'] or r['selected']!=1 or not study.model.risk(atoms[g[i],a],full_easy)['supported']:
        raise ValueError('Opening must pass the original observed scorer')
    return dict(category='first' if j==first else 'later',queries=len(groups),
        legal_opening_exists=True,delay_from_first_query=j-first,positive_query_count=len(positive))


def main():
    folder = study.PUBLIC
    analysis = folder/'analysis.json'
    if run.read(folder/'replay_execution.json')['artifact']['sha256']!=run.sha(analysis):
        raise ValueError('Require completed anchor readout')
    d,refs = run.population(),run.register()
    labels = run.old.prior.arrays(('target_eval','valid','frames'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    records = prior.blocks(d,labels['frames'])
    cuts = run.cuts_by_source()
    truth = {(r['seed'],r['recording']):r['metrics']['direct']['risk']['selected'] for r in
             run.read(study.timing.PAST.PUBLIC/'analysis.json')['records']}
    rows = []
    for seed in run.kin.prior.SEEDS:
        c,b = run.candidates(d,refs,seed)
        for rec,(ix,g) in records.items():
            if truth[seed,rec]:
                continue
            cut = cuts[str(d['sites'][ix[0]])]
            rows.append(inspect(c[ix],b[ix],y[ix],valid[ix],g,cut['easy'],cut['hard'],rec))
    if len(rows)!=18:
        raise ValueError('Require all pre-existing zero-intervention views')
    delay = [r['delay_from_first_query'] for r in rows if r['category']=='later']
    result = run.stable(dict(views=len(rows),categories=dict(Counter(r['category'] for r in rows)),
        median_later_delay_queries=float(np.median(delay)) if delay else None,
        rows_without_any_legal_opening=sum(not r['legal_opening_exists'] for r in rows),
        label_aware_only=True,choices_exported=False,independent_outcomes_opened=False,
        original_rules_unchanged=True,analysis_sha256=run.sha(analysis)))
    run.once(folder/'opportunity_diagnostic.json',result)
    print(result)


if __name__=='__main__':
    main()
