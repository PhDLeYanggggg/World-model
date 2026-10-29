"""Same-context, cross-source reference support without query-recording donors."""
import json
import numpy as np

from src.evaluation import m3w_selected_analogue as previous
from src.world_model.m3w_shared_train_risk import eligible

ARMS = ('origin', 'estimation', 'pooled_mean', 'pooled_nearest32', 'pooled_shuffled32')


def pool(packets):
    if len(packets) != 4: raise ValueError('Exactly four registered fitting sources required')
    fields = {k: [] for k in ('x','envelope','labels','recordings','eligible','sources')}
    spans = {}; census = {}; offset = 0
    for source in sorted(packets):
        x,env,y,_,sites,recs,frames,pr = packets[source]
        if x.ndim != 2 or x.shape[1] != 380 or not np.isfinite(x).all():
            raise ValueError('Frozen380 causal schema required')
        if pr['training_site'] != source or set(sites.tolist()) != {source}:
            raise ValueError('Source identity differs')
        mask = eligible(packets[source]); known = np.isfinite(y).all(1)
        names = np.asarray([json.dumps([source,str(r)],separators=(',',':')) for r in recs])
        spans[source] = (offset,offset+len(x)); offset += len(x)
        for key,value in zip(fields,(x,env,y,names,mask,np.repeat(source,len(x)))):
            fields[key].append(value)
        census[source] = dict(rows=len(x),features=x.shape[1],recordings=len(set(names.tolist())),
            eligible=int(mask.sum()),known_eligible=int((mask&known&(env>0)).sum()),
            unknown_eligible=int((mask&~known&(env>0)).sum()),
            observed_log_scale_quantiles=np.quantile(x[:,354],[0,.5,1]).tolist(),
            input_hashes={name:previous.fingerprint(a) for name,a in zip(
                ('x','envelope','primary','temporal','sites','recordings','frames'),packets[source][:-1])})
    bank = {key:np.concatenate(values) for key,values in fields.items()}
    usable = bank['eligible'] & (bank['envelope']>0)
    known = np.isfinite(bank['labels']).all(1)
    for source in census:
        rows = bank['sources'] == source
        counts = [int((usable & known & (bank['recordings']!=rec)).sum())
                  for rec in sorted(set(bank['recordings'][rows].tolist()))]
        census[source]['min_pooled_donors_after_record_exclusion'] = min(counts)
        census[source]['pooled_supported_all_recordings'] = min(counts) >= 32
    return bank,spans,census


def retrieve(bank, query, heartbeat=lambda **_:None):
    out = previous.retrieve(bank['x'],bank['envelope'],bank['labels'],bank['recordings'],
                            bank['eligible'],query,heartbeat=heartbeat)
    out['predictions'] = dict(zip(ARMS[2:],(out['predictions'][k] for k in previous.MODES)))
    for receipt in out['references']:
        source,rec = json.loads(receipt['recording'])
        receipt.update(query_source=source,local_recording=rec)
        donor = bank['eligible'] & (bank['envelope']>0) & np.isfinite(bank['labels']).all(1)
        donor &= bank['recordings'] != receipt['recording']
        receipt['known_donors_by_source'] = {s:int((donor&(bank['sources']==s)).sum())
                                           for s in sorted(set(bank['sources'].tolist()))}
    return out


def summarize(rows,expected):
    mapped = []
    names = dict(zip(ARMS,('origin','estimation')+previous.MODES))
    for row in rows:
        mapped.append(dict(context=row['context'],source=row['source'],seed=row['seed'],
                           references=row['references'],scores={names[k]:v for k,v in row['scores'].items()}))
    result = previous.summarize(mapped,expected)
    result['equal_view_metrics'] = {old:result['equal_view_metrics'][new] for old,new in names.items()}
    result['information_screen'] = {
        k.replace('source_mean','pooled_mean').replace('shuffled_nearest32','pooled_shuffled32'):v
        for k,v in result['information_screen'].items()}
    nearest = result['equal_view_metrics']['pooled_nearest32']['easy']['full_cohort_MSE']
    original = result['equal_view_metrics']['origin']['easy']['full_cohort_MSE']
    result.update(reference_bank='four_same_context_inner_fit_sources_excluding_query_recording',
                  magnitude_better_than_origin=nearest is not None and original is not None and nearest < original-1e-12)
    return result
