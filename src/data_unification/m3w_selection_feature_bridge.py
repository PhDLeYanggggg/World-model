"""Exact past-only bridge to the actual partial-context detector forecasters."""
import numpy as np
from src.data_unification.m3w_model_selection_inputs import INPUT_FIELDS
from src.world_model.m3w_european_source_forecast import pack_scene
from src.world_model.m3w_observation_quality import masked_neighbors,repair_geometry
from src.world_model.m3w_detector_history_features import lookup_history,features,NAMES

FIELDS=('geometry','detector_history','history','origin','width','agents','frames',
        'input_rows','neighbor_agent_ids','partial_neighbor_count')
REPRESENTATIONS=('disabled','current','noise','history')


def pack_record(inputs,raw):
    if set(inputs)!=set(INPUT_FIELDS):raise ValueError('Exact causal input-only schema required')
    blocks=[];targets=[];neighbors=[];partial=[]
    offsets=inputs['query_offsets']
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        scene={k:v[lo:hi] for k,v in inputs.items() if k!='query_offsets'}
        g,ids=pack_scene(scene);context=masked_neighbors(scene)
        np.testing.assert_array_equal(ids,context['target'])
        blocks.append(repair_geometry(g,context));targets.append(lo+ids)
        neighbors.append(context['agent_ids']);partial.append(context['partial_neighbors'])
    ids=np.concatenate(targets) if targets else np.empty(0,np.int64)
    np.testing.assert_array_equal(ids,np.flatnonzero(inputs['target_eligible']))
    boxes,confidence,grid=lookup_history(raw,inputs['agent_id'][ids],inputs['query_frame'][ids],inputs['history_boxes'][ids])
    if (grid>inputs['query_frame'][ids,None]).any():raise ValueError('Future detection lookup forbidden')
    h=np.asarray(inputs['history_xy'][ids]);np.testing.assert_array_equal(h,(boxes[...,:2]+boxes[...,2:])/2)
    result=dict(geometry=np.concatenate(blocks) if blocks else np.empty((0,476),np.float32),
        detector_history=features(boxes,confidence),history=h.copy(),origin=h[:,-1].copy(),
        width=(boxes[:,-1,2]-boxes[:,-1,0]).copy(),agents=np.asarray(inputs['agent_id'][ids]).copy(),
        frames=np.asarray(inputs['query_frame'][ids]).copy(),input_rows=ids,
        neighbor_agent_ids=np.concatenate(neighbors) if neighbors else np.empty((0,8),np.int64),
        partial_neighbor_count=np.concatenate(partial) if partial else np.empty(0,np.int64))
    if set(result)!=set(FIELDS) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Finite declared causal bridge fields required')
    return result


def validate_lineage(rows,roles):
    """Validate receipt semantics before checkpoint hashes and train IDs are checked."""
    source={r['locality_group'] for r in roles['recordings'] if r['role']=='source_training'}
    reserved={r['locality_group'] for r in roles['recordings'] if r['role']!='source_training'}
    expected={(f'single{f}_seed{s}',a) for f in range(3) for s in (17,29,43) for a in REPRESENTATIONS}
    keys=[(r['trial'],r['representation']) for r in rows]
    if len(keys)!=36 or set(keys)!=expected:raise ValueError('All nine actual producers/four representations required')
    for r in rows:
        identity=r['parent_trial'];norm=r['normalizer'];fit=set(identity['fit_sites'])
        if (len(fit)!=4 or not fit<=source or fit&reserved or norm['fit_sites']!=identity['fit_sites']
                or norm['future_support_used'] or r['fit']['step']!=4000 or r['fit']['held_rows_sampled']
                or r['fit']['parameters']!=89026 or identity['seed'] not in (17,29,43)
                or r['trial']!=f'single{identity["fold"]}_seed{identity["seed"]}'):
            raise ValueError('Source-only producer and source-fitted normalization required')
        mean,std=np.asarray(norm['mean']),np.asarray(norm['std'])
        if mean.shape!=(64,) or std.shape!=(64,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any() or norm['clip']!=8:
            raise ValueError('Frozen64-field source normalizer required')
    return keys
