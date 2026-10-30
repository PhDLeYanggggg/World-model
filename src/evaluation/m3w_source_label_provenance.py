"""Outcome provenance only; never export these diagnostics as inference features."""
import numpy as np

REASONS=('beyond_supplied_record_max_frame','after_last_scoped_agent_detection',
         'within_agent_span_other_agents_observed','within_agent_span_no_detection_frame')
KEY=np.dtype([('agent','<i8'),('frame','<i8')])


def key_values(agents,frames):
    values=np.empty(np.shape(frames),dtype=KEY);values['agent']=agents;values['frame']=frames
    return values


def audit_rows(raw,agents,frames,current_xy,target,valid):
    agents,frames=np.asarray(agents),np.asarray(frames);n=len(agents)
    target,valid=np.asarray(target),np.asarray(valid)
    if (not len(raw) or agents.shape!=(n,) or frames.shape!=(n,) or not n
            or target.shape!=(n,12,2) or valid.shape!=(n,12) or valid.dtype!=bool
            or np.shape(current_xy)!=(n,2) or not np.isfinite(target[valid]).all()
            or not np.isfinite(current_xy).all() or agents.dtype.kind not in 'iu' or frames.dtype.kind not in 'iu'):
        raise ValueError('Nonempty frozen query identities and separately masked audit labels required')
    a,f=raw['agent'],raw['frame']
    if np.any((a[1:]<a[:-1])|((a[1:]==a[:-1])&(f[1:]<=f[:-1]))):
        raise ValueError('Unique sorted scoped raw agent/frame rows required')
    boxes=np.column_stack([raw[k] for k in ('x_min','y_min','x_max','y_max')])
    if not np.isfinite(boxes).all():raise ValueError('Finite original raw boxes required')
    keys=key_values(a,f)
    def lookup(aa,ff):
        query=key_values(aa,ff);pos=np.searchsorted(keys,query)
        bounded=np.minimum(pos,len(keys)-1);found=(pos<len(keys))&(keys[bounded]==query)
        centers=(boxes[bounded,...,:2]+boxes[bounded,...,2:])/2
        return found,centers
    present,current=lookup(agents,frames)
    if not present.all() or not np.array_equal(current,current_xy):raise ValueError('Frozen current input fails raw alignment')
    requested=frames[:,None]+12*np.arange(1,13,dtype=np.int64)
    observed,xy=lookup(agents[:,None],requested);missing=~observed
    unique,starts,counts=np.unique(a,return_index=True,return_counts=True)
    positions=np.searchsorted(unique,agents)
    last=f[starts[positions]+counts[positions]-1,None]
    recorded=np.unique(f);where=np.searchsorted(recorded,requested)
    frame_present=(where<len(recorded))&(recorded[np.minimum(where,len(recorded)-1)]==requested)
    beyond=missing&(requested>recorded[-1])
    ended=missing&~beyond&(requested>last)
    interior=missing&~beyond&~ended
    reason_masks=(beyond,ended,interior&frame_present,interior&~frame_present)
    if not np.array_equal(sum(m.astype(np.int8) for m in reason_masks),missing.astype(np.int8)):
        raise AssertionError('Missing-label provenance must form a disjoint step partition')
    mismatch=valid!=observed;common=valid&observed
    coordinate_mismatch=np.any(xy!=target,axis=-1)&common
    count=observed.sum(1);none=count==0
    return dict(rows=n,requested_future_steps=n*12,observed_future_steps=int(observed.sum()),
        missing_future_steps=int(missing.sum()),complete_future_rows=int((count==12).sum()),
        partial_future_rows=int(((count>0)&(count<12)).sum()),missing_future_rows=int(none.sum()),
        final_step_observed_rows=int(observed[:,-1].sum()),
        missing_steps_by_reason={r:int(m.sum()) for r,m in zip(REASONS,reason_masks)},
        missing_final_step_by_reason={r:int(m[:,-1].sum()) for r,m in zip(REASONS,reason_masks)},
        queries_with_any_reason_non_disjoint={r:int(m.any(1).sum()) for r,m in zip(REASONS,reason_masks)},
        wholly_missing_queries_with_any_reason_non_disjoint={r:int((m.any(1)&none).sum()) for r,m in zip(REASONS,reason_masks)},
        stored_mask_mismatch_steps=int(mismatch.sum()),
        raw_present_but_stored_missing_steps=int((observed&~valid).sum()),
        stored_present_but_raw_missing_steps=int((valid&~observed).sum()),
        known_coordinate_mismatch_steps=int(coordinate_mismatch.sum()),
        raw_query_frame_range=[int(frames.min()),int(frames.max())],
        raw_record_detection_frame_range=[int(recorded[0]),int(recorded[-1])],
        all_stored_masks_and_known_coordinates_match=not bool(mismatch.any() or coordinate_mismatch.any()),
        missing_reason_is_export_evidence_not_physical_exit_or_video_end=True,
        outputs_not_inference_features=True)


def combine(records):
    if not records:raise ValueError('Nonempty registered recording audit required')
    fields=('rows','requested_future_steps','observed_future_steps','missing_future_steps','complete_future_rows',
        'partial_future_rows','missing_future_rows','final_step_observed_rows','stored_mask_mismatch_steps',
        'raw_present_but_stored_missing_steps','stored_present_but_raw_missing_steps','known_coordinate_mismatch_steps')
    maps=('missing_steps_by_reason','missing_final_step_by_reason','queries_with_any_reason_non_disjoint',
          'wholly_missing_queries_with_any_reason_non_disjoint')
    return dict(recordings=len(records),**{k:sum(r[k] for r in records) for k in fields},
        **{k:{reason:sum(r[k][reason] for r in records) for reason in REASONS} for k in maps},
        all_stored_masks_and_known_coordinates_match=all(r['all_stored_masks_and_known_coordinates_match'] for r in records))
