"""Descriptive causal-input slices of already frozen forecast pairs; no selection."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_european_partial_neighbor_refit as run
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_partial_neighbor_refit import masks,slice_metrics,paired_localities


def main():
    cfg,reg,data,receipts,jobs,identity=run.load()
    partial=run.prepared(data,receipts,identity)
    path=run.PUBLIC/'prediction_freeze.json'; run.committed(path)
    frozen=json.loads(path.read_text())
    changed=np.any(data['geometry'][:,38:294]!=partial['geometry'][:,38:294],axis=1)
    assert int(changed.sum())==282529
    old_mask=data['geometry'][:,230:294].reshape(-1,8,8)>0
    new_mask=partial['geometry'][:,230:294].reshape(-1,8,8)>0
    old_slots=old_mask.sum((1,2)); new_slots=new_mask.sum((1,2))
    support=dict(old_mean_current_neighbors=float(old_mask.any(2).sum(1).mean()),
        new_mean_current_neighbors=float(new_mask.any(2).sum(1).mean()),
        old_mean_valid_neighbor_slots=float(old_slots.mean()),
        new_mean_valid_neighbor_slots=float(new_slots.mean()),
        changed_queries=int(changed.sum()),
        changed_queries_fewer_valid_slots=int(((new_slots<old_slots)&changed).sum()),
        changed_queries_more_valid_slots=int(((new_slots>old_slots)&changed).sum()),
        changed_queries_equal_valid_slots=int(((new_slots==old_slots)&changed).sum()))
    roster=sorted(set(data['sites'])); rows=[]
    for job,ref in zip(jobs,frozen['predictions']):
        assert job['key']==ref['key']
        for arm in ('partial','legacy'): assert run.artifact(ROOT/ref[arm]['path'])==ref[arm]
        with np.load(ROOT/ref['partial']['path'],allow_pickle=False) as z: ids,p=z['ids'].copy(),z['prediction'].copy()
        with np.load(ROOT/ref['legacy']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['ids']); old=z['prediction'].copy()
        origin=data['origin'][ids,None]
        a,_=native_errors(p.astype(float)+origin,data['target_eval'][ids],data['valid'][ids],np.ones(len(ids)))
        b,_=native_errors(old.astype(float)+origin,data['target_eval'][ids],data['valid'][ids],np.ones(len(ids)))
        cv=data['baseline_ade'][ids,1]; reference=data['baseline_ade'][ids,job['design']['baseline_index']]
        sub=masks(cv,job['design']['easy_cut'],job['design']['hard_cut'])
        for site in sorted(set(data['sites'][ids])):
            for name,c in [('partial_context_changed',changed[ids]),('no_observed_input_change',~changed[ids])]:
                for subset in ('all','positive_easy','hard'):
                    take=(data['sites'][ids]==site)&c
                    result=slice_metrics(a,b,reference,cv,take&sub[subset])
                    rows.append(dict(trial=job['key'],site=str(site),group=name,subset=subset,metric=result))
    summaries={}
    for group in ('partial_context_changed','no_observed_input_change'):
        for subset in ('all','positive_easy','hard'):
            rr=[r for r in rows if r['group']==group and r['subset']==subset]
            summaries[group+'_'+subset]=paired_localities(rr,roster,'gain_vs_legacy_percent',
                cfg['bootstrap_draws'],cfg['bootstrap_seed'])
    doc=dict(prediction_freeze_sha256=run.digest(path),causal_grouping_only=True,
        descriptive_not_primary=True,thresholds_selected=False,input_support=support,rows=rows,summaries=summaries)
    run.immutable_json(run.PUBLIC/'input_slices.json',doc)
    lines=['# Input-Support Diagnostic', '',
        'Descriptive source slices, not a new selection criterion or a causal mediation analysis.',
        'Groups depend only on whether the observed neighbor geometry changed. No outcome defines a group.',
        'All models and primary endpoints remain fixed. Inference for both groups uses the trained full model.', '',
        '| Input group/subset | Equal-locality ADE gain vs legacy (%) | Locality interval |', '|---|---:|---|']
    for key,v in summaries.items():
        lines.append(f"| {key} | {v['point']} | {v['ci95']} |")
    lines += ['', 'An unchanged input row can still change prediction because training changed shared model weights.',
        'A contrast between these groups cannot separate normalization, nearest-agent membership, mask support,',
        'detector noise or the true usefulness of interactions. Undefined fixed-roster percentages stay undefined.',
        'Slice percentage gains have different denominators and locality weights; they are not additive components of the primary.',
        'There is no seed selection, threshold search, risk-policy refitting or independent-role access.']
    lines += ['', '## Observed Support', '', '| Quantity | Value |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in support.items()]
    lines += ['', 'These counts describe observed inputs, not interaction relevance or annotation accuracy.']
    (run.PUBLIC/'input_slices.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=len(rows),summaries={k:v['point'] for k,v in summaries.items()})))


if __name__=='__main__': main()
