"""Test a frozen event-head readout at identical first-query intervention counts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_shortlist_cost as study

run,np,anchor = study.run,study.np,study.anchor


def main():
    pub = study.PUBLIC
    original = pub/'analysis.json'
    if run.read(pub/'replay_execution.json')['artifact']['sha256']!=run.sha(original):
        raise ValueError('Require the complete fixed-shortlist study')
    d,refs = run.population(),run.register()
    lab = run.old.prior.arrays(('target_eval','valid','frames'))
    y,valid = lab['target_eval']-d['origin'][:,None],lab['valid']
    records = study.prior.blocks(d,lab['frames'])
    cuts = run.cuts_by_source()
    parent,controls,old = study.banks()
    parent = {c['seed']:c for c in parent}
    p = run.parent
    rows = []
    for cell in run.read(pub/'freeze.json')['cells']:
        seed = cell['seed']
        c,b = run.candidates(d,refs,seed)
        direct,co = study.timing.parents(d,parent[seed],controls)
        previous = direct.copy()
        weights = {}
        for f in cell['folds']:
            s = f['identity']['held']
            ix = d['sites']==s
            with np.load(run.checked(f['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['global_ids'][ix])
                previous[ix] = z['choices'][5]
                weights[s] = z['coefficients'].copy()
        for rec,(ix,g) in records.items():
            s = str(d['sites'][ix[0]])
            e,h = cuts[s]['easy'],cuts[s]['hard']
            floor,bank = c[ix],b[ix]
            raw,scale = run.model.descriptors(d['history'][ix],bank,floor,d['crowd'][ix])
            with np.load(run.checked(old[seed,s]),allow_pickle=False) as z:
                rank = z['coefficients']
            dc,examples = study.proposals(raw,scale,bank,floor,g,co[s],rank)
            np.testing.assert_array_equal(dc,direct[ix])
            action = dc.copy()
            for q,i,j,x in examples:
                if not (dc[q]>=0).any():
                    k = int((x@weights[s])[:,4].argmax())
                    action[i[k]] = j[k]
                break
            v = previous[ix]
            if any((action[q]>=0).sum()!=(v[q]>=0).sum() for q in g):
                raise ValueError('Event choice must match all prior queries and counts')
            paths = {name:run.model.selected_paths(a,bank,floor) for name,a in
                     (('risk',v),('event',action))}
            out = p.readout.score_recording({p.CV:floor,**paths},
                dict(primary=('risk','event',p.CV)),y[ix],valid[ix],d['queries'][ix],rec,
                easy_cut=e,hard_cut=h)
            zero = not (dc>=0).any()
            hits = {}
            if zero:
                _,atoms,_ = anchor.model.labels(floor,bank,y[ix],valid[ix],e)
                targets = study.model.targets(atoms,scale,examples)
                for name,n in (('first',1),('first_two',2),('any',len(examples))):
                    hits[name] = any(t[(x@weights[s])[:,4].argmax(),4]>0.
                                     for (_,_,_,x),t in zip(examples[:n],targets[:n]))
            rows.append(dict(seed=seed,source=s,zero=zero,hit=hits,changed=not np.array_equal(v,action),
                risk={k:out['metrics'][k]['risk'] for k in paths},contrast=out['comparisons']['primary']))
    sources = sorted({r['source'] for r in rows})
    zero = [r for r in rows if r['zero']]
    if len(rows)!=102 or len(zero)!=18 or len(sources)!=14:
        raise ValueError('Require the original recording population')
    results = {}
    for k in ('risk','event'):
        v = [r['risk'][k] for r in rows]
        results[k] = dict(switches=sum(x['selected'] for x in v),passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                x['known_easy_net_degradation']>.02 for x in v),
            zero_pass=sum(r['risk'][k]['supported'] for r in zero))
    saved = run.read(original)['risk']['diagnostic_risk']
    if any(v!=saved[k] for k,v in results['risk'].items()):
        raise ValueError('Prior risk choice must replay exactly')
    grouped = [[r['contrast'] for r in rows if r['source']==s and r['seed']==k]
               for s in sources for k in run.kin.prior.SEEDS]
    x = np.asarray([p.readout.merge_comparisons(v)['completion_lower_pp'] for v in grouped]).reshape(14,3).mean(1)
    draw = np.random.default_rng(73107).integers(14,size=(3000,14))
    result = run.stable(dict(views=102,zero_views=18,changed_views=sum(r['changed'] for r in rows),
        recall={k:sum(r['hit'][k] for r in zero) for k in ('first','first_two','any')},
        risk=results,paired_lower_pp=dict(point=float(x.mean()),
            interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist()),
        analysis_sha256=run.sha(original),posthoc_development_only=True,
        thresholds_retuned=False,independent_outcomes_opened=False))
    run.once(pub/'event_choice_diagnostic.json',result)
    print(result)


if __name__=='__main__':
    main()
