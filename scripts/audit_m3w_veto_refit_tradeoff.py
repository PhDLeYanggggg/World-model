"""Account for released-action utility and easy degradation on frozen outputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_veto_risk_refit as study

run,np = study.run,study.np


def main():
    path = study.PUBLIC/'analysis.json'
    base = run.read(path)
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256']!=run.sha(path):
        raise ValueError('Require exact frozen scoring')
    d,refs = run.population(),run.register()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    cuts = run.cuts_by_source()
    records = study.prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
    all_rows = []
    for cell in run.read(study.PUBLIC/'freeze.json')['cells']:
        seed = cell['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            original,target = z['choices'][[0,2]]
            released = z['releases'][1]
        for ix,_ in records.values():
            source = str(d['sites'][ix[0]])
            known = valid[ix].any(1)
            ref,_ = run.model.errors(cv[ix],y[ix],valid[ix])
            easy = known&(ref>0.)&(ref<=cuts[source]['easy'])
            hard = known&(ref>cuts[source]['hard'])
            a,b = [run.model.selected_paths(ch[ix],bank[ix],cv[ix]) for ch in (original,target)]
            ea,_ = run.model.errors(a,y[ix],valid[ix])
            eb,_ = run.model.errors(b,y[ix],valid[ix])
            j = np.flatnonzero(target[ix]>=0)
            use = np.zeros(len(ix),bool)
            use[j] = released[ix[j],target[ix[j]]]
            diff = np.where(known,ea-eb,0.)
            weight = 1./d['crowd'][ix]
            ref_easy = float(ref[easy].sum())
            signed = float((eb[easy]-ref[easy]).sum())
            mixed = signed+float(diff[easy&use].sum())
            all_rows.append(dict(seed=seed,source=source,known_gain_on_release=float(weight[use]@diff[use]),
                known_gain_elsewhere=float(weight[~use]@diff[~use]),hard_gain=float(weight[hard]@diff[hard]),
                hard_ref=float(weight[hard]@ref[hard]),net_easy_exceeds=ref_easy>0 and signed>.02*ref_easy,
                after_replacing_release_rows_exceeds=ref_easy>0 and mixed>.02*ref_easy))
    if len(all_rows)!=102 or sum(r['net_easy_exceeds'] for r in all_rows)!=base['risk']['veto_only']['easy_net_over_2']:
        raise ValueError('The attribution must replay all recorded easy-net violations')
    sources = sorted({r['source'] for r in all_rows})
    groups = [[r for r in all_rows if r['source']==s and r['seed']==k]
              for s in sources for k in run.kin.prior.SEEDS]
    values = np.asarray([100*sum(r['hard_gain'] for r in g)/sum(r['hard_ref'] for r in g)
                         for g in groups]).reshape(len(sources),3).mean(1)
    draws = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    run.once(study.PUBLIC/'tradeoff_attribution.json',dict(source=run.artifact(path),views=len(all_rows),
        known_gain_mass_on_new_release_rows=sum(r['known_gain_on_release'] for r in all_rows),
        known_gain_mass_elsewhere=sum(r['known_gain_elsewhere'] for r in all_rows),
        hard_known_gain_vs_original_pp=dict(point=float(values.mean()),
            interval95=np.quantile(values[draws].mean(1),[.025,.975]).tolist()),
        easy_net_exceeds=sum(r['net_easy_exceeds'] for r in all_rows),
        easy_net_exceeds_after_restoring_original_only_on_release_rows=sum(
            r['after_replacing_release_rows_exceeds'] for r in all_rows),
        limited_views_with_defined_selected_easy_after_refit=sum(
            r['previously_limited'] and r['metrics']['veto_only']['risk']['completion_easy_risk_upper'] is not None
            for r in base['records']),
        posthoc_bookkeeping_only_not_matched_policy=True,no_outcome_aware_choices_exported=True))


if __name__=='__main__':
    main()
