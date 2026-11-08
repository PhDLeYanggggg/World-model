"""Read frozen anchor forecasts with explicit missing slice support."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_ols_anchor_denoising as study

run,prev,np = study.run,study.prev,study.np


def interval(cells,draws):
    a = np.stack(cells,axis=1)
    missing = int((~np.isfinite(a)).sum())
    if missing:
        return dict(point=None,interval95=None,missing_source_seed_cells=missing)
    return dict(**study.old.metric.intervals(a,draws),missing_source_seed_cells=0)


def score(jobs,cut,frozen):
    if frozen['registration'] != run.sha(study.REG) or not all(frozen[k] for k in (
            'source_disjoint','matched_draws','past_only_predictions')):
        raise ValueError('Require the original committed prediction freeze')
    d = study.data(jobs,lambda **_:None)
    queries = study.old.fitting_layout(jobs[0]['identity'],d)
    _,paths0,a,b,_,available = study.model.design(d['geometry'][:,:16].reshape(-1,8,2))
    y,v = d['target'],d['valid']; known,endpoint = v.any(1),v[:,-1]
    t = np.where(v[...,None],y,0.)
    sites,src = np.unique(d['sites'],return_inverse=True); w = np.empty(len(src))
    for q in queries:
        w[q] = 1./len(q)
    with np.load(run.checked(frozen['coefficients']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); f = z['factors']

    def error(p):
        e = np.linalg.norm(p-t,axis=2)
        return np.where(v,e,0.).sum(1)/v.sum(1).clip(1),e[:,-1]

    cv,_ = error(paths0['cv']); mass = np.bincount(src,weights=w*cv)
    if len(sites) != 9 or (mass <= 0.).any():
        raise ValueError('Primary contrast requires every source to have CV reference mass')
    weight = 100*w/mass[src]
    easy = known & (cv > 0.) & (cv <= cut['easy'])

    def mean(x,mask):
        num = np.bincount(src,weights=w*np.where(mask,x,0.),minlength=9)
        den = np.bincount(src,weights=w*mask,minlength=9)
        return np.divide(num,den,out=np.full(9,np.nan),where=den > 0.)

    absolute = {k:[] for k in study.ARMS}; effects = {f'{p}_vs_{r}':[] for p,r in study.CONTRASTS}
    joint = available & (np.linalg.norm(a,axis=2).max(1) > 1e-6)
    for alpha,beta in f:
        ref = paths0['ols4']; alpha,beta = alpha[:,None,None],beta[:,None,None]
        matched_a = alpha*a*joint[:,None,None]; matched_b = alpha*b*joint[:,None,None]
        np.testing.assert_allclose(np.linalg.norm(matched_a,axis=2).max(1),
                                   np.linalg.norm(matched_b,axis=2).max(1),atol=1e-10)
        p = dict(**paths0,learned_anchor=ref+alpha*a,learned_slope=ref+beta*b,
                 matched_anchor=ref+matched_a,matched_slope=ref+matched_b)
        e = {k:error(z) for k,z in p.items()}
        for k,(ade,fde) in e.items():
            absolute[k].append(dict(ade=mean(ade,known),fde=mean(fde,endpoint),easy_ade=mean(ade,easy),
                                    worst_source_p95=max(float(np.quantile(ade[(src==s)&known],.95)) for s in range(9))))
        for k,r in study.CONTRASTS:
            g = e[r][0]-e[k][0]; debit = np.linalg.norm(p[r]-p[k],axis=2).max(1)
            effects[f'{k}_vs_{r}'].append(dict(known=np.bincount(src,weights=weight*np.where(known,g,0.)),
                lower=np.bincount(src,weights=weight*np.where(known,g,-debit)),
                harm=np.bincount(src,weights=weight*np.where(known,np.maximum(-g,0.),debit))))
    draws = np.random.default_rng(73107).integers(9,size=(3000,9))
    support = {str(s):dict(known=int(((src==i)&known).sum()),fde=int(((src==i)&endpoint).sum()),
                           easy=int(((src==i)&easy).sum())) for i,s in enumerate(sites)}
    return dict(contrasts={k:{f:interval([r[f] for r in rows],draws) for f in ('known','lower','harm')} for k,rows in effects.items()},
        absolute={k:dict(**{f:interval([r[f] for r in rows],draws) for f in ('ade','fde','easy_ade')},
                         worst_source_p95=max(r['worst_source_p95'] for r in rows)) for k,rows in absolute.items()},
        source_support=support,rows=len(y),queries=len(queries),fits=len(frozen['fits']),seeds=study.prev.SEEDS,
        matched_direction_rows=int(joint.sum()),source_groups=9,whole_roster=True,reporting_amendment_only=True,
        development_targets=False,external_access=False,independent_confirmation=False,
        original_cv_rule_unchanged=True,deployment=False,stage5c_executed=False,smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('score','replay'))
    phase = p.parse_args().phase
    cut,jobs = study.register()
    frozen = prev.receipt(study.PUBLIC,'train')
    script = Path(__file__).resolve()
    if subprocess.check_output(['git','show',f'HEAD:{script.relative_to(ROOT)}'],cwd=ROOT) != script.read_bytes():
        raise ValueError('Commit the missing-support reporting amendment before scoring')
    start = time.monotonic(); out = study.PUBLIC/'analysis.json'
    if phase == 'replay':
        prev.receipt(study.PUBLIC,'score')
    run.once(out,run.stable(score(jobs,cut,frozen)))
    receipt = study.PUBLIC/f'{phase}_execution.json'
    if not receipt.exists():
        run.once(receipt,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out),
            reporter=run.artifact(script),frozen_training=run.artifact(study.PUBLIC/'train_execution.json')))
    elif run.read(receipt)['artifact'] != run.artifact(out):
        raise ValueError('Support-aware replay differs')


if __name__ == '__main__':
    main()
