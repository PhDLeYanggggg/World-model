"""Check whether shrinking the already selected OLS4 directions could help."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_ols_replacement as parent
from src.evaluation import m3w_ols_replacement_amplitude as probe

run,prev,np = parent.run,parent.prev,parent.np
PUBLIC = parent.PUBLIC.parent/'native_ols_replacement_amplitude_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(),Path(probe.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_ols_replacement_amplitude.py')


def register():
    parent.register()
    freeze, old = (prev.receipt(parent.PUBLIC,k) for k in ('infer','replay'))
    src = (*FILES,parent.REG,*(parent.PUBLIC/f'{k}_execution.json' for k in ('infer','replay')))
    run.once(REG,run.stable(dict(bindings=[run.artifact(p) for p in src],range=[0.,1.],steps=32,
        primary='ols_hindcast_common_amplitude_hindsight',fixed_recipient_path_directions=True,
        cv_risk_unchanged=True,external_access=False,no_deployment=True)))
    return freeze,old


def score(freeze,old):
    d = run.population()
    lab = run.old.prior.arrays(('target_eval','valid','frames'))
    blocks = prev.old.capacity.cal.prior.h.previous.prior.blocks(d,lab['frames'])
    cv,ols,bank,_ = parent.model.rollout(d['history'],12)
    y,valid = lab['target_eval']-d['origin'][:,None],lab['valid']
    cv_error,_ = parent.errors(cv,y,valid)
    cv_error = np.where(valid.any(1),cv_error,0.)
    groups = sorted(set(d['sites']))
    seeds = list(prev.SEEDS)
    m = len(groups)*len(seeds)
    masses = np.zeros(m)
    packed = {a:[] for a in parent.model.ARMS}
    for f in freeze['cells']:
        with np.load(run.checked(f['artifact']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choices = z['choices']
        for ix,_ in blocks.values():
            group = groups.index(d['sites'][ix[0]])*len(seeds)+seeds.index(f['seed'])
            _, i,n = np.unique(d['queries'][ix],return_inverse=True,return_counts=True)
            w = 1./n[i]
            masses[group] += float(w @ cv_error[ix])
            for j,a in enumerate(parent.model.ARMS):
                on = choices[j,ix] >= 0
                at = ix[on]
                b = bank[at,choices[j,at]]
                packed[a].append((w[on],np.full(len(at),group),*probe.components(ols[at],b,y[at],valid[at])))
    if (masses <= 0.).any():
        raise ValueError('Every source-seed cell needs the registered CV denominator')
    draws = np.random.default_rng(73107).integers(len(groups),size=(3000,len(groups)))

    def interval(x):
        v = x.reshape(len(groups),len(seeds)).mean(1)
        return dict(point=float(v.mean()),interval95=np.quantile(v[draws].mean(1),[.025,.975]).tolist())

    results = {}
    for a,rows in packed.items():
        w,k,*terms = (np.concatenate(t) for t in zip(*rows,strict=True))
        w = 100*w/masses[k]
        if len(w) != 9675:
            raise ValueError('Fixed selected directions changed')

        def per_cell(v):
            return np.bincount(k,weights=v*w,minlength=m)

        def common(z):
            return tuple(per_cell(v).mean() for v in probe.value_slope(terms,z))

        full,_ = probe.value_slope(terms,1.)
        check = old['grouped']['effects'][f'{a}_vs_{parent.OLS}']['completion_lower_pp']
        np.testing.assert_allclose(per_cell(full).mean(),check['point'],rtol=1e-10,atol=1e-10)
        alpha,_,slope = probe.maximize(common)
        va,da = probe.value_slope(terms,float(alpha))
        _,zero = probe.value_slope(terms,0.)
        opt,vt,_ = probe.maximize(lambda z:probe.value_slope(terms,z))
        results[a] = dict(registered_full_gain_pp=check['point'], shared_alpha=float(alpha),
            shared_hindsight_gain_pp=interval(per_cell(va)), slope_at_zero_pp=interval(per_cell(zero)),
            shared_slope_at_optimum_pp=float(per_cell(da).mean()),
            positive_slope_source_seed_cells=int((per_cell(zero) > 0.).sum()),
            positive_slope_rows=int((zero > 0.).sum()),
            rowwise_oracle_positive_amplitude=int((opt > 0.).sum()),rowwise_oracle_gain_pp=interval(per_cell(vt)))
    return dict(arms=results,recordings=len(blocks),source_groups=len(groups),seeds=seeds,
        frozen_directions_per_arm=9675,external_access=False,independent_confirmation=False,
        no_new_policy=True,threshold_or_risk_change=False,stage5c_executed=False,smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','score','replay'))
    a = p.parse_args()
    freeze,old = register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Register the diagnostic before running')
    start = time.monotonic()
    out = PUBLIC/'analysis.json'
    if a.phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
        raise ValueError('Require the original diagnostic result')
    run.once(out,run.stable(score(freeze,old)))
    mark = PUBLIC/f'{a.phase}_execution.json'
    if not mark.exists():
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))
    elif run.read(mark)['artifact'] != run.artifact(out):
        raise ValueError('Amplitude diagnostic replay changed')


if __name__ == '__main__':
    main()
