"""Fixed-amplitude causal controls, independent of future labels and cost heads."""
import numpy as np

from src.evaluation.m3w_amplitude_opportunity import FRACTIONS

GUARDS = ('all_past_eligible', 'last_velocity_nonzero')


def choices(history, paths, floor, action, guard):
    h,p,b=map(np.asarray,(history,paths,floor));n=len(h)
    if (not n or h.shape!=(n,8,2) or p.shape!=(n,7,12,2) or b.shape!=(n,12,2)
            or action not in range(1,7) or guard not in GUARDS
            or not all(np.isfinite(v).all() for v in (h,p,b))):
        raise ValueError('Eight past states, frozen causal paths and one original amplitude required')
    use=np.any(p[:,action]!=b,axis=(1,2))
    if guard=='last_velocity_nonzero':
        use &= np.any(h[:,-1]!=h[:,-2],axis=1)
    return np.where(use,action,-1).astype(np.int64)


def name(action,guard):
    if action not in range(1,7) or guard not in GUARDS:
        raise ValueError('Fixed original amplitude/guard only')
    return f'{guard}_a{action}'


def variants():
    return [(name(a,g),a,g) for g in GUARDS for a in range(1,7)]


def interval(rows,key,sources,*,seeds=None,resamples=3000,seed=62468):
    expected=set(sources);groups={s:[r for r in rows if r['source']==s] for s in expected}
    if set(r['source'] for r in rows)!=expected or len(expected)!=8:
        raise ValueError('All eight original source components required')
    for rr in groups.values():
        if seeds is None:
            if len(rr)!=1 or any('seed' in r for r in rr):raise ValueError('Static controls have no cost-head replicas')
        elif len(rr)!=len(seeds) or {r['seed'] for r in rr}!=set(seeds):
            raise ValueError('Every original risk seed exactly once per source')
    if any(r[key] is None for r in rows):
        return dict(point=None,interval95=None,status='required_view_not_estimable')
    means={s:float(np.mean([r[key] for r in groups[s]])) for s in sorted(expected)}
    x=np.array(list(means.values()));draw=np.random.default_rng(seed).integers(0,8,(resamples,8))
    return dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist(),
        by_source=means,source_blocks=8,resamples=resamples,risk_head_seeds_averaged=seeds is not None,
        static_baseline_forecaster_seeds=[17],independent_confirmation=False)
