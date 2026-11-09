"""Hindsight capacity when all causal proposals can be selected."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_feasibility as audit
from scripts import run_m3w_cv_pooled_coverage as study

np,torch,base,prior = study.np,study.torch,study.base,study.prior
read,sha,artifact,once,stable = study.read,study.sha,study.artifact,study.once,study.stable
LIMITS = ('cap8','cap32','uncapped')
SECONDS = 1.5


def one(design, args, cap, inherited=None):
    # A previous verified path remains feasible at a larger per-query cap.
    z, record = audit.model.solve(design,cap,seconds=SECONDS)
    candidates = []
    if z is not None:
        candidates.append(np.column_stack((design['row'][z],design['arm'][z])).tolist())
    if inherited is not None:
        if record['status'] not in ('unresolved',):
            raise ValueError('A relaxed cap ruled out a verified witness')
        candidates.append(inherited)
    for a in candidates:
        r = audit.verify(a,*args,cap)
        if r['verified']:
            return dict(status='verified',actions=a,edits=r['edits'],gain_pp=r['gain_pp'])
    return dict(status=record['status'],actions=None)


def main():
    PRIVATE,OUT = study.PRIVATE,study.OUT
    with (PRIVATE/'capacity.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        t0 = time.monotonic()
        fp,rp = OUT/'freeze.json',OUT/'readout.json'
        frozen, scores = read(fp),read(rp)
        if (scores['freeze']!=sha(fp) or frozen['registration']!=sha(OUT/'registration.json')
                or scores['a_labels_opened'] or scores['independent_confirmation'] or not frozen['before_held_scoring']):
            raise ValueError('Expected the verified fitting-only pooled readout')
        _,donor,_ = base.register()
        data = base.fit_population(lambda **v:None)
        source = artifact(rp)
        code = artifact(Path(__file__))
        summary = {k:Counter() for k in LIMITS}
        new = Counter()
        switches = {k:[] for k in LIMITS}
        for k,c in enumerate(frozen['cells']):
            ident = c['identity']
            hi,h = base.subset(data,ident['held'])
            seed = ident['seed']
            cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
            bank = base.expanded(cv,raw)
            for r in np.unique(h['recording']):
                ix = h['recording']==r
                x,o,b = cv[ix],ols[ix],bank[ix]
                y,v,q = (h[k][ix] for k in ('target','valid','query'))
                design = audit.model.coefficients(x,o,b,y,v,q,c['cut'][0])
                max_query = int(design['counts'].max())
                caps = dict(zip(LIMITS,(8,32,max(32,max_query)),strict=True))
                args = (x,o,b,y,v,q,*c['cut'])
                binding = dict(cell=k,recording=int(r),parent=source,code=code)
                path = PRIVATE/f'capacity_{k}_{r}.json'
                if not path.exists():
                    z, inherit = {},None
                    for name,n in caps.items():
                        z[name] = one(design,args,n,inherit)
                        inherit = z[name]['actions']
                    once(path,stable(dict(binding=binding,witness=z)))
                rec = read(path)
                if rec['binding']!=binding:
                    raise ValueError('Checkpoint binding changed')
                z = rec['witness']
                for name,a in z.items():
                    summary[name][a['status']] += 1
                    if a['status']=='verified':
                        proof = audit.verify(a['actions'],*args,caps[name])
                        if not proof['verified'] or proof['edits']!=a['edits']:
                            raise ValueError('Frozen hindsight witness failed independent scoring')
                        switches[name].append(a['edits']/len(q))
                new['cap8_to_cap32'] += int(z['cap8']['status']!='verified' and z['cap32']['status']=='verified')
                new['cap32_to_uncapped'] += int(z['cap32']['status']!='verified' and z['uncapped']['status']=='verified')
                prior.donor.prior.prior.source.atomic(PRIVATE/'capacity_heartbeat.json',
                    dict(pid=os.getpid(),cell=k,recording=int(r),seconds=round(time.monotonic()-t0,3)))
            print(dict(pid=os.getpid(),cell=k,seed=seed),flush=True)
        if any(sum(v.values())!=72 for v in summary.values()):
            raise ValueError('Expected all 72 views')
        once(OUT/'hindsight_capacity.json',stable(dict(parent=source,code=code,
            verified_counts={k:dict(v) for k,v in summary.items()},new_witnesses=dict(new),
            median_verified_switch_percent={k:100*float(np.median(v)) if v else None for k,v in switches.items()},
            seconds_per_lp_mip=SECONDS,hindsight=True,independent_confirmation=False,a_labels_opened=False,
            original_2pct=True)))
        print(summary,flush=True)


if __name__=='__main__':
    main()
