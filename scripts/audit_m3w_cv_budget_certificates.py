"""Certify recording edit-count intervals using a dual and verified witness deletion."""
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_minimum_budget as study
from src.evaluation import m3w_cv_budget_certificates as cert

exp,base,prior,np = study.exp,study.base,study.prior,study.np
OUT,PRIVATE,parent = study.OUT,study.PRIVATE,study.parent


def calculate(cv,ols,bank,y,v,q,cut,incumbent):
    d = parent.design.coefficients(cv,ols,bank,y,v,q,cut[0])
    low = cert.lower_bound(d)
    edits = np.asarray(incumbent)
    terms,need = cert.witness_terms(cv,ols,bank,y,v,q,cut[0],edits)
    reduced = edits[cert.prune(terms,need)]
    cap = int(np.unique(q,return_counts=True)[1].max())
    z = parent.old.verify(reduced,cv,ols,bank,y,v,q,*cut,cap)
    if not z['verified']:
        reduced = edits
        z = parent.old.verify(reduced,cv,ols,bank,y,v,q,*cut,cap)
    if not z['verified'] or low>z['edits']:
        raise ValueError('Invalid upper or lower edit certificate')
    return dict(lower=low,upper=z['edits'],actions=reduced.tolist())


def report(rows):
    eligible = [r for r in rows if r['original']=='verified']
    if (len(rows),len(eligible))!=(72,53):
        raise ValueError('Expected original complete full-bank roster')
    table = {}
    for f in study.FRACTIONS:
        outcome = []
        for r in rows:
            limit = int(f*r['n'])
            k = ('undefined_support' if r['original']=='undefined_support' else
                 'verified' if r['upper'] is not None and r['upper']<=limit else
                 'excluded' if r['original']!='verified' or r['lower']>limit else 'unresolved')
            outcome.append(k)
        table[str(f)] = {k:outcome.count(k) for k in ('verified','excluded','undefined_support','unresolved')}
    return dict(views=len(rows),capacity=table,exact_count=sum(r['lower']==r['upper'] for r in eligible),
        median_lower_percent=float(np.median([100*r['lower']/r['n'] for r in eligible])),
        median_upper_percent=float(np.median([100*r['upper']/r['n'] for r in eligible])),
        prior_verified=len(eligible))


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'certificates.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        t = time.monotonic()
        inputs = (exp.OUT/'freeze.json',parent.OUT/'readout.json',parent.OUT/'nested_verification.json')
        f,r,n = map(exp.read,inputs)
        if (n['parent']['sha256']!=exp.sha(inputs[1]) or r['independent_confirmation']
                or r['a_labels_opened'] or not r['original_2pct']):
            raise ValueError('Require development-only verified full bank')
        binding = dict(parents=[exp.artifact(x) for x in inputs],
                       files=[exp.artifact(Path(__file__)),exp.artifact(Path(cert.__file__))])
        _,donor,_ = base.register()
        d = base.fit_population(lambda **v:None)
        rows = []
        for i,c in enumerate(f['cells']):
            ident = c['identity']
            hi,h = base.subset(d,ident['held'])
            cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
            b = base.expanded(cv,raw)
            for rec in np.unique(h['recording']):
                ix = h['recording']==rec
                src = exp.read(parent.PRIVATE/f'nested_{i}_{rec}.json')
                if src['parent']!=exp.sha(inputs[1]):
                    raise ValueError('Changed parent witness')
                z = src['results']['full_relaxed']
                result = dict(lower=None,upper=None)
                if z['status']=='verified':
                    args = (cv[ix],ols[ix],b[ix],*(h[k][ix] for k in ('target','valid','query')),c['cut'])
                    result = study.cached(PRIVATE/f'cert_{i}_{rec}.json',binding,
                                          lambda:calculate(*args,z['actions']),False)
                    proof = parent.old.verify(result['actions'],*args[:-1],*c['cut'],
                                              int(np.unique(args[5],return_counts=True)[1].max()))
                    if not proof['verified'] or proof['edits']!=result['upper']:
                        raise ValueError('Deletion incumbent failed replay')
                rows.append(dict(original=z['status'],n=int(ix.sum()),lower=result['lower'],upper=result['upper']))
                prior.donor.prior.prior.source.atomic(PRIVATE/'certificates_heartbeat.json',
                    dict(pid=os.getpid(),cell=i,views=len(rows),seconds=round(time.monotonic()-t,3)))
            print(dict(cell=i,views=len(rows)),flush=True)
        out = dict(**binding,**report(rows),independent_confirmation=False,a_labels_opened=False,original_2pct=True)
        exp.once(OUT/'capacity_certificates.json',exp.stable(out))
        print(out,flush=True)


if __name__=='__main__':
    main()
