"""Export anonymous scalar replay for the budget and trained-policy tables."""
import argparse
import hashlib
import inspect
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_budget_certificates as bounds
from scripts import build_m3w_review_replay as pack
from scripts import train_m3w_cv_minimum_budget as study

exp,base,prior,np = study.exp,study.base,study.prior,study.np


def collect(cells,d,donor):
    score = prior.prior.parent.readout
    pairs = {f'{a}_vs_{k}':(b,a) for a in study.ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(teacher_at_match=('cost','teacher_matched'),guarded_vs_teacher=('teacher','teacher_guarded'))
    rows,budget = [],[]
    for i,c in enumerate(cells):
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            act = {k:z[k] for k in study.ARMS}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            c0,o = cv[ix],ols[ix]
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            paths = {base.CV:c0,base.OLS:o,**{k:exp.slate.path_for(c0,bank[ix],a[ix]) for k,a in act.items()}}
            r = score.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = exp.weight(q)
            terms = {k:prior.study.motion.full_masses(paths[b],paths[a],c0,y,v,w) for k,(b,a) in pairs.items()}
            site, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(site),seed=seed,comparisons=terms))
            old = exp.read(bounds.parent.PRIVATE/f'nested_{i}_{rec}.json')['results']['full_relaxed']
            cert = exp.read(study.PRIVATE/f'cert_{i}_{rec}.json')['result'] if old['status']=='verified' else {}
            budget.append(dict(original=old['status'],n=int(ix.sum()),lower=cert.get('lower'),upper=cert.get('upper')))
    return rows,budget


def build(path):
    rp,fp,cp = [study.OUT/k for k in ('training_readout.json','training_freeze.json','capacity_certificates.json')]
    r,f,cap = map(exp.read,(rp,fp,cp))
    if (r['freeze']!=exp.sha(fp) or any(z['a_labels_opened'] or z['independent_confirmation']
                                       or not z['original_2pct'] for z in (r,cap))):
        raise ValueError('Need the development-only frozen tables')
    _,donor,_ = base.register()
    data = base.fit_population(lambda **v:None)
    rows,budget = collect(f['cells'],data,donor)
    roster = {x['recording']:x['source'] for x in rows}
    if prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)!=r['summary']:
        raise ValueError('Policy table did not reproduce')
    rb = bounds.report(budget)
    if rb!={k:cap[k] for k in rb}:
        raise ValueError('Budget table did not reproduce')
    rec = {k:f'R{i:02d}' for i,k in enumerate(sorted(roster))}
    src = {k:f'S{i:02d}' for i,k in enumerate(sorted(set(roster.values())))}
    seeds = {k:i for i,k in enumerate(base.SEEDS,1)}
    packet = dict(seeds=list(seeds.values()),roster={rec[k]:src[v] for k,v in roster.items()},
                  rows=pack.trim(rows,rec,src,seeds),budget=budget)
    expected = dict(policy=dict(r['summary'],seeds=list(seeds.values())),budget=rb)
    budget_code = inspect.getsource(bounds.report).replace('study.FRACTIONS',repr(bounds.study.FRACTIONS))
    script = '''from pathlib import Path
import hashlib
import json
import numpy as np
from src.evaluation.m3w_grouped_recording_readout import grouped_readout

'''+budget_code+'''
root = Path(__file__).resolve().parent
read = lambda name: json.loads((root/name).read_text())
for name,digest in read('manifest.json').items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError(f'Manifest mismatch: {name}')
p = read('statistics.json')
for row in p['rows']:
    for m in row['metrics'].values():
        risk = m['risk']
        keys = ('completion_all_risk_upper','completion_easy_risk_upper','known_easy_net_degradation')
        risk['supported'] = all(risk[k] is not None and risk[k]<=.02 for k in keys)
actual = dict(policy=grouped_readout(p['rows'],p['roster'],p['seeds']),budget=report(p['budget']))
if actual!=read('expected.json'):
    raise ValueError('Anonymous scalar tables did not replay')
print(json.dumps({'views':actual['policy']['views'],'budget':actual['budget']},sort_keys=True))
'''
    files = {name:(pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md':b'Run python replay.py. Replays scalar policy scores and count bounds, not training or pair geometry.\n',
                  'replay.py':script.encode(),'statistics.json':pack.raw_json(packet),'expected.json':pack.raw_json(expected),
                  'requirements.txt':f'numpy=={np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(path,files)
    print(exp.sha(path),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=study.PRIVATE/'anonymous_minimum_budget.zip')
    build(p.parse_args().out)
