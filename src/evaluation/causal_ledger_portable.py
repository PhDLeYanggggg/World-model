"""Standalone reproduction of archived ledger aggregate evidence."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def mean(xs):
    return math.fsum(xs)/len(xs)


def quantile(xs, p):
    x = sorted(xs); at = (len(x)-1)*p; i = math.floor(at); j = math.ceil(at)
    return x[i]+(at-i)*(x[j]-x[i])


def check(x,y):
    if not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-9):
        raise ValueError('Aggregate differs from the archived analysis')


def reconstruct(evidence):
    if evidence['scope']!='exposed_source_development_aggregate' or evidence['budget']!=.02:
        raise ValueError('Changed study scope or original tolerance')
    boot = evidence['bootstrap']; contrasts = {}; risk = {}
    if len(boot)!=3000 or any(len(b)!=14 or any(type(i) is not int or not 0<=i<14 for i in b) for b in boot):
        raise ValueError('Retain all original bootstrap draws and source groups')
    for label,v in evidence['contrasts'].items():
        groups = v['source_seed_masses']
        if len(groups)!=14 or any(len(g)!=3 for g in groups):
            raise ValueError('Every source and forecast seed is required')
        source = []
        for g in groups:
            if any(m[0]<=0 or m[2]<0 or not all(map(math.isfinite,m)) for m in g):
                raise ValueError('Unknown paired mass or unsupported reference')
            source.append(mean([100*(d-u)/r for r,d,u in g]))
        draw = [mean([source[i] for i in b]) for b in boot]
        value = [mean(source),quantile(draw,.025),quantile(draw,.975)]
        for x,y in zip(value,v['archived'],strict=True):
            check(x,y)
        contrasts[label] = [round(x,12) for x in value]
    for label,v in evidence['policies'].items():
        rows = v['recording_views']
        if len(rows)!=102:
            raise ValueError('Unsupported or empty recording views cannot be dropped')
        got = dict(switches=sum(n for n,*_ in rows),
            passes=sum(all(x is not None and x<=.02 for x in (a,e,s)) for _,a,e,s in rows),
            undefined_easy=sum(e is None for _,_,e,_ in rows),
            easy_over_2=sum(e is not None and e>.02 for _,_,e,_ in rows))
        if got != v['archived']:
            raise ValueError('Original 2% risk accounting did not replay')
        risk[label] = got
    credit = evidence['credit']; aggregates = {}
    for key, v in credit.items():
        if len(v['recording_views'])!=102:
            raise ValueError('All paired credit views required')
        value = [math.fsum(row[i] for row in v['recording_views']) for i in range(len(v['archived']))]
        for x,y in zip(value,v['archived'],strict=True):
            check(x,y)
        aggregates[key] = [round(x,10) for x in value]
    return dict(scope=evidence['scope'],comparisons=contrasts,risk=risk,credit=aggregates,
        independent_confirmation=False,full_training_reproduction=False,submission_ready=False)


def outputs(result):
    s = io.StringIO(newline=''); w = csv.writer(s,lineterminator='\n')
    w.writerow(('comparison','completion_lower_pp','ci95_low','ci95_high'))
    for key,v in result['comparisons'].items():
        w.writerow([key,*(f'{x:.9f}' for x in v)])
    r = io.StringIO(newline=''); w = csv.writer(r,lineterminator='\n')
    fields = ('switches','passes','undefined_easy','easy_over_2'); w.writerow(('policy',*fields))
    for key,v in result['risk'].items():
        w.writerow([key,*(v[f] for f in fields)])
    return {'paired_comparisons.csv':s.getvalue(),'recording_risks.csv':r.getvalue(),
            'replayed.json':json.dumps(result,indent=2,allow_nan=False)+'\n'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',type=Path,required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parent; m = json.loads((root/'manifest.json').read_text())
    def digest(b): return hashlib.sha256(b).hexdigest()
    for name, spec in m['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or digest((root/p).read_bytes())!=spec['sha256']:
            raise ValueError('Archive file changed or outside package')
    result = reconstruct(json.loads((root/'evidence.json').read_text())); out = outputs(result)
    if args.verify and {k:digest(v.encode()) for k,v in out.items()}!=m['expected_outputs']:
        raise ValueError('Outputs differ from the archive manifest')
    args.output.mkdir(parents=True,exist_ok=False)
    for k,v in out.items():
        (args.output/k).write_text(v,encoding='utf-8')
    print(json.dumps(dict(contrasts=len(result['comparisons']),policies=len(result['risk']),
                          independent_confirmation=False)))


if __name__=='__main__':
    main()
