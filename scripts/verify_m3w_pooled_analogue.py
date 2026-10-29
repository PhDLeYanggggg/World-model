"""Receipt arithmetic independent of the pooled inference runner."""
import argparse
import json
from pathlib import Path

from scripts import verify_m3w_selected_analogue as previous
from scripts import run_m3w_pooled_analogue as run

NAMES = dict(zip(run.api.ARMS,previous.ARMS))


def compatible_rows(rows):
    return [dict(r,scores={NAMES[k]:v for k,v in r['scores'].items()}) for r in rows]


def verify(rows,summary,expected):
    old={k:v for k,v in summary.items() if k not in ('reference_bank','magnitude_better_than_origin')}
    old['equal_view_metrics']={NAMES[k]:v for k,v in old['equal_view_metrics'].items()}
    old['information_screen']={k.replace('pooled_mean','source_mean').replace('pooled_shuffled32','shuffled_nearest32'):v
                                for k,v in old['information_screen'].items()}
    result=previous.verify(compatible_rows(rows),old,expected)
    checks=0
    for row in rows:
        for ref in row['references']:
            source,rec=json.loads(ref['recording'])
            previous.require(source==row['source']==ref['query_source'] and rec==ref['local_recording'],
                             'Recording namespace changed')
            counts=ref['known_donors_by_source']
            previous.require(len(counts)==4 and source in counts and
                all(isinstance(v,int) and v>=0 for v in counts.values()) and
                sum(counts.values())==ref['known_donors'],'Pooled donor attribution differs')
            checks+=1
    a=summary['equal_view_metrics']['pooled_nearest32']['easy']['full_cohort_MSE']
    b=summary['equal_view_metrics']['origin']['easy']['full_cohort_MSE']
    previous.require(summary['magnitude_better_than_origin']==(a is not None and b is not None and a<b-1e-12),
                     'Magnitude comparison changed')
    result.update(pooled_bank_attribution_checks=checks,magnitude_comparison_checked=True)
    return result


def execute(check):
    reg,_,_=run.admission();complete=json.loads((run.PUBLIC/'complete.json').read_text())
    previous.require(complete['registration_sha256']==run.sha(run.REG) and
                     complete['summary_sha256']==run.sha(run.PUBLIC/'summary.json'),'Freeze changed')
    audit=json.loads(run.AUDIT.read_text());rows=[]
    for receipt in complete['groups']:
        p=run.ROOT/receipt['path']
        previous.require(not p.is_symlink() and p.resolve().is_relative_to((run.PUBLIC/'groups').resolve()) and
                         run.sha(p)==receipt['sha256'],'Group binding changed')
        row=json.loads(p.read_text());previous.require(row['registration_sha256']==run.sha(run.REG),'Registration changed')
        old=json.loads((run.previous.PUBLIC/'groups'/p.name).read_text())
        for arm in ('origin','estimation'):
            previous.compare(row['scores'][arm],old['scores'][arm])
            previous.require(row['prediction_sha256'][arm]==old['prediction_sha256'][arm],'Frozen control changed')
        previous.require(row['parent_cohort_sha256']==old['parent_cohort_sha256'],'Selection changed')
        sources=set(audit['contexts'][row['context']])
        previous.require(all(set(r['known_donors_by_source'])==sources for r in row['references']),
                         'Donors crossed context boundary')
        rows.append(row)
    result=verify(rows,json.loads((run.PUBLIC/'summary.json').read_text()),run.previous.previous.parent.expected(reg))
    result.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),summary_sha256=run.sha(run.PUBLIC/'summary.json'),
        verifier_sha256=run.sha(Path(__file__)),prior_control_comparisons=144,
        result_source='cached_verified_separate_receipt_arithmetic',new_inference=False,independent_replication=False)
    p=run.PUBLIC/'verification.json'
    if check:previous.require(result==json.loads(p.read_text()),'Verification differs')
    else:run.once(p,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');execute(p.parse_args().check)
