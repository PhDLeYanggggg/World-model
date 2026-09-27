"""Independently verify exchange partitions, identities and locality reductions."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selection_exchange as run
from scripts.verify_m3w_fixed_floor_tail import reduce_check

TESTS = ['tests/test_m3w_selection_exchange.py', 'tests/test_m3w_fixed_floor_probe.py',
    'tests/test_m3w_causal_descriptor_head.py','tests/test_m3w_causal_descriptor_protocol.py']


def main():
    cfg, data, _, _, identity = run.load()
    completion = json.loads((run.PUBLIC/'completion.json').read_text())
    assert completion['identity'] == identity
    assert json.loads((run.PUBLIC/'replay.json').read_text())['exact']
    rows=[]; queries=changed=checks=0
    for ref in completion['groups']:
        assert run.parent.base.artifact(ROOT/ref['path']) == ref
        d=json.loads((ROOT/ref['path']).read_text()); assert d['identity']==identity
        assert run.parent.base.artifact(ROOT/d['parent_head']['path'])==d['parent_head']
        head=json.loads((ROOT/d['parent_head']['path']).read_text())
        with np.load(ROOT/head['artifacts']['decisions']['path'],allow_pickle=False) as z:
            ids=z['ids'].copy(); new=z['descriptor'].copy(); old=z['control_matched_count'].copy(); elig=z['eligible'].copy()
        groups={}
        for i,rid in enumerate(ids):
            key=(str(data['sites'][rid]),str(data['recordings'][rid]),int(data['frames'][rid]))
            groups.setdefault(key,[]).append(i)
        assert len(groups)==d['queries']['queries']; nc=0
        for positions in groups.values():
            assert sum(new[i] for i in positions)==sum(old[i] for i in positions)
            nc+=any(new[i]!=old[i] for i in positions)
        assert nc==d['queries']['changed_queries']; queries+=len(groups); changed+=nc
        for r in d['rows']:
            at=data['sites'][ids]==r['site']; parts={
                'common':at & new & old,'new_only':at & new & ~old,
                'control_only':at & old & ~new,'unselected_eligible':at & elig & ~new & ~old,
                'excluded':at & ~elig}
            for key,mask in parts.items():
                s=r['sums'][key]
                assert s['rows']==int(mask.sum())
                assert s['known']+s['unknown']==s['rows']
                assert s['useful_rows']+s['harmful_rows']+s['zero_gain_rows']==s['known']
                checks+=3
            a,b=r['sums']['new_only'],r['sums']['control_only']
            delta=a['benefit_sum']-b['benefit_sum']-a['harm_sum']+b['harm_sum']
            np.testing.assert_allclose(delta,r['old_error_sum']-r['new_error_sum'],atol=1e-8)
            np.testing.assert_allclose(r['metric']['net_ADE_gain_percent'],100*delta/r['old_error_sum'],atol=1e-10)
            checks+=2
        rows+=d['rows']
    summary=json.loads((run.PUBLIC/'summary.json').read_text()); reductions=0
    for key,metric in summary['metrics'].items():
        reductions+=reduce_check(metric,rows,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    parent=json.loads((run.parent.PUBLIC/'summary.json').read_text())['paired']['control_matched_count']['ADE_gain_percent']
    for r in rows:
        assert r['metric']['new_only_rows']==r['metric']['control_only_rows']
    for site,value in parent['by_site'].items():
        np.testing.assert_allclose(value,summary['metrics']['net_ADE_gain_percent']['by_site'][site],atol=1e-11)
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt'; log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.parent.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png') and p.name!='verification.json'}
    proc=subprocess.run([sys.executable,'scripts/report_m3w_selection_exchange.py'],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode: raise RuntimeError(proc.stderr)
    assert all(run.parent.base.digest(run.PUBLIC/p)==h for p,h in artifacts.items())
    seal=json.loads((run.parent.PUBLIC/'verification.json').read_text()); bindings=dict(seal['source_bindings'])
    bindings.update(identity['bindings'])
    extras=['scripts/report_m3w_selection_exchange.py','scripts/verify_m3w_selection_exchange.py',*TESTS]
    bindings.update({p:run.parent.base.digest(ROOT/p) for p in extras})
    run.parent.base.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.parent.base.artifact(log),replay_exact=True,
        groups=len(completion['groups']),held_views=len(rows),independent_current_query_checks=queries,
        changed_query_views=changed,partition_and_accounting_checks=checks,locality_reductions=reductions,
        parent_contrast_reconstructed=True,report_figure_byte_reproducible=True,
        full_legacy_suite='not_run',cold_raw_rebuild=False,independent_confirmation=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(tests=tests,query_checks=queries,changed_queries=changed,accounting_checks=checks,reductions=reductions)))


if __name__=='__main__': main()
