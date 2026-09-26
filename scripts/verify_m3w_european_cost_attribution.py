"""Verify frozen inference receipt, exact factor algebra and scoped regressions."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_cost_attribution as run
from scripts.report_m3w_european_cost_attribution import aggregate
import numpy as np


def main():
    identity=run.registration()
    done=json.loads((run.PUBLIC/'completion_checks.json').read_text())
    replay=json.loads((run.PUBLIC/'replay_receipt.json').read_text())
    assert done['identity']==replay['identity']==identity
    assert done['groups']==replay['groups'] and replay['all_passed'] and done['new_fits']==0
    rows=[]; checks=0
    for ref in done['groups']:
        assert run.artifact(ROOT/ref['path'])==ref
        row=json.loads((ROOT/ref['path']).read_text()); rows.append(row)
        for fold in row['folds']:
            assert fold['held'] not in fold['training_sites']
            assert run.artifact(ROOT/fold['conditional_checkpoint']['path'])==fold['conditional_checkpoint']
            for sub in fold['diagnosis']['subsets'].values():
                if sub.get('status')=='not_estimable': continue
                for key in ('easy','all_harm'):
                    d=sub[key]
                    np.testing.assert_allclose(d['membership_squared']+d['severity_squared']+d['cross_term'],d['MSE'],rtol=1e-10,atol=1e-10)
                    assert d['label_assisted_E_MSE']==d['severity_squared']; checks+=2
                if sub['severity_weighted_Brier'] is not None:
                    np.testing.assert_allclose(sub['severity_weighted_Brier']*sub['easy_expert_squared_mean'],sub['easy']['membership_squared'],rtol=1e-10,atol=1e-10); checks+=1
                np.testing.assert_allclose(sum(sub['strata'][s]['easy_MSE_contribution'] for s in ('easy','outside_easy')),sub['easy']['MSE'],rtol=1e-10,atol=1e-10); checks+=1
    assert len(rows)==36 and sum(len(r['folds']) for r in rows)==replay['held_folds']==144
    assert aggregate(rows)==json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    svg=run.PUBLIC/'factor_terms.svg'; before=run.digest(svg)
    subprocess.run([sys.executable,'scripts/plot_m3w_european_cost_attribution.py'],cwd=ROOT,check=True)
    assert before==run.digest(svg)
    old=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    tests=sorted(set(old['test_files'])|{'tests/test_m3w_cost_factor_attribution.py','tests/test_m3w_cost_factor_reporting.py'})
    xml=run.PRIVATE/'tests.xml'
    p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-1700:],flush=True)
    assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    required=('protocol.md','results.md','conclusions.md','operation_zh.md','factor_terms.svg')
    assert all((run.PUBLIC/p).is_file() for p in required)
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/p).stat().st_size<2**20 for p in artifacts)
    bindings=sorted(set(run.FILES+tests+['scripts/report_m3w_european_cost_attribution.py','scripts/plot_m3w_european_cost_attribution.py',str(Path(__file__).relative_to(ROOT))]))
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings},test_files=tests,tests=count,
        groups=len(rows),full_expert_replays=replay['full_expert_replays'],independent_algebra_checks=checks,
        deterministic_svg=True,full_legacy_suite='not_run',new_fits=0,policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(tests=count,files=len(tests),algebra_checks=checks,public_artifacts=len(artifacts),source_bindings=len(bindings))))


if __name__=='__main__': main()
