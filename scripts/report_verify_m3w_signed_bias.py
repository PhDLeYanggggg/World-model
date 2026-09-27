"""Report and independently verify the fitting-only analytic score probe."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_m3w_signed_bias as run
TESTS=['tests/test_m3w_signed_bias_probe.py','tests/test_m3w_signed_bias_verification.py',
       'tests/test_m3w_selected_risk_diagnosis.py']


def main():
    ident=run.binding()
    completion=json.loads((run.PUBLIC/'completion.json').read_text())
    replay=json.loads((run.PUBLIC/'replay.json').read_text())
    assert completion['identity']==replay['identity']==ident and replay['exact_replay']
    assert completion['groups']==replay['groups'] and completion['summary']==replay['summary']
    assert run.base.artifact(ROOT/completion['summary']['path'])==completion['summary']
    data=json.loads((run.PUBLIC/'summary.json').read_text());records=[];checked=0
    for ref in completion['groups']:
        assert run.base.artifact(ROOT/ref['path'])==ref
        d=json.loads((ROOT/ref['path']).read_text());assert d['identity']==ident
        assert not d['held_labels_used'] and not d['policy_changed'];roles=d['roles']
        run.base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
        assert run.base.artifact(ROOT/d['parent_fit']['path'])==d['parent_fit']
        for arm,f in d['fits'].items():
            mass=f['curvature_mass'];assert .5<=mass<=1.+1e-12
            for axis in (0,1):
                mu=f['weighted_residual_sum'][axis]/mass
                np.testing.assert_allclose(mu,f['unconstrained_offset'][axis],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(max(mu,0),f['nonnegative_offset'][axis],rtol=1e-12,atol=1e-12)
                checked+=2
            gain=sum(f['weighted_residual_sum'][j]*f['nonnegative_offset'][j]-.5*mass*f['nonnegative_offset'][j]**2 for j in (0,1))
            np.testing.assert_allclose(f['fitting_objective_before']-f['fitting_objective_after'],gain,rtol=1e-9,atol=1e-12)
            assert not f['selected_policy_changed'] and not f['held_labels_used']
            assert not f['fitting_loss_reduction_is_downstream_lift'];checked+=1
        records.append(d)
    assert len(records)==108
    for arm,s in data['summary'].items():
        rr=[d['fits'][arm] for d in records]
        for axis in (0,1):
            vals=[r['nonnegative_offset'][axis] for r in rr]
            np.testing.assert_allclose(s['offsets_mean'][axis],sum(vals)/len(vals),rtol=1e-12,atol=1e-12)
            assert s['positive_offsets_per_axis'][axis]==sum(x>0 for x in vals)
        for key in ('before','after'):
            expected=sum(r['fitting_objective_'+key] for r in rr)/len(rr)
            np.testing.assert_allclose(expected,s['loss_'+key+'_mean'],rtol=1e-12,atol=1e-12)
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt';log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    lines=['# Fitting-Only Signed-Risk Bias Probe','','## Material Passport','',
        '- fresh_run:216 exact two-axis nonnegative intercept fits;108fixedsourcegroups.',
        '- cached_verified: existing Torch neural heads. No new neural-network training.',
        '- Future labels used only on the two fitting sources. Held evaluation is not_run; no policy action changed.',
        '- Already-opened development track. Independent selection, calibration and confirmation stay closed.','',
        '## Exact Fitting Result','','| Parent head | Positive all/easy offsets /108 | Mean all/easy offset | Median fitting loss reduction % | Mean objective before | Mean objective after |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,s in data['summary'].items():
        lines.append(f"| {arm} | {s['positive_offsets_per_axis']} | {s['offsets_mean']} | {s['median_relative_loss_reduction_percent']:.8f} | {s['loss_before_mean']:.8f} | {s['loss_after_mean']:.8f} |")
    lines+=['','Each intercept minimizes the exact source/query/subset-weighted quadratic while keeping predictions and all other parameters fixed. The score offset is constrained nonnegative. A nonpositive unconstrained optimum becomes zero.',
        'Fitting loss improvement follows mathematically from this projected minimization: it is not evidence of downstream lift, calibration, new neural dynamics or deployment quality.',
        'These offsets are signed-score units normalized by each fitting cost scale. They are not FDE percentages, failure probabilities, identified cost components or calibrated uncertainty.',
        '','## Seed Breakdown','','| Parent arm | Forecast seed | Fitted heads | Mean all/easy offset |','|---|---:|---:|---:|']
    for arm,s in data['summary'].items():
        for seed,r in s['by_seed'].items():lines.append(f"| {arm} | {seed} | {r['heads']} | {r['offset_mean']} |")
    lines+=['','## Verification','',f'- {tests} scoped tests across{len(TESTS)}files pass.',
        f'- {checked} independent saved-parameter/quadratic checks; full108group fit replay is exact.',
        '- The analytic loss is checked against the actual parent Torch loss, not merely an independently retyped formula. Autograd verifies its derivative on synthetic fixtures.',
        '- Unknown fitting labels contribute zero objective weight; source and known-query balance are checked. Empty subsets contribute zero objective and zero curvature mass.',
        f"- Fresh fitting: {completion['seconds']:.2f}s, peakRSS{completion['peak_RSS_bytes']}bytes, PID{completion['pid']}.",
        f"- Exact replay: {replay['seconds']:.2f}s, peakRSS{replay['peak_RSS_bytes']}bytes, PID{replay['pid']}.",
        '- Config/protocol, frozen parent checkpoint references and fit-group hashes are recorded. Large data and checkpoint files stay local.',
        '','## Next Decision','',
        'If nonnegative fitted offsets are nonzero, a separate preregistered policy experiment can test them with an identical retained-count control. If offsets vanish or are negligible, the exact objective already centers these errors and the next repair must address conditional/source-shift errors rather than a global intercept.',
        'No held policy comparison is run in this probe. No threshold is selected from the diagnostic outcomes. The original2%screen and undefined selected-risk cases are unchanged.',
        '','## Limits','',
        'No fitting-loss percentage is a generalization percentage. Source/seed summaries repeat development contexts. No claim of independent calibration, physical safety, seconds, metric scale, human gold, true3D or foundation modeling. Obs8/pred12 raw-frame stride12 remains image-local detector-silver. Stage5C/SMC are disabled.',
        'Full legacy integration tests and cold raw reconstruction are not_run. The research objective is not complete.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    parent_path=run.diagnosis.PUBLIC/'verification.json'
    parent=json.loads(parent_path.read_text());bindings=dict(parent['source_bindings'])
    for p,h in bindings.items():assert run.base.digest(ROOT/p)==h,p
    for p,h in parent['artifacts'].items():assert run.base.digest(run.diagnosis.PUBLIC/p)==h,p
    bindings.update(ident['bindings']);bindings.update({p:run.base.digest(ROOT/p) for p in ['scripts/report_verify_m3w_signed_bias.py',*TESTS]})
    artifacts={p.name:run.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json') and p.name!='verification.json'}
    run.base.immutable_json(run.PUBLIC/'verification.json',dict(parent_diagnostic_seal=run.base.artifact(parent_path),
        source_bindings=bindings,artifacts=artifacts,tests=tests,test_files=len(TESTS),test_log=run.base.artifact(log),
        independent_parameter_checks=checked,fit_groups=108,analytic_fits=216,exact_fit_replay=True,
        actual_parent_Torch_loss_checked=True,held_evaluation='not_run',new_neural_training=False,new_policy=False,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(verified=True,tests=tests,parameter_checks=checked,analytic_fits=216)))


if __name__=='__main__':main()
