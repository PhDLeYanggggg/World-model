"""Separate aggregate reader for the fixed-routing training control."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_leaf_geometry_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(a,b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8), (a,b)


def verify_policy(p):
    b,h,r,er,eh,u = [p[k] for k in ('selected_known_benefit_mass','selected_known_harm_mass',
        'selected_known_reference_mass','selected_known_easy_reference_mass','selected_known_easy_harm_mass',
        'selected_unknown_envelope_mass')]
    equal(p['selected_net_gain_lower_mass'],b-h-u)
    equal(p['all_selected_risk_upper'],(h+u)/r if r>0 else None)
    equal(p['easy_selected_risk_upper'],(eh+u)/er if er>0 else None)
    equal(p['all_budget_slack_mass'],.02*r-h)
    equal(p['easy_budget_slack_mass'],.02*er-eh)
    expected = (r>0 and er>0 and b-h-u>0 and p['all_selected_risk_upper']<=.02+1e-12
        and p['easy_selected_risk_upper']<=.02+1e-12 and p['easy_degradation_upper'] is not None
        and p['easy_degradation_upper']<=.02+1e-12)
    assert p['finite_completion_supported'] == expected
    return 6


def main():
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for path,h in reg['bindings'].items():assert sha(ROOT/path)==h
    done = json.loads((PUBLIC/'complete.json').read_text())
    summary = json.loads((PUBLIC/'summary.json').read_text())
    cfg = json.loads((ROOT/'configs/m3w_european_leaf_geometry_v1.json').read_text())
    assert sha(PUBLIC/'summary.json')==done['summary_sha256']
    assert done['exact_refit_replay'] and done['exact_inference_replay'] and done['exact_readout_replay']
    groups = []
    checks = 0
    for ref in done['groups']:
        assert sha(ROOT/ref['path'])==ref['sha256']
        g = json.loads((ROOT/ref['path']).read_text());groups.append(g)
        r = g['result'];p = r['policies']
        checks += sum(verify_policy(v) for v in p.values())
        equal(r['signed_MSE_change'],r['scores']['refit']-r['scores']['original']);checks+=1
        for suffix in ('','_matched'):
            a,b = p['original'+suffix],p['refit'+suffix]
            equal(r['contrasts']['utility'+suffix+'_percent_full_known_reference'],
                100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/r['full_known_reference_mass'])
            checks+=1
        assert p['original_matched']['selected_count']==p['refit_matched']['selected_count'];checks+=1
        assert g['training']['known_training_rows']>0 and g['training']['original_leaves_reconstructed']>0
    assert len(groups)==summary['groups']==72
    for name in ('original','refit','original_matched','refit_matched'):
        vals = [g['result']['policies'][name] for g in groups]
        risk = [v['easy_selected_risk_upper'] for v in vals if v['easy_selected_risk_upper'] is not None]
        expected = dict(selected=sum(v['selected_count'] for v in vals),
            unknown_selected=sum(v['selected_unknown'] for v in vals),
            complete_support=sum(v['finite_completion_supported'] for v in vals),
            defined_easy_risk=len(risk),violations=sum(v>.02+1e-12 for v in risk),
            worst_easy_upper=max(risk) if risk else None)
        assert summary[name]==expected;checks+=6
    sites = sorted({g['source'] for g in groups})
    assert len(sites)==12
    for key in ('signed_MSE_change','utility_percent_full_known_reference','utility_matched_percent_full_known_reference'):
        buckets = {s:[] for s in sites}
        for g in groups:
            val = g['result'][key] if key=='signed_MSE_change' else g['result']['contrasts'][key]
            if val is not None:buckets[g['source']].append(val)
        means = np.array([math.fsum(v)/len(v) for v in buckets.values() if v])
        draw = np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(means),size=(cfg['bootstrap_resamples'],len(means)))
        samples = np.array([math.fsum(means[i])/len(means) for i in draw])
        equal(summary[key]['mean'],math.fsum(means)/len(means))
        for a,b in zip(summary[key]['CI95'],np.quantile(samples,[.025,.975])):equal(a,b)
        checks+=3
    expected = (summary['signed_MSE_change']['CI95'][1]<0 and
        summary['utility_percent_full_known_reference']['CI95'][0]>0 and
        summary['utility_matched_percent_full_known_reference']['CI95'][0]>0 and
        summary['refit']['complete_support']>=summary['original']['complete_support'] and
        summary['refit']['violations']<=summary['original']['violations'] and
        summary['refit']['worst_easy_upper'] is not None and
        summary['refit']['worst_easy_upper']<=summary['original']['worst_easy_upper'])
    assert summary['advance_to_transfer']==expected;checks+=1
    manifest = json.loads((PUBLIC/'checkpoint_manifest.json').read_text())
    assert manifest['checkpoints']==[g['checkpoint'] for g in groups]
    assert done['remote_verification']['checkpoints_verified']==72
    assert sum(g['checkpoint']['bytes'] for g in groups)==done['checkpoint_bytes']
    files = [Path(__file__),ROOT/'tests/test_m3w_leaf_geometry_readout.py']
    result = dict(aggregate_checks=checks,source_groups=72,
        already_completed_scalar_checks=done['scalar_checks'],
        original_leaf_means_reconstructed=sum(g['training']['original_leaves_reconstructed'] for g in groups),
        target_roundoff_rows=sum(g['training']['training_target_roundoff_rows'] for g in groups),
        complete_sha256=sha(PUBLIC/'complete.json'),summary_sha256=sha(PUBLIC/'summary.json'),
        bindings={str(p.relative_to(ROOT)):sha(p) for p in files},
        independent_aggregate_arithmetic=True,independent_model_implementation=False,
        independent_confirmation=False,deployment_changed=False)
    value=json.dumps(result,indent=2)+'\n';path=PUBLIC/'verification.json'
    if path.exists():assert path.read_text()==value
    else:
        with path.open('x') as f:f.write(value)
    print(value)


if __name__=='__main__':
    main()
