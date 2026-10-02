"""Independent NumPy/scalar check of frozen source calibration outputs.

No fitting, optimizer, policy search, Torch import or local numerical cache.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selected_set_calibration_v1'
PARENT = PUBLIC.parent/'european_component_calibration_v1'
REMOTE = '/users/k24101830/m3w/european_selected_set_calibration_v1'
CAP = 128*2**20


def digest(value):
    return hashlib.sha256(value).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def actions(p, env, moving, support, cal, mode):
    if mode not in ('harm','reference','joint'):
        raise ValueError('Unknown component mode')
    q = np.array(p,dtype=float,copy=True)
    env = np.asarray(env,float)
    if q.shape != (len(env),5) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError('Invalid causal moments')
    h,eh,r,er = cal['margins']
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in (h,eh,r,er))
    if mode != 'reference':
        q[:,1] = np.minimum(env,q[:,1]+h*env)
        q[:,4] = np.minimum(env,q[:,4]+eh*env)
        q[:,1] = np.maximum(q[:,1],q[:,4])
    if mode != 'harm':
        q[:,2] *= 1-r
        q[:,3] = np.minimum(q[:,3]*(1-er),q[:,2])
    disabled = (q[:,2] <= 0) | (q[:,3] <= 0)
    if not cal['supported']:
        disabled[:] = True
    q[disabled,0] = 0
    return np.asarray(moving,bool) & np.asarray(support,bool) & (q[:,0]-q[:,1] > 0) & (
        q[:,1]-.02*q[:,2] <= 0) & (q[:,4]-.02*q[:,3] <= 0)


def scalar_bounds(y, chosen, env):
    y, chosen, env = np.asarray(y,float),np.asarray(chosen),np.asarray(env,float)
    assert y.shape == (len(chosen),5) and chosen.dtype == bool and env.shape == chosen.shape
    known = np.isfinite(y).all(1)
    assert (known | np.isnan(y).all(1)).all() and (y[known] >= 0).all()
    assert np.isfinite(env).all() and (env >= 0).all()
    ids = np.flatnonzero(known & chosen)
    mass = [math.fsum(float(y[i,j]) for i in ids) for j in range(5)]
    b,h,r,er,eh = mass
    u = math.fsum(float(env[i]) for i in np.flatnonzero(chosen & ~known))
    eb = math.fsum(float(y[i,0]) for i in ids if y[i,3] > 0)
    full_er = math.fsum(float(y[i,3]) for i in np.flatnonzero(known))
    lower = b-h-u
    easy_numerator = eh-eb+u
    easy_upper = (max(easy_numerator,0.) if (~known).any() else easy_numerator)/full_er if full_er > 0 else None
    ar = (h+u)/r if r > 0 else None
    e = (eh+u)/er if er > 0 else None
    reasons = []
    if ar is None or e is None: reasons.append('selected_reference_undefined')
    if ar is not None and ar > .02+1e-12: reasons.append('all_positive_harm_upper_exceeds_budget')
    if e is not None and e > .02+1e-12: reasons.append('easy_positive_harm_upper_exceeds_budget')
    if lower <= 0: reasons.append('positive_completion_utility_not_supported')
    if easy_upper is None or easy_upper > .02+1e-12: reasons.append('easy_preservation_not_supported')
    return dict(known_rows=int(known.sum()),unknown_rows=int((~known).sum()),
        selected_count=int(chosen.sum()),selected_unknown=int((chosen & ~known).sum()),
        selected_unknown_envelope_mass=u,selected_known_benefit_mass=b,
        selected_known_harm_mass=h,selected_known_reference_mass=r,
        selected_known_easy_reference_mass=er,selected_known_easy_harm_mass=eh,
        selected_known_easy_benefit_mass=eb,full_known_easy_reference_mass=full_er,
        all_budget_slack_mass=.02*r-h,easy_budget_slack_mass=.02*er-eh,
        selected_net_gain_lower_mass=lower,all_selected_risk_upper=ar,
        easy_selected_risk_upper=e,easy_degradation_upper=easy_upper,
        finite_completion_supported=not reasons,reasons=reasons,
        known_partial_label_costs_held_fixed=True,per_row_label_inference_gate=False,
        population_safety_guarantee=False)


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    for k,a in actual.items():
        b = expected[k]
        if isinstance(a,float):
            assert b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10), (k,a,b)
        else:
            assert a == b, (k,a,b)
    return len(actual)


def fold_actions(z, folds, mode):
    rec = z['recordings'].astype(str)
    assert {f['held_recording'] for f in folds} == set(rec)
    assert len(folds) == len(set(rec))
    out = np.zeros(len(rec),bool)
    for f in folds:
        name, cal = f['held_recording'],f['calibration']
        assert set(cal['recordings']) == set(rec)-{name}
        ix = rec == name
        out[ix] = actions(z['p'][ix],z['env'][ix],z['moving'][ix],z['support'][ix],cal,mode)
    return out


def verify_group(z, group, old):
    meta = json.loads(str(z['meta_json']))
    assert group['identity'] == meta['identity']
    assert group['source'] == meta['source'] == old['identity']['source']
    for k,h in meta['array_hashes'].items(): assert array_hash(z[k]) == h
    assert old['identity']['target_hash'] == array_hash(z['y'])
    assert old['identity']['prediction_hash'] == array_hash(z['p'])
    assert old['source'] == meta['parent_source']
    known = np.isfinite(z['y']).all(1)
    full_ref = math.fsum(float(z['y'][i,2]) for i in np.flatnonzero(known))
    assert math.isclose(full_ref,group['full_known_reference'],rel_tol=1e-10,abs_tol=1e-10)
    assert group['rows_in_packet'] == len(z['p'])
    rows = group['result']['rows']
    assert len(rows) == 6 and {(r['mode'],r['role']) for r in rows} == {
        (m,k) for m in ('harm','reference','joint') for k in ('source_oof','source_resubstitution')}
    checks, views = 0, []
    for r in rows:
        mode, role = r['mode'], r['role']
        if role == 'source_oof':
            a = fold_actions(z,r['folds'],mode)
            b = fold_actions(z,old['folds'],mode)
            assert array_hash(b) == old['oof_action_hashes'][mode]
        else:
            assert set(r['final']['recordings']) == set(z['recordings'])
            a = actions(z['p'],z['env'],z['moving'],z['support'],r['final'],mode)
            b = actions(z['p'],z['env'],z['moving'],z['support'],old['final'],mode)
        assert not (a & ~b).any()
        assert array_hash(a) == r['action_hash'] and array_hash(b) == r['parent_action_hash']
        assert int(a.sum()) == r['selected'] and int(b.sum()) == r['parent_selected']
        assert int((b & ~a).sum()) == r['removed']
        aa,bb = scalar_bounds(z['y'],a,z['env']),scalar_bounds(z['y'],b,z['env'])
        checks += compare(aa,r['selected_set'])+compare(bb,r['parent'])
        views.append(dict(mode=mode,role=role,source=group['source'],
            parent=bb,selected_set=aa,removed=int((b & ~a).sum())))
    return dict(scalar_checks=checks,action_hash_checks=12,rows=views)


def verify_summary(summary, groups, cfg):
    expected_keys = {m+'_'+r for m in ('harm','reference','joint')
                     for r in ('source_oof','source_resubstitution')}
    assert set(summary['comparisons']) == expected_keys
    assert summary['source_groups'] == len(groups) == 72
    assert summary['source_only'] is True and summary['neural_updates'] == 0
    assert all(summary[k] is False for k in ('transfer_evaluated','independent_roles_read',
        'deployment_promoted','stage5c_executed','smc_enabled'))
    checks = 0
    for key, shown in summary['comparisons'].items():
        mode, role = key.split('_',1)
        rr = [(g,r) for g in groups for r in g['result']['rows'] if r['mode'] == mode and r['role'] == role]
        assert shown['views'] == len(rr) == 72
        for arm in ('parent','selected_set'):
            rows = [r[arm] for _,r in rr]
            risks = [r['easy_selected_risk_upper'] for r in rows]
            expected = dict(complete_support_pass=sum(r['finite_completion_supported'] for r in rows),
                selected_occurrences=sum(r['selected_count'] for r in rows),
                selected_unknown_occurrences=sum(r['selected_unknown'] for r in rows),
                nonempty_views=sum(r['selected_count'] > 0 for r in rows),
                easy_risk_defined=sum(x is not None for x in risks),
                easy_risk_undefined=sum(x is None for x in risks),
                easy_risk_violations=sum(x is not None and x > .02+1e-12 for x in risks),
                easy_risk_worst=max((x for x in risks if x is not None),default=None))
            checks += compare(expected,shown[arm])
        for lost in (True,False):
            field = 'lost_complete_support' if lost else 'gained_complete_support'
            assert shown[field] == sum(r['parent']['finite_completion_supported'] == lost and
                r['selected_set']['finite_completion_supported'] != lost for _,r in rr)
            checks += 1
        for name in ('utility_change_percent_full_reference','common_defined_easy_risk_change'):
            values = []
            for g,r in rr:
                a,b = r['selected_set'],r['parent']
                if name.startswith('utility'):
                    v = 100*(a['selected_net_gain_lower_mass']-b['selected_net_gain_lower_mass'])/g['full_known_reference'] if g['full_known_reference'] > 0 else None
                else:
                    v = a['easy_selected_risk_upper']-b['easy_selected_risk_upper'] if a['easy_selected_risk_upper'] is not None and b['easy_selected_risk_upper'] is not None else None
                values.append((g['source'],v))
            defined = [(s,v) for s,v in values if v is not None]
            sites = sorted({s for s,_ in defined})
            point = np.array([math.fsum(v for s,v in defined if s == site)/sum(s == site for s,_ in defined) for site in sites])
            report = shown[name]
            assert report['defined_views'] == len(defined)
            assert report['undefined_views'] == 72-len(defined)
            assert report['locality_count'] == len(sites)
            if not sites:
                assert report['mean'] is None and report['CI95'] is None
            else:
                rng = np.random.default_rng(cfg['bootstrap_seed'])
                boots = point[rng.integers(0,len(sites),(cfg['bootstrap_resamples'],len(sites)))].mean(1)
                np.testing.assert_allclose(report['mean'],point.mean(),rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(report['CI95'],np.quantile(boots,[.025,.975]),rtol=1e-10,atol=1e-10)
                if len(defined) < 72: assert report['strict_all_views_mean'] is None
            checks += 5
    return checks


def validate_bundle(bundle, manifest, reg_hash, job):
    assert bundle['manifest'] == manifest
    assert bundle['registration_sha256'] == reg_hash
    complete = bundle['complete']
    assert complete['registration_sha256'] == reg_hash
    assert complete['job_id'] == job and complete['source_heads'] == 72 and complete['exact_replay'] is True
    assert complete['new_neural_updates'] == 0 and complete['transfer_evaluated'] is False
    assert complete['independent_roles_read'] is False
    a = bundle['accounting']
    assert a['returncode'] == 0
    assert any(row.split('|')[:3] == [job,'COMPLETED','0:0'] for row in a['stdout'].splitlines())
    names = [r['group'] for r in manifest['packets']]
    assert len(names) == len(set(names)) == 72
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+',n) for n in names)
    assert [r['group'] for r in complete['groups']] == names
    assert set(bundle['outputs']) == set(bundle['inputs']) == set(names)
    assert digest(bundle['summary_text'].encode()) == complete['summary_sha256']
    total = 0
    for refs, files, encoded in ((manifest['packets'],bundle['inputs'],True),
                                (complete['groups'],bundle['outputs'],False)):
        for r in refs:
            raw = base64.b64decode(files[r['group']],validate=True) if encoded else files[r['group']].encode()
            total += len(raw)
            assert len(raw) == r['bytes'] and digest(raw) == r['sha256']
    assert total <= CAP
    return total


def fetch():
    from scripts.manage_m3w_easy_gradient_diagnostic import call_remote
    manifest = json.loads((PUBLIC/'input_manifest.json').read_text())
    reg_hash = digest((PUBLIC/'registration.json').read_bytes())
    job = json.loads((PUBLIC/'submission_receipt.json').read_text())['job_id']
    code = r'''
import base64,gzip,hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['home'])
assert r==pathlib.Path('/users/k24101830/m3w/european_selected_set_calibration_v1')
assert json.loads((r/'.owner.json').read_text())['experiment']==r.name
assert json.loads((r/'submission_receipt.json').read_text())['job_id']==p['job']
c=json.loads((r/'complete.json').read_text())
a=subprocess.run(['sacct','-j',p['job'],'-X','--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=25)
assert a.returncode==0 and any(x.split('|')[:3]==[p['job'],'COMPLETED','0:0'] for x in a.stdout.splitlines())
m=json.loads((r/'input_manifest.json').read_text());ins={};outs={};total=0
for refs,folder,target,encoded in ((m['packets'],'inputs',ins,True),(c['groups'],'groups',outs,False)):
    for e in refs:
        name=e['group'];assert pathlib.PurePath(name).name==name and name not in ('.','..')
        raw=(r/folder/(name+('.npz' if encoded else '.json'))).read_bytes()
        total+=len(raw);assert total<=p['cap']
        assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
        target[name]=base64.b64encode(raw).decode() if encoded else raw.decode()
b=dict(complete=c,manifest=m,inputs=ins,outputs=outs,summary_text=(r/'summary.json').read_text(),
    registration_sha256=hashlib.sha256((r/'registration.json').read_bytes()).hexdigest(),
    accounting=dict(returncode=a.returncode,stdout=a.stdout))
raw=json.dumps(b).encode();assert len(raw)<2*p['cap']
print(json.dumps(dict(payload=base64.b64encode(gzip.compress(raw)).decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
'''
    reply = call_remote(code,dict(home=REMOTE,job=job,cap=CAP))
    if reply['returncode'] != 0:
        raise RuntimeError(json.dumps(reply))
    packed = json.loads(reply['stdout'])
    assert packed['bytes'] <= 2*CAP
    raw = gzip.decompress(base64.b64decode(packed['payload'],validate=True))
    assert len(raw) == packed['bytes'] and digest(raw) == packed['sha256']
    bundle = json.loads(raw)
    total = validate_bundle(bundle,manifest,reg_hash,job)
    return bundle,total


def register():
    files = [Path(__file__),ROOT/'tests/test_m3w_selected_set_readout.py',PUBLIC/'readout_verification.md']
    return dict(experiment_registration_sha256=digest((PUBLIC/'registration.json').read_bytes()),
        bindings={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in files},
        input_manifest_sha256=digest((PUBLIC/'input_manifest.json').read_bytes()),
        parent_calibration_manifest_sha256=digest((PARENT/'calibration_freeze.json').read_bytes()),
        no_training=True,no_policy_change=True,independent_roles_read=False,
        local_numeric_cache=False,memory_input_cap_bytes=CAP)


def once(path,value):
    text = json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists(): assert path.read_text() == text
    else:
        with path.open('x') as f: f.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['register','verify'])
    args = parser.parse_args()
    reg = register()
    if args.phase == 'register': once(PUBLIC/'readout_registration.json',reg);return
    assert json.loads((PUBLIC/'readout_registration.json').read_text()) == reg
    subprocess.run(['git','diff','--exit-code','HEAD','--',str(PUBLIC/'readout_registration.json')],check=True,cwd=ROOT)
    subprocess.run(['git','cat-file','-e','HEAD:'+str((PUBLIC/'readout_registration.json').relative_to(ROOT))],check=True,cwd=ROOT)
    bundle,total = fetch()
    old = {}
    for r in json.loads((PARENT/'calibration_freeze.json').read_text())['groups']:
        raw = (ROOT/r['path']).read_bytes();assert digest(raw) == r['sha256']
        old[Path(r['path']).stem] = json.loads(raw)
    checks, hashes, readout, groups, details = 0, 0, [], [], []
    for name in bundle['inputs']:
        group = json.loads(bundle['outputs'][name]);groups.append(group)
        with np.load(io.BytesIO(base64.b64decode(bundle['inputs'][name])),allow_pickle=False) as z:
            result = verify_group(z,group,old[name])
        checks += result['scalar_checks'];hashes += result['action_hash_checks']
        readout.extend(result['rows'])
        for r in result['rows']:
            if r['mode'] != 'joint' or r['role'] != 'source_oof': continue
            a,b = r['selected_set'],r['parent']
            details.append(dict(group=name,source=group['source'],parent=b,selected_set=a,
                known_easy_risk_parent=b['selected_known_easy_harm_mass']/b['selected_known_easy_reference_mass'] if b['selected_known_easy_reference_mass'] > 0 else None,
                known_easy_risk_selected=a['selected_known_easy_harm_mass']/a['selected_known_easy_reference_mass'] if a['selected_known_easy_reference_mass'] > 0 else None))
    summary = json.loads(bundle['summary_text'])
    summary_checks = verify_summary(summary,groups,json.loads((ROOT/'configs/m3w_european_selected_set_calibration_v1.json').read_text()))
    receipt = dict(job_id=bundle['complete']['job_id'],accounting=bundle['accounting'],
        source_groups=72,scalar_checks=checks,action_hash_checks=hashes,
        summary_count_and_bootstrap_checks=summary_checks,
        frozen_row_comparisons=len(readout),remote_input_and_output_bytes=total,
        remote_complete_sha256=digest(json.dumps(bundle['complete'],sort_keys=True).encode()),
        summary_sha256=bundle['complete']['summary_sha256'],
        readout_registration_sha256=digest((PUBLIC/'readout_registration.json').read_bytes()),
        numerical_arrays_saved_locally=False,models_fit=False,policy_changed=False,
        independent_roles_read=False,exact_scheduled_replay=True,
        scalar_tolerance=dict(relative=1e-10,absolute=1e-10))
    # Only lightweight aggregate evidence is persisted, never inputs or folds.
    assert len(bundle['summary_text'].encode()) < 64*2**10
    once(PUBLIC/'verified_summary.json',summary)
    once(PUBLIC/'readout_verification.json',receipt)
    # Source joint aggregates expose observed versus unknown-completion risk;
    # they do not create a per-row inference rule or a new scientific endpoint.
    details_text = json.dumps(details,indent=2,allow_nan=False)+'\n'
    assert len(details_text.encode()) < 256*2**10
    once(PUBLIC/'joint_oof_details.json',details)
    print(json.dumps(dict(receipt=receipt,summary=summary),indent=2))


if __name__ == '__main__':
    main()
