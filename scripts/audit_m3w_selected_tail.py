"""Replay frozen controls and account for selection-conditional risk errors."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from scripts import evaluate_m3w_isolated_harm_fp64 as parent
from src.evaluation import m3w_selected_tail_audit as audit

run, api = parent.run, parent.api
ROOT = run.ROOT
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selected_tail_audit_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selected_tail_audit_v1'
REG = PUBLIC/'registration.json'


def register():
    reg, cfg, ancestor = run.admission()
    files = [Path(__file__).resolve(), ROOT/'src/evaluation/m3w_selected_tail_audit.py',
             ROOT/'tests/test_m3w_selected_tail_audit.py', PUBLIC/'protocol.md']
    doc = dict(bindings={str(p.relative_to(ROOT)): run.base.sha(p) for p in files},
        parent_registration_sha256=run.base.sha(run.REG),
        parent_freeze_sha256=run.base.sha(run.PUBLIC/'training_freeze.json'),
        parent_readout_sha256=run.base.sha(parent.PUBLIC/'complete.json'),
        expected_views=144, arms=list(audit.ARMS), diagnostic_only=True,
        new_training=False, independent_roles_read=False, deployment_changed=False)
    run.base.once(REG, doc)
    return reg, cfg, ancestor, doc


def admit():
    reg, cfg, ancestor, doc = register()
    for rel in [*doc['bindings'], str(REG.relative_to(ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=ROOT, capture_output=True)
        if p.returncode or p.stdout != (ROOT/rel).read_bytes():
            raise ValueError('Commit fixed diagnostic before computation')
    return reg, cfg, ancestor, doc


def old_rows():
    complete = json.loads((parent.PUBLIC/'complete.json').read_text())
    rows = {}
    for ref in complete['groups']:
        path = ROOT/ref['path']
        if path.is_symlink() or not path.resolve().is_relative_to((parent.PUBLIC/'groups').resolve()) or run.base.sha(path) != ref['sha256']:
            raise ValueError('Changed parent readout')
        row = json.loads(path.read_text())
        key = row['scope'], row['context'], row['source'], row['seed']
        if key in rows:
            raise ValueError('Duplicate parent view')
        rows[key] = row
    if len(rows) != 144:
        raise ValueError('Complete parent census required')
    return rows


def report(summary, complete):
    lines = ['# Where the Frozen Policy Underestimates Selected Risk', '',
        '## Material Passport', '',
        f"Fresh frozen-model inference and accounting:144/144 views in{complete['seconds']:.2f}s.",
        'All prediction hashes and original selections reproduce. No new training,',
        'threshold selection or deployment. Original TRAIN packets only; held',
        'recordings are already exposed. This is not independent confirmation.', '',
        '## Easy-Constraint Error Accounting', '',
        'Sums below are source-scale-normalized and repeat rows across contexts/seeds.',
        'They are not ADE/FDE, independent samples, calibrated probabilities, or causal',
        'effects. Each row obeys observed=predicted+harm error+reference error.', '']
    for scope in ('fit', 'held'):
        lines += [f'### {scope.title()} Own-Selected Cohort', '',
            '| Arm | Selected | Unknown | Predicted excess | Harm underestimated | Reference overestimated x0.02 | Observed excess |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for arm in audit.ARMS:
            c = summary[scope][arm]['own']; v = c['easy']
            lines.append('| '+arm+' | '+str(c['rows'])+' | '+str(c['unknown'])+' | '+
                ' | '.join(f'{v[k]:.6f}' for k in ('predicted_excess', 'harm_underestimate', 'reference_overestimate', 'observed_excess'))+' |')
        lines += ['', '| Arm | Violating known rows | Still allowed with perfect reference | Still allowed with perfect harm | Violating views |',
            '|---|---:|---:|---:|---:|']
        for arm in audit.ARMS:
            v = summary[scope][arm]['own']['easy']
            lines.append('| '+arm+' | '+' | '.join(str(v[k]) for k in ('violating_rows',
                'violating_rows_still_allowed_with_true_reference', 'violating_rows_still_allowed_with_true_harm', 'violating_views'))+' |')
        lines += ['', 'Perfect-component replacements use future outcomes, are diagnostic only,',
            'and are never legal inference inputs or deployable policies.', '',
            '| Arm | Known selected in fit-radius bins <=p50 / p50-p90 / p90-p99 / >=p99 | Known easy-violating rows in same bins |',
            '|---|---|---|']
        for arm in audit.ARMS:
            v = summary[scope][arm]
            lines.append('| '+arm+' | '+str(v['radius_known_selected'])+' | '+str(v['radius_known_easy_violating_rows'])+' |')
        lines += ['', 'Bin assignment uses right-sided boundaries: values equal to a boundary go',
            'in the higher bin. Labels in the header are shorthand, not exact quantile',
            'occupancy. Ties/empty bins retained. Radius is not a density/support proof.', '',
            '| Isolated branch cohort | Observed easy harm | Harm on rows exceeding model supremum | Sum minimum shortfall | Such rows |',
            '|---|---:|---:|---:|---:|']
        for cohort in ('eligible', 'anchor', 'isolated'):
            c = summary[scope+'_ceilings'][cohort]['easy']
            lines.append('| '+cohort+' | '+' | '.join(f'{c[k]:.6f}' for k in
                ('observed_harm', 'observed_harm_above_supremum', 'minimum_shortfall', 'above_supremum_rows'))+' |')
        lines += ['', 'A realized label above a conditional-mean ceiling does not prove the true',
            'conditional expectation is above it. This is a feasibility stress diagnosis,',
            'not a reason to introduce unbounded corrections or claim safety.', '']
    lines += ['## Boundaries and Reproduction', '',
        'All six controls, all sources and all seeds are retained. Exact raw sums,',
        'fixed anchor cohorts, unknown counts and fit-only edges remain in group JSON.',
        'The prior completion-bound safety gates remain failed. This diagnosis has no',
        'new success gate and does not upgrade the model or recertify Stage37.', '',
        '```bash', '.venv-pytorch/bin/python -m scripts.audit_m3w_selected_tail check', '```', '',
        'Complete outputs are immutable. Check verifies artifact hashes, identities,',
        'arithmetic and counts, not a new model fit. Full execution used native arm64',
        'CPU4/workers0, CREATE read-only TRAIN streaming, no new scheduler job.',
        'Obs8/pred12 raw stride12, dataset-local/pixel coordinates, detector tracks.',
        'No metric, seconds, physical safety, human gold, true3D or foundation claim.',
        'Stage5C and SMC disabled. No reserved role opened.', '']
    return '\n'.join(lines)


def verify():
    reg, _, _, _ = admit()
    parent.models(reg)
    old = old_rows()
    complete = json.loads((PUBLIC/'complete.json').read_text())
    if complete['registration_sha256'] != run.base.sha(REG):
        raise ValueError('Changed diagnostic registration')
    rows = []
    for ref in complete['groups']:
        p = ROOT/ref['path']
        if p.is_symlink() or not p.resolve().is_relative_to((PUBLIC/'groups').resolve()) or run.base.sha(p) != ref['sha256']:
            raise ValueError('Changed audit receipt')
        row = json.loads(p.read_text())
        original = old[row['scope'], row['context'], row['source'], row['seed']]
        if row['prediction_hashes'] != original['prediction_hashes'] or row['registration_sha256'] != run.base.sha(REG):
            raise ValueError('Frozen inference or provenance changed')
        audit.validate_view(row['result'])
        for arm in audit.ARMS:
            if row['result']['arms'][arm]['cohorts']['own']['rows'] != original['result']['policies'][arm]['selected_count']:
                raise ValueError('Frozen selection changed')
        rows.append(row)
    expected = set(old)
    actual = {(r['scope'], r['context'], r['source'], r['seed']) for r in rows}
    summary = audit.summarize(rows)
    if actual != expected or summary != json.loads((PUBLIC/'summary.json').read_text()) or complete['summary_sha256'] != run.base.sha(PUBLIC/'summary.json'):
        raise ValueError('Complete aggregate does not replay')
    if (PUBLIC/'conclusions.md').read_text() != report(summary, complete):
        raise ValueError('Report differs from evidence')
    print('Verified144 frozen prediction identities, selections, exact risk accounting and report')


def execute(resume=False):
    reg, _, ancestor, _ = admit()
    _, documents = parent.models(reg)
    _, previous = run.ancestor_eval.models()
    old_local = run.parent.previous_eval.frozen_models()
    old = old_rows()
    if (PUBLIC/'complete.json').exists():
        raise ValueError('Complete diagnostic immutable')
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rows, refs, started = [], [], time.monotonic()
    binding = run.base.sha(REG)

    def beat(**kw):
        row = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = PRIVATE/'heartbeat.tmp'
        tmp.write_text(json.dumps(row)+'\n')
        os.replace(tmp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        if time.monotonic()-started > 43200:
            raise TimeoutError('12h budget reached; preserve receipts and resume')

    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = run.parent.prior.packets(items, ancestor, beat)
            radial = {s: (audit.radius(fit[s][0], fit[s][-1]), audit.radius(held[s][0], fit[s][-1])) for s in fit}
            edges = {s: audit.fitting_bins(radial[s][0], api.previous.prior.eligible(fit[s])) for s in fit}
            for seed in reg['seeds']:
                states = [api.core.read_checkpoint(ROOT/documents[context, seed, a]['checkpoint']['path']) for a in api.ARMS]
                api.assert_matched(states)
                anchor, risk = [api.core.read_checkpoint(ROOT/previous[context, seed, a]['checkpoint']['path']) for a in ('shared_utility', 'shared_risk')]
                for source in sorted(fit):
                    local = api.core.read_checkpoint(ROOT/old_local[context, seed, 'local', source]['checkpoint']['path'])
                    for index, (scope, args) in enumerate((('fit', fit[source]), ('held', held[source]))):
                        path = PUBLIC/'groups'/(scope+'_'+context+'_'+source+'_seed'+str(seed)+'.json')
                        original = old[scope, context, source, seed]
                        if path.exists() and not resume:
                            raise ValueError('Explicit resume required')
                        if path.exists():
                            row = json.loads(path.read_text())
                            if row['registration_sha256'] != binding or row['prediction_hashes'] != original['prediction_hashes']:
                                raise ValueError('Interrupted receipt changed')
                            audit.validate_view(row['result'])
                        else:
                            x, env, y = args[:3]
                            p = dict(local=api.previous.old.predict(local, x, env)[0],
                                anchor=api.previous.predict(anchor, source, x, env),
                                old_risk=api.previous.predict(risk, source, x, env))
                            p.update({s['arm']: api.predict(s, source, x, env) for s in states})
                            hashes = {a: api.previous.old.parent.forest.fingerprint(v) for a, v in p.items()}
                            if hashes != original['prediction_hashes']:
                                raise ValueError('Fresh inference differs from frozen parent')
                            eligible = api.previous.prior.eligible(args)
                            result = audit.view(p, y, env, eligible, radial[source][index], edges[source], args[-1]['scale'])
                            audit.validate_view(result)
                            for a in audit.ARMS:
                                if result['arms'][a]['cohorts']['own']['rows'] != original['result']['policies'][a]['selected_count']:
                                    raise ValueError('Action replay differs from parent')
                            row = dict(registration_sha256=binding, scope=scope, context=context, source=source, seed=seed,
                                prediction_hashes=hashes, result=result, result_source='fresh_run_frozen_policy_diagnostic',
                                preprocessing_source='cached_verified_inner_fit_only', independent_roles_read=False,
                                deployment_changed=False, new_training=False)
                            run.base.once(path, row)
                        rows.append(row)
                        refs.append(dict(path=str(path.relative_to(ROOT)), sha256=run.base.sha(path)))
                        beat(state='view_complete', views=len(rows), total=144)
        summary = audit.summarize(rows)
        run.base.once(PUBLIC/'summary.json', summary)
        complete = dict(registration_sha256=binding, groups=refs, seconds=time.monotonic()-started,
            summary_sha256=run.base.sha(PUBLIC/'summary.json'), pid=os.getpid(),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            new_training=False, independent_roles_read=False, deployment_changed=False)
        run.base.once(PUBLIC/'complete.json', complete)
        content = report(summary, complete)
        (PUBLIC/'conclusions.md').write_text(content)
        beat(state='complete', views=144, seconds=complete['seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'check'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'check':
        verify()
    else:
        execute(args.resume)
