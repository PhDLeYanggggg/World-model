"""Diagnose frozen harm-head derivatives on fitting rows, never new holdouts."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from scripts import run_m3w_full_range_harm as parent
from src.evaluation import m3w_harm_jacobian_audit as audit
from src.evaluation.m3w_selected_tail_audit import actions

ROOT, run = parent.ROOT, parent.run
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_harm_jacobian_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_harm_jacobian_v1'
REG = PUBLIC/'registration.json'


def register():
    reg, cfg, ancestor = parent.admission()
    files = [Path(__file__).resolve(), ROOT/'src/evaluation/m3w_harm_jacobian_audit.py',
        ROOT/'tests/test_m3w_harm_jacobian_audit.py', PUBLIC/'protocol.md']
    doc = dict(bindings={str(p.relative_to(ROOT)): run.base.sha(p) for p in files},
        parent_registration_sha256=run.base.sha(parent.REG),
        parent_freeze_sha256=run.base.sha(parent.PUBLIC/'training_freeze.json'),
        parent_readout_sha256=run.base.sha(parent.PUBLIC/'readout/complete.json'),
        expected_views=72, scope='fit', arms=list(audit.ARMS), cohorts=list(audit.COHORTS),
        independent_roles_read=False, held_inference=False, new_training=False, deployment_changed=False)
    parent.once(REG, doc)
    return reg, cfg, ancestor


def admit():
    reg, cfg, ancestor = register()
    doc = json.loads(REG.read_text())
    for rel in [*doc['bindings'], str(REG.relative_to(ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=ROOT, capture_output=True)
        if p.returncode or p.stdout != (ROOT/rel).read_bytes():
            raise ValueError('Commit fixed diagnostic before computation')
    return reg, cfg, ancestor


def originals():
    complete = json.loads((parent.PUBLIC/'readout/complete.json').read_text())
    result = {}
    for ref in complete['groups']:
        p = ROOT/ref['path']
        if p.is_symlink() or not p.resolve().is_relative_to((parent.PUBLIC/'readout/groups').resolve()) or run.base.sha(p) != ref['sha256']:
            raise ValueError('Changed parent receipt')
        row = json.loads(p.read_text())
        if row['scope'] == 'fit':
            key = row['context'], row['source'], row['seed']
            if key in result:
                raise ValueError('Duplicate fitting identity')
            result[key] = row
    if len(result) != 72:
        raise ValueError('Complete frozen fit roster required')
    return result


def report(summary, complete):
    lines = ['# Frozen Harm-Head Gradient Diagnosis', '',
        f"Fresh autograd and inference on72/72 fit views completed in{complete['seconds']:.2f}s.",
        'Both frozen predictors reproduce their original prediction hashes. No new',
        'fit, holdout inference, threshold, deployment or independent evidence.',
        'CREATE TRAIN files include internal held rows during transport; they were',
        'discarded before forwarding. Cached checkpoints and preprocessing were verified.', '',
        '## Same-Cohort Results', '',
        '| Fixed cohort | Arm | Rows | Unknown | Easy zero-J rows | Easy underprediction | At zero J | At hard H lower clip | At hard q lower clip |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort in audit.COHORTS:
        for arm in audit.ARMS:
            r = summary[arm][cohort]; e = r['easy']
            lines.append('| '+cohort+' | '+arm+' | '+str(r['rows'])+' | '+str(r['unknown'])+' | '+
                str(e['zero_jacobian_rows'])+' | '+' | '.join(f'{e[k]:.8f}' for k in
                ('positive_underprediction', 'underprediction_zero_jacobian',
                 'underprediction_hard_h_lower', 'underprediction_hard_q_lower'))+' |')
    lines += ['', 'J is the actual autograd Jacobian of predicted easy harm with respect to',
        'both risk logits. H is all-harm; q is the easy/all-harm ratio. Clip columns',
        'can overlap and must not be summed. Values use frozen fitting source scales;',
        'repeated occurrences are not independent samples, ADE/FDE or probabilities.', '',
        '## Interpretation Boundary', '',
        'These are local derivatives at final weights. Zero J prevents direct output',
        'correction for that input at that instant; shared weights can still move from',
        'other rows. This does not prove irreversibility or a sole optimization cause.',
        'Nonzero J does not imply useful parameter gradients or convergence. A realized',
        'harm above a predicted cap does not establish an impossible conditional mean.',
        'Original support and independent-evidence gates remain failed.', '',
        'Obs8/pred12, raw stride12; detector labels, pixel/dataset-local coordinates.',
        'No metric/seconds/physical-safety/human-gold/true3D/foundation claim.',
        'Stage5C and SMC disabled.', '',
        '## Reproduction', '', '```bash',
        '.venv-pytorch/bin/python -m scripts.audit_m3w_harm_jacobians check', '```', '',
        'Check replays immutable receipts and accounting, not another training run.', '']
    return '\n'.join(lines)


def verify():
    reg, _, _ = admit()
    parent.models(reg); parent.prior.parent.models(reg)
    old = originals()
    complete = json.loads((PUBLIC/'complete.json').read_text())
    if complete['registration_sha256'] != run.base.sha(REG):
        raise ValueError('Changed diagnostic registration')
    rows = []
    for ref in complete['groups']:
        p = ROOT/ref['path']
        if p.is_symlink() or not p.resolve().is_relative_to((PUBLIC/'groups').resolve()) or run.base.sha(p) != ref['sha256']:
            raise ValueError('Changed diagnostic receipt')
        r = json.loads(p.read_text()); key = r['context'], r['source'], r['seed']
        if r['registration_sha256'] != run.base.sha(REG) or r['scope'] != 'fit':
            raise ValueError('Changed scope or registration')
        for arm, original_name in (('full_range', 'full_range'), ('isolated_odds', 'isolated_risk')):
            if r['prediction_hashes'][arm] != old[key]['prediction_hashes'][original_name]:
                raise ValueError('Frozen inference differs')
            for cohort, policy in (('full_range', 'full_range'), ('isolated_odds', 'isolated_risk'), ('anchor', 'anchor')):
                if r['result'][arm][cohort]['rows'] != old[key]['result']['policies'][policy]['selected_count']:
                    raise ValueError('Frozen cohort differs')
        rows.append(r)
    if {(r['context'], r['source'], r['seed']) for r in rows} != set(old):
        raise ValueError('Diagnostic identity mismatch')
    summary = audit.summarize(rows)
    if complete['summary_sha256'] != run.base.sha(PUBLIC/'summary.json') or summary != json.loads((PUBLIC/'summary.json').read_text()):
        raise ValueError('Aggregate replay differs')
    if (PUBLIC/'conclusions.md').read_text() != report(summary, complete):
        raise ValueError('Report differs from evidence')
    print('Verified72 fit-only Jacobian receipts, frozen predictions, fixed cohorts and report')


def execute(resume=False):
    reg, _, ancestor = admit()
    _, new = parent.models(reg)
    _, old_models = parent.prior.parent.models(reg)
    original = originals()
    if (PUBLIC/'complete.json').exists():
        raise ValueError('Complete diagnostic immutable')
    parent.api.torch.set_num_threads(4); parent.api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rows, refs, started = [], [], time.monotonic()
    binding = run.base.sha(REG)

    def beat(**kw):
        d = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = PRIVATE/'heartbeat.tmp'; tmp.write_text(json.dumps(d)+'\n')
        os.replace(tmp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(d)+'\n')
        print(json.dumps(d), flush=True)
        if time.monotonic()-started > 43200:
            raise TimeoutError('12h budget; resume without changing scope')

    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, discarded_held = run.parent.prior.packets(items, ancestor, beat)
            del discarded_held
            for seed in reg['seeds']:
                states = dict(full_range=parent.api.core.read_checkpoint(ROOT/new[context, seed]['checkpoint']['path']),
                    isolated_odds=parent.api.core.read_checkpoint(ROOT/old_models[context, seed, 'isolated_risk']['checkpoint']['path']))
                parent.api.assert_control_match(states['full_range'], states['isolated_odds'])
                for source, args in sorted(fit.items()):
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}.json'
                    prior = original[context, source, seed]
                    if path.exists() and not resume:
                        raise ValueError('Explicit resume required')
                    if path.exists():
                        row = json.loads(path.read_text())
                        if row['registration_sha256'] != binding:
                            raise ValueError('Changed interrupted registration')
                    else:
                        x, env, y = args[:3]
                        fingerprint = parent.api.previous.old.parent.forest.fingerprint
                        for name, value in zip(('x', 'envelope', 'primary', 'temporal', 'sites', 'recordings', 'frames'), args[:-1]):
                            if fingerprint(value) != states['full_range']['input_hashes'][source][name]:
                                raise ValueError('Fitting packet row alignment differs')
                        values = {arm: audit.collect(impl, states[arm], source, x, env) for arm, impl in
                            (('full_range', parent.api), ('isolated_odds', parent.prior.api))}
                        hashes = {arm: fingerprint(v['prediction']) for arm, v in values.items()}
                        for arm, old_name in (('full_range', 'full_range'), ('isolated_odds', 'isolated_risk')):
                            if hashes[arm] != prior['prediction_hashes'][old_name]:
                                raise ValueError('Autograd forward differs from frozen original')
                        parent.api.np.testing.assert_array_equal(values['full_range']['anchor'], values['isolated_odds']['anchor'])
                        eligible = parent.api.previous.prior.eligible(args)
                        cohorts = dict(eligible=eligible, anchor=actions(values['full_range']['anchor'], eligible),
                            **{a: actions(v['prediction'], eligible) for a, v in values.items()})
                        result = {a: {c: audit.account(v, y, mask, args[-1]['scale'], a)
                            for c, mask in cohorts.items()} for a, v in values.items()}
                        row = dict(registration_sha256=binding, scope='fit', context=context, source=source,
                            seed=seed, prediction_hashes=hashes, result=result,
                            result_source='fresh_run_frozen_fit_autograd', held_inference=False,
                            preprocessing_source='cached_verified_inner_fit_only', new_training=False,
                            independent_roles_read=False, deployment_changed=False)
                        parent.once(path, row)
                    rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=run.base.sha(path)))
                    beat(state='fit_view_complete', views=len(rows), total=72)
        summary = audit.summarize(rows)
        parent.once(PUBLIC/'summary.json', summary)
        complete = dict(registration_sha256=binding, groups=refs, pid=os.getpid(),
            seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            summary_sha256=run.base.sha(PUBLIC/'summary.json'), new_training=False,
            independent_roles_read=False, held_inference=False, deployment_changed=False)
        parent.once(PUBLIC/'complete.json', complete)
        (PUBLIC/'conclusions.md').write_text(report(summary, complete))
        beat(state='complete', views=72, seconds=complete['seconds'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('register', 'run', 'check'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'check':
        verify()
    else:
        execute(args.resume)
