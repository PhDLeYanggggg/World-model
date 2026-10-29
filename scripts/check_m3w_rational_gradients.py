"""Verify the targeted gradient obstruction after fitting, without held inference."""
import argparse
import fcntl
import json
import subprocess

from scripts import run_m3w_rational_harm as run
from src.evaluation import m3w_harm_jacobian_audit as audit
from src.evaluation.m3w_selected_tail_audit import actions

PUBLIC = run.PUBLIC/'gradient_recheck'
REG = PUBLIC/'registration.json'
ARMS = ('full_range', 'rational')
COHORTS = ('eligible', 'anchor', 'full_range', 'rational')


def registration():
    reg, _, ancestor = run.admission()
    files = ['scripts/check_m3w_rational_gradients.py', 'tests/test_m3w_rational_gradient_recheck.py',
             str((PUBLIC/'protocol.md').relative_to(run.ROOT))]
    doc = dict(bindings={p: run.run.base.sha(run.ROOT/p) for p in files},
        repair_registration_sha256=run.run.base.sha(run.REG),
        repair_freeze_sha256=run.run.base.sha(run.PUBLIC/'training_freeze.json'),
        scope='fit_only', new_training=False, held_inference=False, independent_roles_read=False)
    run.once(REG, doc)
    return reg, ancestor, doc


def admit():
    reg, ancestor, doc = registration()
    for rel in [*doc['bindings'], str(REG.relative_to(run.ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=run.ROOT, capture_output=True)
        if p.returncode or p.stdout != (run.ROOT/rel).read_bytes():
            raise ValueError('Commit gradient verification before executing')
    return reg, ancestor


def compact(values, y, cohort, scale):
    # Only shared Jacobian/error fields are used, not either mapping's clip flags.
    all_fields = audit.account(values, y, cohort, scale, 'isolated_odds')
    out = {k: all_fields[k] for k in ('rows', 'known', 'unknown')}
    for target in ('all', 'easy'):
        out[target] = {k: v for k, v in all_fields[target].items() if not k.startswith('underprediction_hard_')}
    audit.validate(out)
    return out


def aggregate(rows):
    if len(rows) != 72 or len({(r['context'], r['source'], r['seed']) for r in rows}) != 72:
        raise ValueError('Complete unique72 fit views required')
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for cohort in COHORTS:
            rr = [r['result'][arm][cohort] for r in rows]
            for r in rr:
                audit.validate(r)
            value = {k: sum(r[k] for r in rr) for k in ('rows', 'known', 'unknown')}
            for target in ('all', 'easy'):
                value[target] = {k: sum(r[target][k] for r in rr) for k in rr[0][target]}
            audit.validate(value)
            out[arm][cohort] = value
    return out


def main(action, resume=False):
    if action == 'register':
        registration(); return
    reg, ancestor = admit()
    _, new = run.models(reg)
    _, old = run.range_run.models(reg)
    parents = run.range_rows()
    binding = run.run.base.sha(REG)
    if action == 'check':
        finish = json.loads((PUBLIC/'complete.json').read_text())
        rows = []
        for ref in finish['groups']:
            p = run.ROOT/ref['path']
            if p.is_symlink() or not p.resolve().is_relative_to((PUBLIC/'groups').resolve()) or run.run.base.sha(p) != ref['sha256']:
                raise ValueError('Changed gradient receipt')
            row = json.loads(p.read_text())
            if row['registration_sha256'] != binding or row['scope'] != 'fit':
                raise ValueError('Changed diagnostic binding')
            parent = parents['fit', row['context'], row['source'], row['seed']]
            if row['prediction_hashes']['full_range'] != parent['prediction_hashes']['full_range']:
                raise ValueError('Frozen comparator inference differs')
            for arm in ARMS:
                if row['result'][arm]['full_range']['rows'] != parent['result']['policies']['full_range']['selected_count']:
                    raise ValueError('Frozen comparator cohort differs')
            rows.append(row)
        expected = {(c, s, seed) for scope, c, s, seed in parents if scope == 'fit'}
        if {(r['context'], r['source'], r['seed']) for r in rows} != expected:
            raise ValueError('Missing fitting identity')
        if finish['registration_sha256'] != binding or finish['summary_sha256'] != run.run.base.sha(PUBLIC/'summary.json') or aggregate(rows) != json.loads((PUBLIC/'summary.json').read_text()):
            raise ValueError('Gradient aggregate differs')
        print('Verified72 post-repair fitting-only Jacobian views'); return
    if (PUBLIC/'complete.json').exists():
        raise ValueError('Complete verification immutable')
    started, beat = run.runtime()
    rows, refs = [], []
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = run.run.parent.prior.packets(items, ancestor, beat)
            del held
            for seed in reg['seeds']:
                states = {a: run.api.core.read_checkpoint(run.ROOT/d[context, seed]['checkpoint']['path'])
                          for a, d in (('full_range', old), ('rational', new))}
                for source, args in sorted(fit.items()):
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}.json'
                    if path.exists() and not resume:
                        raise ValueError('Explicit resume required')
                    if path.exists():
                        row = json.loads(path.read_text())
                        if row['registration_sha256'] != binding:
                            raise ValueError('Changed interrupted binding')
                    else:
                        x, env, y = args[:3]
                        values = {a: audit.collect(api, states[a], source, x, env)
                            for a, api in (('full_range', run.range_run.api), ('rational', run.api))}
                        fp = run.api.previous.old.parent.forest.fingerprint
                        hashes = {a: fp(v['prediction']) for a, v in values.items()}
                        if hashes['full_range'] != parents['fit', context, source, seed]['prediction_hashes']['full_range']:
                            raise ValueError('Full-range prediction identity changed')
                        if hashes['rational'] != fp(run.api.predict(states['rational'], source, x, env)):
                            raise ValueError('Autograd and ordinary forward differ')
                        eligible = run.api.previous.prior.eligible(args)
                        cohorts = dict(eligible=eligible, anchor=actions(values['rational']['anchor'], eligible),
                            **{a: actions(v['prediction'], eligible) for a, v in values.items()})
                        result = {a: {c: compact(v, y, mask, args[-1]['scale']) for c, mask in cohorts.items()}
                                  for a, v in values.items()}
                        row = dict(registration_sha256=binding, scope='fit', context=context, seed=seed,
                            source=source, prediction_hashes=hashes, result=result,
                            result_source='fresh_run_post_repair_fit_autograd', held_inference=False,
                            independent_roles_read=False, deployment_changed=False)
                        run.once(path, row)
                    rows.append(row); refs.append(dict(path=str(path.relative_to(run.ROOT)), sha256=run.run.base.sha(path)))
                    beat(state='post_repair_gradient_view', views=len(rows), total=72)
    run.once(PUBLIC/'summary.json', aggregate(rows))
    run.once(PUBLIC/'complete.json', dict(registration_sha256=binding, groups=refs,
        summary_sha256=run.run.base.sha(PUBLIC/'summary.json'), seconds=run.time.monotonic()-started,
        pid=run.os.getpid(), held_inference=False, independent_roles_read=False))
    beat(state='gradient_recheck_complete', views=72)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('register', 'run', 'check'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    main(args.action, args.resume)
