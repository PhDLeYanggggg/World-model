"""Frozen self/peer risk conditioning on original TRAIN inner-fit recordings."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import resource
import subprocess
import time

from scripts import run_m3w_query_context_risk as run
from src.evaluation import m3w_query_conditioning as audit
from src.evaluation.m3w_selected_tail_audit import actions

PUBLIC = run.ROOT/'outputs/publication_readiness_2026_09/european_query_conditioning_v2'
PRIVATE = run.ROOT/'data/stage_cvpr2027_experiments/european_query_conditioning_v2'
REG = PUBLIC/'registration.json'


def register():
    reg, _, ancestor = run.admission()
    files = [__file__, run.ROOT/'src/evaluation/m3w_query_conditioning.py',
             run.ROOT/'tests/test_m3w_query_conditioning.py', PUBLIC/'protocol.md',
             PUBLIC.parent/'european_query_conditioning_v1/additivity_reproduction.json',
             PUBLIC.parent/'european_query_conditioning_v1/interruption.md']
    from pathlib import Path
    run.once(REG, dict(bindings={str(Path(p).resolve().relative_to(run.ROOT)): run.sha(Path(p)) for p in files},
        parent_registration_sha256=run.sha(run.REG),
        parent_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        parent_readout_sha256=run.sha(run.PUBLIC/'complete.json'),
        fit_views=72, gradient_probe_groups=36, batches_per_group=8,
        common_primary_cohort='self_frozen_selection', no_persistent_training=True,
        held_inference=False, independent_roles_read=False, deployment_changed=False))
    return reg, ancestor


def admission():
    reg, ancestor = register(); doc = json.loads(REG.read_text())
    for rel in [*doc['bindings'], str(REG.relative_to(run.ROOT))]:
        p = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=run.ROOT, capture_output=True)
        if p.returncode or p.stdout != (run.ROOT/rel).read_bytes():
            raise ValueError('Commit fixed diagnostic before execution')
    return reg, ancestor


def refs_read(refs, folder):
    rows = []
    for ref in refs:
        p = run.ROOT/ref['path']
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or run.sha(p) != ref['sha256']:
            raise ValueError('Changed diagnostic receipt')
        row = json.loads(p.read_text())
        if row['registration_sha256'] != run.sha(REG) or row['scope'] != 'fit':
            raise ValueError('Changed scientific scope')
        rows.append(row)
    return rows


def check():
    reg, _ = admission(); run.models(reg)
    finish = json.loads((PUBLIC/'complete.json').read_text())
    rows, probes = refs_read(finish['groups'], PUBLIC/'groups'), refs_read(finish['probes'], PUBLIC/'probes')
    if {(r['context'], r['source'], r['seed']) for r in rows} != run.expected(reg):
        raise ValueError('Changed fit roster')
    expected_probes = {(c, s, a) for c in reg['contexts'] for s in reg['seeds'] for a in run.api.ARMS}
    if {(p['context'], p['seed'], p['arm']) for p in probes} != expected_probes:
        raise ValueError('Changed gradient probe roster')
    summary = audit.summarize(rows, probes)
    if (finish['registration_sha256'] != run.sha(REG) or finish['summary_sha256'] != run.sha(PUBLIC/'summary.json')
            or summary != json.loads((PUBLIC/'summary.json').read_text())):
        raise ValueError('Summary replay differs')
    for p in probes:
        if len(p['probes']) != 8:
            raise ValueError('Missing registered probe')
    paired = {(p['context'], p['seed'], p['arm']): p for p in probes}
    for context in reg['contexts']:
        for seed in reg['seeds']:
            left, right = [paired[context, seed, a]['probes'] for a in run.api.ARMS]
            if any(a['draw_hashes'] != b['draw_hashes'] for a, b in zip(left, right)):
                raise ValueError('Self/peer counterfactuals used different samples')
    print('Verified72 fit views,36x8 matched probe batches and576 disposable optimizer steps')


def execute(resume=False):
    reg, ancestor = admission(); _, models = run.models(reg)
    if (PUBLIC/'complete.json').exists():
        raise ValueError('Completed diagnostic immutable')
    run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic(); refs, prefs = [], []
    binding = run.sha(REG)
    def beat(**kw):
        value = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = PRIVATE/'heartbeat.tmp'; tmp.write_text(json.dumps(value)+'\n'); os.replace(tmp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
        if time.monotonic()-started > 43200:
            raise TimeoutError('12h boundary; resume unchanged diagnostic')
    def persist(path, row):
        if path.exists() and not resume: raise ValueError('Explicit resume required')
        run.once(path, row)
        return dict(path=str(path.relative_to(run.ROOT)), sha256=run.sha(path))
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, discarded_held = run.parent.run.parent.prior.packets(items, ancestor, beat)
            del discarded_held
            for seed in reg['seeds']:
                states = {a: run.api.core.read_checkpoint(run.ROOT/models[context, seed, a]['checkpoint']['path']) for a in run.api.ARMS}
                run.api.assert_matched(states['self'], states['peer']); common = {}
                batches = audit.prepare_probes(fit, states['self'], count=8)
                for source, args in sorted(fit.items()):
                    x, env, y = args[:3]
                    for name, a in zip(('x', 'envelope', 'primary', 'temporal', 'sites', 'recordings', 'frames'), args[:-1]):
                        if run.api.fingerprint(a) != states['self']['input_hashes'][source][name]:
                            raise ValueError('Fitting row alignment changed')
                    values, stats, hashes = {}, {}, {}
                    for arm in run.api.ARMS:
                        values[arm], stats[arm] = audit.collect(states[arm], source, x, env, args[5], args[6])
                        predicted = run.api.predict(states[arm], source, x, env, args[5], args[6])
                        run.api.np.testing.assert_array_equal(values[arm]['prediction'], predicted)
                        hashes[arm] = run.api.fingerprint(predicted)
                    run.api.np.testing.assert_array_equal(values['self']['anchor'], values['peer']['anchor'])
                    eligible = run.api.parent.previous.prior.eligible(args)
                    cohorts = dict(eligible=eligible, anchor=actions(values['self']['anchor'], eligible),
                                   **{a: actions(v['prediction'], eligible) for a, v in values.items()})
                    common[source] = cohorts['self']
                    result = {a: {c: audit.account(v, y, mask, args[-1]['scale']) for c, mask in cohorts.items()}
                              for a, v in values.items()}
                    row = dict(context=context, source=source, seed=seed, scope='fit', result=result,
                        prediction_hashes=hashes, context_stats=stats, registration_sha256=binding,
                        common_mask_sha256=run.api.fingerprint(cohorts['self']),
                        result_source='fresh_run_frozen_inner_fit_autograd', held_inference=False,
                        independent_roles_read=False, deployment_changed=False)
                    refs.append(persist(PUBLIC/'groups'/f'{context}_{source}_seed{seed}.json', row))
                    beat(state='fit_view_complete', views=len(refs), total=72)
                for arm in run.api.ARMS:
                    out = []
                    for i, batch in enumerate(batches):
                        masks = {s: run.api.torch.from_numpy(common[s][d['row_ids']]) for s, d in batch.items()}
                        out.append(audit.gradient_probe(states[arm], batch, masks))
                        beat(state='gradient_probe', context=context, seed=seed, arm=arm, batch=i+1, total=8)
                    row = dict(context=context, seed=seed, arm=arm, scope='fit', registration_sha256=binding,
                        checkpoint_sha256=models[context, seed, arm]['checkpoint']['sha256'], probes=out,
                        result_source='fresh_run_frozen_fit_autograd_and_disposable_counterfactuals',
                        no_persistent_training=True, held_inference=False, independent_roles_read=False)
                    prefs.append(persist(PUBLIC/'probes'/f'{context}_head{seed}_{arm}.json', row))
                    if run.sha(run.ROOT/models[context, seed, arm]['checkpoint']['path']) != row['checkpoint_sha256']:
                        raise ValueError('Frozen checkpoint changed')
        summary = audit.summarize(refs_read(refs, PUBLIC/'groups'), refs_read(prefs, PUBLIC/'probes'))
        run.once(PUBLIC/'summary.json', summary)
        run.once(PUBLIC/'complete.json', dict(groups=refs, probes=prefs, registration_sha256=binding,
            summary_sha256=run.sha(PUBLIC/'summary.json'), pid=os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            no_persistent_training=True, held_inference=False, independent_roles_read=False))
        beat(state='complete', views=72, probe_groups=36)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('register', 'run', 'check'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if a.action == 'register': register()
    elif a.action == 'check': check()
    else: execute(a.resume)
