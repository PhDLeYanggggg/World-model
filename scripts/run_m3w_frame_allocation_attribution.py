"""Frozen-query neural-use and score attribution; never a new policy search."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_frame_action_allocation as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_query_selected_attribution as api

NAME = 'european_frame_allocation_attribution_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
REG = PUBLIC / 'registration.json'
sha, once, write = parent.sha, parent.once, parent.write
FILES = (Path(__file__), ROOT / 'src/evaluation/m3w_query_selected_attribution.py',
    ROOT / 'src/evaluation/m3w_selected_action_attribution.py',
    ROOT / 'tests/test_m3w_query_selected_attribution.py', PUBLIC / 'protocol.md')


def admission(register=False):
    assets = parent.admission()
    complete = json.loads((parent.PUBLIC / 'complete.json').read_text())
    if complete['summary_sha256'] != sha(parent.PUBLIC / 'summary.json'):
        raise ValueError('Frozen complete main readout required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_complete_sha256=sha(parent.PUBLIC / 'complete.json'),
        parent_summary_sha256=sha(parent.PUBLIC / 'summary.json'), groups=72,
        post_main_readout_diagnostic=True, original_choices_unchanged=True, new_training_updates=0,
        primary='joint_matched_actual_neural_use_and_incremental_neural_use',
        bootstrap_resamples=3000, source_fitting_only=True, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Registered query attribution changed: '+rel)
            parent.BASE.engine.committed(ROOT / rel)
        parent.BASE.engine.committed(REG)
        verification = json.loads((parent.PUBLIC / 'verification.json').read_text())
        if not verification['all72groups_exact'] or verification['summary_sha256'] != sha(parent.PUBLIC / 'summary.json'):
            raise ValueError('Complete separate main replay must finish first')
    return assets


def summarize(rows, cfg):
    expected = {(c, s, seed) for c, ss in json.loads(parent.REG.read_text())['contexts'].items()
        for s in ss for seed in cfg['seeds']}
    if len(rows) != 72 or {(r['context'], r['source'], r['seed']) for r in rows} != expected:
        raise ValueError('Complete72 frozen-choice groups required')
    interval = lambda getter: parent.evaluation.bank.legacy.interval(rows, getter, cfg)
    effects = {a: {mode+'_'+bound: interval(lambda r, a=a, mode=mode, bound=bound:
        r['arms'][a]['actual_neural_use'][mode][bound]) for mode in ('full', 'matched')
        for bound in ('known_pp', 'lower_pp', 'upper_pp')} for a in parent.api.ARMS}
    attribution = {}
    for a in parent.api.ARMS:
        values = [r['result']['arms'][a] for r in rows]
        channels = {}
        for channel in ('all', 'easy'):
            v = [r['channels'][channel] for r in values]
            ratios = [record['channels'][channel]['predicted_harm_mass']/record['channels'][channel]['observed_harm_mass']
                for r in values for record in r['records'].values() if record['channels'][channel]['observed_harm_mass'] > 0]
            channels[channel] = dict(observed_positive_excess_row_views=sum(r['observed_positive_excess_rows'] for r in v),
                witness_counts={k: sum(r['witness_counts'][k] for r in v) for k in api.old.WITNESSES},
                observed_harm_mass_bookkeeping=sum(r['observed_harm_mass'] for r in v),
                predicted_harm_mass_bookkeeping=sum(r['predicted_harm_mass'] for r in v),
                within_record_predicted_over_observed_harm_quantiles=np.quantile(ratios, [0, .1, .5, .9, 1]).tolist() if ratios else None,
                fully_known_queries_with_observed_excess=sum(r['query_channels'][channel]['fully_known_selected_queries_with_observed_excess'] for r in values),
                fully_known_query_witness_counts={k: sum(r['query_channels'][channel]['witness_counts'][k] for r in values) for k in api.old.WITNESSES})
        attribution[a] = dict(channels=channels,
            fully_known_selected_query_views=sum(r['fully_known_selected_queries'] for r in values),
            selected_unknown_query_views=sum(r['selected_unknown_queries'] for r in values),
            empty_query_views=sum(r['empty_queries'] for r in values))
    return dict(groups=72, result_source='fresh_causal_inference_and_post_readout_frozen_choice_diagnostics',
        all72legacy_independent_attributions_exact=all(r['legacy_attribution_exact'] for r in rows),
        all288frozen_action_hashes_exact=all(r['all4action_hashes_exact'] for r in rows),
        actual_neural_use=effects,
        joint_minus_independent_incremental_neural_use={bound: interval(lambda r, bound=bound:
            r['incremental_neural_use'][bound]) for bound in ('known_pp', 'lower_pp', 'upper_pp')},
        neural_deleted_joint_minus_independent={bound: interval(lambda r, bound=bound:
            r['neural_deleted_joint_minus_independent'][bound]) for bound in ('known_pp', 'lower_pp', 'upper_pp')},
        attribution=attribution, raw_masses_bookkeeping_not_cross_domain_metric=True,
        row_excess_not_necessarily_a_query_violation=True, unknowns_not_imputed=True,
        post_main_readout_diagnostic=True, new_training_updates=0, new_policy_constructed=False,
        independent_confirmation=False, causal_identification=False, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def execute(verify=False, resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC / 'complete.json').exists() and not verify:
        raise ValueError('Completed diagnosis immutable; use verify')
    parent.BASE.torch.set_num_threads(4); parent.BASE.torch.set_num_interop_threads(1)
    start = time.monotonic()
    def beat(**event):
        if time.monotonic()-start > 43200 or shutil.disk_usage(ROOT).free < 10*2**30:
            raise RuntimeError('12h/10GiB guard; resume the same diagnostic')
        v = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE / 'heartbeat.json', v)
        with (PRIVATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(v)+'\n')
        print(json.dumps(v), flush=True)
    prior = aggregate_groups(NAME) if verify else {}
    main, legacy = aggregate_groups(parent.NAME), aggregate_groups(parent.source.parent.NAME)
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, _ = parent.BASE.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            state = parent.BASE.api.completion.base.read_checkpoint(ROOT / models[context, seed, 'admitted_score']['checkpoint']['path'])
            for name, packet in sorted(packets.items()):
                key = f'{context}_{name}_seed{seed}_fit.json'; path = PUBLIC / 'groups' / key
                if path.exists() and not verify:
                    if not resume:
                        raise ValueError('Explicit partial diagnostic resume required')
                    row = json.loads(path.read_text())
                    if row['registration_sha256'] != sha(REG):
                        raise ValueError('Partial diagnostic registration changed')
                else:
                    p, independent, _ = parent.infer(packet, state, store, legacy[key])
                    if parent.BASE.engine.fingerprint(p) != main[key]['prediction_sha256']:
                        raise ValueError('Frozen main prediction differs')
                    cp = parent.BASE.api.completion.base.read_checkpoint(parent.PRIVATE / 'progress' / (key+'.pt.gz'))
                    groups = parent.api.query_groups(packet['recordings'], packet['frames'], packet['ids'])
                    if cp['completed_queries'] != len(groups):
                        raise ValueError('Complete original action checkpoint required')
                    choices = cp['choices']
                    for a in parent.api.ARMS:
                        if parent.BASE.engine.fingerprint(choices[a]) != main[key]['action_sha256'][a]:
                            raise ValueError('Frozen main action hash differs')
                    np.testing.assert_array_equal(choices['independent'], independent)
                    if api.old.analyze(packet, p, independent, easy_cut=job['design']['easy_cut']) != legacy[key]['result']['attribution']['admitted_score']:
                        raise ValueError('Original full independent attribution differs')
                    arms = {a: api.analyze(packet, p, c, easy_cut=job['design']['easy_cut']) for a, c in choices.items()}
                    deleted = {a: np.where(c == 6, -1, c) for a, c in choices.items()}
                    result = dict(arms=arms, incremental_neural_use=api.incremental_neural_use(packet,
                        independent, choices['joint_matched']),
                        neural_deleted_joint_minus_independent=parent.evaluation.bank.cross_bank_bounds(packet,
                            deleted['independent'], packet, deleted['joint_matched']))
                    row = dict(context=context, source=name, seed=seed, result=result,
                        legacy_attribution_exact=True, all4action_hashes_exact=True,
                        prediction_sha256=main[key]['prediction_sha256'], action_sha256=main[key]['action_sha256'],
                        ids_sha256=main[key]['ids_sha256'], registration_sha256=sha(REG),
                        result_source='fresh_inference_and_frozen_choice_post_readout_diagnostic', held_scored=False)
                    if verify:
                        if row != prior[key]:
                            raise ValueError('Complete separate diagnostic replay differs')
                    else:
                        once(path, row)
                rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='diagnostic_group_complete', groups=len(rows), total=72, context=context, source=name, seed=seed)
        del packets, state
    summary = summarize(rows, cfg)
    if verify:
        if summary != json.loads((PUBLIC / 'summary.json').read_text()):
            raise ValueError('Diagnostic summary replay differs')
        once(PUBLIC / 'verification.json', dict(all72groups_exact=True,
            all288action_hashes_exact=True, summary_sha256=sha(PUBLIC / 'summary.json'),
            complete_sha256=sha(PUBLIC / 'complete.json')))
    else:
        once(PUBLIC / 'summary.json', summary)
        once(PUBLIC / 'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC / 'summary.json'), new_training_updates=0,
            independent_roles_opened=False, deployment_changed=False))
    write(PRIVATE / ('verification_runtime.json' if verify else 'runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72diagnostics_verified' if verify else 'all72diagnostics_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(register=True) if a.phase == 'register' else execute(a.phase == 'verify', a.resume)
