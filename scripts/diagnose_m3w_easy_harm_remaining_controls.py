"""TRAIN-only direct-replay sweep; does not grant an acceptance exception."""
import argparse
import json
import os
from pathlib import Path
import platform

from scripts import train_m3w_easy_harm_deviance as run
from scripts.diagnose_m3w_easy_harm_control import compare

NAME = 'control_diagnostic_v2'


def remaining_refs(manifest, observation):
    pending = set(observation['missing'])
    expected = {run.key(r, a) for r in manifest['heads'] for a in ('quadratic', 'easy_deviance')}
    if len(pending) != len(observation['missing']) or not pending <= expected:
        raise ValueError('Unique registered missing identities required')
    refs = []
    for row in manifest['heads']:
        arms = [run.key(row, a) in pending for a in ('quadratic', 'easy_deviance')]
        if any(arms) and not all(arms):
            raise ValueError('This diagnostic requires wholly unaccepted pairs')
        if all(arms):
            refs.append(row)
    accepted = {Path(r['path']).stem for r in observation['fits']}
    if accepted & pending or accepted | pending != expected:
        raise ValueError('Complete observed accepted/missing partition required')
    return refs


def summary(rows):
    return dict(
        identities=len(rows),
        original_new_exact=sum(r['comparisons']['old_direct_vs_new_direct']['all_control_fields_exact'] for r in rows),
        original_historical_exact=sum(r['comparisons']['old_direct_vs_historical']['all_control_fields_exact'] for r in rows),
        new_historical_exact=sum(r['comparisons']['new_direct_vs_historical']['all_control_fields_exact'] for r in rows),
        metadata_failures=[r['name'] for r in rows if any(
            not c[k]['exact'] for c in r['comparisons'].values()
            for k in ('initial_model','preprocess','settings','input_hashes','seed','step',
                      'sampler_rng','torch_rng','draw_hash','row_draws','queries'))],
        acceptance_changed=False, scientific_lift_established=False,
        validation_scored=False, independent_roles_read=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute required before numerical imports')
    root, source, cfg, reg, manifest = run.verify(args.root)
    binding = json.loads((root/(NAME+'_registration.json')).read_text())
    for rel, digest in binding['code_bindings'].items():
        assert run.sha(root/'code'/rel) == digest
    assert run.sha(root/'registration.json') == binding['training_registration_sha256']
    observation = root/run.PUBLIC/'partial_training_observation_20261006T1256Z.json'
    assert run.sha(observation) == binding['observation_sha256']
    refs = remaining_refs(manifest, json.loads(observation.read_text()))
    assert [run.key(r, 'quadratic') for r in refs] == binding['identities']
    from src.world_model import m3w_easy_harm_deviance_training as api
    from src.world_model.m3w_preprocess_portability import frozen_preprocess
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    home = root/run.PRIVATE/NAME
    public = root/run.PUBLIC/NAME
    home.mkdir(parents=True, exist_ok=True)
    old_docs = {}
    for ref in json.loads((source/run.OLD_PUBLIC/'training_freeze.json').read_text())['fits']:
        path = source/ref['path']; assert run.sha(path) == ref['sha256']
        doc = json.loads(path.read_text())
        if doc['arm'] == 'none':
            old_docs[doc['identity']['group'], doc['identity']['seed']] = doc
    rows = []
    for ref in refs:
        name = run.key(ref, 'quadratic'); output = public/(name+'.json')
        if output.exists():
            row = json.loads(output.read_text())
            assert row['registration_sha256'] == run.sha(root/(NAME+'_registration.json'))
            for cp in row['checkpoints'].values():
                assert run.sha(root/cp['path']) == cp['sha256']
            rows.append(row); continue
        data = run.packet(source, ref)
        def heartbeat(**value):
            print(json.dumps(dict(identity=name, **value)), flush=True)
        kw = dict(settings=cfg['head_training'], seed=ref['identity']['seed'],
                  identity=ref['identity'], heartbeat=heartbeat,
                  checkpoint_guard=lambda: run.quota(root, source, cfg))
        paths = {k: home/name/(k+'.pt.gz') for k in ('old_direct', 'new_direct')}
        states = {}
        with frozen_preprocess(api.core, data[-1]):
            path = paths['old_direct']
            states['old_direct'] = api.parent.fit(*data, arm='none', path=path,
                                                resume=path.exists(), **kw)
            path = paths['new_direct']
            states['new_direct'] = api.fit(*data, arm='quadratic', path=path,
                experiment_sha256=run.sha(root/'registration.json'), resume=path.exists(), **kw)
        old = old_docs[ref['group'], ref['identity']['seed']]
        cp = source/old['checkpoint']['path']; assert run.sha(cp) == old['checkpoint']['sha256']
        states['historical'] = api.core.read_checkpoint(cp)
        pairs = [('old_direct','new_direct'), ('old_direct','historical'), ('new_direct','historical')]
        failed = root/run.PRIVATE/'heads'/name/'checkpoint.pt.gz'
        if failed.exists():
            states['saved_new'] = api.core.read_checkpoint(failed)
            assert states['saved_new']['identity'] == ref['identity']
            pairs += [('old_direct','saved_new'), ('new_direct','saved_new')]
        row = dict(name=name, identity=ref['identity'], job_id=os.environ['SLURM_JOB_ID'],
            registration_sha256=run.sha(root/(NAME+'_registration.json')),
            result_source='fresh_run_TRAIN_only_direct_replay_diagnostic',
            comparisons={a+'_vs_'+b: compare(states[a], states[b], api.core.exact) for a,b in pairs},
            checkpoints={k: dict(path=str(path.relative_to(root)), sha256=run.sha(path), bytes=path.stat().st_size)
                         for k,path in paths.items()},
            historical_checkpoint=old['checkpoint'],
            saved_new_checkpoint_sha256=run.sha(failed) if failed.exists() else None,
            verification_updates=4000, validation_scored=False, independent_roles_read=False,
            acceptance_changed=False)
        run.once(output, row); rows.append(row)
        heartbeat(state='identity_diagnostic_complete', completed=len(rows),
                  exact={k:v['all_control_fields_exact'] for k,v in row['comparisons'].items()})
    out = dict(summary=summary(rows),
        results=[dict(path=str((public/(r['name']+'.json')).relative_to(root)),
                      sha256=run.sha(public/(r['name']+'.json'))) for r in rows],
        job_id=os.environ['SLURM_JOB_ID'], node=platform.node(), torch=api.torch.__version__,
        verification_updates=4000*len(rows), cpu_threads=4, workers=0,
        registration_sha256=run.sha(root/(NAME+'_registration.json')))
    run.once(public/'complete.json', out)
    print(json.dumps(out['summary']), flush=True)


if __name__ == '__main__':
    main()
