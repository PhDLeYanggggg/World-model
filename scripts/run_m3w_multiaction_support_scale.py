"""Single-factor support-unit conditioning; original scientific assets stay frozen."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import resource

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_multiaction_record_policy as run
from src.world_model.m3w_record_support_units import installed
from src.evaluation.m3w_multiaction_record_inference import predict

SOURCE_PUBLIC = run.PUBLIC
NAME = 'european_multiaction_support_scale_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_multiaction_support_scale_v1.json'


def configure():
    original_cfg = json.loads((ROOT/'configs/m3w_multiaction_record_policy_v1.json').read_text())
    cfg = json.loads(CONFIG.read_text())
    if (cfg['training'] != original_cfg['training'] or cfg['risk_budget'] != original_cfg['risk_budget']
            or cfg['support_scale_unit'] != 'minimum_real_easy_reference'):
        raise ValueError('Support unit alone may change; preserve optimizer, model and2%budget')
    verification = json.loads((SOURCE_PUBLIC/'verification.json').read_text())
    if not verification['exact_replay'] or verification['summary_sha256'] != run.sha(SOURCE_PUBLIC/'summary.json'):
        raise ValueError('Exactly verified preceding decision readout required')
    run.NAME, run.PUBLIC, run.PRIVATE, run.CONFIG = NAME, PUBLIC, PRIVATE, CONFIG
    run.REG = PUBLIC/'registration.json'
    run.FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_record_support_units.py',
        ROOT/'tests/test_m3w_record_support_units.py', PUBLIC/'protocol.md',
        ROOT/'src/evaluation/m3w_multiaction_record_inference.py',
        ROOT/'tests/test_m3w_multiaction_record_inference.py',
        ROOT/'src/world_model/m3w_multiaction_record_policy.py',
        ROOT/'scripts/run_m3w_multiaction_record_policy.py', SOURCE_PUBLIC/'training_freeze.json',
        SOURCE_PUBLIC/'summary.json', SOURCE_PUBLIC/'verification.json', SOURCE_PUBLIC/'support_unit_diagnosis.json')
    run.api.predict = predict


def neutral(resume=False):
    cfg, _, reg, _, source, jobs, identity, parents = run.admission()
    finish = PUBLIC/'neutral_replay.json'
    if finish.exists():
        raise ValueError('Reuse immutable completed neutral receipt')
    started, beat, guard = run.runtime(cfg)
    context, items = sorted(reg['contexts'].items())[0]; seed, arm = 17, 'physical_neural'
    packets = run.context_packets(context, items, source, jobs, identity, beat)['fit']
    data = run.prepare(context, seed, arm, packets, parents[context, seed, arm])
    path = PRIVATE/'neutral'/'checkpoint.pt.gz'
    with installed(False):
        result = run.api.fit(data, arm=arm, settings=cfg['training'], seed=seed, path=path,
            heartbeat=beat, guard=guard, resume=resume)
    old_freeze = json.loads((SOURCE_PUBLIC/'training_freeze.json').read_text())
    reference = None
    for ref in old_freeze['fits']:
        p = ROOT/ref['path']; row = json.loads(p.read_text())
        if (row['context'], row['seed'], row['arm']) == (context, seed, arm):
            if run.sha(p) != ref['sha256'] or run.sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
                raise ValueError('Original neutral comparator changed')
            reference = row['checkpoint']
            old = run.api.read_checkpoint(ROOT/reference['path'])
            break
    if reference is None:
        raise ValueError('Original full500-update comparator required')
    for key in result:
        if key not in ('seconds', 'identity'):
            run.api.exact(result[key], old[key])
    run.once(finish, dict(context=context, seed=seed, arm=arm, updates=500,
        exact_all_scientific_state_excluding_registration_and_runtime=True,
        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=run.sha(path), bytes=path.stat().st_size),
        reference=reference, pid=run.os.getpid(), seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        registration_sha256=run.sha(run.REG), independent_roles_opened=False))
    beat(state='full500_neutral_replay_exact')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'neutral', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    configure(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register':
            run.load()
        elif args.phase == 'neutral':
            neutral(args.resume)
        elif args.phase in ('pilot', 'train'):
            if args.phase == 'train' and not json.loads((PUBLIC/'neutral_replay.json').read_text())['exact_all_scientific_state_excluding_registration_and_runtime']:
                raise ValueError('Full neutral replay required before primary optimization')
            with installed():
                run.train(args.phase == 'pilot', args.resume)
        else:
            run.evaluate(args.phase == 'verify', args.resume)


if __name__ == '__main__':
    main()
