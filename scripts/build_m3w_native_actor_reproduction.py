"""Export and clean-run native actor aggregate evidence, not data or models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_m3w_blinded_reproduction import archive_bytes, clean_run, scan_files
from src.evaluation import native_actor_portable as portable

HOME = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = HOME/'native_actor_reproduction_v1'
ARCHIVE = ROOT/'data/stage_cvpr2027_experiments/native_actor_reproduction_v1/evidence_bundle.zip'
PINNED = {
    'moving_prefix_actor_v1': 'ae140b4cad607c07883446a4f0b8c31fe8512cbe7d262ebcd1f78b329f91e83e',
    'moving_visual_actor_v1': 'bd6894ea230320d605c9c4d17516a3c09ab7d821d4a4068e82d28f4a263c6767',
    'moving_camera_frame_v1': 'af4b8ef696a49f529f7da6845105686d21ddbc90e5b7d0f408f26ea05652e1b7',
}


def digest(data): return hashlib.sha256(data).hexdigest()
def blob(value): return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def make_files():
    experiments = []; labels = ('Prefix-balanced actor', 'Image-conditioned actor', 'Query-view actor')
    for (folder, expected), label in zip(PINNED.items(), labels):
        raw = (HOME/folder/'analysis.json').read_bytes()
        if digest(raw) != expected: raise ValueError('Changed frozen source evidence')
        a = json.loads(raw)
        if folder == 'moving_prefix_actor_v1':
            rows = a['sources']; old = a['primary']
            primary = dict(known_pp=old['equal_source_paired_known_pp'],
                completion_lower_pp=old['equal_source_paired_completion_lower_pp'],
                interval95=old['lower_exploratory_bootstrap_CI'],
                fold_known_means=old['fold_known_means_pp'],
                positive_known_sources=old['positive_source_known_contrasts'],
                component_progression_screen_pass=old['actor_component_progression_screen_pass'])
        else:
            rows = [r for r in a['sources'] if r['contrast'] == 'primary']; primary = a['primary']
        aliases = {name: f'locality{i+1:02d}' for i, name in enumerate(sorted(r['source'] for r in rows))}
        packed = []
        for row in rows:
            prefixes = []
            for s in row['summary']:
                prefixes.append(dict(known_pp=s['paired_known_pp'], lower_pp=s['paired_completion_lower_pp'],
                    upper_pp=s['paired_completion_upper_pp'], **{k: s[k] for k in
                        ('rows', 'known_rows', 'complete_rows', 'partial_rows', 'wholly_unknown_rows')}))
            packed.append(dict(source=aliases[row['source']], producer_fold=row['producer_fold'], prefixes=prefixes))
        seed = 39017 if folder == 'moving_prefix_actor_v1' else 39107
        registration = HOME/folder/('registration.json' if folder == 'moving_prefix_actor_v1' else 'registration_v2.json')
        cfg = json.loads(registration.read_text())['config']
        if cfg['bootstrap_seed'] != seed or cfg['bootstrap_resamples'] != 3000:
            raise ValueError('Changed registered bootstrap seed or budget')
        experiments.append(dict(label=label, sources=packed, archived_primary=primary, bootstrap_seed=seed))
    evidence = dict(schema_version=1, scope=portable.SCOPE, risk_budget=.02,
        experiments=experiments, bootstrap_indices_by_seed={str(seed): np.random.default_rng(seed).integers(0, 14, (3000, 14)).tolist()
            for seed in (39017, 39107)},
        **{k: False for k in portable.FALSE_CLAIMS})
    result = portable.reconstruct(evidence); generated = {k: v.encode() for k, v in portable.outputs(result).items()}
    files = {'reproduce.py': Path(portable.__file__).read_bytes(),
        'README.md': (ROOT/'reproducibility/native_actor_evidence/README.md').read_bytes(),
        'evidence.json': (json.dumps(evidence, separators=(',', ':'), allow_nan=False)+'\n').encode(),
        **{'expected/'+k: v for k, v in generated.items()}}
    scan_files(files)
    manifest = dict(package_scope='deidentified_native_actor_aggregate_v1',
        files={k: dict(sha256=digest(v), bytes=len(v)) for k, v in sorted(files.items())},
        expected_outputs={k: digest(v) for k, v in generated.items()},
        anonymity_certified=False, full_training_reproduction=False, independent_confirmation=False, submission_ready=False)
    files['manifest.json'] = blob(manifest); scan_files(files)
    return files


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    files = make_files(); raw = archive_bytes(files)
    if args.verify:
        if ARCHIVE.read_bytes() != raw: raise ValueError('Archive differs from frozen aggregate reconstruction')
    else:
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with ARCHIVE.open('xb') as out: out.write(raw)
    verified = clean_run(ARCHIVE)
    report = dict(result_source='fresh_packaging_and_isolated_aggregate_replay',
        source_evidence='cached_verified', source_sha256=PINNED,
        archive_sha256=digest(raw), archive_bytes=len(raw), archive_members=len(files),
        clean_execution=verified, direct_marker_scan='pass_fixed_scan_not_anonymity_guarantee',
        fixed_bootstrap_draws=3000, resampled_sources=14, experiments=3,
        new_training=False, new_prediction_or_target_readout=False, new_random_bootstrap=False,
        full_training_reproduction=False, independent_confirmation=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    text = blob(report); target = PUBLIC/'verification.json'
    if args.verify:
        if target.read_bytes() != text: raise ValueError('Clean verification differs from saved receipt')
    else:
        PUBLIC.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as out: out.write(text)
    print(json.dumps(dict(archive_bytes=len(raw), files=len(files), clean_exit=verified['exit_code'],
        source_experiments=3, full_training_reproduction=False, submission_ready=False)))


if __name__ == '__main__': main()
