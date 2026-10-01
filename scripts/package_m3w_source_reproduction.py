"""Build and execute a de-identified source-only reproduction package in memory."""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_m3w_selected_pool_source as source

PACKAGE = ROOT / 'reproducibility' / 'selected_pool_source'


def build():
    rows, cfg = source.load_source_rows()
    figure = source.build(rows, cfg)
    saved = json.loads((source.run.PUBLIC/'source_figure_data.json').read_text())
    assert all(saved[k] == v for k, v in figure.items())
    selected = sorted((r for r in rows if r['role'] == 'source_oof' and r['mode'] == 'joint'), key=lambda r: r['view'])
    sites = {s: f'locality{i+1:02d}' for i, s in enumerate(sorted({r['site'] for r in selected}))}
    points = {r['view']: r for r in figure['points']}
    packed = []
    fields = ('rows', 'known', 'unknown', 'envelope', 'unknown_envelope', 'truth', 'original', 'adjusted')
    for i, r in enumerate(selected):
        expected = {k: v for k, v in points[r['view']].items() if k not in ('view', 'site', 'head_seed', 'source_screen')}
        packed.append(dict(view=f'view{i+1:03d}', site=sites[r['site']], head_seed=r['head_seed'],
            source_screen=r['source_screen'], expected=expected,
            pools={name: {key: r['statistics'][name][key] for key in fields} for name in ('raw', 'kept', 'removed')}))
    interval = copy.deepcopy(figure['bias_interval'])
    interval['all_views']['undefined_localities'] = [sites[s] for s in interval['all_views']['undefined_localities']]
    descriptive = interval['defined_only_descriptive']
    descriptive['by_locality'] = {sites[s]: v for s, v in descriptive['by_locality'].items()}
    evidence = dict(schema='selected_pool_source_aggregate_v1', scope='source_oof_joint_only',
        risk_budget=cfg['risk_budget'], independent_confirmation=False, rows=packed,
        bootstrap=dict(draws=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'], unit='locality'),
        expected_interval=interval)
    files = {name: (PACKAGE/name).read_bytes() for name in ('verify.py', 'README.md')}
    files['evidence.json'] = (json.dumps(evidence, indent=2, allow_nan=False)+'\n').encode()
    forbidden = ('/Users/', '/users/', 'yangyue', 'k24101830', 'PhDLeYanggggg', 'github.com/', 'eu-locality-', 'single0_', 'single1_')
    assert not any(s in v.decode() for s in forbidden for v in files.values())
    manifest = dict(scope=evidence['scope'], artifacts={k: dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k, v in files.items()},
        source_manifest_sha256=source.run.digest(source.run.PUBLIC/'source_manifest.json'),
        source_replay_sha256=source.run.digest(source.run.PUBLIC/'source_replay.json'),
        exporter_sha256=source.run.digest(__file__), transfer_outputs_read=False, raw_data_included=False,
        checkpoints_included=False, author_identifiers_removed=True, anonymity_guaranteed=False)
    files['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    assert sum(map(len, files.values())) < 2*2**20
    return files


def isolated_verify(files):
    result = subprocess.run([sys.executable, '-I', '-B', '-c', files['verify.py'].decode(), '--stdin'],
        input=files['evidence.json'].decode(), capture_output=True, text=True, timeout=30, cwd='/')
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['torch_imported'] is False
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store-create', action='store_true')
    args = parser.parse_args()
    files = build(); verification = isolated_verify(files)
    result = dict(verification=verification, local_numerical_artifacts_written=False,
        package_bytes=sum(map(len, files.values())), manifest=json.loads(files['manifest.json']))
    if args.store_create:
        from scripts import manage_m3w_selected_pool_create as manager
        code = r'''
import base64,hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path('/users/k24101830/m3w/european_selected_pool_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
target=root/'source_reproduction_v1';target.mkdir(exist_ok=True)
assert set(p['files'])=={'verify.py','README.md','evidence.json','manifest.json'}
total=0
for name,r in p['files'].items():
    raw=base64.b64decode(r['base64']);total+=len(raw);assert total<2*2**20
    assert hashlib.sha256(raw).hexdigest()==r['sha256']
    f=target/name
    if f.exists():assert f.read_bytes()==raw
    else:
        with f.open('xb') as stream:stream.write(raw)
    assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
print(json.dumps(dict(files_verified=4,bytes=total,training_run=False)))
'''
        result['remote_storage'] = manager.remote(code, dict(files={k: dict(base64=base64.b64encode(v).decode(),
            sha256=hashlib.sha256(v).hexdigest()) for k, v in files.items()}))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
