"""Export the frozen paper's headline inputs and verify them offline."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'
DRAFT = PRIVATE/'evidence_manuscript_v166'/'build'
TEMPLATE = ROOT/'reproducibility/main_claims'
SOURCES = ('streaming_menu_capacity_v1/analysis.json', 'recording_count_v2/analysis.json',
           'moving_record_peak_allocation_v1/readout.json', 'wepdtof_external_transfer_v1/external_readout.json')


def digest(raw):
    return sha256(raw).hexdigest()


def as_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def verify_draft(values, directory):
    definitions = '\n'.join((directory/p).read_text() for p in ('direct_count_values.tex', 'evidence_values.tex'))
    claims = {
        'DirectCountStrict': values['opendd']['hindsight_passes'], 'DirectCountViews': values['opendd']['views'],
        'DirectCountCount': values['opendd']['past_only_passes'], 'RecordPeakExcluded': values['moving']['ols4_excluded'],
        'RecordPeakViews': values['moving']['views'], 'WepExtSources': values['wepdtof']['sources'],
        'WepExtFixedDual': values['wepdtof']['fixed_joint_dual_passes'], 'WepExtViews': values['wepdtof']['views'],
    }
    for k, v in claims.items():
        if re.findall(rf'\\newcommand\{{\\{k}\}}\{{([^}}]+)\}}', definitions) != [str(v)]:
            raise ValueError(f'Draft macro no longer matches the frozen source: {k}')


def export():
    binding = json.loads((DRAFT/'source_bindings.json').read_text())['source_bindings']
    raw = [(PUBLIC/p).read_bytes() for p in SOURCES]
    if any(digest(b) != binding[k] for b, k in zip(raw, SOURCES, strict=True)):
        raise ValueError('An evidence input has changed since the paper build')
    cap, past, moving, external = map(json.loads, raw)
    if not (cap['labels_hindsight_only'] and moving['hindsight_only'] and external['external_scoring_completed']
            and not external['external_tuning'] and not external['dronecrowd_confirmation_access']
            and all(z['original_2pct'] for z in (cap, past, moving, external))
            and not any(z['confirmation_access'] for z in (cap, past, moving))):
        raise ValueError('The study roles or risk rules have changed')
    if cap['by_site'].keys() != past['by_site'].keys():
        raise ValueError('OpenDD evaluation sites differ between the two studies')
    o = []
    for name, part in sorted(cap['by_site'].items()):
        status = part['arms']['nested_stream']
        if sum(status.values()) != part['views']:
            raise ValueError('Missing hindsight statuses')
        o.append(dict(hindsight_status=status, past_only=[
            [v['seed'], v['scores']['recording']['views'], v['scores']['recording']['dual_positive']]
            for v in sorted(past['by_site'][name], key=lambda v: v['seed'])]))
    m = defaultdict(list)
    for row in moving['rows']:
        a = row['identity']
        m[a['source'], a['recording']].append([a['seed'], row['arbitrary']['record'][1]])
    w = defaultdict(dict)
    for row in external['rows']:
        if row['clip'] in w[row['source']]:
            raise ValueError('Duplicate external recording')
        rows = []
        for seed, view in sorted(row['seeds'].items()):
            v = view['arms']['fixed_joint']
            rows.append([int(seed), v['risk']['supported'], view['reference_mass'],
                         v['lower_vs_cv'], v['lower_vs_ols4'], v['dual_pass']])
        w[row['source']][row['clip']] = rows
    return dict(opendd=o, moving=[sorted(r) for _, r in sorted(m.items())],
                wepdtof=[[r for _, r in sorted(g.items())] for _, g in sorted(w.items())])


def package(destination):
    items = {name: (TEMPLATE/name).read_bytes() for name in ('README.md', 'verify.py')}
    items['evidence.json'] = as_json(export())
    items['manifest.json'] = as_json(dict(members={k: digest(v) for k, v in items.items()},
        scope='frozen aggregate replay; no training, thresholds or label readout'))
    for raw in items.values():
        if re.search(rb'/Users/|/home/|/cephfs/|github\.com|kcl\.ac\.uk|yangyue|PhDLeYang|moving_source_component', raw, re.I):
            raise ValueError('Identity or source path in package')
    with TemporaryDirectory() as d:
        p = Path(d)
        for k, v in items.items():
            (p/k).write_bytes(v)
        report = subprocess.run([sys.executable, '-I', '-B', str(p/'verify.py')], cwd=p, check=True,
                                capture_output=True, text=True, timeout=20)
    values = json.loads(report.stdout)
    verify_draft(values, DRAFT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix('.pending.zip')
    with ZipFile(pending, 'w', compression=ZIP_DEFLATED) as z:
        for k, v in sorted(items.items()):
            info = ZipInfo(k, (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, v)
    pending.replace(destination)
    print(json.dumps(dict(counts=values, archive_sha256=digest(destination.read_bytes())), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PRIVATE/'submission_admin'/'main_claims_replay.zip')
    package(p.parse_args().output)
