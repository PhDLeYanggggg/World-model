"""Build one standalone reviewer archive from hash-verified scalar replays."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package_m3w_main_claims as main

ROOT, PUBLIC = main.ROOT, main.PUBLIC
LOCAL = main.PRIVATE / 'submission_admin'
OUT = PUBLIC / 'reviewer_evidence_v1'
TEMPLATE = ROOT / 'reproducibility/reviewer_evidence'
SOURCES = {
    'headline': ('main_claims_replay', '6d2f280fac09591ba7b0729755ded98967901db885f1e0d3d37464c24a12a8a6',
                 ('README.md', 'manifest.json', 'verify.py', 'evidence.json')),
    'budget': ('matched_budget_evidence_replay', '5c71826115f7a71c5fc19c4e72208b1a1e97efa1dae6e6863f9f48039dbe8ef7',
               ('README.txt', 'scalars.json', 'verify.py')),
    'external': ('wepdtof_scalar_replay', 'a23d64df3e183862fb42f853405fd2f50eae3303494a855d9a8177d0c44a2186',
                 ('README.md', 'manifest.json', 'verify.py', 'evidence.json', 'resamples.json')),
}
PRIVATE_MARKERS = re.compile(
    rb'/Users/|/home/|/cephfs/|https?://|github\.com|kcl\.ac\.uk|yangyue|Le\s+Yang|PhDLeYang|'
    rb'data/stage_|rdb[1-7]_', re.IGNORECASE)


def inherited(group, inputs=LOCAL):
    name, digest, members = SOURCES[group]
    path = inputs / (name+'.zip')
    if main.digest(path.read_bytes()) != digest:
        raise ValueError('Archived replay differs: '+group)
    with ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(members):
            raise ValueError('Unexpected replay members: '+group)
        return {f'{group}/{k}': archive.read(k) for k in members}


def development():
    p = PUBLIC / 'opendd_pair_disagreement_v1/diagnostic.json'
    q = PUBLIC / 'opendd_support_rerank_v1/readout.json'
    a, b = (json.loads(r.read_text()) for r in (p, q))
    if (any(r['independent_confirmation'] or r['external_outcomes_read'] or not r['original_2pct_unchanged']
            for r in (a, b)) or a['learned']['unique_recordings'] != 493 or b['directions'] != 126
            or b['summary']['additive']['matched'] != 8874
            or b['freeze_sha256'] != main.digest(q.with_name('freeze.json').read_bytes())):
        raise ValueError('Require the existing matched development evidence')
    pairs = a['learned']['comparisons']
    contrasts = {}
    for key in ('joint_minus_additive', 'joint_minus_same_family'):
        ref = pairs[key]['paired']
        sites = sorted(ref['known_pp']['by_site'])
        contrasts[key] = {k: [ref[field]['by_site'][s] for s in sites]
                          for k, field in (('observed', 'known_pp'), ('lower', 'lower_pp'))}
    for key, ref in b['intervals'].items():
        contrasts[key] = dict(observed=ref['ade_available_pp']['sites'], lower=ref['ade_lower_pp']['sites'])
    return dict(locations=7, recordings=493, views=8874, seeds=[17, 29, 43], contrasts=contrasts)


def members(inputs=LOCAL):
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    for group in SOURCES:
        items.update(inherited(group, inputs))
    if json.loads(items['headline/evidence.json']) != main.export():
        raise ValueError('Main replay no longer matches the verified draft sources')
    items['opendd_decisions.json'] = main.as_json(development())
    if any(PRIVATE_MARKERS.search(b) or not k.isascii() for k, b in items.items()):
        raise ValueError('External link or identifying marker in a reviewer file')
    items['manifest.json'] = main.as_json(dict(members={k: main.digest(v) for k, v in items.items()}))
    return items


def encode(path, items):
    with ZipFile(path, 'x', compression=ZIP_DEFLATED) as archive:
        for name, raw in sorted(items.items()):
            z = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            z.compress_type = ZIP_DEFLATED
            archive.writestr(z, raw)


def package(path):
    items = members()
    with TemporaryDirectory(prefix='offline-review-') as tmp:
        root = Path(tmp)
        a, b = root/'a.zip', root/'b.zip'
        encode(a, items)
        encode(b, items)
        archive = a.read_bytes()
        if archive != b.read_bytes():
            raise ValueError('Reviewer archive is not deterministic')
        extracted = root/'clean'
        with ZipFile(a) as z:
            z.extractall(extracted)
        run = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=extracted,
                             capture_output=True, text=True, timeout=120, check=True)
    result = json.loads(run.stdout)
    main.verify_draft(result['headline'], main.DRAFT)
    if path.exists() and path.read_bytes() != archive:
        raise FileExistsError('Refusing to overwrite a different reviewer archive')
    if not path.exists():
        path.write_bytes(archive)
    receipt = dict(archive_sha256=main.digest(archive), archive_bytes=len(archive), members=len(items),
        headline=result['headline'], exact_opendd_development=result['opendd_decision_development'],
        deterministic_rebuild=True, clean_extraction_stdlib_pass=True, direct_identity_marker_scan_pass=True,
        source_archive_sha256={k: v[1] for k, v in SOURCES.items()},
        manuscript_unchanged=True, external_outcomes_opened=False, submission_ready=False, upload_performed=False)
    OUT.mkdir(exist_ok=True)
    (OUT/'receipt.json').write_bytes(main.as_json(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=LOCAL/'reviewer_evidence.zip')
    package(parser.parse_args().output)
