"""Verify completed label sidecars without changing forecasts, actions or risk rules."""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_fitting_gain_labels as build


def main():
    started = time.monotonic(); _, reg = build.registration()
    assert reg == json.loads((build.PUBLIC/'registration.json').read_text())
    done = json.loads((build.PUBLIC/'completion.json').read_text())
    replay = json.loads((build.PUBLIC/'replay.json').read_text())
    assert replay['replay_exact'] and replay['groups'] == done['groups'] == 108
    assert done['records'] == replay['records']
    assert done['counts_repeated_fitting_views'] == replay['counts_repeated_fitting_views']
    totals = dict.fromkeys(done['counts_repeated_fitting_views'], 0)
    ids_all = set(); rows = []; checks = 0
    for ref in done['records']:
        path = ROOT/ref['path']; assert build.old.base.artifact(path) == ref
        doc = json.loads(path.read_text()); labels = doc['labels']
        assert doc['registration_sha256'] == done['registration_sha256'] == build.digest(build.PUBLIC/'registration.json')
        assert build.old.base.artifact(ROOT/labels['path']) == labels
        with np.load(ROOT/labels['path'], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        assert set(a) == {'ids', 'known', 'easy', 'reference', 'candidate', 'signed_gain', 'positive_benefit', 'positive_harm'}
        ids, known = a['ids'], a['known']; assert known.dtype == bool
        assert all(v.shape == ids.shape for v in a.values()) and np.all(np.diff(ids) > 0)
        assert build.old.base.inter.array_hash(ids) == doc['source_identity']['fitting_ids_hash']
        old = np.column_stack((a['easy'], a['reference'], a['positive_harm']))
        assert build.old.base.inter.array_hash(old) == doc['source_identity']['input_hashes']['targets']
        for k in set(a)-{'ids', 'known'}:
            assert np.array_equal(np.isfinite(a[k]), known) and np.isnan(a[k][~known]).all()
        r, n, g, b, h = (a[k][known] for k in ('reference', 'candidate', 'signed_gain', 'positive_benefit', 'positive_harm'))
        np.testing.assert_allclose(r-n, g, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(b-h, g, rtol=1e-12, atol=1e-12)
        assert (r >= 0).all() and (n >= 0).all() and (b >= 0).all() and (h >= 0).all()
        assert np.all((b == 0) | (h == 0)) and set(np.unique(a['easy'][known])) <= {0., 1.}
        roles = doc['source_identity']['roles']; fit = set(roles['training_sites'])
        assert fit == set(doc['fitting_sources']) and len(fit) == 2
        assert not any(fit & set(roles[k]) for k in ('producer_sites', 'controller_sites', 'held_sites'))
        stats = dict(rows=len(ids), known=int(known.sum()), unknown=int((~known).sum()),
                     beneficial=int((g > 0).sum()), harmful=int((g < 0).sum()), tied=int((g == 0).sum()),
                     easy=int((a['easy'] == 1).sum()))
        assert stats == doc['stats']
        for k in totals: totals[k] += stats[k]
        ids_all.update(map(int, ids)); rows.append(dict(group=doc['group'], **stats)); checks += 1
    assert totals == done['counts_repeated_fitting_views']
    run = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_m3w_fitting_gain_labels.py'],
                         cwd=ROOT, text=True, capture_output=True)
    if run.returncode: raise RuntimeError(run.stdout+run.stderr)
    result = dict(result_source='fresh_run_label_completion_and_full_exact_replay', input_source='cached_verified',
        verified_groups=checks, repeated_fitting_counts=totals, unique_source_row_ids=len(ids_all),
        unique_rows_are_not_independent_samples=True, sidecar_bytes=done['label_bytes'],
        build_seconds=done['seconds'], replay_seconds=replay['seconds'],
        targeted_tests='10 passed' if '10 passed' in run.stdout else run.stdout.strip(),
        parameter_updates=0, policy_actions_computed=False, held_outcomes_used=False,
        independent_roles_read=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    build.immutable(build.PUBLIC/'verification_report.json', result)
    text = f'''# Signed Gain Label Completion

## Material Passport

Fresh label construction and full exact array replay, using cached_verified
forecasts, training scales and fitting roles. All{checks} groups retain original
row IDs and occurrence/reference/positive-harm target hashes. No new training,
policy, threshold, held readout or independent confirmation data access.

## Verified Asset

- Repeated fitting row occurrences: {totals['rows']:,}; known: {totals['known']:,}; unknown: {totals['unknown']:,}.
- Beneficial: {totals['beneficial']:,}; harmful: {totals['harmful']:,}; exact ties: {totals['tied']:,}.
- Unique source row IDs: {len(ids_all):,}. Overlapping windows and repeated seeds are NOT independent observations.
- Label-only sidecars: {done['label_bytes']:,} bytes, stored privately and excluded from Git.
- Full build/resume: {done['seconds']:.2f}s; full replay: {replay['seconds']:.2f}s; preceding one-group pilot is separate.
- Ten targeted synthetic tests pass. Original IDs/targets and signed arithmetic independently checked in108 groups.

Unknown labels stay NaN. Benefit and harm are mutually exclusive and their
difference equals signed gain. Supervision is separated from inference: these
sidecars expose no new deployable feature and are not consumed by the ongoing
conditional-risk component diagnostic. No signed-utility ranking model or
improvement has been evaluated by this construction.

## Why This Matters

The prior easy-risk packet alone could not distinguish a highly beneficial
switch from a tiny beneficial switch: both have zero positive harm. That is a
missing diagnostic target, not proof of why deployment fails and not proof
that prior utility learning was absent. These sidecars permit a later registered
fitting-only check of beneficial ranking without opening held outcomes.

## Boundaries

This is data completion, not efficacy, independent generalization, calibration
or risk-control evidence. Keep the deployed floor and the failed-risk result.
Observation8/prediction12, raw stride12, image-local detector silver. No metric,
seconds, human-gold, physical-safety, true3D, foundation or submission-ready
claim. Stage5C/SMC remain disabled. Full legacy integration and cold raw rebuild
are not_run; the exact replay covers labels derived from frozen forecasts only.
'''
    path = build.PUBLIC/'results.md'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    artifacts = {p.name: build.digest(p) for p in build.PUBLIC.iterdir()
                 if p.suffix in ('.md', '.json') and p.name != 'verification.json'}
    build.immutable(build.PUBLIC/'verification.json', dict(source_bindings={**reg['bindings'],
        str(Path(__file__).relative_to(ROOT)): build.digest(Path(__file__))}, artifacts=artifacts,
        verified_groups=108, tests=10, seconds=time.monotonic()-started, deployment_changed=False))
    print(json.dumps(result))


if __name__ == '__main__': main()
