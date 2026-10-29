"""Verify complete frozen TRAIN inference and distinguish fitting from transfer."""
import argparse
import json
import math
import statistics

from scripts import run_m3w_train_census as run


def check_rows(rows, expected, registration_sha, documents, development):
    seen = set()
    sampling = {}
    for row in rows:
        ident = row['identity']
        key = ident['group'], ident['source'], ident['seed'], row['mode']
        census = row['census']
        if (key not in expected or key in seen
                or row['registration_sha256'] != registration_sha
                or row['checkpoint_sha256'] != documents[key]['checkpoint']['sha256']
                or not census['inference_replay_exact'] or census['inference_outcome_filter']
                or census['new_training'] or census['independent_roles_read']
                or row['optimizer_steps'] != 0 or row['independent_roles_read']
                or row['deployment_changed'] or not row['sampling_coverage']['sampler_exact']):
            raise ValueError('Changed census identity, provenance or inference rule')
        dev = development[key[:3]]['result']
        if (row['cached_development_policy'] != dev['policies'][key[3]]
                or row['cached_development_diagnostics'] != dev['diagnostics'][key[3]]):
            raise ValueError('Development comparison is not the frozen aggregate')
        if key[:3] in sampling and sampling[key[:3]] != row['sampling_coverage']:
            raise ValueError('Matched arms must have identical sampling coverage')
        sampling[key[:3]] = row['sampling_coverage']
        full = census['full']
        if census['full_TRAIN_rows'] != full['known_rows'] + full['unknown_rows']:
            raise ValueError('Full TRAIN row count changed')
        for num, den, ratio in (
                ('selected_known_harm_mass', 'selected_known_reference_mass', 'known_all_risk'),
                ('selected_known_easy_harm_mass', 'selected_known_easy_reference_mass', 'known_easy_risk')):
            value = full[num] / full[den] if full[den] > 0 else None
            observed = full[ratio]
            if ((value is None) != (observed is None)
                    or value is not None and not math.isclose(value, observed, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError('Risk ratio differs from independently summed masses')
        seen.add(key)
    if seen != expected or len(rows) != len(expected):
        raise ValueError('Missing or duplicated fixed census identities')


def details(rows):
    selected = [r for r in rows if r['mode'] == 'budget_scaled']
    coverage = [r['sampling_coverage'] for r in selected]
    full = [r['census']['full'] for r in selected]
    fractions = [r['unseen_easy_harm_mass_fraction'] for r in coverage
                 if r['unseen_easy_harm_mass_fraction'] is not None]
    return dict(
        matched_identities=len(selected),
        sampler_exact_all=all(r['sampler_exact'] for r in coverage),
        views_with_unseen_queries=sum(r['unseen_queries'] > 0 for r in coverage),
        median_unseen_query_fraction=statistics.median(r['unseen_queries']/r['queries'] for r in coverage),
        median_unseen_easy_harm_mass_fraction=statistics.median(fractions),
        maximum_unseen_easy_harm_mass_fraction=max(fractions),
        median_top_one_percent_easy_harm_share=statistics.median(
            r['top_one_percent_rows_easy_harm_share'] for r in full),
        known_easy_defined_views=sum(r['known_easy_risk'] is not None for r in full),
        known_easy_supported_but_completion_fails=sum(
            r['known_easy_risk'] is not None and r['known_easy_risk'] <= .02+1e-12
            and r['easy_selected_risk_upper'] is not None
            and r['easy_selected_risk_upper'] > .02+1e-12 for r in full),
        total_TRAIN_row_occurrences=sum(r['census']['full_TRAIN_rows'] for r in selected),
        repeated_occurrences_not_independent_samples=True,
        new_predictions=False, new_training=False, deployment_changed=False)


def validate():
    sha = run.base.sha
    reg = json.loads(run.REG.read_text())
    complete = json.loads((run.PUBLIC/'complete.json').read_text())
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    for path, digest in reg['bindings'].items():
        if sha(run.ROOT/path) != digest:
            raise ValueError('Changed registered census source')
    if (complete['summary_sha256'] != sha(run.PUBLIC/'summary.json')
            or complete['registration_sha256'] != sha(run.REG)
            or reg['parent_freeze_sha256'] != sha(run.training.PUBLIC/'training_freeze.json')
            or reg['development_complete_sha256'] != sha(run.training.PUBLIC/'readout/complete.json')
            or reg['input_manifest_sha256'] != sha(run.base.PRIVATE/'create_train_input_manifest.json')
            or complete['optimizer_steps'] != 0 or complete['new_development_inference']
            or complete['independent_roles_read'] or complete['deployment_changed']):
        raise ValueError('Changed inputs or role boundaries')
    documents, _ = run.heads.admission()
    development = run.cached_development()
    rows = []
    for ref in complete['groups']:
        path = run.ROOT/ref['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((run.PUBLIC/'groups').resolve())
                or sha(path) != ref['sha256']):
            raise ValueError('Changed census group')
        rows.append(json.loads(path.read_text()))
    expected = {(r['group'],r['source'],r['seed'],arm) for r in reg['expected'] for arm in run.ARMS}
    if len(expected) != 288 or len(reg['expected']) != 72:
        raise ValueError('Complete registered 288-view census required')
    check_rows(rows, expected, sha(run.REG), documents, development)
    from src.world_model.m3w_cost_head_census import summarize
    if summarize(rows) != summary:
        raise ValueError('Census summary does not replay')
    return complete, summary, details(rows)


def content(complete, summary, extra):
    table = ['| Frozen arm | TRAIN known easy violations /72 | Development known easy violations /72 | TRAIN pass, development fail /72 | TRAIN completion support /72 |',
             '|---|---:|---:|---:|---:|']
    for arm in run.ARMS:
        value = summary[arm]
        table.append('| '+arm+' | '+' | '.join(str(value[k]) for k in (
            'TRAIN_known_easy_violations','development_known_easy_violations',
            'TRAIN_known_easy_pass_development_fail','TRAIN_supported_views'))+' |')
    return f'''# Full TRAIN Census: Fitting Improves, Generalization Does Not

## Evidence

Fresh_run: all 288 frozen-head full-TRAIN inferences completed in
{complete['seconds']:.2f}s. Models, source packets and previous development
aggregates are cached_verified, not freshly trained or newly evaluated.
There are 72 matched identities per arm, not 288 independent datasets.
All TRAIN rows, including unknown outcomes, remain in inference. No target is
an input or action filter. Repeated predictions and the 2,000-step sampling
histories replay exactly. No optimizer, new development inference, independent
selection/calibration/confirmation access or deployment change occurred.

{chr(10).join(table)}

The unchanged 2% risk is positive selected harm/reference error, not net ADE.
The scaled policy has 2/72 known easy violations on full TRAIN, compared with
58/72 for its unit control; it still has 58/72 on held-recording development.
56/72 scaled views pass the known TRAIN calculation but fail on development.
This supports prioritizing out-of-recording harm prediction over another global
penalty increase. It does not identify a unique causal mechanism or restore
independence to exposed development results.

## Unknown Outcomes Still Block Safety

The scaled model has 66/72 TRAIN completion-upper easy violations, four undefined
known easy ratios, and only 2/72 full finite-completion supports. In
{extra['known_easy_supported_but_completion_fails']}/72 views a known-label easy
calculation passes but its completion-upper easy calculation fails. Unknown
outcomes cannot be discarded or treated as zero harm. Full TRAIN fitting success
on known labels is not a safety guarantee, nor evidence of new world dynamics.

## Coverage and Rare Harm

Across the 72 matched sampling identities, {extra['views_with_unseen_queries']}
leave at least one known TRAIN query unseen in 2,000 updates. Median unseen-query
fraction is {extra['median_unseen_query_fraction']:.6f}. Median unseen easy-harm
mass fraction is {extra['median_unseen_easy_harm_mass_fraction']:.6f}; its maximum
is {extra['maximum_unseen_easy_harm_mass_fraction']:.6f}. The largest 1% of TRAIN
row easy-harm values contain a median
{extra['median_top_one_percent_easy_harm_share']:.6f} of its total easy-harm mass.
These coverage statistics do not establish that unseen rows caused development
failure. The census includes {extra['total_TRAIN_row_occurrences']:,} repeated
TRAIN row occurrences per arm; overlapping views are not independent samples.

## Next Repair Boundary

Keep existing fits and failed development results frozen. Any next change must
test harm generalization inside TRAIN using whole recordings, rather than tune
against these already-exposed development labels. Check available recording
support first, retain unknown-label completion accounting, and compare against
the existing negative sampling/calibration controls before commissioning fits.
Increasing model size or repeating the same threshold search is not justified
by this census. The separate prospective contract repair prevents impossible
same-count safety requirements; it does not rescue the current unsafe model.

## Reproduction

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_train_census --check
```

This verifies all receipts, registered source hashes, checkpoint bindings,
sampling equality, development aggregates, row totals and risk mass ratios.
No new inference is needed for report verification. Observation8/prediction12,
dataset-local coordinates and raw-frame conventions are unchanged. Not true3D,
not a foundation model, no metric/seconds claim; Stage5C and SMC remain disabled.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    complete, summary, extra = validate()
    artifacts = {'conclusions.md':content(complete,summary,extra),
                 'verified_details.json':json.dumps(extra,indent=2,allow_nan=False)+'\n'}
    for name, text in artifacts.items():
        path = run.PUBLIC/name
        if args.check or path.exists():
            if path.read_text() != text:
                raise ValueError('Preserve conflicting census report')
        else:
            with path.open('x') as stream:
                stream.write(text)
    print(json.dumps(dict(status='checked' if args.check else 'reported',views=288,
                         optimizer_steps=0,new_inference=False)))


if __name__ == '__main__':
    main()
