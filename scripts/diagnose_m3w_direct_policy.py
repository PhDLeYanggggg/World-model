"""Posthoc arithmetic diagnosis; no model, threshold or dataset selection."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import report_m3w_direct_policy as report

run = report.run


def describe(rows):
    out = dict(result_source='cached_verified_posthoc_arithmetic',
        thresholds_changed=False, new_fit=False, independent_confirmation=False, scopes={})
    for scope in ('fit', 'inner_held'):
        group = [r for r in rows if r['scope'] == scope]
        if not group:
            raise ValueError('Both fit and inner-held cohorts required')
        den = [r['result']['full_known_reference_mass'] for r in group]
        if min(den) <= 0:
            raise ValueError('Cannot hide an undefined whole-view denominator')
        terms = {}
        for arm in run.readout.POLICIES:
            terms[arm] = {name: statistics.fmean(
                100*r['result']['policies'][arm]['selected_'+name+'_mass']/d
                for r, d in zip(group, den)) for name in ('known_benefit', 'known_harm', 'unknown_envelope')}
        reference, primary = terms['origin'], terms['hard_record']
        out['scopes'][scope] = dict(unique_source_recordings=len({(r['source'], rec)
                for r in group for rec in r['result']['recordings']}),
            unique_sources=len({r['source'] for r in group}), mass_terms_equal_view_pp=terms,
            primary_known_benefit_retained=(primary['known_benefit']/reference['known_benefit']
                                          if reference['known_benefit'] > 0 else None),
            primary_known_harm_retained=(primary['known_harm']/reference['known_harm']
                                       if reference['known_harm'] > 0 else None),
            primary_known_net_change_pp=(primary['known_benefit']-primary['known_harm']
                                        -reference['known_benefit']+reference['known_harm']))
    return out


def execute(check=False):
    _, _, _, checked = report.validate()
    refs = json.loads((run.PUBLIC/'complete.json').read_text())['groups']
    rows = [json.loads((run.ROOT/r['path']).read_text()) for r in refs]
    out = describe(rows)
    out.update(complete_sha256=checked['complete_sha256'], script_sha256=run.sha(Path(__file__)),
        note='Equal-view normalized masses, not pooled raw units or independent observations')
    path = run.PUBLIC/'failure_decomposition.json'
    if check:
        report.require(out == json.loads(path.read_text()), 'Diagnosis replay differs')
    else:
        run.once(path, out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true')
    execute(p.parse_args().check)
