"""Check whether any distinct easy action can satisfy the original ratio."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_recipient_budget_capacity as prior

run, np, cal = prior.run, prior.np, prior.cal
PUBLIC = prior.PUBLIC.parent/'native_easy_support_certificate_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_easy_support_certificate.py')


def assess(cv, bank, target, valid, can_switch, easy_cut):
    err = prior.model.errors
    base, _ = err(cv, target, valid)
    easy = valid.any(1) & (base > 0.) & (base <= easy_cut)
    ix = np.flatnonzero(easy & can_switch)
    best = np.full(len(ix), np.inf)
    options = 0
    for path in bank.transpose(1, 0, 2, 3):
        used = (path[ix] != cv[ix]).any(axis=(1, 2))
        error, _ = err(path[ix], target[ix], valid[ix])
        ratio = np.maximum(error-base[ix], 0.)/base[ix]
        best = np.minimum(best, np.where(used, ratio, np.inf))
        options += int(used.sum())
    return dict(easy_rows=int(easy.sum()), rows_with_distinct_easy_path=int(np.isfinite(best).sum()),
                distinct_easy_paths=options, minimum_risk_ratio=float(best.min()) if options else None,
                feasible=bool((best <= .02).any()))


def main():
    cut, controller = prior.register()
    source = prior.PUBLIC/'refinement.json'
    run.checked(run.read(prior.PUBLIC/'refinement_execution.json')['artifact'])
    parents = (prior.REG, source, prior.PUBLIC/'refinement_execution.json', *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in parents], risk=.02,
        scope='full_recipient_up_to_original_query_cap', outcome_access='exposed_development_only')))
    if sys.argv[1:] == ['register']:
        return
    if sys.argv[1:] != ['score']:
        raise ValueError('Use register or score')
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the case-study design before scoring')
    start = time.monotonic()
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = cal.prior.h.previous.prior.blocks(d, lab['frames'])
    cases = []
    checked = 0
    for c in controller['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(c['cache']), allow_pickle=False) as f:
            coef = dict(zip(f['sources'], f['excluded_source_coefficients']))
        for r, (ix, groups) in records.items():
            at = ix[np.concatenate(groups)]
            site = d['sites'][at[0]]
            sizes = np.array([len(g) for g in groups])
            off = np.r_[0, sizes.cumsum()]
            paths, cap, _ = prior.query_design(d, at, off, cv[at], bank[at], coef[site])
            found = assess(cv[at], paths, lab['target_eval'][at]-d['origin'][at, None],
                           lab['valid'][at], np.repeat(cap > 0, sizes), cut['easy'])
            p = prior.PRIVATE/'refinement'/f'{seed}_{r}.json'
            if p.exists():
                bound = run.read(p)['result']
            else:
                bound = run.read(prior.PRIVATE/f'{seed}_{r}.json')['outcomes']['all_recipients']
            if found['feasible'] != bound['feasible'] or (
                    not found['feasible'] and not bound['proved_infeasible']):
                raise ValueError('Analytic feasibility and the solved certificate disagree')
            if not found['feasible']:
                cases.append((str(r), str(site), seed, found))
            checked += 1
    if checked != 102 or len(cases) != 102-run.read(source)['summary']['slates']['all_recipients']['feasible']:
        raise ValueError('Incomplete case roster')
    aliases = {k: i+1 for i, k in enumerate(sorted({r for r, *_ in cases}))}
    doc = dict(verified_views=checked, feasible_views=checked-len(cases),
        failed_recordings=len(aliases), failed_sources=len({s for _, s, *_ in cases}),
        failed_views=[dict(case=aliases[r], seed_index=prior.g.prior.SEEDS.index(seed)+1, **v)
                      for r, _, seed, v in cases], external_data_opened=False, deployable=False,
        criterion='exists_a_distinct_easy_action_within_original_cap_and_risk')
    out = PUBLIC/'analysis.json'
    run.once(out, run.stable(doc))
    mark = PUBLIC/'execution.json'
    if not mark.exists():
        run.once(mark, dict(seconds=time.monotonic()-start, artifact=run.artifact(out)))
    elif run.read(mark)['artifact']['sha256'] != hashlib.sha256(out.read_bytes()).hexdigest():
        raise ValueError('Case-study receipt differs')
    print(json.dumps(doc), flush=True)


if __name__ == '__main__':
    main()
