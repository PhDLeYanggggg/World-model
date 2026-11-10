"""Verify the corrected WEPDTOF comparator and aggregate diagnosis."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'equal_steps.json',
           'equal_step_verify.py', 'breakpoints.json', 'prior_evidence.zip'}


def replay(data):
    summary, sources = data['summary'], data['source_summaries']
    fields = ('queries', 'edits', 'one_edit_zero_pair_term', 'two_edit_no_lower_penalty_alternative',
              'two_edit_finite_crossover', 'changed_options', 'strict_crossover_below_registered_weight',
              'crossover_equal_registered_weight')
    if (data['registered_pair_weight'] != .10 or len(sources) != 13 or data['edited_identity_views'] != 36
            or tuple(summary[k] for k in fields) != (198, 366, 30, 10, 158, 0, 0, 0)
            or any(sum(r.get(k, 0) for r in sources) != summary[k] for k in fields)):
        raise ValueError('Unexpected frozen-query counts')
    thresholds = [r['finite_crossover_quantiles'][0] for r in sources if r['finite_crossover_quantiles']]
    minimum = summary['finite_crossover_quantiles'][0]
    if minimum != min(thresholds) or minimum <= data['registered_pair_weight']:
        raise ValueError('Observed first crossover contradicts the fixed weight')
    return dict(edited_queries=summary['queries'], one_edit=summary['one_edit_zero_pair_term'],
                no_lower_penalty=summary['two_edit_no_lower_penalty_alternative'],
                finite_crossovers=summary['two_edit_finite_crossover'], changed_options=summary['changed_options'],
                minimum_crossover=minimum, no_pair_penalty_identity_views=data['edited_identity_views'])


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected reviewer archive members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed reviewer archive member')
    step = runpy.run_path(str(root/'equal_step_verify.py'))['replay'](
        json.loads((root/'equal_steps.json').read_text()))
    row = replay(json.loads((root/'breakpoints.json').read_text()))
    if step['additive_identity_views'] != row['no_pair_penalty_identity_views']:
        raise ValueError('Step and breakpoint identity counts disagree')
    step['no_pair_penalty_identity_views'] = step.pop('additive_identity_views')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for name in names:
            (folder/name).write_bytes(z.read(name))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, wepdtof_equal_steps=step, wepdtof_penalty_breakpoints=row)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
