"""Integrate the completed cost-loss repair without changing prior evidence."""
import argparse
import hashlib
import json

from scripts import build_m3w_evidence_manuscript_v4 as previous
from scripts import report_m3w_easy_harm_readout_v3 as report

ROOT, BASE = previous.ROOT, previous.BASE
OUTPUT = BASE/'evidence_manuscript_v5'
READOUT = 'european_easy_harm_deviance_v1/readout_verified_v3/'
SOURCES = {
    'summary': (READOUT+'summary.json', 'bd26d64b345d38c8d191978b11f6f058149e211cda0af32bff3b1d98f8896239'),
    'complete': (READOUT+'complete.json', 'd8122ac408794b7d26839d20866d1ff6d15314c4aee9cb607433058953a039ae'),
    'verification': (READOUT+'report_verification.json', 'd11659e3d8d0525ec71d25732c7a737fd4e15e96cccf60d2cf189733cc2b65d5'),
    'diagnosis': (READOUT+'selected_harm_diagnosis.json', '3d50efc8714d0b1d627ef1dd8b8919186537810f72f1777d5e1d49995e6286ab'),
    'disclosure': ('llm_usage_statement_draft.md', '77d0b7d4797b98ac848fadd5eba0cbeda82a1d7c1ddeeda9fc47db2b4376db63'),
    'template': ('evidence_manuscript_v4/manuscript.template.md', 'acee5c36335e446edf975576e0d4bb1a5f649b8a42f72332c3e3857eb8d218c7'),
}
require = previous.require


def load_sources():
    docs = {}
    for key, (path, digest) in SOURCES.items():
        raw = (ROOT/BASE/path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, 'Changed source: '+path)
        docs[key] = json.loads(raw) if path.endswith('.json') else raw.decode()
    old = previous.artifacts(*previous.load_sources(), docs['template'])
    complete = docs['complete']
    require(len(complete['groups']) == 72, 'Complete fixed grid required')
    seen = set()
    for ref in complete['groups']:
        path = (ROOT/ref['path']).resolve()
        require(path.is_relative_to((ROOT/BASE/READOUT/'groups').resolve())
                and path not in seen, 'Foreign or duplicate group')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], 'Changed group')
        seen.add(path)
    return old, docs


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Missing or duplicate manuscript anchor: '+old[:70])
    return text.replace(old, new, 1)


def artifacts(old, docs):
    s, c, v, d = [docs[k] for k in ('summary', 'complete', 'verification', 'diagnosis')]
    require(s['groups'] == 72 and s['localities'] == 12 and s['trained_cost_heads'] == 144,
            'Changed loss-comparison population')
    require(s['advance_to_transfer_design'] is False and v['advance'] is False,
            'Do not promote the failed loss repair')
    require(c['summary_sha256'] == SOURCES['summary'][1]
            and v['complete_sha256'] == SOURCES['complete'][1]
            and d['complete_sha256'] == SOURCES['complete'][1], 'Unlinked evidence')
    require(d['budget'] == .02 and d['identity_checks'] == 432, 'Changed diagnosis contract')
    require(s['bootstrap_resamples'] == 3000 and v['summary_recomputed_exact'], 'Unverified statistics')
    for item in (c, v, d):
        require(item['deployment_changed'] is False, 'Deployment role changed')
    require(c['new_training'] is False and c['independent_roles_read'] is False
            and d['hypothesis_test'] is False and d['causal_attribution'] is False,
            'Do not relabel readout or descriptive diagnosis')
    require(s['safety']['easy_deviance']['easy_upper_violations'] == 71
            and s['safety']['easy_deviance']['known_easy_violations'] == 59
            and d['arms']['easy_deviance']['harm_underpredicted_views'] == 72,
            'Do not suppress selected-risk failure')
    contrasts, safety = report.tables(s)
    paper = old['manuscript.md']
    paper = replace_once(paper,
        'English development-evidence manuscript, 7 October 2026. This revision integrates\n'
        'the completed temporal-auxiliary experiment and its TRAIN replay. It replaces the\n'
        'pending-experiment narrative in the previous draft without rewriting that dated\n'
        'record. It is not an anonymous or submission-ready manuscript.',
        'English development-evidence manuscript, 28 October 2026. This revision adds the\n'
        'completed one-factor easy-harm loss comparison and selected-risk diagnosis. Earlier\n'
        'dated exports remain unchanged. It is not an anonymous or submission-ready manuscript.')
    paper = replace_once(paper,
        'explanation. These controlled negative results distinguish prediction quality,',
        'explanation. A subsequent 144-head matched loss comparison replaces only the easy-harm\n'
        'squared loss with continuous-cost deviance. Its full paired utility falls by\n'
        '0.902716 percentage points versus the quadratic control, and 71 of 72 views\n'
        'violate the completion-upper easy-risk budget. These controlled negative results distinguish prediction quality,')
    paper = replace_once(paper,
        'TRAIN-selected cases. This motivates a targeted loss experiment, but\ndoes not prove that modifying a loss will repair the policy.',
        'TRAIN-selected cases. A completed one-factor loss experiment improves neither the required\n'
        'matched utility contrast nor the risk gate; changing this loss alone is insufficient.')
    paper = replace_once(paper,
        'The aggregate export underlying this revision\ndoes not reopen trajectory labels or perform new model inference.',
        'This manuscript export reuses the completed, fixed-final development readout\n'
        'and does not itself reopen trajectory labels or perform new model inference.')
    paper = replace_once(paper, '## 5. Experiments and Results', '''### 4.6 One-factor easy-harm cost-loss repair

The successor keeps the five-moment encoder, frozen forecast candidates, TRAIN
preprocessing, query draws, optimizer budget and decision rule fixed. Only the
fifth positive-easy-harm moment loss changes from quadratic error to normalized
continuous-cost deviance. There is no temporal auxiliary in either arm. Each of
24 TRAIN contexts has three head seeds and two arms, giving 144 fixed-final heads.
This tests one loss repair, not end-to-end trajectory training or risk calibration.

The pre-readout control amendment retains 54 historical-exact quadratic controls
and 18 exact original-implementation replays. Historical bitwise reproduction of
all controls failed, and its numerical cause remains unresolved. The amendment
does not erase that limitation. Recovery trained 17 missing candidates; the final
readout did not freshly train all 144 heads. Full and same-count comparisons
against both quadratic and original controls remain the four primary contrasts.

## 5. Experiments and Results''')
    new_section = f'''### 5.5 Changing the easy-harm loss does not repair the policy

The fixed six-arm readout is complete: 72 source/head views, 12 localities and
3,000 paired-locality bootstrap draws, with unchanged unknown-outcome handling.

{contrasts}

**Table 7.** Easy-harm deviance minus each retained comparator. The signs, units,
locality aggregation and nominal development interpretation match Table 4.
The quadratic full contrast is negative with its interval below zero. The
quadratic same-count contrast includes zero. Positive comparisons to the original
forest do not replace either failed primary comparison. Original-selected MSE
contrasts remain undefined in the complete readout; missing views are not dropped.

{safety}

**Table 8.** All full and matched policies retain their selected-risk and support
failures. Deviance has 59/72 known-label violations, 71/72 completion-upper
violations and only one fully supported finite view. Selecting fewer total
occurrences than the quadratic arm does not establish safety. These counts do
not represent 72 independent sites.

A descriptive post-readout accounting finds easy-harm underprediction in all
72 quadratic and all 72 deviance views. The median predicted/observed harm ratio
is 0.014273 for quadratic and 0.048558 for deviance; neither ratio is calibrated.
Median known selected easy-risk ratios are 7.382428% and 5.517979%, respectively,
both above the 2% budget. These medians concern each policy's own selected
population and are not causal estimates of changing the loss. Missing-label
conservatism cannot explain the already-observed known-label violations.

Using the Section 4.5 identity on development-selected known rows gives, for
deviance, -1.081269 pp predicted slack +12.295870 pp harm underprediction
-0.568737 pp reference overprediction = +10.645864 pp realized excess after
equal-locality averaging. All 432 per-view identities verify. This is a
descriptive analysis of cached aggregates, not a new inferential test. It
narrows the next target to selected-population objective mismatch; it does not
identify a unique causal failure or prove that decision-focused learning works.

'''
    paper = replace_once(paper, '## 6. Uncertainty and Failure Analysis',
                         new_section+'## 6. Uncertainty and Failure Analysis')
    start = paper.index('A successor experiment changes only the easy-positive-harm moment loss')
    end = paper.index('The present intervention is deterministic selection', start)
    paper = paper[:start]+'''The completed easy-harm deviance experiment fails the original advancement
rule. It is not promoted to transfer or deployment. The 54+18 control provenance
is retained, including incomplete historical bitwise reproduction. A fresh
allocated verification job rehashed all 144 final checkpoint payloads and 145
metadata files before readout. Historical scheduler accounting remains unavailable;
an explicit execution-only amendment uses the new verification job's actual
successful exit instead of inventing the historical job status. Scientific
inputs, comparisons and thresholds did not change.

'''+paper[end:]
    paper = replace_once(paper,
        '216 training receipts and 72 readout group receipts. It integrates the earlier\n'
        'development tables, completed temporal contrasts and TRAIN diagnostic into one\n'
        'manuscript.',
        '216 temporal training receipts and 72 temporal readout receipts, plus six new\n'
        'pinned loss-repair/report/template sources and 72 loss-repair group receipts.\n'
        'It integrates completed development and descriptive TRAIN evidence; the\n'
        'newly reported scientific readout is distinct from this cached-result export.')
    paper = replace_once(paper,
        'Original failures, execution amendments and support failures remain accessible.',
        'The loss-repair readout reverified all frozen model payloads, exact repeated\n'
        'inference and strong-control outputs, with 9,656,592 scalar checks. This\n'
        'algorithmic verification does not create independent observations.\n\n'
        'Original failures, execution amendments and support failures remain accessible.')
    paper = replace_once(paper,
        'underestimated before unseen-domain evaluation. A successful method must close',
        'underestimated before unseen-domain evaluation. Replacing one easy-harm loss\n'
        'with deviance does not resolve the selected-risk failure or beat the matched\n'
        'quadratic policy. A successful method must close')
    disclosure = docs['disclosure'].split('\n---\n', 1)[0].strip()
    require(disclosure.startswith('# LLM Usage Statement'), 'Disclosure heading required')
    paper = replace_once(paper, '## References', '#'+disclosure+'\n\n## References')
    paper = paper.replace('](figures/', '](../evidence_manuscript_v4/figures/')
    evidence = dict(assembly_source='fresh_run', numerical_results_source='cached_verified',
        prior_evidence=json.loads(old['evidence.json']),
        source_hashes={str(BASE/p):h for p, h in SOURCES.values()},
        completed_loss_comparison=s, selected_harm_diagnosis=d,
        new_training=False, new_inference=False, new_bootstrap=False,
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    return {'manuscript.md':paper, 'evidence.json':json.dumps(evidence, indent=2, allow_nan=False)+'\n',
            'tables.md':old['tables.md']+'\n\n# Completed Loss Repair\n\n'+contrasts+'\n\n'+safety+'\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = artifacts(*load_sources())
    folder = ROOT/OUTPUT
    if not args.check:
        folder.mkdir(parents=True, exist_ok=True)
    for name, text in content.items():
        path = folder/name
        if args.check:
            require(path.read_bytes() == text.encode(), 'Changed export: '+name)
        else:
            path.write_bytes(text.encode())
    print(json.dumps(dict(status='checked' if args.check else 'exported', tables=8,
                          new_inference=False, submission_ready=False)))


if __name__ == '__main__':
    main()
