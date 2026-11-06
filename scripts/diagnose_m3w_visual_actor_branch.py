"""Post-readout branch diagnostics; no new model choice or outcome evaluation."""
import hashlib
import json
from pathlib import Path
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_visual_actor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_visual_actor_v1'
SOURCE = ROOT/'outputs/publication_readiness_2026_09/moving_producer_crossfit_v1'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(ref):
    path = ROOT/ref['path']
    if sha(path) != ref['sha256']:
        raise ValueError('Changed diagnostic input: '+ref['path'])
    return path


def moments(x, state, ids):
    z = (torch.from_numpy(np.asarray(x[ids]).copy())-state['image_mean'])/state['image_std']
    projected = F.linear(F.silu(F.linear(z.clamp(-8, 8),
        state['image_encoder.0.weight'])), state['image_encoder.2.weight'])
    response = torch.tanh(projected)
    return dict(rows=len(ids), input_element_clipped_fraction=float((z.abs() > 8).float().mean()),
        rows_with_any_clipped_input_fraction=float((z.abs() > 8).any(1).float().mean()),
        branch_saturated_fraction=float((response.abs() >= .95).float().mean()),
        offset_mean_absolute=float((.1*response).abs().mean()),
        offset_max_absolute=float((.1*response).abs().max()),
        first_weight_norm=float(state['image_encoder.0.weight'].norm()),
        final_weight_norm=float(state['image_encoder.2.weight'].norm()))


def main():
    if not list((PRIVATE/'replays').glob('replay_*.json')):
        raise ValueError('Finish exact original prediction/scalar replay before posthoc diagnostics')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    population = read(PUBLIC/'population.json')
    source = read(SOURCE/'population.json')
    sites = np.load(checked(source['files']['sites']), mmap_mode='r', allow_pickle=False)
    ids = np.load(checked(population['files']['global_ids']), allow_pickle=False)
    np.testing.assert_array_equal(ids, np.load(checked(source['files']['global_ids']), allow_pickle=False))
    contexts = {a: np.load(checked(population['files'][k]), mmap_mode='r', allow_pickle=False)
        for a, k in (('quality_geometry', 'quality'), ('motion_geometry', 'motion'))}
    a = read(PUBLIC/'analysis.json')
    effect = {r['source']: r['summary'][-1]['paired_known_pp'] for r in a['sources'] if r['contrast'] == 'primary'}
    jobs, sources, bindings = [], {}, {}
    with torch.no_grad():
        for path in sorted((PUBLIC/'fits').glob('*.json')):
            fit = read(path)
            saved = torch.load(checked(fit['checkpoint']), map_location='cpu', weights_only=False)
            state = saved['model']; arm = fit['arm']; x = contexts[arm]
            fitting = moments(x, state, saved['train_ids'])
            donor_ids = np.flatnonzero(np.isin(sites, fit['job']['donor_sites']))
            donor = moments(x, state, donor_ids)
            jobs.append(dict(job=fit['job']['key'], arm=arm, fitting=fitting, donor=donor,
                final_fitting_minibatch_objective=fit['result']['losses'][-1]['loss']))
            bindings[str(path.relative_to(ROOT))] = sha(path)
            for site in fit['job']['donor_sites']:
                sources.setdefault(site, {})[arm] = moments(x, state, np.flatnonzero(sites == site))
    for site, values in sources.items():
        values['paired_known_pp'] = effect[site]
    result = dict(result_source='fresh_posthoc_checkpoint_and_past_input_diagnostic',
        diagnostic_script_sha256=sha(Path(__file__)),
        primary_analysis_sha256=sha(PUBLIC/'analysis.json'), fitting_receipts=bindings,
        jobs=jobs, sources=sources, target_arrays_opened=False,
        training_objectives_read_from_existing_public_receipts=True,
        new_outcome_scoring=False, new_model_or_threshold_selection=False,
        independent_confirmation=False, diagnostic_not_causal_ablation=True)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    output = PUBLIC/'branch_diagnostics.json'
    if output.exists() and output.read_text() != text:
        raise ValueError('Changed deterministic posthoc diagnosis')
    output.write_text(text)
    lines = ['# Posthoc Image-Branch Diagnosis', '',
        'All fourteen donors remain. These diagnostics inspect past inputs and',
        'frozen checkpoints after the primary screen, not new targets. Clipping',
        'or saturation associations are not causal explanations or retrained',
        'ablations. Fitting loss is not validation performance.', '',
        '| Source | Motion - quality known pp | Quality clipped inputs % | Motion clipped inputs % | Quality saturation % | Motion saturation % |',
        '|---|---:|---:|---:|---:|---:|']
    for site, row in sorted(sources.items()):
        q, m = row['quality_geometry'], row['motion_geometry']
        lines.append(f'| {site} | {row["paired_known_pp"]:+.4f} | {100*q["input_element_clipped_fraction"]:.4f} | '
            f'{100*m["input_element_clipped_fraction"]:.4f} | {100*q["branch_saturated_fraction"]:.4f} | {100*m["branch_saturated_fraction"]:.4f} |')
    lines += ['', '| Fitting design | Quality final minibatch loss | Motion final minibatch loss |',
        '|---|---:|---:|']
    for job in sorted({r['job'] for r in jobs}):
        q, m = [next(r for r in jobs if r['job'] == job and r['arm'] == arm) for arm in contexts]
        lines.append(f'| {job} | {q["final_fitting_minibatch_objective"]:.6f} | {m["final_fitting_minibatch_objective"]:.6f} |')
    nonzero = all(r['fitting']['final_weight_norm'] > 0 for r in jobs)
    lines += ['', f'All twelve trained final projections are nonzero:{nonzero}. The offset is bounded',
        'in latent feature units, not image trajectory error. A nonsaturated',
        'branch can still be harmful; a large input clip fraction does not prove',
        'it causes the held-source error. No new deployment or risk-rule change.']
    (PUBLIC/'branch_diagnostics.md').write_text('\n'.join(lines)+'\n')
    print('Posthoc frozen branch diagnostics complete; no new outcomes or selection.')


if __name__ == '__main__':
    main()
