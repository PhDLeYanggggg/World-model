"""Verify nested training lineage and replay frozen policy-matched choices."""

from collections import Counter
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_onpolicy_donors as trial
from scripts.audit_m3w_incumbent_slate import probabilities

run, np, torch = trial.run, trial.np, trial.torch


def verify_producers(reg):
    sites = tuple(trial.parents.parent.source.SITES)
    counts, ranks, teachers = set(), set(), set()
    teacher_stats = Counter()
    frozen = run.read(trial.OUT/'producer_freeze.json')
    if frozen['registration'] != reg or len(frozen['cells']) != 63:
        raise ValueError('Incomplete producer freeze')
    for artifact in frozen['cells'].values():
        receipt = run.read(run.checked(artifact))
        m = receipt['model']
        trial.roles.verify_role(m['identity'], sites, receipt['role']['omitted'])
        state = torch.load(run.checked(m['artifact']), weights_only=True, map_location='cpu')
        assert state['identity'] == m['identity'] and state['step'] == 500
        assert state['identity']['registration'] == reg
        assert m['identity']['teachers'] == receipt['dependencies']
        assert np.isfinite(state['trace']).all()
        counts.add(m['artifact']['path'])
        excluded = receipt['role']['omitted']
        assert set(receipt['dependencies']) == set(trial.roles.fit_roster(sites, excluded))
        for target, dep in receipt['dependencies'].items():
            tm = run.read(run.checked(dep))
            ident = tm['identity']
            assert ident['label_site'] == target and ident['omitted'] == excluded
            assert ident['cutoff_fit'] == trial.roles.fit_roster(sites, (*excluded, target))
            rank = torch.load(run.checked(ident['rank']), weights_only=True, map_location='cpu')
            trial.roles.verify_role(rank['identity'], sites, (*excluded, target))
            assert rank['identity']['registration'] == reg and rank['step'] == 700
            assert rank['identity']['seed'] == receipt['role']['seed'] == ident['seed']
            assert np.isfinite(rank['trace']).all()
            ranks.add(ident['rank']['path'])
            run.checked(tm['artifact'])
            teachers.add(tm['artifact']['path'])
            teacher_stats.update(tm['audit'])
    assert (len(ranks), len(counts), len(teachers)) == (105, 63, 315)
    return dict(rank_models=len(ranks), count_models=len(counts), teacher_caches=len(teachers),
                updates=105*700+63*500, teacher_recording_status=dict(teacher_stats))


def donor_counts(freeze):
    total, by_cell = Counter(), {}
    for key, cell in freeze['cells'].items():
        model = cell['models']['point']['identity']
        held = model['held']
        counts = Counter()
        for site, artifact in model['data'].items():
            meta = run.read(run.checked(artifact))
            ident = meta['identity']
            assert ident['held'] == held and ident['donor'] == site and held != site
            assert ident['cutoff_fit'] == trial.roles.fit_roster(trial.parents.parent.source.SITES, (held, site))
            for k in ('count', 'rank'):
                net = torch.load(run.checked(ident[k]), weights_only=True, map_location='cpu')
                trial.roles.verify_role(net['identity'], trial.parents.parent.source.SITES, (held, site))
            with np.load(run.checked(meta['artifact']), allow_pickle=False) as z:
                for y, valid, anchor in zip(z['y'], z['valid'], z['anchor'], strict=True):
                    choices = y[valid]
                    positives = int(anchor)+int(choices.sum())
                    counts.update(slates=1, options=len(choices), anchor_positive=int(anchor),
                        option_positive=int(choices.sum()), all_negative=int(positives == 0),
                        all_positive=int(positives == len(choices)+1),
                        heterogeneous=int(0 < positives < len(choices)+1),
                        could_rescue=int(not anchor and choices.any()), could_harm=int(anchor and not choices.all()))
            assert meta['audit'] == cell['donor_audit'][site]
        assert (counts['slates'], counts['options'], counts['anchor_positive']) == (
            cell['donor_slates'], cell['donor_options'], cell['incumbent_positives'])
        by_cell[key] = dict(counts)
        total.update(counts)
    return dict(total=dict(total), by_cell=by_cell, independent_samples=False,
                reason='Repeated donors across outer folds and seeds')


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, _ = trial.register()
    producer_audit = verify_producers(reg)
    freeze = run.read(trial.OUT/'head_freeze.json')
    readout = run.read(trial.OUT/'readout.json')
    run.checked(readout['freeze'])
    assert readout['registration'] == freeze['registration'] == reg
    total, per_cell, traces = Counter(), {}, []
    calibration = {k: dict(anchor=[], chosen=[]) for k in trial.previous.MODELS}
    margins = {k: [] for k in trial.previous.MODELS}
    for args, _ in trial.previous.matched.context_cells():
        held, seed, _, d, bank, menu, _, _, cut = args
        key = f'{held}_{seed}'
        cell = freeze['cells'][key]
        width = 4*menu[0].shape[-1]+10
        nets = {k: trial.study.load(m, lambda: trial.model.ListRanker(width), 400)
                for k, m in cell['models'].items()}
        for m in cell['models'].values():
            assert m['identity']['held'] == held and held not in m['identity']['train']
            assert np.isfinite(m['trace']).all()
            traces.extend(m['trace'])
        picks = run.read(run.checked(cell['choices']))['picks']
        assert trial.previous.choose(args, nets) == picks
        counts = Counter()
        for row, chosen in zip(trial.previous.prior.slate_rows(args), picks, strict=True):
            if chosen is None:
                continue
            ai, r, anchor, pairs, _, features = row
            outcomes, full = [], []
            for plan in [anchor, *(trial.old.model.exchanged(anchor, *p) for p in pairs)]:
                prediction, s = trial.old.measure(d, ai, bank, menu[3], r, plan, cut)
                outcomes.append(bool(s['dual_positive']))
                full.append(trial.previous.prior.prior.certificate(bank[0][ai], prediction, d['y'][ai],
                                                                   d['valid'][ai], cut[0], s)[0])
            counts.update(eligible=1, anchor_original=int(outcomes[0]), anchor_full_grid=int(full[0]),
                          any_full_grid=int(any(full)))
            with torch.inference_mode():
                x = trial.model.with_incumbent(features[None])[0]
                for k, net in nets.items():
                    logits = net(x)
                    j = chosen[f'{k}_optional']
                    p = logits.sigmoid().numpy()
                    calibration[k]['anchor'].append((float(p[0]), outcomes[0]))
                    calibration[k]['chosen'].append((float(p[j]), outcomes[j]))
                    margins[k].append(float(logits[1:].max()-logits[0]))
                    counts.update({f'{k}_switch': int(j != 0), f'{k}_keep': int(j == 0),
                        f'{k}_rescued': int(not full[0] and full[j]), f'{k}_destroyed': int(full[0] and not full[j]),
                        f'{k}_missed_feasible': int(any(full) and not full[j])})
        per_cell[key] = dict(counts)
        total.update(counts)
    for k in trial.previous.MODELS:
        expected = readout['summary']['arms'][f'{k}_optional']
        assert total[f'{k}_switch'] == readout['summary']['switches'][f'{k}_optional']
        assert total[f'{k}_rescued'] == expected['rescued_full_grid']
        assert total[f'{k}_destroyed'] == expected['destroyed_full_grid']
    output = dict(code=run.artifact(Path(__file__)), readout=run.artifact(trial.OUT/'readout.json'),
        producer_lineage=producer_audit, choice_replay_cells=len(per_cell), selector_loss_values=len(traces),
        all_recorded_losses_finite=True, donor_counts=donor_counts(freeze), eligible_counts=dict(total),
        per_cell=per_cell, sigmoid_score_diagnostics={k: {name: probabilities(rows) for name, rows in values.items()}
            for k, values in calibration.items()},
        score_margins={k: np.quantile(v, [0., .25, .5, .75, 1.]).tolist() for k, v in margins.items()},
        new_parameters=0, new_external_outcomes=False, diagnostic_only=True, calibrated_probabilities=False)
    run.once(trial.OUT/'audit.json', run.stable(output))
    invocations = []
    for file in sorted(trial.PRIVATE.glob('events_*.jsonl')):
        events = [json.loads(line) for line in file.read_text().splitlines()]
        last = events[-1]
        invocations.append(dict(log=run.artifact(file), pid=last['pid'], terminal_event=last['phase'],
            command=last.get('command'), seconds=last['seconds'], peak_rss_bytes=last['peak_rss_bytes'],
            updates=last.get('result', {}).get('updates')))
    run.once(trial.OUT/'runtime.json', run.stable(dict(invocations=invocations,
        location='native_arm64_local_cpu', threads=4, interop=1, workers=0,
        pilot_result='full_local_run_feasible', hpc_jobs_submitted=0,
        python=sys.version, executable=sys.executable, architecture=platform.machine(),
        torch=str(torch.__version__), numpy=str(np.__version__))))
    print(dict(producers=producer_audit, donor_counts=output['donor_counts']['total'],
               eligible_counts=dict(total), replayed_cells=len(per_cell)), flush=True)


if __name__ == '__main__':
    main()
