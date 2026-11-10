"""Diagnose frozen incumbent-slate predictions; never select a new policy."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_incumbent_slate as trial

run, np, torch = trial.run, trial.np, trial.torch


def probabilities(rows):
    if not rows:
        return dict(n=0, status='not_estimable')
    p, y = np.asarray(rows, dtype=float).T
    bins, ece = [], 0.
    for i in range(10):
        mask = np.minimum((p*10).astype(int), 9) == i
        if mask.any():
            predicted, observed = float(p[mask].mean()), float(y[mask].mean())
            ece += mask.mean()*abs(predicted-observed)
            bins.append(dict(bin=i, n=int(mask.sum()), predicted=predicted, observed=observed))
    return dict(n=len(p), mean_prediction=float(p.mean()), positive_rate=float(y.mean()),
                brier=float(((p-y)**2).mean()), ece=float(ece), bins=bins)


def donor_counts(freeze):
    total, by_cell = Counter(), {}
    for key, cell in freeze['cells'].items():
        identity = cell['models']['point']['identity']
        held, seed = identity['held'], identity['seed']
        counts = Counter()
        for site, bound in identity['data'].items():
            name = '_'.join(sorted((held, site)))+f'_{seed}_{site}'
            risk_path = trial.parents.PRIVATE/'donors'/(name+'.json')
            slate_path = trial.prior.PRIVATE/'donors'/f'{held}_{site}_{seed}.json'
            raw_path = trial.exp.PRIVATE/'donors'/(name+'.npz')
            assert run.sha(risk_path) == bound['features_and_transition']
            assert run.sha(slate_path) == bound['slate_labels']
            raw_meta = run.read(raw_path.with_suffix('.json'))
            assert run.sha(raw_path) == raw_meta['cache_sha256']
            risk = run.read(risk_path)
            assert risk['identity']['previous'] == run.sha(raw_path.with_suffix('.json'))
            slates = run.read(slate_path)
            option = torch.tensor(risk['target']) == 3
            with np.load(raw_path, allow_pickle=False) as z:
                event = torch.from_numpy(z['event'])
            anchors = trial.model.incumbent_targets(option, event, slates['lengths'])
            offset = 0
            for n, anchor in zip(slates['lengths'], anchors, strict=True):
                choices = option[offset:offset+n]
                positives = int(choices.sum())+int(anchor)
                counts.update(slates=1, options=n, anchor_positive=int(anchor),
                    option_positive=int(choices.sum()),
                    all_negative=int(positives == 0), all_positive=int(positives == n+1),
                    heterogeneous=int(0 < positives < n+1),
                    could_rescue=int(not anchor and choices.any()),
                    could_harm=int(anchor and not choices.all()))
                offset += n
        assert counts['slates'] == cell['donor_slates']
        assert counts['options'] == cell['donor_options']
        assert counts['anchor_positive'] == cell['incumbent_positives']
        by_cell[key] = dict(counts)
        total.update(counts)
    return dict(total=dict(total), by_cell=by_cell,
                independent_samples=False, reason='Donors repeat across outer folds and seeds')


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    freeze = run.read(trial.OUT/'head_freeze.json')
    readout = run.read(trial.OUT/'readout.json')
    assert readout['freeze_sha256'] == run.sha(trial.OUT/'head_freeze.json')
    summaries, overall = {}, Counter()
    calibration = {k: dict(anchor=[], chosen=[]) for k in trial.MODELS}
    score_margin = {k: [] for k in trial.MODELS}
    for args, _ in trial.matched.context_cells():
        held, seed, _, d, bank, menu, _, _, cut = args
        key = f'{held}_{seed}'
        meta = freeze['cells'][key]
        picks = run.read(run.checked(meta['choices']))['picks']
        counts, nets = Counter(), None
        for row, chosen in zip(trial.prior.slate_rows(args), picks, strict=True):
            if chosen is None:
                continue
            ai, r, anchor, pairs, _, features = row
            if nets is None:
                nets = {k: trial.study.load(m, lambda: trial.model.ListRanker(features.shape[-1]+1),
                                           trial.old.STEPS) for k, m in meta['models'].items()}
            plans = [anchor, *(trial.old.model.exchanged(anchor, *p) for p in pairs)]
            outcomes, full = [], []
            for plan in plans:
                p, s = trial.old.measure(d, ai, bank, menu[3], r, plan, cut)
                outcomes.append(bool(s['dual_positive']))
                full.append(trial.prior.prior.certificate(bank[0][ai], p, d['y'][ai],
                                                         d['valid'][ai], cut[0], s)[0])
            counts.update(eligible=1, anchor_original=int(outcomes[0]),
                          anchor_full_grid=int(full[0]), any_full_grid=int(any(full)))
            with torch.inference_mode():
                z = trial.model.with_incumbent(features[None])[0]
                for k, net in nets.items():
                    logits = net(z)
                    predictions = trial.model.choices(logits)
                    assert all(predictions[a] == chosen[f'{k}_{a}'] for a in predictions)
                    p = logits.sigmoid().numpy()
                    ix = predictions['optional']
                    calibration[k]['anchor'].append((float(p[0]), outcomes[0]))
                    calibration[k]['chosen'].append((float(p[ix]), outcomes[ix]))
                    score_margin[k].append(float(logits[1:].max()-logits[0]))
                    counts.update({f'{k}_switch': int(ix != 0),
                        f'{k}_keep': int(ix == 0),
                        f'{k}_rescued': int(not full[0] and full[ix]),
                        f'{k}_destroyed': int(full[0] and not full[ix]),
                        f'{k}_missed_feasible': int(any(full) and not full[ix])})
            counts.update(optional_choice_disagreement=int(chosen['point_optional'] != chosen['pair_optional']))
        summaries[key] = dict(counts)
        overall.update(counts)
    for k in trial.MODELS:
        expected = readout['summary']['arms'][f'{k}_optional']
        assert overall[f'{k}_switch'] == readout['summary']['switches'][f'{k}_optional']
        assert overall[f'{k}_rescued'] == expected.get('rescued_full_grid', 0)
        assert overall[f'{k}_destroyed'] == expected.get('destroyed_full_grid', 0)
    donors = donor_counts(freeze)
    score = {k: {name: probabilities(rows) for name, rows in values.items()}
             for k, values in calibration.items()}
    margins = {k: np.quantile(v, [0., .25, .5, .75, 1.]).tolist() for k, v in score_margin.items()}
    output = dict(readout_sha256=run.sha(trial.OUT/'readout.json'),
        audit_code=run.artifact(Path(__file__)), eligible_counts=dict(overall), by_cell=summaries,
        donor_counts=donors, sigmoid_score_diagnostics=score, best_exchange_minus_anchor_logit_quantiles=margins,
        diagnostic_only=True, calibrated_probabilities=False, new_parameters=0,
        new_external_outcomes=False, source='fresh_posthoc_audit_of_frozen_development_predictions')
    run.once(trial.OUT/'failure_diagnostic.json', run.stable(output))
    print(dict(eligible_counts=dict(overall), donor_counts=donors['total'],
               score_diagnostics={k: {name: {f: v for f, v in values.items() if f != 'bins'}
                                      for name, values in detail.items()} for k, detail in score.items()},
               logit_margins=margins), flush=True)


if __name__ == '__main__':
    main()
