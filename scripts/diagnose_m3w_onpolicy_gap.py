"""Compare in-sample donor fit with frozen held-location choices; no retuning."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_onpolicy_donors as trial
from scripts.audit_m3w_incumbent_slate import probabilities

run, np, torch = trial.run, trial.np, trial.torch


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    freeze = run.read(trial.OUT/'head_freeze.json')
    checked = run.read(trial.OUT/'audit.json')
    totals = {k: Counter() for k in trial.previous.MODELS}
    cells = {}
    probability = {k: dict(anchor=[], chosen=[]) for k in totals}
    for args, _ in trial.previous.matched.context_cells():
        held, seed, width = args[0], args[1], args[6]
        key = f'{held}_{seed}'
        cell = freeze['cells'][key]
        feature_width = 4*args[5][0].shape[-1]+10
        nets = {k: trial.study.load(m, lambda: trial.model.ListRanker(feature_width), 400)
                for k, m in cell['models'].items()}
        counts = {k: Counter() for k in nets}
        for bound in cell['models']['point']['identity']['data'].values():
            meta = run.read(run.checked(bound))
            with np.load(run.checked(meta['artifact']), allow_pickle=False) as z:
                feat = torch.from_numpy(z['x'])
                valid = torch.from_numpy(z['valid'])
                labels = torch.from_numpy(np.column_stack((z['anchor'], z['y'])))
            for start in range(0, len(feat), 128):
                x, v, y = feat[start:start+128], valid[start:start+128], labels[start:start+128]
                with torch.inference_mode():
                    x = trial.exp.design(x.flatten(0, 1), width, 'aligned').reshape_as(x)
                    x = trial.model.with_incumbent(x)
                    mask = torch.cat((torch.ones((len(v), 1), dtype=torch.bool), v), dim=1)
                    for k, net in nets.items():
                        logits = net(x).masked_fill(~mask, -torch.inf)
                        j = logits.argmax(1)
                        chosen = y[torch.arange(len(y)), j]
                        anchor = y[:, 0]
                        counts[k].update(slates=len(x), switches=int((j != 0).sum()),
                            anchor_positive=int(anchor.sum()), chosen_positive=int(chosen.sum()),
                            rescued=int((~anchor & chosen).sum()), destroyed=int((anchor & ~chosen).sum()),
                            oracle_positive=int((y & mask).any(1).sum()),
                            missed_feasible=int(((y & mask).any(1) & ~chosen).sum()))
                        p = logits.sigmoid()
                        probability[k]['anchor'].extend(zip(p[:, 0].tolist(), anchor.tolist(), strict=True))
                        probability[k]['chosen'].extend(zip(p[torch.arange(len(y)), j].tolist(), chosen.tolist(), strict=True))
        cells[key] = {k: dict(c) for k, c in counts.items()}
        for k in totals:
            totals[k].update(counts[k])
    result = dict(code=run.artifact(Path(__file__)), freeze=run.artifact(trial.OUT/'head_freeze.json'),
        audit=run.artifact(trial.OUT/'audit.json'), train_counts={k: dict(v) for k, v in totals.items()},
        train_score_diagnostics={k: {name: probabilities(v) for name, v in rows.items()} for k, rows in probability.items()},
        held_score_diagnostics=checked['sigmoid_score_diagnostics'], per_cell=cells,
        label='original_dual_pass_not_full_grid', train_role='in_sample_repeated_donors',
        held_role='outer_location_exposed_development', independent_confirmation=False,
        new_training=False, no_threshold_search=True, deployment_changed=False)
    run.once(trial.OUT/'generalization_diagnostic.json', run.stable(result))
    print(dict(counts=result['train_counts'],
        probabilities={k: {name: {f: z for f, z in v.items() if f != 'bins'} for name, v in rows.items()}
                       for k, rows in result['train_score_diagnostics'].items()}), flush=True)


if __name__ == '__main__':
    main()
