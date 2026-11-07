# Learned Forecasts Do Not Beat The Matched Causal Menu Conservatively

Both selectors see the same past histories and use the same
source-excluded cost model and ranking rule. The mixed menu adds
EqMotion, Past Context Transformer and the self-kinematic forecast.
The eight forecasts are the fixed native bank; neither menu uses
the additional shared-motion transform. Nothing was refit on the
scored recording or changed in the running external experiment.

| Policy | Actual switches | Learned switches | Recording passes / 102 | Undefined easy risk | All-risk >2% |
|---|---:|---:|---:|---:|---:|
| Causal joint | 9,675 | 0 | 0 | 15 | 102 |
| Mixed joint | 9,675 | 7,110 | 3 | 24 | 87 |
| Causal independent | 9,675 | 0 | 0 | 12 | 102 |
| Mixed independent | 9,675 | 7,132 | 4 | 23 | 88 |
| Causal, own screen | 78 | 0 | 0 | 84 | 18 |
| Mixed, own screen | 90 | 27 | 3 | 84 | 15 |
| Each menu, common-screen control | 56 | 0 | 0 | 88 | 14 |

There are zero count mismatches across the 4,839 repeated query
views. In the primary joint comparison, adding learned forecasts
improves available-label utility by **+0.450 pp** [0.161, 0.797].
The conservative row-support lower contrast is **-0.314 pp**
[-1.183, 0.298], so it does not establish superiority. The analogous
independent comparison gives -0.263 pp [-1.140, 0.373] for the lower
contrast. The source-paired intervals use 3,000 draws of the 14
source groups after averaging the three forecast seeds. Neither
joint policy establishes positive conservative utility over OLS4.

The mixed joint arm also worsens relative-path error: the observed
paired ADE reduction relative to the causal joint arm is
**-0.390 pixels** [-0.850, -0.060], and the full-grid lower contrast
is -1.853 pixels [-3.365, -0.635]. The full-grid upper mean is
positive, so the lower envelope does not prove a loss under every
completion of missing pair steps.

Available-label ADE changes from 32.739 to 32.670 pixels and
available-endpoint FDE from 70.004 to 69.787. The corresponding
hard ADE improves from 108.371 to 108.027, but easy ADE worsens
from 7.547 to 7.562 and worst signed recording easy degradation
increases from 81.269% to 132.430%. These pooled errors are
descriptive, based on the same 368,148 known rows (278,448 observed
endpoints) across repeated seeds, and do not replace recording checks.

## What The Screen Keeps

The own-screen comparisons operate at different switch counts.
The mixed arm improves the lower utility against the causal arm
by just +0.00203 pp [0.00006, 0.00540] and still only passes
3/102 recording views. The 56 common-screen switches are an
equal-count comparison with real, nonempty support: the frozen
choices match exactly in all three seeds, and every choice is a
causal path. This equality gives zero measured utility difference,
not a risk pass or an exemption for unsupported views.

The previous [hindsight bank ablation](../native_neural_bank_ablation_v1/README.md)
found one recording where the learned menu unlocks positive capacity.
This new control does not show that the past-only selector can use
such opportunities conservatively. The next method comparison should
predict incremental benefit over the selected causal option while
preserving the original CV-based harm rule. An extra candidate
family or a lower available-only ADE is not sufficient evidence.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_causal_menu_control.py freeze
.venv-pytorch/bin/python scripts/run_m3w_causal_menu_control.py score
.venv-pytorch/bin/python scripts/run_m3w_causal_menu_control.py replay
~~~

The new choices freeze in 14 seconds, the new score takes 20 seconds,
and an independent re-score from those choices reproduces the
same aggregate SHA-256. It also verifies the original mixed joint,
mixed independent, screened mixed, CV and OLS4 risk summaries exactly.
Nine targeted tests pass. The forecast bank and coefficients are
verified cache; choice inference and scoring are fresh. No new model
training, external access, deployment, Stage5C or SMC.
This is the 8/12 native annotation-step development task in pixels.

[Protocol](protocol.md) |
[Registration](registration.json) |
[Metrics](analysis.json) |
[Freeze](freeze_execution.json) |
[Replay](replay_execution.json)
