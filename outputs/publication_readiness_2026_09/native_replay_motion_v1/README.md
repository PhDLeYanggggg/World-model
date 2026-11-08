# Can short causal backtests improve a motion forecast?

I trained a mixture of five causal motion experts: OLS4, the last
displacement, OLS8, median velocity and bounded acceleration. Each
forecast starts at the last observed position. The mixture cannot move
farther from OLS4 than 0.35 times the past RMS step size per predicted
step of lead. The replay arm sees the experts' errors in short forecasts
entirely within the eight observed steps. The history arm sees the same
positions without those replay errors, and the global mixture learns
fixed weights per training fold. These are forecast changes, not another
ranking of frozen interventions.

## Results

Three seeds, nine held fitting sources, three matched arms and 800
updates per fit: 81 fresh Torch fits, 69,746 windows in 1,176 queries.
67,854 windows have at least one future label and 45,330 have all 12.
Paired intervals use 3,000 source resamples after seed averaging.
The table uses the observed CV error denominator; these are points
on that scale, not percent reductions in ADE or FDE.

| Contrast | Available labels | Full-grid lower gain [95% CI] |
| --- | ---: | ---: |
| Replay vs OLS4 | +1.771 | -0.030 [-0.687, +0.603] |
| History only vs OLS4 | +1.535 | -0.345 [-1.631, +0.643] |
| Global mixture vs OLS4 | +0.0004 | -0.255 [-0.717, -0.006] |
| Replay vs history only | +0.237 | -1.188 [-2.419, -0.244] |
| Replay vs global mixture | +1.771 | +0.032 [-0.528, +0.607] |

Available-label ADE/FDE are 88.501/225.579 image pixels for replay and
91.597/234.452 for OLS4. The conservative bound of a direct comparison
is not the difference of two bounds against OLS4. Its negative value
does not prove actual unobserved trajectories were worse. None of
these contrasts establishes a gain from replay.

## What limited the result?

Within the fixed full-grid denominator, known steps contribute
+1.562 points; missing steps cost 1.593 under the worst-case
disagreement bound. Complete windows contribute +1.225 points and
incomplete windows -1.255; these are contributions to the overall
score, not separate subgroup averages. Replay retained 87.9% to 89.7%
mean OLS4 weight and made a median final-step edit of only 0.046 to
0.072 pixels. The finite menu of five experts and ten equal-weight
pairs has +21.332 points in hindsight [15.060, 27.730], using labels
to choose per row. This is not a learned policy or an upper bound
over continuous mixtures. It shows the fixed menu contains helpful
edits but does not show that current inputs can identify them.

The unguarded replay arm is not safe: its worst supported known-easy
positive harm relative to CV ranges from 181.67% to 211.24% across
seeds, and one fitting source lacks an easy denominator in each
seed. This diagnostic is not a recording-level safety certification.
OLS4 remains the reference; I am not promoting this mixture.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_replay_motion.py train
.venv-pytorch/bin/python scripts/report_m3w_replay_motion.py score
.venv-pytorch/bin/python scripts/report_m3w_replay_motion.py replay
.venv-pytorch/bin/python scripts/report_m3w_replay_motion_headroom.py
```

Interrupted training uses `train --resume`. Local native arm64 CPU,
four compute threads, no loader workers, checkpoints every 100
updates. Training took 40.7 seconds. The original readout needed a
[JSON-only repair](readout_amendment.md); all training checkpoints
and predictions were retained. Independent scoring replay matched
the aggregate file exactly, and the post-hoc decomposition ran
twice with the same result. Ten targeted tests passed.

[Protocol](protocol.md) | [Results](analysis.json) |
[Hindsight diagnostic](headroom.json) | [Replay receipt](replay_execution.json)

No external calibration or confirmation outcomes were accessed.
The original 2% rule and sealed 30 DroneCrowd groups are unchanged.
This is 8/12 annotation-step, image-pixel evidence; no metric,
seconds, true 3D, foundation, Stage5C or SMC claim.
