# Fixed-Schedule Control For Risk-Head Optimization

We examined whether constant-step optimization explains the failed transport of
an intervention-risk head. The two outputs regress signed all-agent and easy-
agent risk-budget excess, with conservative supervision for unknown outcomes.
A complete-fitting gradient diagnostic found aligned shared task gradients in
all nine inspected frozen model views; projecting conflicting gradients was
therefore not supported by this diagnostic.

We then trained matched constant and cosine-schedule continuations in three
source-separated contexts. Each model retained the same 32,130 parameters,
causal inputs, warm initialization, labels, loss, sampled queries and 2,000 updates.
The fixed cosine schedule reduced the learning rate from .001 to .00001. Model
weights froze before the exposed nested-C development readout. Constant controls
reproduced earlier weights exactly; a complete treatment refit and prediction
replays were also exact. No reserved-domain confirmation was performed.

| Schedule | Known ADE gain over floor % | Hard gain % | Intervention % | Worst easy-net degradation % | Full risk/support |
|---|---:|---:|---:|---:|---:|
| Constant | 3.6006 | 4.6257 | 15.0144 | 38.9046 | 0/17 |
| Cosine | 3.7549 | 4.7483 | 16.5645 | 44.6540 | 0/17 |

The known paired contrast was +0.1543 pp, with missing-label completion bounds
[+0.0275, +0.2811] pp. These are not sampling confidence intervals. Easy-risk
violations increased from 13 to 15, and both full-fitting task MSEs were worse in
all three contexts. Thus a small development utility gain does not establish
safe transport, and the fixed-schedule repair is not deployed. It does not
demonstrate convergence, rule out benefits from additional training, or establish
irreducible causal-feature insufficiency.

The original 2% budgets and support rules remain unchanged. Seven exposed SOURCE
units and one producer seed are development evidence, not independent tests;
partial observed ADE is not full-roster 12-step ADE. The original 24-group primary
remains not estimable. This is a risk-head optimization control, not a new world-
dynamics contribution. Image-pixel obs8/pred12 has no verified metric/seconds
interpretation. Stage5C/SMC remain disabled and submission_ready remains false.
