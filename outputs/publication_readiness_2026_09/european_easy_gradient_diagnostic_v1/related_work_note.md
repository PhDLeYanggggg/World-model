# Gradient Diagnostics and Prior Work

Fresh primary-source reading on 28 September 2026. Scope below is partial;
this is not a systematic novelty search or a claim of reading entire papers.

## Gradient Surgery

Yu et al., NeurIPS 2020, define conflict through negative gradient cosine and
project conflicting task gradients. Their analysis also considers magnitude
imbalance and curvature; a negative cosine alone is not evidence of harmful
training. Read scope: Sections 2.2-2.4 and Algorithm 1. Our frozen-state cosine
diagnostic is therefore established methodology, not a novel M3W contribution.
We have not measured curvature or reproduced their sufficient conditions, and
our Euclidean gradient statistics do not establish how AdamW behaved.
[Original paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf).

## Gradient Normalization

Chen et al., ICML 2018, adjust task weights using gradient magnitudes on shared
parameters and relative training rates. Read scope: Sections 3.1-3.2. This
motivates measuring actual shared-layer gradients instead of inferring influence
from loss values. It does not imply that equal norms are optimal for our risk
constraint, or that adaptive weights would fix source shift or selected risk.
Applying loss balancing alone would not establish methodological novelty.
[Original paper](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf).

## Remaining M3W Question

Our unresolved question is useful neural intervention under a fixed reference
forecast and explicit scene-level risk constraints. Neither cited optimization
method certifies the realized selected-harm ratio, identifies unknown outcomes,
or substitutes for independent-source calibration and confirmation. A future
repair must demonstrate count-matched allocation value and unchanged easy-case
protection, not merely reduced loss or more aligned gradients. This diagnostic
performs no optimization, policy selection or held evaluation.
