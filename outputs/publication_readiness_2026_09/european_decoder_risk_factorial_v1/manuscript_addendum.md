# Exploratory Results: Prediction Accuracy and Safe Intervention Diverge

## Design

We crossed shared versus source-specific output decoding with presence versus
absence of a source-wise empirical-risk penalty. All models retained identical
source normalization, pooled priors, primary moment/score supervision, decision
regret, initialization and per-source query draws. Six fixed contexts, three
cost-head seeds and four arms produced72 fresh fits, each with2,000 updates.
The18 shared-risk controls exactly reproduced prior weights, optimizer states,
dual variables and training traces. Source decoding added495 parameters to the
13,149-parameter shared model, including792 unused auxiliary parameters in both.

All final models were frozen before evaluation. We used original TRAIN packets,
holding out entire recordings internally and fitting normalizers only on the
remaining recordings. All five policies, including unchanged local controls,
were evaluated on72 fit and72 held views. Unknown outcomes were retained in
inference and completion bounds. No independent selection, calibration or
confirmation data were used.

## Findings

Removing risk penalties substantially improved signed-score prediction. On held
recordings the risk-minus-utility MSE difference was5.757[3.125,8.603] for shared
decoding and11.209[3.047,21.856] for source decoding. This improvement did not
produce a safe intervention policy: known easy-risk violations increased from
40 to61/72 and42 to54/72, respectively. Source-specific utility decoding improved
fit MSE over shared utility decoding, but did not improve held MSE; its matched-
count paired completion-lower utility was-0.580pp[-1.101,-0.058].

Compared with local controls, source-utility decoding lowered held signed-score
MSE by2.062[0.957,3.349] while reducing same-count lower utility by0.569pp
[0.110,1.079]. Full completion support was only3/72. These results illustrate
that improved aggregate cost prediction is insufficient for useful risk-limited
intervention. They do not establish an improved forecasting method.

## Interpretation and Limitations

Intervals are nominal95% intervals from3,000 locality bootstrap draws, averaging
repeated contexts/seeds within12 localities. Internal held recordings have prior
research exposure and pooled training couples source estimates. We therefore
do not interpret these intervals as independent confirmation. Utility is a
pair-specific completion-lower contrast normalized by known reference error mass,
not ADE/FDE. The budget constrains positive selected harm, not just net easy error.
Additional decoder capacity and known source identities preclude claims of
parameter-matched or zero-shot generalization. Neither registered repair advanced.

The work uses8 observations and12 predictions at a stride of12 raw frames in
dataset-local coordinates. It does not establish seconds-level, metric, true3D,
foundation-model or physical-safety performance. No latent generative execution
or SMC was run. These findings belong in exploratory failure analysis, not as
the main positive claim of a submission.

For positioning against established multi-task weighting and gradient-interference
work, see [primary-source notes](literature_position.md). Neither uncertainty
weighting nor PCGrad is implemented in this experiment.
