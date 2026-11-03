# Selected-Ratio And Joint-Prediction Prior Work

Fresh primary-source reading on3November2026 while the new-source inference
continues. No empirical score,training objective or gate is changed here.

## Certification Is Established Prior Work

[Yu and Liu,2026preprint](https://arxiv.org/html/2606.08517v1)
jointly study selected risk,acceptance and utility,including a random acceptance
denominator and non-monotone finite-grid selection. Section3.4requires i.i.d.
certification examples,bounded loss/utility,a separate certification split and
a sufficient sample-size condition. For illustration,the stated
`ceil(32*log(32*m/delta)/pi_min)` is2,068at`m=1,delta=.05,pi_min=.1` and
14,672at`m=15,delta=.05,pi_min=.02`. Here`pi_min`is acceptance,not M3W's2%
harm budget. These are this theorem's sufficient conditions,not an impossibility
result or new M3W gates. Overlapping windows cannot manufacture independent
certification examples. The preprint is not a certificate for our experiment.

[Angelopoulos,2026preprint](https://arxiv.org/abs/2602.20151)
already treats algorithmic stability and non-monotone,multidimensional risk
control. It is already cited in the v9 draft. Non-monotonicity alone is not
an unaddressed novelty claim.

[Hegazy etal.,2026preprint](https://arxiv.org/abs/2610.06432)
analyze false acceptance in adaptive generator-verifier stopping. Its FDR
objective does not automatically certify trajectory error or every-record2%
preservation. Reusing exposed proxy results needs care; the held confirmation
reservation is unchanged. These papers are read,not implemented controls here.

[Liu etal.,ICML2026](https://proceedings.mlr.press/v306/liu26g.html)
study classification underfitting and expert identifiability in multi-expert
deferral. This is a relevant prior,not proof that our continuous easy-harm
regression has the same cause. PiCCE is not implemented in the new study.

## The Local Target Remains Different And Unproved

The implemented M3W risk denominator is selected baseline ADE mass,not simply
the number of accepted examples. Its numerator is baseline-relative positive
harm,with explicit bounds for wholly unlabelled futures. Raw loss may not have
the cited global bounds; source components are not certified i.i.d. sites.
Mapping those assumptions to this target requires a derivation and empirical
evidence,not renaming our predictor a conformal certifier.

## Joint Error Is Not A New Idea

[Weng etal.,ICCV2023](https://arxiv.org/abs/2305.06292)
already motivate joint trajectory metrics beyond per-agent ADE/FDE. Our budget
allocation across agents is not by itself proof of coherent joint futures.
The new past-neighbor relative-motion diagnostic is a separate image-plane
proxy,not their published multimodal JADE/JFDE or a collision certificate.
It can miss an arbitrarily large shared translation,so absolute error remains
required. It cannot rescue a failed primary or safety gate.

The fixed same-proposal shared-motion experiment continues unchanged. Useful
evidence must come from causal information and matched intervention controls,
not a generic novelty claim about routing,ratios,joint metrics or CRC.
