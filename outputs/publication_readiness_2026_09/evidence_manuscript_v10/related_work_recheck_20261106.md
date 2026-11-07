# Primary-Source Scope And Contribution Recheck

This is targeted reading during the full model run,not a systematic review or
new model result. Sources are official proceedings or author preprints. No
third-party summary supports the claims. The live actor matrix remains unscored;
this reading does not tune its registered endpoints.

## Established Work And Reading Depth

| Primary source | Inspected material | Relevant scope | Not automatically inherited |
|---|---|---|---|
| [Mao et al.,ICML2024](https://proceedings.mlr.press/v235/mao24d.html) | Official abstract/proceedings metadata | Regression expert deferral with continuous costs and single/two-stage consistency analysis | A per-baseline cost head alone is not new;our selected-reference ratios are not its bounded-loss theorem |
| [Shah et al.,ICML2022](https://proceedings.mlr.press/v162/shah22a/shah22a.pdf) | Problem,Definition1and Section4theorem statements | Selective regression can worsen subgroup error;representation conditions support monotone subgroup MSE | Recording-relative positive harm is a different estimand;more fallback not always helping is already known |
| [Liu et al.,ICML2026](https://proceedings.mlr.press/v306/liu26g.html) | Official PDF first four pages,Sections2and3 | Multi-expert classification surrogate aggregation can cause underfitting;PiCCE addresses that setting | Categorical expert identification does not prove failure of a continuous ADE regressor or geometric input representation |
| [Angelopoulos et al.,ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf) | Main algorithm and exchangeable,bounded-above,monotone-loss assumptions | Expected-risk control with its stated conditions | Our selected-agent error/reference ratio can be non-monotone/unbounded;no direct per-recording2%guarantee |
| [Farinhas et al.,ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/de04896f011beff76c91e094f72727f4-Paper-Conference.pdf) | Abstract and main-theorem assumptions | Weighted non-exchangeable control of bounded monotone losses | Distribution shift does not license arbitrary conditional or recording-wise certification |
| [Bahari et al.,CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bahari_Certified_Human_Trajectory_Prediction_CVPR_2025_paper.html) | Official abstract/first-page indexed excerpt;direct CVF access403 | Robust trajectory outputs under bounded input perturbations | Observation-noise certification is not our baseline-relative ADE preservation;no reproduction claimed |
| [Kuipers et al.,CDC2024](https://arxiv.org/html/2403.16871v2) | Formulation and SectionsI/II/IV | Joint trajectory regions under changes in agents' behavior policies,with density-ratio information | Our action changes a forecast,not human behavior;its simulation/coverage result is not our real-data evidence |

The ICML2026PDF is retained only in the ignored private literature directory.
It has25pages;the stated first four were inspected,not the entire paper.
Its official linked PDF was retrieved locally after the web parser rejected
the octet-stream MIME type. No PDF/third-party code is committed.

## Additional Abstract Screening

[Top-k deferral,ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6dc129f0383f9fa270347bb09f0030d4-Abstract-Conference.html)
assigns multiple experts to an example. That is not our one-forecast-per-agent,
coupled scene-budget decision. [Budgeted expert deferral](https://arxiv.org/abs/2510.26706)
reduces expert-query expense during training,not deployment intervention count.
Publication status beyond the accessible preprint was not established here.
Neither screened work is implemented or benchmarked in this project.

## Consequences

The question under investigation is whether past-only observations identify
neural gain and positive harm reliably enough for joint query decisions under
partial labels and source shift. The original2%/support/NA contract has not
been met. Camera compensation,an extra encoder,more fallback or more experts
is not sufficient as a main contribution. Existing consistency/calibration
theorems do not supply the missing empirical evidence.

The synthetic camera example concerns information erased by a one-sided
coordinate map. It is distinct from Liu et al.'s classification expert-
identification issue. It justifies stopping an unscored incomplete trial and
retaining both views. It does not identify the cause of every prior real-data
failure or become a novel general impossibility theorem.

Required method evidence still includes selected-tail gain/harm accuracy,
joint versus independent/query-uniform choices under matched intervention or
risk budgets,and actual combination-consistency diagnostics. The current
joint allocator pools predicted error budgets;it is not a collision optimizer.
Three forecaster seeds and independent confirmation remain missing. No metric,
seconds,physical-safety,foundation or submission-readiness claim is introduced.
Stage5C execution and SMC remain off.

