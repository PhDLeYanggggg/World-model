# Observation Noise Is Established Prior Work

Fresh primary-source access:30October2026. This note records reading scope;
it is not a new empirical result or an implemented-comparator claim.

## Detector History And Predictive Uncertainty

[Bhattacharyya,Fritz and Schiele,arXiv:1711.09026v2](https://arxiv.org/pdf/1711.09026)
is the accessed20June2018preprint of *Long-Term On-Board Prediction of People
in Traffic Scenes under Uncertainty*. Pages1-3,including Section3.1,describe
past bounding-box sequences and odometry-conditioned future box prediction;
the method models observation/model uncertainty. Thus using box history or
uncertainty is not new by itself. That mobile-camera task differs from our
fixed SOURCE midpoint task. Its Bayesian sampling/odometry components are not
implemented here and do not authorize generative execution or SMC. We do not
transfer its timing or physical interpretations to our raw-frame data.

## Tracking Corruption And Observation Length

[Monti etal.,arXiv:2203.04781v1](https://arxiv.org/pdf/2203.04781),accessed
9March2022preprint,pp1-3,Sections1,3.2and4,explicitly motivates forecasting
with tracking noise and distills an eight-observation teacher into a shorter-
observation student using spatiotemporal attention. Our fixed8/12protocol
does not reproduce that two-observation experiment or distillation objective.
This literature prevents treating attention,tracking-noise awareness or
history variation as sufficient novelty. The CVF accepted-paper HTML fetch
returned403;the full text read here is the identified preprint,not a certified
semantic comparison with every detail of the accepted version.

## A Recent Noise-Agnostic Framework

The official indexed abstract for [NATRA,ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_NATRA_Noise-Agnostic_Framework_for_Trajectory_Prediction_with_Noisy_Observations_ICCV_2025_paper.html)
describes trajectory forecasting under noisy observations and an explicit
denoising framework. Access scope is indexed official abstract only:direct
full-page access returned403. No full-method claim,implementation,noise
guarantee or performance comparison is inferred from that abstract. This
remaining comparator must be reviewed from the exact cited version before
any manuscript-level technical comparison or reproduction.

## What This Experiment Can Establish

The current512parameter input branch is a controlled repair,not the main
novel method. It tests actual detector-derived inputs under fixed producer
budgets,original roles,matched disabled/current/noise arms and retained
missing labels. Adding such features may improve a proposal without making
its learned selection safe. The main M3W claim still needs a useful neural
increment,causal risk-aware joint decisions and untouched calibration and
confirmation. No paper's accuracy or uncertainty construction supplies our
approved every-record2%contract automatically. Supplied detector confidence
is not a calibrated probability of human-motion correctness or human gold.
