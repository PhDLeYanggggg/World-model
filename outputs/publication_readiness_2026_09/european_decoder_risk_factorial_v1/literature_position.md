# Objective and Decoder Sharing: Literature Position

Primary sources checked on2026-10-29. These papers motivate a diagnosis, not a
claim that their mechanisms have already been demonstrated in M3W.

Kendall, Gal and Cipolla, CVPR2018, show that relative task-loss weighting can
strongly affect multi-task learning and propose homoscedastic-uncertainty
weighting. This experiment does not implement that weighting. Its risk-versus-
utility comparison isolates one fixed penalty contribution while preserving
the same moment/score and decision-regret losses. The evaluation risk budget
does not change. Better fit after removing a penalty would not prove safety.
[Official paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html).

Yu et al., NeurIPS2020, study detrimental interference among task gradients
and introduce PCGrad. M3W does not implement PCGrad here and has not measured
the paper's full interference conditions. Source-specific decoding retains
a shared encoder, so a decoder benefit alone would not establish conflicting
encoder gradients as the cause. Direct gradient attribution would require
a separate TRAIN-only measurement with a fixed objective and batch.
[Official paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

Sharing an encoder, assigning source-specific heads, and changing loss weights
are standard design choices, not a standalone novelty claim. The possible M3W
contribution remains reliable, baseline-relative scene intervention under
limited outcome support and recording/domain transfer. It is unproven until
matched comparisons, safety/support and independent evidence all succeed.

This study has495 extra source-decoder parameters and uses known source
identity. It cannot establish parameter-matched superiority or unseen-source
generalization. Nominal intervals use previously exposed TRAIN recordings.
No new trajectory model, generative latent execution or SMC is involved.
