# Metadata-Only Cache Eligibility

No new forecast or future-label readout is used by this matrix.

| Target | Donor context | Blind pooling allowed | Reasons |
|---|---|---|---|
| eqmotion_fold1_seed17_critic0 | eqmotion_fold0_seed17_critic1 | False | donor_teacher_trained_on_target_critic_sources, different_teacher, different_training_defined_target_cut |
| eqmotion_fold1_seed17_critic0 | eqmotion_fold0_seed17_critic2 | False | donor_is_target_outer_H, donor_teacher_trained_on_target_critic_sources, different_teacher, different_training_defined_target_cut |
| eqmotion_fold0_seed17_critic1 | eqmotion_fold1_seed17_critic0 | False | donor_teacher_trained_on_target_critic_sources, different_teacher, different_training_defined_target_cut |
| eqmotion_fold0_seed17_critic1 | eqmotion_fold0_seed17_critic2 | False | donor_is_target_outer_H |
| eqmotion_fold0_seed17_critic2 | eqmotion_fold1_seed17_critic0 | False | different_teacher, different_training_defined_target_cut |
| eqmotion_fold0_seed17_critic2 | eqmotion_fold0_seed17_critic1 | False | donor_is_target_outer_H |
