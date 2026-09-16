# Causal Recording Integrity and Counterfactual Checks

Fresh checks on 142402 indexed windows, with 144 sampled real-recording counterfactual cases.

| recording | windows checked | future-corruption cases | checks passed |
| --- | ---: | ---: | --- |
| eth_eth | 2614 | 16 | True |
| eth_hotel | 8945 | 16 | True |
| ucy_zara01 | 11843 | 16 | True |
| ucy_zara02 | 26873 | 16 | True |
| ucy_zara03 | 4320 | 16 | True |
| ucy_students01 | 21384 | 16 | True |
| ucy_students03 | 64020 | 16 | True |
| ucy_arxiepiskopi1 | 1440 | 16 | True |
| pets09_s2l1 | 963 | 16 | True |

Source positions were reconstructed independently of legacy teacher/features and compared to raw input. All indexed history/future spans were checked for one agent, continuous timestamps and exact horizon. Artifact hashes were verified before use.

For sampled real windows, every position after the current frame, for every agent, was replaced by NaN. All inference inputs, causal baseline rollouts and neighbor features had to remain exactly unchanged. This is a causal-access test, not a prediction accuracy experiment.

In 36 additional scene queries, target-agent membership and all scene inputs also remained unchanged after future corruption. Agents with sufficient observed history are included even if their future is absent; future availability is a separate loss-only mask.

This does not certify a future train/validation/test split, teacher fitting, all possible input perturbations, or independent confirmation. Those steps have not run. No metric/seconds, Stage5C or SMC claim.
