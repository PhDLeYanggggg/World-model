# Stage32 Cross-Domain Eval Matrix

- source: `fresh_run`

| direction | all | t50 | hard | easy | regret | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDD_train_to_SDD_test | 0.084218 | 0.078622 | 0.084560 | 0.000000 | 4.658241 | 0.050000 |
| SDD_train_to_external_test | -0.337476 | -1.018801 | -0.095699 | 1.376132 | 0.063891 | 0.049780 |
| external_train_to_external_test | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| external_train_to_SDD_test | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.843625 | 0.000000 |
| SDD_external_train_to_SDD_test | 0.045934 | 0.044813 | 0.046300 | 250699.064457 | 5.651678 | 0.050000 |
| SDD_external_train_to_external_test | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
