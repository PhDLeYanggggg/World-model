# Stage 25 Metrics

| model | t50 improvement | hard/failure improvement | easy degradation | harm over fallback | selector regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| global_strongest_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.843625 |
| per_sample_oracle_diagnostic | 0.285146 | 0.264502 | 0.000000 | -6.843625 | 0.000000 |
| stage24_validation_selected_selector | -0.432650 | 0.000000 | 11.328798 | 0.000000 | 29.636851 |
| regret_selector | 0.013573 | 0.010930 | 0.009655 | -0.281194 | 6.562431 |
| soft_label_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.843625 |
| hierarchical_selector | 0.000000 | -0.040336 | 0.068384 | 1.037770 | 7.881394 |
| failure_assisted_selector | 0.010378 | 0.007118 | 0.009405 | -0.183136 | 6.660489 |
| conservative_fallback_selector:regret_selector | 0.013573 | 0.010930 | 0.009655 | -0.281194 | 6.562431 |
| BPSG-MA_v1_fallback | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.843625 |
