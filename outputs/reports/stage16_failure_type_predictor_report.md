# Stage 16 Failure Type Predictor Report

- train_records: `473`
- test_records: `122`
- failure AUROC: `0.734694`
- failure AUPRC: `0.609011`
- ECE: `0.136296`
- Brier: `0.215701`
- correction-needed F1: `0.685315`
- residual direction cluster accuracy: `0.229508`
- residual magnitude bucket accuracy: `0.508197`
- hard/failure recall: `1.000000`
- easy false alarm rate: `0.616438`

Inputs are causal past features and baseline rollout diagnostics only; oracle residuals are labels, not inference inputs.
