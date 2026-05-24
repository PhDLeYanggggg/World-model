# M3W Bounded Optimization Report

This report records the local-small research loop after repairing the PyTorch runtime. It is not a full/medium result and it is not a CCF-A submission-ready claim.

## Runtime

- Initial torch training failed with `OMP: Error #179: Function Can't open SHM failed`.
- The runtime was repaired by forcing sequential CPU execution:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `VECLIB_MAXIMUM_THREADS=1`
  - `MKL_THREADING_LAYER=SEQUENTIAL`
  - `KMP_INIT_AT_FORK=FALSE`
  - `KMP_DUPLICATE_LIB_OK=TRUE`
  - `KMP_AFFINITY=disabled`
  - `KMP_BLOCKTIME=0`
- MPS is not available in the current environment.
- DataLoader multiprocessing remains disabled.

## Trials

| trial | backend | t+50 improvement | hard/failure improvement | easy degradation | failure AUROC | interaction AUROC | outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| torch_small | torch_cpu_sequential | 0.0009519330 | 0.0000501921 | 0.0007841103 | 0.6274410644 | 0.4048163178 | underfit; no JEPA/Transformer contribution |
| torch_small_optimized | torch_cpu_sequential | 0.0959744761 | 0.0702400714 | 0.0202588776 | 0.9208336109 | 0.6383899178 | better t+50/hard, but easy gate narrowly fails and Stage26 is stronger |
| torch_small_hard_focus | torch_cpu_sequential | 0.0797643166 | 0.0459157386 | 0.0081676650 | 0.9346414494 | 0.6537175646 | easy preserved, but hard/failure and t+50 below Stage26 |

## Best Deployable Model

Stage26 remains the best deployable model:

- t+50 improvement: `0.14583655843823773`
- hard/failure improvement: `0.11232167634621226`
- easy degradation: `0.01808836280803794`

## Conclusion

M3W is now a real executable research track with a repaired torch runtime, but it is not yet a CCF-A submission candidate. The strongest result remains the Stage26 feature-complete cost-aware selector. M3W needs a demonstrated JEPA or Transformer downstream lift, a stronger hard/failure result, and ablation evidence before it can be positioned as the main method.
