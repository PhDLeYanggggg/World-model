# Risk Training Losses

Only the neural risk objective changed. These are minibatch fitting losses, not held-out accuracy.
Raw MSE and underharm4 loss values are not directly comparable objectives.

| Head | Updates | First loss | Last logged loss | Fit seconds | Known unique rows |
|---|---:|---:|---:|---:|---:|
| neural_complement0_seed17_all | 2000 | 1.67599583 | 0.90017039 | 1.263 | 92036 |
| neural_complement0_seed17_easy | 2000 | 0.00074042 | 0.00209822 | 1.212 | 92036 |
| neural_complement0_seed29_all | 2000 | 1.05887103 | 1.23039007 | 2.051 | 92376 |
| neural_complement0_seed29_easy | 2000 | 0.00311463 | 0.00198180 | 1.443 | 92376 |
| neural_complement0_seed43_all | 2000 | 1.03439260 | 2.26175427 | 1.802 | 92170 |
| neural_complement0_seed43_easy | 2000 | 0.00139159 | 0.00080570 | 1.872 | 92170 |
| neural_complement1_seed17_all | 2000 | 1.28745389 | 0.54249680 | 1.555 | 128437 |
| neural_complement1_seed17_easy | 2000 | 0.01019619 | 0.00241783 | 1.827 | 128437 |
| neural_complement1_seed29_all | 2000 | 2.11603689 | 0.62657654 | 1.688 | 128514 |
| neural_complement1_seed29_easy | 2000 | 0.00380179 | 0.00530943 | 1.605 | 128514 |
| neural_complement1_seed43_all | 2000 | 1.79216480 | 0.55432385 | 1.690 | 128263 |
| neural_complement1_seed43_easy | 2000 | 0.00179345 | 0.00262052 | 2.213 | 128263 |
| neural_complement2_seed17_all | 2000 | 1.04899967 | 0.52147943 | 1.926 | 83023 |
| neural_complement2_seed17_easy | 2000 | 0.00210981 | 0.00109999 | 1.819 | 83023 |
| neural_complement2_seed29_all | 2000 | 1.46586609 | 0.70781302 | 1.812 | 83289 |
| neural_complement2_seed29_easy | 2000 | 0.00118715 | 0.00114671 | 1.824 | 83289 |
| neural_complement2_seed43_all | 2000 | 0.91093969 | 0.85232347 | 2.121 | 83084 |
| neural_complement2_seed43_easy | 2000 | 0.00117967 | 0.00120780 | 1.794 | 83084 |
| damping097_complement0_seed17_all | 2000 | 1.23625112 | 0.76236534 | 1.618 | 92036 |
| damping097_complement0_seed17_easy | 2000 | 0.00056133 | 0.00062718 | 1.370 | 92036 |
| damping097_complement0_seed29_all | 2000 | 0.78966826 | 0.96945369 | 1.759 | 92376 |
| damping097_complement0_seed29_easy | 2000 | 0.00099344 | 0.00057739 | 1.685 | 92376 |
| damping097_complement0_seed43_all | 2000 | 0.86639857 | 2.18576217 | 1.553 | 92170 |
| damping097_complement0_seed43_easy | 2000 | 0.00068639 | 0.00048817 | 1.306 | 92170 |
| damping097_complement1_seed17_all | 2000 | 1.23745751 | 0.49988878 | 1.898 | 128437 |
| damping097_complement1_seed17_easy | 2000 | 0.00182712 | 0.00120347 | 2.477 | 128437 |
| damping097_complement1_seed29_all | 2000 | 2.00198460 | 0.54006964 | 2.068 | 128514 |
| damping097_complement1_seed29_easy | 2000 | 0.00136555 | 0.00152136 | 1.805 | 128514 |
| damping097_complement1_seed43_all | 2000 | 1.56909490 | 0.51747453 | 1.643 | 128263 |
| damping097_complement1_seed43_easy | 2000 | 0.00122116 | 0.00123396 | 1.849 | 128263 |
| damping097_complement2_seed17_all | 2000 | 1.01863110 | 0.51487243 | 1.942 | 83023 |
| damping097_complement2_seed17_easy | 2000 | 0.00129189 | 0.00108396 | 1.435 | 83023 |
| damping097_complement2_seed29_all | 2000 | 1.44677067 | 0.61728191 | 2.070 | 83289 |
| damping097_complement2_seed29_easy | 2000 | 0.00116367 | 0.00096746 | 1.638 | 83289 |
| damping097_complement2_seed43_all | 2000 | 0.89626515 | 0.72134197 | 1.699 | 83084 |
| damping097_complement2_seed43_easy | 2000 | 0.00112453 | 0.00100301 | 1.742 | 83084 |

The 100-update pilot is included in 72,000 updates. Every new head has exactly the old head's training draws, known-label support, preprocessing and initialization constants.
All fits are completed before outer-source readout. No early stopping or checkpoint selection from that readout.
