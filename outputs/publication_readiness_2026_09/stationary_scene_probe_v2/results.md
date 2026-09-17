# Static-Scene Start and Trajectory Probes

Fit-only, eight observed/twelve predicted native steps; no new deployment or confirmation.
72 models per version: 36 start classifiers plus 36 multi-output trajectory regressors.
Initial version and corrected shared-corner version are both retained. No model/threshold selection.

## Corrected Run: Every Fixed Setting

Three-seed means below. Gains are stationary-subset ADE versus CV, not the full forecasting primary result.
Brier differences are absolute scores, not trajectory percentages. Linear seeds are identical.

| Held | Features | Family | AUC | Brier lift | Positive Brier seeds | Unrestricted ADE gain % | Endpoint angle, degrees | Fixed-gate gain % by seed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ETH | pooled | linear | 0.2766 | -0.20060 | 0/3 | -6.827 | 127.34 | +0.00000, +0.00000, +0.00000 |
| ETH | pooled | extra_trees | 0.6692 | -0.03195 | 0/3 | -3.306 | 120.50 | +0.00000, +0.00000, +0.00000 |
| ETH | scene | linear | 0.3744 | -0.03655 | 0/3 | -19.567 | 131.01 | -0.28382, -0.28382, -0.28382 |
| ETH | scene | extra_trees | 0.8048 | +0.00617 | 2/3 | -6.918 | 129.79 | +0.00000, +0.00000, +0.00000 |
| ETH | scene_neighbor | linear | 0.1972 | -0.23922 | 0/3 | -15.358 | 140.38 | +0.00000, +0.00000, +0.00000 |
| ETH | scene_neighbor | extra_trees | 0.5090 | -0.07899 | 0/3 | -4.364 | 137.96 | +0.00000, +0.00000, +0.00000 |
| Hotel | pooled | linear | 0.5041 | -0.10187 | 0/3 | -1345.110 | 72.70 | -663.15672, -663.15672, -663.15672 |
| Hotel | pooled | extra_trees | 0.5003 | -0.00820 | 0/3 | -346.695 | 81.15 | -86.21890, -88.82264, -110.10513 |
| Hotel | scene | linear | 0.5611 | -0.06193 | 0/3 | -1637.599 | 88.97 | -524.09788, -524.09788, -524.09788 |
| Hotel | scene | extra_trees | 0.5387 | +0.02894 | 3/3 | -213.066 | 81.41 | -1.85953, +0.00000, -2.13017 |
| Hotel | scene_neighbor | linear | 0.5810 | -0.03795 | 0/3 | -1973.516 | 79.90 | -639.66435, -639.66435, -639.66435 |
| Hotel | scene_neighbor | extra_trees | 0.5023 | +0.01957 | 3/3 | -252.225 | 83.14 | -1.06397, -1.10486, -0.69870 |

All corrected unrestricted regressors are worse than CV. None of the corrected fixed-gate
regressors has positive ADE gain. Zero gain means fallback, not successful prediction.

## Easy Cases and Intervention

All easy labels in this stationary subset have exactly zero CV error. Percentage easy degradation
is undefined; absolute excess error is reported instead. Positive absolute error violates exact
preservation here. A null ratio must never be turned into a pass. Units below use the unchanged
parent normalizer, not verified meters. Endpoint angles exclude undefined/zero vectors;
all-row ADE still includes them and every zero target.

| Held | Features | Family | Fixed-gate switch rates | Easy absolute harm by seed |
| --- | --- | --- | --- | --- |
| ETH | pooled | linear | 0.0000, 0.0000, 0.0000 | 0.000000, 0.000000, 0.000000 |
| ETH | pooled | extra_trees | 0.0000, 0.0000, 0.0000 | 0.000000, 0.000000, 0.000000 |
| ETH | scene | linear | 0.0247, 0.0247, 0.0247 | 0.000000, 0.000000, 0.000000 |
| ETH | scene | extra_trees | 0.0000, 0.0000, 0.0000 | 0.000000, 0.000000, 0.000000 |
| ETH | scene_neighbor | linear | 0.0000, 0.0000, 0.0000 | 0.000000, 0.000000, 0.000000 |
| ETH | scene_neighbor | extra_trees | 0.0000, 0.0000, 0.0000 | 0.000000, 0.000000, 0.000000 |
| Hotel | pooled | linear | 0.4965, 0.4965, 0.4965 | 150.310755, 150.310755, 150.310755 |
| Hotel | pooled | extra_trees | 0.1162, 0.1162, 0.1408 | 23.569471, 25.538618, 29.531142 |
| Hotel | scene | linear | 0.2746, 0.2746, 0.2746 | 84.269929, 84.269929, 84.269929 |
| Hotel | scene | extra_trees | 0.0070, 0.0000, 0.0070 | 0.721169, 0.000000, 0.826130 |
| Hotel | scene_neighbor | linear | 0.2676, 0.2676, 0.2676 | 114.725734, 114.725734, 114.725734 |
| Hotel | scene_neighbor | extra_trees | 0.0176, 0.0211, 0.0141 | 0.415213, 0.345966, 0.385844 |

## Source and Repair Boundaries

The shared-corner correction removes false directional ambiguity; it does not change any native
future coordinate or held row. The two apparent tiny positive gated results in the initial
Hotel scene-neighbor trees disappear after correction. Both versions are archived.
See ../stationary_scene_probe/corner_repair.md for exact provenance.

ETH has 59 nonzero endpoints among 59 changed futures. Hotel has 92 nonzero endpoints among
129 changed futures: 37 windows change and return to their initial recorded endpoint.
Annotation precision, motion and interpolation are not distinguished by that fact alone.

Reference images contain people with unknown capture time and are NOT encoded. Supplied
destinations/groups and video frames are excluded. The XML is an unverified static proxy.
All ETH points and 82.78% of Hotel points project inside the reference dimensions under
the supplied H convention. This numerical check is not an alignment or calibration certificate.
Dataset-local coordinates and native steps remain unverified metric/time quantities.

All 144 fitted models across both versions replay their saved predictions within 1e-12.
Only 31 agents, 45 stationary runs and two physical fit sites support these experiments.
No independent-scene significance or CI is claimed. The unchanged forecasting primary and
raw50 supplement are not replaced by these diagnostic slices. No Stage5C or SMC.
