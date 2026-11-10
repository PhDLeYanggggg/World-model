# Paper candidate: distinguish forecast replacement from prior decisions

I checked the following primary sources against the actual intervention
in the paper. Fallback, joint forecasting, and matched decision controls
are established ideas, not separate novelty claims here.

| Prior work | Decision and evidence | Difference in this study |
| --- | --- | --- |
| [Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html) | Cost-aware regression deferral among multiple experts | Edit a set of actor forecasts within a query and separate action capacity from the selector's realized utility. |
| [Xu et al., 2026](https://arxiv.org/html/2609.15801v1) | Fixed feasible proposals versus persistence; Theorem 2 concerns expected bounded loss in accepted groups under independent target-population calibration units | Keep CV as the fallback and OLS4 as a second comparator. The observed three-part recording risk rule is not a population guarantee. |
| [Wang, 2026](https://arxiv.org/html/2609.14997v1) | Preserve actor marginals and change forecast-mode pairing in a fixed offline AV planner; concentration changes as well | Hold the candidate paths, edit count and radius fixed, and compare recipient/forecast choices. The outcome is prediction error, not the chosen ego action. |
| [Xie and Alsinglawi, 2026](https://arxiv.org/html/2610.08322v1) | Graph-based accept/abstain on complete multivariate forecasts at matched retained coverage | Keep an output for each actor and allow a subset of forecast replacements with partial labels. |
| [Zeng et al., 2026](https://arxiv.org/abs/2609.18442) | Occupancy-informed selective replacement of an ego plan | No ego plan, action-conditioned rollouts, or collision-rate claim here. |

The comparison asks whether better decisions can recover utility from
the same available trajectories. The hindsight and arbitrary-path
bounds test whether that utility exists before crediting a learned
selector. The frozen negative result means the comparison is useful;
it does not show a deployable policy or a general advantage for
joint selection. None of the papers above has been rerun here.

The eight-page v169 candidate adds direct attribution for Xu and Wang.
Relative to v168, one introduction paragraph changes and two references
are appended. The title, abstract, scores, figures, original references
and disclosure stay unchanged. Pages 2 through 7 match pixel for pixel
at 90 DPI. The first and last pages were rendered and inspected. All
16 references are cited, both versions rebuild byte for byte, and four
focused tests passed.

```sh
# Use a Python environment with pypdf and a locally cached Tectonic bundle.
python scripts/build_m3w_related_work_candidate.py --compiler /path/to/tectonic
python -m unittest discover -s tests -p 'test_m3w_related_work_candidate.py'
```

Local candidate: `data/stage_cvpr2027_experiments/evidence_manuscript_v169_related_work/manuscript.pdf`.
The v168 PDF and supplementary ZIP remain intact. [Artifact receipt](receipt.json).
No new outcome access or upload.
