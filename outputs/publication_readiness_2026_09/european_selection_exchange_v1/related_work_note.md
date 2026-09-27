# Decision Quality Is Not Prediction MSE

## Material Passport

Primary-source reading, 2026-09-27. Read the problem formulation and ranking
method sections, not a reproduced public baseline. No author-read attestation.

[Mandi et al., ICML 2022, Decision-Focused Learning: Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html)
formulate decision-focused learning in terms of ranking feasible solutions
and study pointwise, pairwise and listwise surrogate objectives. Their
motivation distinguishes parameter prediction error from downstream regret.
[Primary paper, Sections 3-4](https://proceedings.mlr.press/v162/mandi22a/mandi22a.pdf).

Our decomposition is consistent with that distinction, but does not validate
their method in this application. Utility-aware allocation or a ranking loss
alone cannot be claimed as M3W novelty. The project still needs to establish
the value of source-separated risk control and joint multi-agent intervention
over strong compatible methods, with an independently frozen evaluation.

No optimization or statistical guarantee from this paper is transferred to
the proposed M3W allocator. Public method reproduction remains not_run here.
