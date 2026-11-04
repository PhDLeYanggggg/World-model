# Public Deferral Control: Equation Verified, Real Benchmark Not Yet Run

## Material Passport

Fresh primary-paper reading and a tested loss primitive while the registered
joint-harm study trains. This does not change that study, supply a new score,
or reproduce a published experimental result.

I obtained and read the primary ICML2024 PDF, pages1-7 including Section4,
Eq.(3) and Theorem4.1. The earlier web-tool MIME failure was an access issue,
not absence of the paper. The official proceedings link supplied the PDF;
local SHA256 is06d3b0b82937a157b23084ba8ba6c7110420edec0ecd7e545f57fee18a9a1ed5.
The third-party PDF stays outside Git.

Mao, Mohri and Zhong study regression routing to multiple experts, including
a fixed-predictor second stage. Their logistic surrogate weights each action's
classification loss by the sum of the other actions' costs. Their consistency
analysis assumes bounded regression and expert costs and suitable hypothesis
classes; it is not a guarantee of our recording-level risk constraints.
[Original paper](https://proceedings.mlr.press/v235/mao24d.html).

For a finite action bank with costs c, the implemented per-row primitive is
sum_k[(sum_j c_j-c_k)*(-log2 softmax(r)_k)]. There is no one-hot oracle label.
Synthetic tests verify the exact equation, its stationary population ordering,
and masking of unsupported costs. The causal legality mask permits variable
action banks as an engineering extension, not a proved theorem extension.
Singleton rows have no routing decision. Unknown outcomes are excluded from
this loss, never labelled safe or zero-error. Real data fitting is **not_run**:
the current registered experiment uses its original two losses only.

The next matched-control integration must fix the action roster, available-
label costs, fitting roles and budget prospectively. Unbounded pixel ADE,
partial futures, source dependence and additional relative-risk constraints
prevent automatically inheriting the paper's theorem. Do not clip costs or
change the main estimand merely to make that theorem applicable. This provides
a concrete task-matched public baseline interface, not a new M3W contribution
or an already outperformed comparator.

Decision calibration also separates loss estimation from ordinary probability
calibration. Its stated setting is multiclass labels and decisions based on
the predicted distribution; feature-dependent costs and our multi-agent action
combinations need separate treatment.
[Zhao et al., Sections2-3](https://proceedings.neurips.cc/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf).

LTT explicitly handles multiple, potentially non-monotone risks with held
calibration. Its IID calibration premise is not established by splitting these
dependent source views. The new factorized head is neither LTT nor a
distribution-free risk certificate.
[LTT, Sections1.1/2.4](https://arxiv.org/html/2110.01052v5).
